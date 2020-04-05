#include "baseline.h"
#include "load_and_quantize_data.h"

#include "data_provider_builders.h"
#include "feature_names_converter.h"

#include <catboost/libs/column_description/cd_parser.h>
#include <catboost/libs/data/quantization.h>
#include <catboost/libs/data/borders_io.h>
#include <catboost/libs/helpers/exception.h>
#include <catboost/libs/helpers/int_cast.h>
#include <catboost/libs/helpers/sample.h>
#include <catboost/libs/logging/logging.h>
#include <catboost/private/libs/app_helpers/proceed_pool_in_blocks.h>
#include <catboost/private/libs/options/plain_options_helper.h>
#include <catboost/private/libs/quantization/utils.h>

namespace NCB {
    struct TUnsampledData {
        // unsampled data from external data sources
        TMaybeData<TVector<float>> GroupWeights;
        TMaybeData<TVector<TVector<float>>> MultidimBaseline;
        TMaybeData<TVector<TPair>> Pairs;
        TMaybeData<TVector<ui64>> Timestamps;
    };

    struct TFirstPassQuantizationResult {
        ui32 ObjectCount;
        TRawDataProviderPtr SampleDataProvider;
        TQuantizedFeaturesInfoPtr QuantizedFeaturesInfo;
        TUnsampledData UnsampledData;
    };

    void CheckFeaturesLayoutForUnsupportedFeatures(TFeaturesLayoutPtr featuresLayout) {
        CB_ENSURE(
            !featuresLayout->GetCatFeatureCount(),
            "Categorical features are not supported in block quantization");
        CB_ENSURE(
            !featuresLayout->GetTextFeatureCount(),
            "Categorical features are not supported in block quantization");
        const auto expectedFeatures = featuresLayout->GetFloatFeatureCount();
        CB_ENSURE_INTERNAL(
            featuresLayout->GetExternalFeatureCount() == expectedFeatures,
            "Found unknown features, which are not supported in block quantization");
    }

    class TRawObjectsOrderFirstPassQuantizationVisitor : public IRawObjectsOrderDataVisitor {
    public:
        TRawObjectsOrderFirstPassQuantizationVisitor(
            NJson::TJsonValue plainJsonParams,
            TQuantizedFeaturesInfoPtr quantizedFeaturesInfo,
            THolder<IDataProviderBuilder> builder,
            TRestorableFastRng64* rand,
            NPar::TLocalExecutor* localExecutor)
            : LocalExecutor(localExecutor)
            , PlainJsonParams(std::move(plainJsonParams))
            , QuantizedFeaturesInfo(std::move(quantizedFeaturesInfo))
            , DataBuilder(std::move(builder))
            , DataVisitor(dynamic_cast<NCB::IRawObjectsOrderDataVisitor*>(DataBuilder.Get()))
            , Rand(rand)
            , SampleInvertedSubset(TFullSubset<ui32>(0)) {
            CB_ENSURE_INTERNAL(DataBuilder != nullptr, "no builder provided");
            CB_ENSURE_INTERNAL(
                DataVisitor != nullptr, "failed to cast from IDataProviderBuilder to IRawObjectsOrderDataVisitor");
        }

        void Start(
            bool inBlock,
            const TDataMetaInfo& metaInfo,
            bool haveUnknownNumberOfSparseFeatures,
            ui32 objectCount,
            EObjectsOrder objectsOrder,

            // keep necessary resources for data to be available (memory mapping for a file for example)
            TVector<TIntrusivePtr<IResourceHolder>> resourceHolders) override {
            CB_ENSURE(!inBlock, "block read is not supported, when sampling");

            CB_ENSURE_INTERNAL(metaInfo.FeaturesLayout, "feature layout is unknown on builder start");
            CheckFeaturesLayoutForUnsupportedFeatures(metaInfo.FeaturesLayout);

            CB_ENSURE_INTERNAL(!IsStarted, "started twice");
            IsStarted = true;

            ObjectCount = objectCount;
            CB_ENSURE(ObjectCount > 0, "pool is empty");

            QuantizationOptions = ConstructQuantizationOptions(
                PlainJsonParams, metaInfo, /* bordersFile */ Nothing(), QuantizedFeaturesInfo);

            CB_ENSURE_INTERNAL(
                *metaInfo.FeaturesLayout.Get() == *QuantizedFeaturesInfo->GetFeaturesLayout().Get(),
                "features layout is changed, while constructing quantization options");

            SampleSubset = MakeIncrementalIndexing(
                GetArraySubsetForBuildBorders(
                    ObjectCount,
                    QuantizedFeaturesInfo->GetFloatFeatureBinarization(Max<ui32>()).BorderSelectionType,
                    objectsOrder == EObjectsOrder::RandomShuffled,
                    QuantizationOptions.MaxSubsetSizeForBuildBordersAlgorithms,
                    Rand),
                LocalExecutor);
            SampleInvertedSubset = GetInvertedIndexing(SampleSubset, ObjectCount, LocalExecutor);
            IsFullSubset = SampleSubset.IsFullSubset();
            if (!IsFullSubset) {
                auto* indexedSubset = ::GetIf<TInvertedIndexedSubset<ui32>>(&SampleInvertedSubset);
                CB_ENSURE_INTERNAL(indexedSubset != nullptr, "inverted subset should be either indexed, or full");
                GlobalIdxToSampleIdx = indexedSubset->GetMapping();
            }
            SampleCount = SampleSubset.Size();

            Cursor = NotSet;
            NextCursor = 0;

            NCB::PrepareForInitialization(metaInfo.HasGroupId, ObjectCount, 0, &AllGroupIds);

            DataVisitor->Start(
                inBlock, metaInfo, haveUnknownNumberOfSparseFeatures, SampleCount, objectsOrder, resourceHolders);

            DataVisitor->StartNextBlock(SampleCount);
        }

        void StartNextBlock(ui32 blockSize) override {
            Cursor = NextCursor;
            NextCursor = Cursor + blockSize;
        }

    private:
        ui32 GetSampleIdx(ui32 localObjectIdx) {
            ui32 globalObjectIdx = Cursor + localObjectIdx;
            if (IsFullSubset) {
                return globalObjectIdx;
            }
            return GlobalIdxToSampleIdx[globalObjectIdx];
        }

    public:
        void AddGroupId(ui32 localObjectIdx, TGroupId value) override {
            (*AllGroupIds)[Cursor + localObjectIdx] = value;

            const ui32 sampleIdx = GetSampleIdx(localObjectIdx);
            if (sampleIdx == NotSet) {
                return;
            }
            DataVisitor->AddGroupId(sampleIdx, value);
        }

        void AddSubgroupId(ui32 localObjectIdx, TSubgroupId value) override {
            const ui32 sampleIdx = GetSampleIdx(localObjectIdx);
            if (sampleIdx == NotSet) {
                return;
            }
            DataVisitor->AddSubgroupId(sampleIdx, value);
        }

        void AddTimestamp(ui32 localObjectIdx, ui64 value) override {
            const ui32 sampleIdx = GetSampleIdx(localObjectIdx);
            if (sampleIdx == NotSet) {
                return;
            }
            DataVisitor->AddTimestamp(sampleIdx, value);
        }

        void AddFloatFeature(ui32 localObjectIdx, ui32 flatFeatureIdx, float feature) override {
            const ui32 sampleIdx = GetSampleIdx(localObjectIdx);
            if (sampleIdx == NotSet) {
                return;
            }
            DataVisitor->AddFloatFeature(sampleIdx, flatFeatureIdx, feature);
        }

        void AddAllFloatFeatures(ui32 localObjectIdx, TConstArrayRef<float> features) override {
            const ui32 sampleIdx = GetSampleIdx(localObjectIdx);
            if (sampleIdx == NotSet) {
                return;
            }
            DataVisitor->AddAllFloatFeatures(sampleIdx, features);
        }

        void
        AddAllFloatFeatures(ui32 localObjectIdx, TConstPolymorphicValuesSparseArray<float, ui32> features) override {
            const ui32 sampleIdx = GetSampleIdx(localObjectIdx);
            if (sampleIdx == NotSet) {
                return;
            }
            DataVisitor->AddAllFloatFeatures(sampleIdx, std::move(features));
        }

        ui32 GetCatFeatureValue(ui32 flatFeatureIdx, TStringBuf feature) override {
            // in this case it's impossible to detect, if feature is from skipped object
            return DataVisitor->GetCatFeatureValue(flatFeatureIdx, feature);
        }

        ui32 GetCatFeatureValue(ui32 localObjectIdx, ui32 flatFeatureIdx, TStringBuf feature) override {
            const ui32 sampleIdx = GetSampleIdx(localObjectIdx);
            if (sampleIdx == NotSet) {
                // value here doesn't matter for skipped objects
                return 0;
            }
            return DataVisitor->GetCatFeatureValue(flatFeatureIdx, feature);
        }

        void AddCatFeature(ui32 localObjectIdx, ui32 flatFeatureIdx, TStringBuf feature) override {
            const ui32 sampleIdx = GetSampleIdx(localObjectIdx);
            if (sampleIdx == NotSet) {
                return;
            }
            DataVisitor->AddCatFeature(sampleIdx, flatFeatureIdx, feature);
        }

        void AddAllCatFeatures(ui32 localObjectIdx, TConstArrayRef<ui32> features) override {
            const ui32 sampleIdx = GetSampleIdx(localObjectIdx);
            if (sampleIdx == NotSet) {
                return;
            }
            DataVisitor->AddAllCatFeatures(sampleIdx, features);
        }

        void AddAllCatFeatures(ui32 localObjectIdx, TConstPolymorphicValuesSparseArray<ui32, ui32> features) override {
            const ui32 sampleIdx = GetSampleIdx(localObjectIdx);
            if (sampleIdx == NotSet) {
                return;
            }
            DataVisitor->AddAllCatFeatures(sampleIdx, std::move(features));
        }

        void AddCatFeatureDefaultValue(ui32 flatFeatureIdx, TStringBuf feature) override {
            DataVisitor->AddCatFeatureDefaultValue(flatFeatureIdx, feature);
        }

        void AddTextFeature(ui32 localObjectIdx, ui32 flatFeatureIdx, TStringBuf feature) override {
            const ui32 sampleIdx = GetSampleIdx(localObjectIdx);
            if (sampleIdx == NotSet) {
                return;
            }
            DataVisitor->AddTextFeature(sampleIdx, flatFeatureIdx, feature);
        }

        void AddTextFeature(ui32 localObjectIdx, ui32 flatFeatureIdx, const TString& feature) override {
            const ui32 sampleIdx = GetSampleIdx(localObjectIdx);
            if (sampleIdx == NotSet) {
                return;
            }
            DataVisitor->AddTextFeature(sampleIdx, flatFeatureIdx, feature);
        }

        void AddAllTextFeatures(ui32 localObjectIdx, TConstArrayRef<TString> features) override {
            const ui32 sampleIdx = GetSampleIdx(localObjectIdx);
            if (sampleIdx == NotSet) {
                return;
            }
            DataVisitor->AddAllTextFeatures(sampleIdx, features);
        }

        void
        AddAllTextFeatures(ui32 localObjectIdx, TConstPolymorphicValuesSparseArray<TString, ui32> features) override {
            const ui32 sampleIdx = GetSampleIdx(localObjectIdx);
            if (sampleIdx == NotSet) {
                return;
            }
            DataVisitor->AddAllTextFeatures(sampleIdx, std::move(features));
        }

        void AddTarget(ui32 localObjectIdx, const TString& value) override {
            const ui32 sampleIdx = GetSampleIdx(localObjectIdx);
            if (sampleIdx == NotSet) {
                return;
            }
            DataVisitor->AddTarget(sampleIdx, value);
        }

        void AddTarget(ui32 localObjectIdx, float value) override {
            const ui32 sampleIdx = GetSampleIdx(localObjectIdx);
            if (sampleIdx == NotSet) {
                return;
            }
            DataVisitor->AddTarget(sampleIdx, value);
        }

        void AddTarget(ui32 flatTargetIdx, ui32 localObjectIdx, const TString& value) override {
            const ui32 sampleIdx = GetSampleIdx(localObjectIdx);
            if (sampleIdx == NotSet) {
                return;
            }
            DataVisitor->AddTarget(flatTargetIdx, sampleIdx, value);
        }

        void AddTarget(ui32 flatTargetIdx, ui32 localObjectIdx, float value) override {
            const ui32 sampleIdx = GetSampleIdx(localObjectIdx);
            if (sampleIdx == NotSet) {
                return;
            }
            DataVisitor->AddTarget(flatTargetIdx, sampleIdx, value);
        }

        void AddBaseline(ui32 localObjectIdx, ui32 baselineIdx, float value) override {
            const ui32 sampleIdx = GetSampleIdx(localObjectIdx);
            if (sampleIdx == NotSet) {
                return;
            }
            DataVisitor->AddTarget(sampleIdx, baselineIdx, value);
        }

        void AddWeight(ui32 localObjectIdx, float value) override {
            const ui32 sampleIdx = GetSampleIdx(localObjectIdx);
            if (sampleIdx == NotSet) {
                return;
            }
            DataVisitor->AddWeight(sampleIdx, value);
        }

        void AddGroupWeight(ui32 localObjectIdx, float value) override {
            const ui32 sampleIdx = GetSampleIdx(localObjectIdx);
            if (sampleIdx == NotSet) {
                return;
            }
            DataVisitor->AddWeight(sampleIdx, value);
        }

        void SetGroupWeights(TVector<float>&& groupWeights) override {
            // TODO(vetaleha): fill for sample, if needed for quantization
            UnsampledData.GroupWeights = std::move(groupWeights);
        }

        void SetBaseline(TVector<TVector<float>>&& multidimBaseline) override {
            // TODO(vetaleha): fill for sample, if needed for quantization
            UnsampledData.MultidimBaseline = std::move(multidimBaseline);
        }

        void SetPairs(TVector<TPair>&& pairs) override {
            // TODO(vetaleha): fill for sample, if needed for quantization
            UnsampledData.Pairs = std::move(pairs);
        }

        void SetTimestamps(TVector<ui64>&& timestamps) override {
            // TODO(vetaleha): fill for sample, if needed for quantization
            UnsampledData.Timestamps = std::move(timestamps);
        }

        TMaybeData<TConstArrayRef<TGroupId>> GetGroupIds() const override {
            return AllGroupIds;
        }

        void Finish() override {
            DataVisitor->Finish();
        }

        TFirstPassQuantizationResult GetFirstPassResult() {
            CB_ENSURE_INTERNAL(IsStarted, "First pass was not started by loader");
            CB_ENSURE_INTERNAL(
                !ResultsTaken, "TRawObjectsOrderFirstPassQuantizationVisitor::GetFirstPassResult called twice");
            ResultsTaken = true;

            TRawDataProviderPtr rawDataProvider;
            {
                auto dataProvider = DataBuilder->GetResult();
                CB_ENSURE_INTERNAL(dataProvider, "Failed to create data provider from raw objects builder");

                rawDataProvider = dataProvider->CastMoveTo<TRawObjectsDataProvider>();
                CB_ENSURE_INTERNAL(rawDataProvider, "Failed to cast data provider to TRawDataProviderPtr");
            }

            CB_ENSURE_INTERNAL(
                *rawDataProvider->MetaInfo.FeaturesLayout.Get() == *QuantizedFeaturesInfo->GetFeaturesLayout().Get(),
                "features layout was changed by builder");

            CalcBordersAndNanMode(QuantizationOptions, rawDataProvider, QuantizedFeaturesInfo, Rand, LocalExecutor);

            // TODO(vetaleha): build CatFeaturesPerfectHash, TextProcessingOptions and TextDigitizers

            rawDataProvider->MetaInfo.FeaturesLayout = QuantizedFeaturesInfo->GetFeaturesLayout();

            return TFirstPassQuantizationResult{
                ObjectCount, rawDataProvider, QuantizedFeaturesInfo, std::move(UnsampledData)};
        }

    private:
        bool IsStarted = false;
        bool ResultsTaken = false;

        NPar::TLocalExecutor* LocalExecutor;

        NJson::TJsonValue PlainJsonParams;
        TQuantizationOptions QuantizationOptions;
        TQuantizedFeaturesInfoPtr QuantizedFeaturesInfo;

        THolder<IDataProviderBuilder> DataBuilder;
        IRawObjectsOrderDataVisitor* DataVisitor;
        TRestorableFastRng64* Rand;

        ui32 ObjectCount;
        ui32 SampleCount;

        static constexpr const ui32 NotSet = TInvertedIndexedSubset<ui32>::NOT_PRESENT;

        ui32 Cursor;
        ui32 NextCursor;

        TArraySubsetIndexing<ui32> SampleSubset;
        TArraySubsetInvertedIndexing<ui32> SampleInvertedSubset;
        bool IsFullSubset;
        // mapping available only when SampleSubset is not full
        TConstArrayRef<ui32> GlobalIdxToSampleIdx;

        // group ids for all objects, required for GetGroupIds and data from external sources
        TMaybeData<TVector<TGroupId>> AllGroupIds;
        TUnsampledData UnsampledData;
    };

    class TSecondPassQuantizationBlockConsumer {
    public:
        TSecondPassQuantizationBlockConsumer(
            TFirstPassQuantizationResult firstPassResult,
            TDatasetSubset loadSubset,
            EObjectsOrder objectsOrder,
            NJson::TJsonValue plainJsonParams,
            TRestorableFastRng64* rand,
            NPar::TLocalExecutor* localExecutor)
            : LocalExecutor(localExecutor)
            , FirstPassResult(std::move(firstPassResult))
            , QuantizationOptions(ConstructQuantizationOptions(
                  std::move(plainJsonParams),
                  FirstPassResult.SampleDataProvider->MetaInfo,
                  /* bordersFile */ Nothing(),
                  FirstPassResult.QuantizedFeaturesInfo))
            , QuantizedDataBuilder(CreateDataProviderBuilder(
                  EDatasetVisitorType::QuantizedFeatures,
                  NCB::TDataProviderBuilderOptions{},
                  loadSubset,
                  LocalExecutor))
            , QuantizedDataVisitor(dynamic_cast<NCB::IQuantizedFeaturesDataVisitor*>(QuantizedDataBuilder.Get()))
            , ObjectsOrder(objectsOrder)
            , ObjectOffset(0)
            , Rand(rand) {
            CB_ENSURE_INTERNAL(
                QuantizedDataBuilder, "Failed to create data provider builder for QuantizedFeatures visitor");

            CB_ENSURE_INTERNAL(
                QuantizedDataVisitor, "failed cast of IDataProviderBuilder to IQuantizedFeaturesDataVisitor");
        }

        void ProcessBlock(TDataProviderPtr untypedDataBlock) {
            TRawDataProviderPtr dataBlock = untypedDataBlock->CastMoveTo<TRawObjectsDataProvider>();
            untypedDataBlock.Drop();
            CB_ENSURE_INTERNAL(dataBlock, "failed cast of TDataProvider to TRawDataProvider");

            TQuantizedObjectsDataProviderPtr quantizedObjectsData = Quantize(
                QuantizationOptions,
                dataBlock->ObjectsData,
                FirstPassResult.QuantizedFeaturesInfo,
                Rand,
                LocalExecutor);

            if (ObjectOffset == 0) {
                const auto& quantizedFeaturesInfo = quantizedObjectsData->GetQuantizedFeaturesInfo();
                const auto& featuresLayout = quantizedFeaturesInfo->GetFeaturesLayout();

                TVector<TVector<float>> borders;
                TVector<ENanMode> nanModes;
                TVector<size_t> floatFeatureIndices;
                TVector<size_t> catFeatureIndices;

                for (auto flatFeatureIdx : xrange(featuresLayout->GetExternalFeatureCount())) {
                    const auto featureMetaInfo = featuresLayout->GetExternalFeatureMetaInfo(flatFeatureIdx);

                    if (featureMetaInfo.Type == EFeatureType::Float) {
                        const auto floatFeatureIdx =
                            featuresLayout->GetInternalFeatureIdx<EFeatureType::Float>(flatFeatureIdx);

                        const auto featureBorders = quantizedFeaturesInfo->HasBorders(floatFeatureIdx)
                            ? quantizedFeaturesInfo->GetBorders(floatFeatureIdx)
                            : TVector<float>();
                        const auto featureNanMode = quantizedFeaturesInfo->HasNanMode(floatFeatureIdx)
                            ? quantizedFeaturesInfo->GetNanMode(floatFeatureIdx)
                            : ENanMode::Forbidden;
                        borders.push_back(featureBorders);
                        nanModes.push_back(featureNanMode);
                        floatFeatureIndices.push_back(flatFeatureIdx);
                    } else if (featureMetaInfo.Type == EFeatureType::Categorical) {
                        catFeatureIndices.push_back(flatFeatureIdx);
                    } else {
                        CB_ENSURE_INTERNAL(
                            false,
                            "building quantization results is supported only for numerical and categorical features");
                    }
                }
                QuantizedDataVisitor->Start(
                    FirstPassResult.SampleDataProvider->MetaInfo,
                    FirstPassResult.ObjectCount,
                    ObjectsOrder,
                    {},
                    TPoolQuantizationSchema{
                        std::move(floatFeatureIndices),
                        std::move(borders),
                        std::move(nanModes),
                        FirstPassResult.SampleDataProvider->MetaInfo.ClassLabels,
                        std::move(catFeatureIndices),
                        TVector<TMap<ui32, TValueWithCount>>() // TODO(vetaleha): build CatFeaturesPerfectHash
                    });
            }

            auto groupIds = quantizedObjectsData->GetGroupIds();
            if (groupIds) {
                QuantizedDataVisitor->AddGroupIdPart(ObjectOffset, TUnalignedArrayBuf<TGroupId>(*groupIds));
            }

            auto subgroupIds = quantizedObjectsData->GetSubgroupIds();
            if (subgroupIds) {
                QuantizedDataVisitor->AddSubgroupIdPart(ObjectOffset, TUnalignedArrayBuf<TSubgroupId>(*subgroupIds));
            }

            auto timestamps = quantizedObjectsData->GetTimestamp();
            if (timestamps) {
                QuantizedDataVisitor->AddTimestampPart(ObjectOffset, TUnalignedArrayBuf<ui64>(*timestamps));
            }

            const auto& quantizedFeaturesInfo = quantizedObjectsData->GetQuantizedFeaturesInfo();
            const auto& featuresLayout = quantizedFeaturesInfo->GetFeaturesLayout();
            for (auto flatFeatureIdx : xrange(featuresLayout->GetExternalFeatureCount())) {
                const auto featureMetaInfo = featuresLayout->GetExternalFeatureMetaInfo(flatFeatureIdx);

                if (featureMetaInfo.Type == EFeatureType::Float) {
                    const auto floatFeatureIdx =
                        featuresLayout->GetInternalFeatureIdx<EFeatureType::Float>(flatFeatureIdx);

                    TMaybeData<const IQuantizedFloatValuesHolder*> feature =
                        quantizedObjectsData->GetFloatFeature(*floatFeatureIdx);

                    if (feature) {
                        CB_ENSURE_INTERNAL(
                            feature.GetRef(),
                            "GetFloatFeature returned nullptr for feature " << flatFeatureIdx
                                                                            << " which is not ignored");

                        // add feature to builder
                        auto values = feature.GetRef()->ExtractValues(LocalExecutor);
                        QuantizedDataVisitor->AddFloatFeaturePart(
                            flatFeatureIdx,
                            ObjectOffset,
                            sizeof(IQuantizedFloatValuesHolder::TValueType) * 8,
                            TMaybeOwningConstArrayHolder<ui8>::CreateOwningReinterpretCast(values));
                    } else {
                        CB_ENSURE_INTERNAL(
                            featureMetaInfo.IsIgnored, "If GetFloatFeature returns Nothing(), feature must be ignored");
                    }
                } else if (featureMetaInfo.Type == EFeatureType::Categorical) {
                    const auto catFeatureIdx =
                        featuresLayout->GetInternalFeatureIdx<EFeatureType::Categorical>(flatFeatureIdx);

                    TMaybeData<const IQuantizedCatValuesHolder*> feature =
                        quantizedObjectsData->GetCatFeature(*catFeatureIdx);

                    if (feature) {
                        CB_ENSURE_INTERNAL(
                            feature.GetRef(),
                            "GetCatFeature returned nullptr for feature " << flatFeatureIdx << " which is not ignored");

                        // add feature to builder
                        auto values = feature.GetRef()->ExtractValues(LocalExecutor);
                        QuantizedDataVisitor->AddCatFeaturePart(
                            flatFeatureIdx,
                            ObjectOffset,
                            sizeof(IQuantizedCatValuesHolder::TValueType) * 8,
                            TMaybeOwningConstArrayHolder<ui8>::CreateOwningReinterpretCast(values));
                    } else {
                        CB_ENSURE_INTERNAL(
                            featureMetaInfo.IsIgnored, "If GetCatFeature returns Nothing(), feature must be ignored");
                    }
                } else {
                    CB_ENSURE_INTERNAL(
                        false,
                        "building quantization results is supported only for numerical and categorical features");
                }
            }

            const auto targetDimension = dataBlock->RawTargetData.GetTargetDimension();
            switch (dataBlock->RawTargetData.GetTargetType()) {
                case ERawTargetType::Integer:
                case ERawTargetType::Float: {
                    TVector<TVector<float>> targetNumeric(targetDimension);
                    TVector<TArrayRef<float>> targetNumericRefs(targetDimension);
                    for (auto targetIdx : xrange(targetDimension)) {
                        targetNumeric[targetIdx].yresize(dataBlock->RawTargetData.GetObjectCount());
                        targetNumericRefs[targetIdx] = targetNumeric[targetIdx];
                    }
                    dataBlock->RawTargetData.GetNumericTarget(targetNumericRefs);
                    for (auto targetIdx : xrange(targetDimension)) {
                        QuantizedDataVisitor->AddTargetPart(
                            targetIdx, ObjectOffset, TUnalignedArrayBuf<float>(targetNumericRefs[targetIdx]));
                    }
                } break;
                case ERawTargetType::String: {
                    TVector<TConstArrayRef<TString>> targetStrings;
                    dataBlock->RawTargetData.GetStringTargetRef(&targetStrings);
                    for (auto targetIdx : xrange(targetDimension)) {
                        QuantizedDataVisitor->AddTargetPart(
                            targetIdx,
                            ObjectOffset,
                            TMaybeOwningConstArrayHolder<TString>::CreateNonOwning(targetStrings[targetIdx]));
                    }
                } break;
                case ERawTargetType::None:
                    break;
            }

            const auto& maybeBaseline = dataBlock->RawTargetData.GetBaseline();
            if (maybeBaseline) {
                const auto& baseline = *maybeBaseline;
                for (size_t baselineIdx : xrange(baseline.size())) {
                    QuantizedDataVisitor->AddBaselinePart(
                        ObjectOffset, baselineIdx, TUnalignedArrayBuf<float>(baseline[baselineIdx]));
                }
            }

            // weights
            const auto& weights = dataBlock->RawTargetData.GetWeights();
            if (!weights.IsTrivial()) {
                QuantizedDataVisitor->AddWeightPart(
                    ObjectOffset, TUnalignedArrayBuf<float>(weights.GetNonTrivialData()));
            }

            // groupWeights
            const auto& groupWeights = dataBlock->RawTargetData.GetGroupWeights();
            if (!groupWeights.IsTrivial()) {
                QuantizedDataVisitor->AddGroupWeightPart(
                    ObjectOffset, TUnalignedArrayBuf<float>(groupWeights.GetNonTrivialData()));
            }

            ObjectOffset += dataBlock->GetObjectCount();
        }

        TDataProviderPtr GetResult() {
            CB_ENSURE_INTERNAL(!ResultsTaken, "TSecondPassQuantizationBlockConsumer::GetResult called twice");
            ResultsTaken = true;

            if (FirstPassResult.UnsampledData.GroupWeights) {
                QuantizedDataVisitor->SetGroupWeights(std::move(*FirstPassResult.UnsampledData.GroupWeights));
            }
            if (FirstPassResult.UnsampledData.MultidimBaseline) {
                QuantizedDataVisitor->SetBaseline(std::move(*FirstPassResult.UnsampledData.MultidimBaseline));
            }
            if (FirstPassResult.UnsampledData.Pairs) {
                QuantizedDataVisitor->SetPairs(std::move(*FirstPassResult.UnsampledData.Pairs));
            }
            if (FirstPassResult.UnsampledData.Timestamps) {
                QuantizedDataVisitor->SetTimestamps(std::move(*FirstPassResult.UnsampledData.Timestamps));
            }
            QuantizedDataVisitor->Finish();
            return QuantizedDataBuilder->GetResult();
        }

    private:
        bool ResultsTaken = false;

        NPar::TLocalExecutor* LocalExecutor;

        TFirstPassQuantizationResult FirstPassResult;
        TQuantizationOptions QuantizationOptions;
        THolder<IDataProviderBuilder> QuantizedDataBuilder;
        IQuantizedFeaturesDataVisitor* QuantizedDataVisitor;

        EObjectsOrder ObjectsOrder;
        ui32 ObjectOffset;

        TRestorableFastRng64* Rand;
    };

    TDataProviderPtr ReadAndQuantizeDataset(
        const TPathWithScheme& poolPath,
        const TPathWithScheme& pairsFilePath,        // can be uninited
        const TPathWithScheme& groupWeightsFilePath, // can be uninited
        const TPathWithScheme& timestampsFilePath,   // can be uninited
        const TPathWithScheme& baselineFilePath,     // can be uninited
        const TPathWithScheme& featureNamesPath,     // can be uninited
        const NCatboostOptions::TColumnarPoolFormatParams& columnarPoolFormatParams,
        const TVector<ui32>& ignoredFeatures,
        EObjectsOrder objectsOrder,
        NJson::TJsonValue plainJsonParams,
        TQuantizedFeaturesInfoPtr quantizedFeaturesInfo,
        TDatasetSubset loadSubset,
        TMaybe<TVector<NJson::TJsonValue>*> classLabels,
        NPar::TLocalExecutor* localExecutor) {
        const ui32 BLOCK_SIZE = 10000;

        TVector<NJson::TJsonValue> emptyClassLabels;
        CB_ENSURE_INTERNAL(
            !baselineFilePath.Inited() || classLabels, "ClassLabels must be specified if baseline file is specified");
        if (classLabels) {
            UpdateClassLabelsFromBaselineFile(baselineFilePath, *classLabels);
        } else {
            classLabels = &emptyClassLabels;
        }

        auto datasetLoader = GetProcessor<IDatasetLoader>(
            poolPath, // for choosing processor
            // processor args
            TDatasetLoaderPullArgs{poolPath,
                                   TDatasetLoaderCommonArgs{pairsFilePath,
                                                            groupWeightsFilePath,
                                                            baselineFilePath,
                                                            timestampsFilePath,
                                                            featureNamesPath,
                                                            **classLabels,
                                                            columnarPoolFormatParams.DsvFormat,
                                                            MakeCdProviderFromFile(columnarPoolFormatParams.CdFilePath),
                                                            ignoredFeatures,
                                                            objectsOrder,
                                                            BLOCK_SIZE,
                                                            loadSubset,
                                                            localExecutor}});

        CB_ENSURE(
            EDatasetVisitorType::QuantizedFeatures != datasetLoader->GetVisitorType(), "Data is already quantized");
        CB_ENSURE_INTERNAL(
            datasetLoader->GetVisitorType() == EDatasetVisitorType::RawObjectsOrder,
            "dataset should be loaded by RawObjectsOrder loader");

        NJson::TJsonValue jsonParams;
        NJson::TJsonValue outputJsonParams;
        NCatboostOptions::PlainJsonToOptions(plainJsonParams, &jsonParams, &outputJsonParams);
        NCatboostOptions::TCatBoostOptions catBoostOptions(NCatboostOptions::LoadOptions(jsonParams));

        TRestorableFastRng64 rand(catBoostOptions.RandomSeed);

        TRawObjectsOrderFirstPassQuantizationVisitor firstPassVisitor(
            plainJsonParams,
            quantizedFeaturesInfo,
            CreateDataProviderBuilder(
                datasetLoader->GetVisitorType(), TDataProviderBuilderOptions{}, loadSubset, localExecutor),
            &rand,
            localExecutor);
        datasetLoader->DoIfCompatible(&firstPassVisitor);

        TSecondPassQuantizationBlockConsumer secondPassQuantizer(
            firstPassVisitor.GetFirstPassResult(),
            loadSubset,
            objectsOrder,
            std::move(plainJsonParams),
            &rand,
            localExecutor);

        TAnalyticalModeCommonParams params;
        params.ColumnarPoolFormatParams = columnarPoolFormatParams;
        params.InputPath = poolPath;
        params.FeatureNamesPath = featureNamesPath;
        params.ClassLabels = **classLabels;
        ReadAndProceedPoolInBlocks(
            params,
            BLOCK_SIZE,
            [&](TDataProviderPtr dataBlock) { secondPassQuantizer.ProcessBlock(std::move(dataBlock)); },
            localExecutor);

        return secondPassQuantizer.GetResult();
    }

    TDataProviderPtr ReadAndQuantizeDataset(
        const TPathWithScheme& poolPath,
        const TPathWithScheme& pairsFilePath,        // can be uninited
        const TPathWithScheme& groupWeightsFilePath, // can be uninited
        const TPathWithScheme& timestampsFilePath,   // can be uninited
        const TPathWithScheme& baselineFilePath,     // can be uninited
        const TPathWithScheme& featureNamesPath,     // can be uninited
        const NCatboostOptions::TColumnarPoolFormatParams& columnarPoolFormatParams,
        const TVector<ui32>& ignoredFeatures,
        EObjectsOrder objectsOrder,
        NJson::TJsonValue plainJsonParams,
        TQuantizedFeaturesInfoPtr quantizedFeaturesInfo,
        int threadCount,
        bool verbose,
        TMaybe<TVector<NJson::TJsonValue>*> classLabels) {
        NPar::TLocalExecutor localExecutor;
        localExecutor.RunAdditionalThreads(threadCount - 1);

        TSetLoggingVerboseOrSilent inThisScope(verbose);

        TDataProviderPtr dataProviderPtr = ReadAndQuantizeDataset(
            poolPath,
            pairsFilePath,
            groupWeightsFilePath,
            timestampsFilePath,
            baselineFilePath,
            featureNamesPath,
            columnarPoolFormatParams,
            ignoredFeatures,
            objectsOrder,
            std::move(plainJsonParams),
            std::move(quantizedFeaturesInfo),
            TDatasetSubset::MakeColumns(),
            classLabels,
            &localExecutor);

        return dataProviderPtr;
    }

} // namespace NCB
