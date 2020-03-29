#include "baseline.h"
#include "load_data.h"

#include "cb_dsv_loader.h"
#include "data_provider_builders.h"

#include <catboost/libs/column_description/cd_parser.h>
#include <catboost/libs/data/quantization.h>
#include <catboost/libs/helpers/exception.h>
#include <catboost/libs/helpers/int_cast.h>
#include <catboost/libs/logging/logging.h>
#include <catboost/private/libs/data_util/exists_checker.h>
#include <catboost/private/libs/options/plain_options_helper.h>

#include <util/datetime/base.h>


namespace NCB {

    TDataProviderPtr ReadDataset(
        TMaybe<ETaskType> taskType,
        const TPathWithScheme& poolPath,
        const TPathWithScheme& pairsFilePath, // can be uninited
        const TPathWithScheme& groupWeightsFilePath, // can be uninited
        const TPathWithScheme& timestampsFilePath, // can be uninited
        const TPathWithScheme& baselineFilePath, // can be uninited
        const TPathWithScheme& featureNamesPath, // can be uninited
        const NCatboostOptions::TColumnarPoolFormatParams& columnarPoolFormatParams,
        const TVector<ui32>& ignoredFeatures,
        EObjectsOrder objectsOrder,
        TDatasetSubset loadSubset,
        TMaybe<TVector<NJson::TJsonValue>*> classLabels,
        NPar::TLocalExecutor* localExecutor
    ) {
        CB_ENSURE_INTERNAL(!baselineFilePath.Inited() || classLabels, "ClassLabels must be specified if baseline file is specified");
        if (classLabels) {
            UpdateClassLabelsFromBaselineFile(baselineFilePath, *classLabels);
        }
        auto datasetLoader = GetProcessor<IDatasetLoader>(
            poolPath, // for choosing processor

            // processor args
            TDatasetLoaderPullArgs {
                poolPath,

                TDatasetLoaderCommonArgs {
                    pairsFilePath,
                    groupWeightsFilePath,
                    baselineFilePath,
                    timestampsFilePath,
                    featureNamesPath,
                    classLabels ? **classLabels : TVector<NJson::TJsonValue>(),
                    columnarPoolFormatParams.DsvFormat,
                    MakeCdProviderFromFile(columnarPoolFormatParams.CdFilePath),
                    ignoredFeatures,
                    objectsOrder,
                    10000, // TODO: make it a named constant
                    loadSubset,
                    localExecutor
                }
            }
        );

        TDataProviderBuilderOptions builderOptions;
        builderOptions.GpuDistributedFormat = !loadSubset.HasFeatures && taskType && *taskType == ETaskType::GPU
            && EDatasetVisitorType::QuantizedFeatures == datasetLoader->GetVisitorType()
            && poolPath.Inited() && IsSharedFs(poolPath);
        builderOptions.PoolPath = poolPath;

        THolder<IDataProviderBuilder> dataProviderBuilder = CreateDataProviderBuilder(
            datasetLoader->GetVisitorType(),
            builderOptions,
            loadSubset,
            localExecutor
        );
        CB_ENSURE_INTERNAL(
            dataProviderBuilder,
            "Failed to create data provider builder for visitor of type " << datasetLoader->GetVisitorType()
        );

        datasetLoader->DoIfCompatible(dynamic_cast<IDatasetVisitor*>(dataProviderBuilder.Get()));
        return dataProviderBuilder->GetResult();
    }


    TDataProviderPtr ReadDataset(
        TMaybe<ETaskType> taskType,
        const TPathWithScheme& poolPath,
        const TPathWithScheme& pairsFilePath, // can be uninited
        const TPathWithScheme& groupWeightsFilePath, // can be uninited
        const TPathWithScheme& timestampsFilePath, // can be uninited
        const TPathWithScheme& baselineFilePath, // can be uninited
        const TPathWithScheme& featureNamesPath, // can be uninited
        const NCatboostOptions::TColumnarPoolFormatParams& columnarPoolFormatParams,
        const TVector<ui32>& ignoredFeatures,
        EObjectsOrder objectsOrder,
        int threadCount,
        bool verbose,
        TMaybe<TVector<NJson::TJsonValue>*> classLabels
    ) {
        NPar::TLocalExecutor localExecutor;
        localExecutor.RunAdditionalThreads(threadCount - 1);

        TSetLoggingVerboseOrSilent inThisScope(verbose);

        TDataProviderPtr dataProviderPtr = ReadDataset(
            taskType,
            poolPath,
            pairsFilePath,
            groupWeightsFilePath,
            timestampsFilePath,
            baselineFilePath,
            featureNamesPath,
            columnarPoolFormatParams,
            ignoredFeatures,
            objectsOrder,
            TDatasetSubset::MakeColumns(),
            classLabels,
            &localExecutor
        );

        return dataProviderPtr;
    }


    TDataProviderPtr ReadDataset(
        THolder<ILineDataReader> poolReader,
        const TPathWithScheme& pairsFilePath, // can be uninited
        const TPathWithScheme& groupWeightsFilePath, // can be uninited
        const TPathWithScheme& timestampsFilePath, // can be uninited
        const TPathWithScheme& baselineFilePath, // can be uninited
        const TPathWithScheme& featureNamesPath, // can be uninited
        const TDsvFormatOptions& poolFormat,
        const TVector<TColumn>& columnsDescription, // TODO(smirnovpavel): TVector<EColumn>
        const TVector<ui32>& ignoredFeatures,
        EObjectsOrder objectsOrder,
        TMaybe<TVector<NJson::TJsonValue>*> classLabels,
        NPar::TLocalExecutor* localExecutor
    ) {
        const auto loadSubset = TDatasetSubset::MakeColumns();
        THolder<IDataProviderBuilder> dataProviderBuilder = CreateDataProviderBuilder(
            EDatasetVisitorType::RawObjectsOrder,
            TDataProviderBuilderOptions{},
            loadSubset,
            localExecutor
        );
        CB_ENSURE_INTERNAL(
            dataProviderBuilder,
            "Failed to create data provider builder for visitor of type RawObjectsOrder";
        );

        TCBDsvDataLoader datasetLoader(
            TLineDataLoaderPushArgs {
                std::move(poolReader),

                TDatasetLoaderCommonArgs {
                    pairsFilePath,
                    groupWeightsFilePath,
                    baselineFilePath,
                    timestampsFilePath,
                    featureNamesPath,
                    classLabels ? **classLabels : TVector<NJson::TJsonValue>(),
                    poolFormat,
                    MakeCdProviderFromArray(columnsDescription),
                    ignoredFeatures,
                    objectsOrder,
                    10000, // TODO: make it a named constant
                    loadSubset,
                    localExecutor
                }
            }
        );
        datasetLoader.DoIfCompatible(dynamic_cast<IDatasetVisitor*>(dataProviderBuilder.Get()));
        return dataProviderBuilder->GetResult();
    }

    TDataProviders ReadTrainDatasets(
        TMaybe<ETaskType> taskType,
        const NCatboostOptions::TPoolLoadParams& loadOptions,
        EObjectsOrder objectsOrder,
        bool readTestData,
        TDatasetSubset trainDatasetSubset,
        TMaybe<TVector<NJson::TJsonValue>*> classLabels,
        NPar::TLocalExecutor* const executor,
        TProfileInfo* const profile
    ) {
        if (readTestData) {
            loadOptions.Validate();
        } else {
            loadOptions.ValidateLearn();
        }

        TDataProviders dataProviders;

        if (loadOptions.LearnSetPath.Inited()) {
            CATBOOST_DEBUG_LOG << "Loading features..." << Endl;
            auto start = Now();
            dataProviders.Learn = ReadDataset(
                taskType,
                loadOptions.LearnSetPath,
                loadOptions.PairsFilePath,
                loadOptions.GroupWeightsFilePath,
                loadOptions.TimestampsFilePath,
                loadOptions.BaselineFilePath,
                loadOptions.FeatureNamesPath,
                loadOptions.ColumnarPoolFormatParams,
                loadOptions.IgnoredFeatures,
                objectsOrder,
                trainDatasetSubset,
                classLabels,
                executor
            );
            CATBOOST_DEBUG_LOG << "Loading features time: " << (Now() - start).Seconds() << Endl;
            if (profile) {
                profile->AddOperation("Build learn pool");
            }
        }
        dataProviders.Test.resize(0);

        if (readTestData) {
            CATBOOST_DEBUG_LOG << "Loading test..." << Endl;
            for (int testIdx = 0; testIdx < loadOptions.TestSetPaths.ysize(); ++testIdx) {
                const NCB::TPathWithScheme& testSetPath = loadOptions.TestSetPaths[testIdx];
                const NCB::TPathWithScheme& testPairsFilePath =
                        testIdx == 0 ? loadOptions.TestPairsFilePath : NCB::TPathWithScheme();
                const NCB::TPathWithScheme& testGroupWeightsFilePath =
                    testIdx == 0 ? loadOptions.TestGroupWeightsFilePath : NCB::TPathWithScheme();
                const NCB::TPathWithScheme& testTimestampsFilePath =
                    testIdx == 0 ? loadOptions.TestTimestampsFilePath : NCB::TPathWithScheme();
                const NCB::TPathWithScheme& testBaselineFilePath =
                    testIdx == 0 ? loadOptions.TestBaselineFilePath : NCB::TPathWithScheme();

                TDataProviderPtr testDataProvider = ReadDataset(
                    taskType,
                    testSetPath,
                    testPairsFilePath,
                    testGroupWeightsFilePath,
                    testTimestampsFilePath,
                    testBaselineFilePath,
                    loadOptions.FeatureNamesPath,
                    loadOptions.ColumnarPoolFormatParams,
                    loadOptions.IgnoredFeatures,
                    objectsOrder,
                    TDatasetSubset::MakeColumns(),
                    classLabels,
                    executor
                );
                dataProviders.Test.push_back(std::move(testDataProvider));
                if (profile && (testIdx + 1 == loadOptions.TestSetPaths.ysize())) {
                    profile->AddOperation("Build test pool");
                }
            }
        }

        return dataProviders;
    }


    TDataProviderPtr ReadAndQuantizeDataset(
        TMaybe<ETaskType> taskType,
        const TPathWithScheme& poolPath,
        const TPathWithScheme& pairsFilePath, // can be uninited
        const TPathWithScheme& groupWeightsFilePath, // can be uninited
        const TPathWithScheme& timestampsFilePath, // can be uninited
        const TPathWithScheme& baselineFilePath, // can be uninited
        const TPathWithScheme& featureNamesPath, // can be uninited
        const NCatboostOptions::TColumnarPoolFormatParams& columnarPoolFormatParams,
        const TVector<ui32>& ignoredFeatures,
        EObjectsOrder objectsOrder,
        NJson::TJsonValue plainJsonParams,
        TQuantizedFeaturesInfoPtr quantizedFeaturesInfo,
        TDatasetSubset loadSubset,
        TMaybe<TVector<NJson::TJsonValue>*> classLabels,
        NPar::TLocalExecutor* localExecutor
    ) {
        const ui32 kBufferSize = 10;
        const ui32 kReservoirMaxSize = 50;
        CB_ENSURE_INTERNAL(!baselineFilePath.Inited() || classLabels, "ClassLabels must be specified if baseline file is specified");
        if (classLabels) {
            UpdateClassLabelsFromBaselineFile(baselineFilePath, *classLabels);
        }

        const auto create_loader = [&] {
            return GetProcessor<IDatasetLoader>(
                poolPath, // for choosing processor

                // processor args
                TDatasetLoaderPullArgs {
                    poolPath,

                    TDatasetLoaderCommonArgs {
                        pairsFilePath,
                        groupWeightsFilePath,
                        baselineFilePath,
                        timestampsFilePath,
                        featureNamesPath,
                        classLabels ? **classLabels : TVector<NJson::TJsonValue>(),
                        columnarPoolFormatParams.DsvFormat,
                        MakeCdProviderFromFile(columnarPoolFormatParams.CdFilePath),
                        ignoredFeatures,
                        objectsOrder,
                        kBufferSize,
                        loadSubset,
                        localExecutor
                    }
                }
            );
        };

        auto datasetLoader = create_loader();

        CB_ENSURE(
            EDatasetVisitorType::QuantizedFeatures != datasetLoader->GetVisitorType(),
            "Data is already quantized"
        );

        auto* rawObjectsOrderDatasetLoader =
            dynamic_cast<NCB::IRawObjectsOrderDatasetLoader*>(datasetLoader.Get());

        if (!rawObjectsOrderDatasetLoader) {
            TDataProviderBuilderOptions builderOptions;
            builderOptions.GpuDistributedFormat = !loadSubset.HasFeatures && taskType && *taskType == ETaskType::GPU
                                                  && EDatasetVisitorType::QuantizedFeatures == datasetLoader->GetVisitorType()
                                                  && poolPath.Inited() && IsSharedFs(poolPath);
            builderOptions.PoolPath = poolPath;

            THolder<IDataProviderBuilder> dataProviderBuilder = CreateDataProviderBuilder(
                datasetLoader->GetVisitorType(),
                builderOptions,
                loadSubset,
                localExecutor
            );
            CB_ENSURE_INTERNAL(
                dataProviderBuilder,
                "Failed to create data provider builder for visitor of type " << datasetLoader->GetVisitorType()
            );

            datasetLoader->DoIfCompatible(
                dynamic_cast<IDatasetVisitor*>(dataProviderBuilder.Get()));
            TDataProviderPtr dataProviderPtr = dataProviderBuilder->GetResult();
            dataProviderPtr.Get()->ObjectsData = ConstructQuantizedPoolFromRawPool(
                dataProviderPtr, std::move(plainJsonParams), std::move(quantizedFeaturesInfo));
            return dataProviderPtr;
        } else {
            NJson::TJsonValue jsonParams;
            NJson::TJsonValue outputJsonParams;
            NCatboostOptions::PlainJsonToOptions(plainJsonParams, &jsonParams, &outputJsonParams);
            NCatboostOptions::TCatBoostOptions catBoostOptions(NCatboostOptions::LoadOptions(jsonParams));

            TDataProviderBuilderOptions builderOptions;
            builderOptions.GpuDistributedFormat = !loadSubset.HasFeatures && taskType && *taskType == ETaskType::GPU
                                                  && EDatasetVisitorType::QuantizedFeatures == datasetLoader->GetVisitorType()
                                                  && poolPath.Inited() && IsSharedFs(poolPath);
            builderOptions.PoolPath = poolPath;
            builderOptions.SampleDataset = true;
            builderOptions.SampleSize = kReservoirMaxSize;
            builderOptions.SampleSeed = catBoostOptions.RandomSeed;

            THolder<IDataProviderBuilder> samplingBuilder = CreateDataProviderBuilder(
                datasetLoader->GetVisitorType(),
                builderOptions,
                loadSubset,
                localExecutor
            );
            CB_ENSURE_INTERNAL(
                samplingBuilder,
                "Failed to create data provider builder for visitor of type " << datasetLoader->GetVisitorType()
            );

            datasetLoader->DoIfCompatible(dynamic_cast<IDatasetVisitor*>(samplingBuilder.Get()));
            return samplingBuilder->GetResult();

            //            auto secondPassDatasetLoader = create_loader();
//
//            CB_ENSURE_INTERNAL(datasetLoader->GetVisitorType() == EDatasetVisitorType::RawObjectsOrder,
//                "Loader has unsupported visitor type: " << datasetLoader->GetVisitorType());
//            CB_ENSURE_INTERNAL(datasetLoader->GetVisitorType() == EDatasetVisitorType::RawObjectsOrder,
//                "Second loader has unsupported visitor type: " << secondPassDatasetLoader->GetVisitorType());
//
//            auto* secondPassRawObjectsOrderDatasetLoader =
//                dynamic_cast<NCB::IRawObjectsOrderDatasetLoader*>(secondPassDatasetLoader.Get());
//
//            CB_ENSURE_INTERNAL(secondPassRawObjectsOrderDatasetLoader != nullptr,
//                "Second loader cannot be converted to first loader format");
        }
    }


    TDataProviderPtr ReadAndQuantizeDataset(
        TMaybe<ETaskType> taskType,
        const TPathWithScheme& poolPath,
        const TPathWithScheme& pairsFilePath, // can be uninited
        const TPathWithScheme& groupWeightsFilePath, // can be uninited
        const TPathWithScheme& timestampsFilePath, // can be uninited
        const TPathWithScheme& baselineFilePath, // can be uninited
        const TPathWithScheme& featureNamesPath, // can be uninited
        const NCatboostOptions::TColumnarPoolFormatParams& columnarPoolFormatParams,
        const TVector<ui32>& ignoredFeatures,
        EObjectsOrder objectsOrder,
        NJson::TJsonValue plainJsonParams,
        TQuantizedFeaturesInfoPtr quantizedFeaturesInfo,
        int threadCount,
        bool verbose,
        TMaybe<TVector<NJson::TJsonValue>*> classLabels
    ) {
        NPar::TLocalExecutor localExecutor;
        localExecutor.RunAdditionalThreads(threadCount - 1);

        TSetLoggingVerboseOrSilent inThisScope(verbose);

        TDataProviderPtr dataProviderPtr = ReadAndQuantizeDataset(
            taskType,
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
            &localExecutor
        );

        return dataProviderPtr;
    }

} // NCB
