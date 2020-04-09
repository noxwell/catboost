#pragma once

#include "data_provider.h"
#include "loader.h"
#include "objects.h"

#include <catboost/private/libs/data_util/path_with_scheme.h>
#include <catboost/private/libs/options/load_options.h>

#include <library/threading/local_executor/local_executor.h>

#include <util/generic/maybe.h>
#include <util/generic/vector.h>
#include <util/system/types.h>


namespace NJson {
    class TJsonValue;
}


namespace NCB {
    // use from C++ code
    TDataProviderPtr ReadAndQuantizeDataset(
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
        TMaybe<ui32> blockSize,
        TQuantizedFeaturesInfoPtr quantizedFeaturesInfo,
        TDatasetSubset loadSubset,
        TMaybe<TVector<NJson::TJsonValue>*> classLabels,
        NPar::TLocalExecutor* localExecutor
    );

    // for use from context where there's no localExecutor and proper logging handling is unimplemented
    TDataProviderPtr ReadAndQuantizeDataset(
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
        TMaybe<ui32> blockSize,
        TQuantizedFeaturesInfoPtr quantizedFeaturesInfo,
        int threadCount,
        bool verbose,
        TMaybe<TVector<NJson::TJsonValue>*> classLabels = Nothing()
    );
}
