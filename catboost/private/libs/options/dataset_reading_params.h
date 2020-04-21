#pragma once

#include "load_options.h"

#include <catboost/private/libs/data_util/path_with_scheme.h>

#include <library/json/json_value.h>

#include <util/generic/vector.h>
#include <util/system/types.h>

namespace NLastGetopt {
    class TOpts;
}

namespace NCatboostOptions {
    struct TDatasetReadingParams {
        NCatboostOptions::TColumnarPoolFormatParams ColumnarPoolFormatParams;

        NCB::TPathWithScheme PoolPath;

        TVector<NJson::TJsonValue> ClassLabels;

        NCB::TPathWithScheme PairsFilePath;
        NCB::TPathWithScheme FeatureNamesPath;

        TVector<ui32> IgnoredFeatures;

        void BindParserOpts(NLastGetopt::TOpts* parser);

        void ValidatePoolParams() const;
    };

    void BindColumnarPoolFormatParams(
        NLastGetopt::TOpts* parser,
        NCatboostOptions::TColumnarPoolFormatParams* columnarPoolFormatParams);
}
