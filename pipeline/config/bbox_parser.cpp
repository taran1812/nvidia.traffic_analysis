#include "nvdsinfer_custom_impl.h"
#include <vector>

extern "C" bool NvDsInferParseCustomYoloV8Stub(
    std::vector<NvDsInferLayerInfo> const& outputLayersInfo,
    NvDsInferNetworkInfo const& networkInfo,
    NvDsInferParseDetectionParams const& detectionParams,
    std::vector<NvDsInferParseObjectInfo>& objectList)
{
    // Post-processing handled in Python probe via output-tensor-meta.
    objectList.clear();
    return true;
}
