from .basics import GetDimensions, ListSubsetNode
from .batching import PaddedBatchImages, PaddedBatchListImages
from .composite import NODE_CLASS_MAPPINGS as COMPOSITE_NODE_CLASS_MAPPINGS
from .composite import (
    NODE_DISPLAY_NAME_MAPPINGS as COMPOSITE_NODE_DISPLAY_NAME_MAPPINGS,
)
from .grid_sampler import GridPreview, GridPreviewWidth, GridSampler
from .masking import (
    AddImagesWithAlpha,
    AttachMaskAsAlphaChannel,
    ConcatenateArrays,
    FillMasksWithColor,
    GetStatisticsForMasks,
    LoadMaskFromCSV,
    QuantizeColors,
    SaveMaskAsCSV,
    UnifyMask,
)

# Register Nodes
NODE_CLASS_MAPPINGS = {
    "PaddedBatchImages": PaddedBatchImages,
    "PaddedBatchListImages": PaddedBatchListImages,
    "AttachMaskAsAlpha": AttachMaskAsAlphaChannel,
    "GridSampler": GridSampler,
    "GridPreview": GridPreview,
    "ListSubsetNode": ListSubsetNode,
    "GetDimensions": GetDimensions,
    "GridPreviewWidth": GridPreviewWidth,
    "AddImagesWithAlpha": AddImagesWithAlpha,
    "UnifyMask": UnifyMask,
    "SaveMaskAsCSV": SaveMaskAsCSV,
    "LoadMaskFromCSV": LoadMaskFromCSV,
    "GetStatisticsForMasks": GetStatisticsForMasks,
    "FillMasksWithColor": FillMasksWithColor,
    "QuantizeColors": QuantizeColors,
    "ConcatenateArrays": ConcatenateArrays,
    **COMPOSITE_NODE_CLASS_MAPPINGS,
}
# Configure Names (e.g., for Search)
NODE_DISPLAY_NAME_MAPPINGS = {
    "PaddedBatchImages": "PaddedBatchImages",
    "PaddedBatchListImages": "PaddedBatchListImages",
    "AttachMaskAsAlpha": "Mask to Alpha",
    "GridSampler": "GridSampler",
    "GridPreview": "GridPreview",
    "ListSubsetNode": "ListSubsetNode",
    "GetDimensions": "GetDimensions",
    "GridPreviewWidth": "GridPreviewWidth",
    "AddImagesWithAlpha": "AddImagesWithAlpha",
    "UnifyMask": "UnifyMask",
    "SaveMaskAsCSV": "SaveMaskAsCSV",
    "LoadMaskFromCSV": "LoadMaskFromCSV",
    "GetStatisticsForMasks": "GetStatisticsForMasks",
    "FillMasksWithColor": "FillMasksWithColor",
    "QuantizeColors": "QuantizeColors",
    "ConcatenateArrays": "ConcatenateArrays",
    **COMPOSITE_NODE_DISPLAY_NAME_MAPPINGS,
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
