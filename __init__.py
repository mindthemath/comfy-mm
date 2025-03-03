from .basics import GetDimensions, ListSubsetNode
from .batching import PaddedBatchImages
from .grid_sampler import GridPreview, GridPreviewWidth, GridSampler
from .masking import AddImagesWithAlpha, AttachMaskAsAlphaChannel, UnifyMask

# Register Nodes
NODE_CLASS_MAPPINGS = {
    "PaddedBatchImages": PaddedBatchImages,
    "AttachMaskAsAlpha": AttachMaskAsAlphaChannel,
    "GridSampler": GridSampler,
    "GridPreview": GridPreview,
    "ListSubsetNode": ListSubsetNode,
    "GetDimensions": GetDimensions,
    "GridPreviewWidth": GridPreviewWidth,
    "AddImagesWithAlpha": AddImagesWithAlpha,
    "UnifyMask": UnifyMask,
}
# Configure Names (e.g., for Search)
NODE_DISPLAY_NAME_MAPPINGS = {
    "PaddedBatchImages": "Padded Batch Images",
    "AttachMaskAsAlpha": "Mask to Alpha",
    "GridSampler": "GridSampler",
    "GridPreview": "GridPreview",
    "ListSubsetNode": "ListSubsetNode",
    "GetDimensions": "GetDimensions",
    "GridPreviewWidth": "GridPreviewWidth",
    "AddImagesWithAlpha": "AddImagesWithAlpha",
    "UnifyMask": "UnifyMask",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
