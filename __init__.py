from .batching import PaddedBatchImages
from .grid_sampler import GridPreview, GridSampler, GridPreviewWidth
from .masking import AttachMaskAsAlphaChannel
from .basics import ListSubsetNode, GetDimensions

# Register Nodes
NODE_CLASS_MAPPINGS = {
    "PaddedBatchImages": PaddedBatchImages,
    "AttachMaskAsAlpha": AttachMaskAsAlphaChannel,
    "GridSampler": GridSampler,
    "GridPreview": GridPreview,
    "ListSubsetNode": ListSubsetNode,
    "GetDimensions": GetDimensions,
    "GridPreviewWidth": GridPreviewWidth,
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
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
