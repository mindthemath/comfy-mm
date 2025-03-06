# import torch
# import numpy as np
# import pyvips

# def torch_to_vips(image: torch.Tensor) -> pyvips.Image:
#     """
#     Converts a PyTorch tensor (H, W, C) to a PyVips image.
#     Assumes input tensor is in float format [0,1].
#     """
#     image_np = (image * 255).byte().numpy()  # Convert to uint8 for PyVips
#     height, width, channels = image_np.shape

#     if channels == 1:
#         return pyvips.Image.new_from_memory(image_np.tobytes(), width, height, channels, "uchar")
#     else:
#         return pyvips.Image.new_from_memory(image_np.transpose(2, 0, 1).tobytes(), width, height, channels, "uchar")

# def get_nonzero_mask(image: pyvips.Image) -> pyvips.Image:
#     """
#     Generates a binary mask (H, W, 1) from a PyVips image,
#     where any nonzero pixel is set to 255.
#     """
#     mask = image.bandjoin([image.bandbool("or")])  # Reduce to a single-channel mask
#     return mask[mask > 0].ifthenelse(255, 0)  # Convert to 0-255 binary

# def batch_composite(base: torch.Tensor, overlays: torch.Tensor) -> torch.Tensor:
#     """
#     Composites a batch of overlays onto a single base canvas efficiently.
#     - base: (B, H, W, C) PyTorch tensor
#     - overlays: (B, H, W, C) PyTorch tensor
#     """
#     if base.ndims == 4:
#         base = base.squeeze(0)
#     base_vips = torch_to_vips(base)

#     overlay_vips_list = []
#     mask_vips_list = []

#     for i in range(overlays.shape[0]):
#         overlay_vips = torch_to_vips(overlays[i])
#         mask_vips = get_nonzero_mask(overlay_vips)

#         overlay_vips_list.append(overlay_vips)
#         mask_vips_list.append(mask_vips)

#     # Stack all overlays and masks into single images
#     stacked_overlays = pyvips.Image.arrayjoin(overlay_vips_list, across=1)  # Join along horizontal axis
#     stacked_masks = pyvips.Image.arrayjoin(mask_vips_list, across=1)

#     # Efficiently apply all overlays using a single PyVips operation
#     final_image = base_vips.ifthenelse(stacked_overlays, base_vips, stacked_masks)

#     # Convert back to PyTorch tensor
#     base_np = np.frombuffer(final_image.write_to_memory(), dtype=np.uint8).reshape(
#         final_image.height, final_image.width, final_image.bands
#     )
#     return torch.from_numpy(base_np).float() / 255
import torch


def composite_torch(base: torch.Tensor, overlays: torch.Tensor) -> torch.Tensor:
    """
    Composites a batch of overlays onto a single base image.

    - base: (1, H, W, C) PyTorch tensor (assumed float in range [0,1])
    - overlays: (B, H, W, C) PyTorch tensor (assumed float in range [0,1])

    The last overlay in the batch takes precedence in case of overlap.

    Returns:
        - Composite image of shape (1, H, W, C)
    """
    if base.ndim == 4:
        base = base.squeeze(0)  # Remove batch dimension for processing

    output = base.clone()  # Clone to avoid modifying input

    for i in range(overlays.shape[0]):
        overlay = overlays[i]  # Shape (H, W, C)
        mask = overlay.sum(dim=-1, keepdim=True) > 0  # Nonzero mask (H, W, 1)

        # Expand mask to (1, H, W, 1) to match base shape
        mask = mask.unsqueeze(0)  # Add batch dimension back

        # Apply overlay where mask is True
        output = torch.where(mask, overlay, output)

    return output.unsqueeze(0)  # Restore batch dimension (1, H, W, C)


class CompositeImage:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "base_image": ("IMAGE", {}),  # Shape (1, H, W, C)
                "layers": ("IMAGE", {}),  # Shape (B, H, W, C)
            }
        }

    CATEGORY = "image_processing"
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "composite_image"

    def composite_image(
        self, base_image: torch.Tensor, layers: torch.Tensor
    ) -> torch.Tensor:
        return composite_torch(base_image, layers)


NODE_CLASS_MAPPINGS = {
    "CompositeImage": CompositeImage,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CompositeImage": "Composite Image",
}
