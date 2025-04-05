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
    Composites a batch of overlays onto a single base image using alpha blending.

    - base: (1, H, W, C) PyTorch tensor (float in range [0,1])
    - overlays: (B, H, W, C) PyTorch tensor (float in range [0,1])

    The last overlay in the batch takes precedence in case of overlap.

    Returns:
        - Composite image of shape (1, H, W, C)
    """
    assert overlays.ndim == 4, "Overlays must have shape (B, H, W, C)"

    if base.ndim == 3:
        base = base.unsqueeze(0)

    if base.shape[-1] == 3:
        base = torch.cat([base, torch.ones_like(base[..., :1])], dim=-1)

    output = base.clone().squeeze(0)  # (H, W, 4)

    for overlay in overlays:
        rgb_overlay = overlay[..., :3]
        if overlay.shape[-1] == 3:
            overlay = torch.cat([overlay, torch.ones_like(overlay[..., :1])], dim=-1)
        # Ensure overlay has an alpha channel
        if overlay.shape[-1] != 4:
            raise ValueError("Overlay must have 4 channels (RGBA)")
        alpha_overlay = overlay[..., 3:4]

        rgb_output = output[..., :3]
        alpha_output = output[..., 3:4]

        # Alpha compositing formula
        alpha_comp = alpha_overlay + alpha_output * (1 - alpha_overlay)
        rgb_comp = (
            rgb_overlay * alpha_overlay
            + rgb_output * alpha_output * (1 - alpha_overlay)
        ) / torch.clamp(alpha_comp, min=1e-6)

        output[..., :3] = rgb_comp
        output[..., 3:4] = alpha_comp

    return output.unsqueeze(0)


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

    def composite_image(self, base_image: torch.Tensor, layers: torch.Tensor):
        return (composite_torch(base_image, layers),)


NODE_CLASS_MAPPINGS = {
    "CompositeImage": CompositeImage,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CompositeImage": "Composite Image",
}
