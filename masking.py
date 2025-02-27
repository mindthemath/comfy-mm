import torch


class AttachMaskAsAlphaChannel:  # custom version of JoinImageWithAlpha
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {}),
                "mask": ("MASK", {}),
            },
            "optional": {
                "alpha_value": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0})
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("IMAGE",)
    FUNCTION = "attach_mask"
    CATEGORY = "image"

    def attach_mask(
        self, image: torch.Tensor, mask: torch.Tensor, alpha_value: float = 0.0
    ):
        """
        Converts a 3-channel (RGB) image into a 4-channel (RGBA) image by attaching
        the provided mask as the alpha channel. The alpha is scaled by alpha_value.

        Parameters:
        -----------
        image : torch.Tensor
            Shape (B, H, W, 3). For this node, typically B=1.
        mask : torch.Tensor
            Shape (B, H, W). Values usually in [0, 1].
        alpha_value : float
            Maximum alpha for mask=1. Clamped to [0, 1].

        Returns:
        --------
        (torch.Tensor,)
            A single-element tuple containing the RGBA image: shape (B, H, W, 4).
        """
        # Ensure image has 3 channels
        B, H, W, C = image.shape
        if C != 3:
            raise ValueError(
                f"Input image must have exactly 3 channels (RGB). Got {C}."
            )

        # Ensure mask has shape (B, H, W)
        if mask.ndim != 3:
            raise ValueError(
                f"Mask must have 3 dimensions (B, H, W). Got {mask.shape}."
            )

        # Optionally clamp alpha_value
        alpha_value = max(0.0, min(1.0, alpha_value))

        # Scale the mask by alpha_value
        # mask shape: (B, H, W) -> expand to (B, H, W, 1) for concatenation
        # grab black region of mask as content to keep.
        alpha_channel = (1 - mask).unsqueeze(-1)
        # optionally blend in masked region
        if alpha_value > 0.0:
            alpha_channel += (alpha_value * mask).unsqueeze(-1)

        # Concatenate the alpha channel to create RGBA
        rgba_image = torch.cat([image, alpha_channel], dim=-1)  # (B, H, W, 4)

        return (rgba_image,)
