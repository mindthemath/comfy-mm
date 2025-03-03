from typing import Tuple

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
            image = image[..., :3]

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


class AddImagesWithAlpha:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image1": ("IMAGE", {}),
                "image2": ("IMAGE", {}),
                "alpha1": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0}),
                "alpha2": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("IMAGE",)
    FUNCTION = "add_images"
    CATEGORY = "processing"

    def add_images(
        self, image1: torch.Tensor, image2: torch.Tensor, alpha1: float, alpha2: float
    ):
        """
        Adds two images together with optional alpha blending.

        Parameters:
        -----------
        image1 : torch.Tensor
            Shape (B, H, W, 3). For this node, typically B=1.
        image2 : torch.Tensor
            Shape (B, H, W, 3). For this node, typically B=1.
        alpha1 : float
            Alpha value for image1. Clamped to [0, 1].
        alpha2 : float
            Alpha value for image2. Clamped to [0, 1].

        Returns:
        --------
        (torch.Tensor,)
            A single-element tuple containing the blended image: shape (B, H, W, 3).
        """
        # ensure batch dimension
        if image1.ndim != 4:
            image1 = image1.unsqueeze(0)
        if image2.ndim != 4:
            image2 = image2.unsqueeze(0)
        # ensure alpha channels exist:
        if image1.shape[-1] == 3:
            image1 = torch.cat([image1, torch.ones_like(image1[..., :1])], dim=-1)
        if image2.shape[-1] == 3:
            image2 = torch.cat([image2, torch.ones_like(image2[..., :1])], dim=-1)
        # Ensure images have the same dimensions
        if image2.shape != image1.shape:
            raise ValueError(
                f"Input images must have the same shape. Got {image1.shape} and {image2.shape}."
            )

        # Optionally clamp alpha values
        alpha1 = max(0.0, min(1.0, alpha1))
        alpha2 = max(0.0, min(1.0, alpha2))

        # Blend the images
        # mask pixels by alpha channel before adding.
        image1[..., :3] *= image1[..., 3].unsqueeze(-1)
        image2[..., :3] *= image2[..., 3].unsqueeze(-1)
        blended_image = (alpha1 * image1 + alpha2 * image2).clamp(0.0, 1.0)
        # for the alpha channel, add the two up and clamp to [0, 1]
        blended_image[..., 3] = (image1[..., 3] + image2[..., 3]).clamp(0.0, 1.0)
        return (blended_image,)


class UnifyMask:
    """
    Encodes each batch dimension of a collection of masks as an integer
    and returns a single mask tensor with unique integer values for each mask.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"masks": ("MASK", {})}}

    RETURN_TYPES = ("MASK",)
    FUNCTION = "unify_masks"
    CATEGORY = "masking"

    def unify_masks(self, masks: torch.Tensor) -> Tuple[torch.Tensor]:
        """
        Encodes each batch dimension of a collection of masks as an integer
        and returns a single mask tensor with unique integer values for each mask.

        Parameters:
        -----------
        masks : torch.Tensor
            Shape (B, H, W). Values usually in [0, 1].

        Returns:
        --------
        (torch.Tensor,)
            A single-element tuple containing the unified mask: shape (B, H, W).
        """
        # Ensure mask has shape (B, H, W)
        if masks.ndim != 3:
            raise ValueError(
                f"Mask must have 3 dimensions (B, H, W). Got {masks.shape}."
            )

        # Create a unique integer mask for each batch dimension
        B, H, W = masks.shape
        unified_mask = torch.zeros_like(masks[0], dtype=torch.uint8)
        for i in range(B):
            unified_mask[masks[i] > 0] = i + 1

        return (unified_mask,)


class SaveMaskAsCSV:
    """
    Takes a mask (without a batch dimension) and saves it as a CSV file (no header)
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mask": ("MASK", {}),
                "filename": ("STRING", {"default": "output/mask.csv"}),
            }
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "save_mask_as_csv"
    CATEGORY = "masking"
    OUTPUT_NODE = True

    def save_mask_as_csv(self, mask: torch.Tensor, filename: str) -> Tuple[str]:
        """
        Takes a mask (without a batch dimension) and saves it as a CSV file (no header).

        Parameters:
        -----------
        mask : torch.Tensor
            Shape (H, W). Values usually in [0, 1].
        filename : str
            The name of the CSV file to save.

        Returns:
        --------
        (str,)
            A single-element tuple containing the filename of the saved CSV.
        """
        # Ensure mask has shape (H, W)
        if mask.ndim != 2:
            raise ValueError(f"Mask must have 2 dimensions (H, W). Got {mask.shape}.")

        # Save the mask as a CSV file
        mask_np = mask.cpu().numpy()
        with open(filename, "w") as f:
            for row in mask_np:
                f.write(",".join(map(str, row)) + "\n")

        return (filename,)
