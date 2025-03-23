import torch


def unify_to_rgba(img: torch.Tensor) -> torch.Tensor:
    """
    If the image is 3-channel (RGB), add an alpha channel of 1.0.
    If it's 4-channel (RGBA), keep the alpha as is.
    """
    B, H, W, C = img.shape
    if C == 3:
        alpha = torch.ones((B, H, W, 1), dtype=img.dtype, device=img.device)
        img = torch.cat([img, alpha], dim=-1)
    elif C == 4:
        pass  # already RGBA, do nothing
    else:
        raise ValueError(f"Expected 3 or 4 channels, got {C}.")
    return img


def pad_image(
    img: torch.Tensor,
    target_h: int,
    target_w: int,
    pad_color: torch.Tensor,
    pad_alpha: float,
) -> torch.Tensor:
    """
    Creates a new RGBA image of size (B, target_h, target_w, 4) filled with
    'pad_color' in RGB and 'pad_alpha' in A, then copies 'img' into the top-left.
    Preserves the existing alpha of 'img' in that region.
    """
    img = unify_to_rgba(img)  # Ensure input is RGBA
    B, H, W, C = img.shape  # C=4 after unify_to_rgba

    # Create a new RGBA image
    padded = torch.empty((B, target_h, target_w, 4), dtype=img.dtype, device=img.device)
    # Fill RGB with the padding color
    padded[..., :3] = pad_color
    # Fill alpha with user-specified padded_alpha
    padded[..., 3] = pad_alpha

    # Copy the original image (including its alpha) into the top-left corner
    padded[:, :H, :W, :] = img
    return padded


def pad_to_match(img1: torch.Tensor, img2: torch.Tensor, color: int, pad_alpha: float):
    """
    Pads both images to the same (max) width/height, preserving (and/or adding) alpha.
    The newly padded area uses 'color' for RGB and 'pad_alpha' for the alpha channel.
    """
    # First, unify both images to RGBA so alpha is preserved
    img1 = unify_to_rgba(img1)
    img2 = unify_to_rgba(img2)

    B1, H1, W1, C1 = img1.shape
    B2, H2, W2, C2 = img2.shape

    # Determine new height and width
    new_h = max(H1, H2)
    new_w = max(W1, W2)

    # Convert color (int) to a normalized RGB tensor of shape (1,1,1,3)
    r = ((color >> 16) & 0xFF) / 255.0
    g = ((color >> 8) & 0xFF) / 255.0
    b = (color & 0xFF) / 255.0
    pad_color = torch.tensor([r, g, b], dtype=img1.dtype, device=img1.device).view(
        1, 1, 1, 3
    )

    # Pad images if needed
    if (H1 != new_h) or (W1 != new_w):
        img1 = pad_image(img1, new_h, new_w, pad_color, pad_alpha)
    if (H2 != new_h) or (W2 != new_w):
        img2 = pad_image(img2, new_h, new_w, pad_color, pad_alpha)

    return img1, img2


def batch_two_images(
    image_1: torch.Tensor,
    image_2: torch.Tensor,
    color: int = 0xFFFFFF,
    padded_alpha: float = 1.0,
):
    """
    Batches two images along the batch dimension.
    - Ensures both images are RGBA.
    - Pads smaller images to match the larger size,
        using 'color' as the RGB fill and 'padded_alpha' for alpha in padded regions.
    - Preserves alpha from the original images if they already had it.
    """
    with torch.no_grad():
        img1, img2 = pad_to_match(image_1, image_2, color, padded_alpha)
        # Concatenate along batch dimension
        batched = torch.cat([img1, img2], dim=0)
        # If alpha is 1.0, remove it for a 3-channel image
        if padded_alpha == 1.0:
            batched = batched[..., :3]
    return batched


def batch_list_of_images(
    image_list: list[torch.Tensor],
    color: int = 0xFFFFFF,
    padded_alpha: float = 1.0,
):
    """ """
    with torch.no_grad():
        first_image = image_list[0]
        for image in image_list[1:]:
            first_image = batch_two_images(first_image, image, color, padded_alpha)
    return first_image


class PaddedBatchImages:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image_1": ("IMAGE", {}),
                "image_2": ("IMAGE", {}),
            },
            "optional": {
                "color": (
                    "INT",
                    {"default": 0xFFFFFF, "min": 0x000000, "max": 0xFFFFFF},
                ),  # Default white
                "padded_alpha": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("IMAGE",)
    FUNCTION = "batch_images"
    CATEGORY = "image"

    def batch_images(self, image_1, image_2, color, padded_alpha):
        batched = batch_two_images(image_1, image_2, color, padded_alpha)
        return (batched,)


class PaddedBatchListImages:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {}),
                "bounding_boxes": ("LIST", {"default": None}),
                "box_scale": ("FLOAT", {"default": 1.0}),
            },
            "optional": {
                "color": (
                    "INT",
                    {"default": 0xFFFFFF, "min": 0x000000, "max": 0xFFFFFF},
                ),  # Default white
                "padded_alpha": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("IMAGE",)
    FUNCTION = "batch_list_images"
    CATEGORY = "image"

    def batch_list_images(self, image, bounding_boxes, box_scale, color, padded_alpha):
        # bounding boxes is a list of tuples (x_min, x_max, y_min, y_max) for segments
        # to be taken out of the image to create a list of images. image of shape (1, H, W, C)
        image_list = [
            image[
                :,
                int(y_min * box_scale) : int(y_max * box_scale),
                int(x_min * box_scale) : int(x_max * box_scale),
                :,
            ]
            for x_min, x_max, y_min, y_max in bounding_boxes
        ]
        batched = batch_list_of_images(image_list, color, padded_alpha)
        return (batched,)
