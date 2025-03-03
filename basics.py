from typing import Tuple

import torch


class GetDimensions:
    """
    A ComfyUI node that extracts the shape of an image tensor and returns BATCH, ROWS, COLS, and CHANNELS.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"image": ("IMAGE", {})}}

    RETURN_TYPES = ("INT", "INT", "INT", "INT")
    RETURN_NAMES = ("BATCH", "ROWS", "COLS", "CHANNELS")
    FUNCTION = "extract_shape"

    def extract_shape(self, image: torch.Tensor) -> Tuple[int, int, int, int]:
        shape = image.shape

        if len(shape) == 4:  # (B, C, H, W)
            batch, channels, rows, cols = shape
        elif len(shape) == 3:  # (C, H, W) -> Assume batch of 1
            batch, rows, cols, channels = 1, shape[1], shape[2], shape[0]
        elif len(shape) == 2:  # (H, W) -> Grayscale image, assume single-channel
            batch, rows, cols, channels = 1, shape[0], shape[1], 1
        else:
            batch, rows, cols, channels = 1, 0, 0, 0  # Invalid shape handling

        return batch, rows, cols, channels


class ListSubsetNode:
    """
    A ComfyUI node that extracts a subset of a list based on user-specified indices.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input_list": ("LIST", {}),
                "indices": (
                    "STRING",
                    {"default": "-1", "multiline": True},
                ),
                "offset": ("INT", {"default": 0}),
            }
        }

    RETURN_TYPES = ("LIST",)
    FUNCTION = "subset_list"

    def subset_list(self, input_list: list, indices: str, offset: int) -> Tuple[list]:
        # Parse the indices string into a list of integers
        try:
            index_list = [
                int(i.strip()) + offset
                for i in indices.split(",")
                if i.strip().isdigit() or i.strip() == "-1"
            ]
        except ValueError:
            return ([],)  # Return empty list on parsing failure

        # If -1 is in the index list, return the full input list
        if -1 in index_list:
            return (input_list,)

        # Select elements based on valid indices
        output_list = [input_list[i] for i in index_list if 0 <= i < len(input_list)]
        return (output_list,)
