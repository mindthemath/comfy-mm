import numpy as np
import torch


class GridSampler:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "grid_x": ("INT", {"default": 8, "min": 1, "max": 256, "step": 1}),
                "grid_y": ("INT", {"default": 8, "min": 1, "max": 256, "step": 1}),
            }
        }

    RETURN_TYPES = ("GRID_DATA", "FLOAT")
    RETURN_NAMES = ("grid_data", "aspect_ratio")
    CATEGORY = "image"
    FUNCTION = "sample_grid"

    def sample_grid(self, image, grid_x, grid_y):
        # Convert tensor to numpy array
        img = image[0].cpu().numpy()
        img_height, img_width = img.shape[:2]

        # Calculate aspect ratio
        aspect_ratio = img_width / img_height

        # Calculate grid cell dimensions
        cell_width = img_width / grid_x
        cell_height = img_height / grid_y

        grid_points = []

        for i in range(grid_y):
            for j in range(grid_x):
                # Calculate cell boundaries
                x_start = int(j * cell_width)
                x_end = int((j + 1) * cell_width)
                y_start = int(i * cell_height)
                y_end = int((i + 1) * cell_height)

                # Extract cell region
                cell = img[y_start:y_end, x_start:x_end]

                # Calculate center coordinates (relative 0-1)
                center_x = (j + 0.5) / grid_x
                center_y = (i + 0.5) / grid_y

                # Calculate average RGB
                avg_rgb = cell.mean(axis=(0, 1))
                grid_points.append(
                    [center_x, center_y, avg_rgb[0], avg_rgb[1], avg_rgb[2]]
                )

        return (np.array(grid_points), aspect_ratio)


class GridPreview:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "grid_data": ("GRID_DATA",),
                "aspect_ratio": ("FLOAT", {"default": 1.0}),
                "ppi": ("INT", {"default": 300, "min": 1, "max": 1200, "step": 1}),
                "inches": (
                    "FLOAT",
                    {"default": 8.0, "min": 0.1, "max": 100.0, "step": 0.1},
                ),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    CATEGORY = "image"
    FUNCTION = "render_preview"

    def render_preview(self, grid_data, aspect_ratio, ppi, inches):
        # Calculate output dimensions
        target_width = int(ppi * inches)
        target_height = int(target_width / aspect_ratio)

        # Create output image
        output = np.zeros((target_height, target_width, 3), dtype=np.float32)

        # Convert grid_data to numpy array
        points = np.array(grid_data)

        # Calculate cell dimensions based on the data
        x_positions = sorted(list(set([point[0] for point in grid_data])))
        y_positions = sorted(list(set([point[1] for point in grid_data])))
        grid_x = len(x_positions)
        grid_y = len(y_positions)

        cell_width = target_width / grid_x
        cell_height = target_height / grid_y

        # Process in smaller chunks for memory efficiency
        chunk_size = 100  # Adjust this value based on available memory
        for i in range(0, len(points), chunk_size):
            chunk = points[i : i + chunk_size]

            # Convert normalized coordinates to pixel positions
            x_centers = (chunk[:, 0] * target_width).astype(np.int32)
            y_centers = (chunk[:, 1] * target_height).astype(np.int32)

            # Calculate cell boundaries
            half_width = int(cell_width / 2)
            half_height = int(cell_height / 2)

            for j, (x, y, r, g, b) in enumerate(
                zip(x_centers, y_centers, chunk[:, 2], chunk[:, 3], chunk[:, 4])
            ):
                x_start = max(0, x - half_width)
                x_end = min(target_width, x + half_width)
                y_start = max(0, y - half_height)
                y_end = min(target_height, y + half_height)

                output[y_start:y_end, x_start:x_end] = [r, g, b]

        # Convert to tensor
        output = torch.from_numpy(output).unsqueeze(0)
        return (output,)


class GridPreviewWidth:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "grid_data": ("GRID_DATA",),
                "aspect_ratio": ("FLOAT", {"default": 1.0}),
                "target_width": (
                    "INT",
                    {"default": 30000, "min": 1000, "max": 1_000_000, "step": 50},
                ),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    CATEGORY = "image"
    FUNCTION = "render_preview"

    def render_preview(self, grid_data, aspect_ratio, ppi, inches):
        # Calculate output dimensions
        target_width = int(ppi * inches)
        target_height = int(target_width / aspect_ratio)

        # Create output image
        output = np.zeros((target_height, target_width, 3), dtype=np.float32)

        # Convert grid_data to numpy array
        points = np.array(grid_data)

        # Calculate cell dimensions based on the data
        x_positions = sorted(list(set([point[0] for point in grid_data])))
        y_positions = sorted(list(set([point[1] for point in grid_data])))
        grid_x = len(x_positions)
        grid_y = len(y_positions)

        cell_width = target_width / grid_x
        cell_height = target_height / grid_y

        # Process in smaller chunks for memory efficiency
        chunk_size = 100  # Adjust this value based on available memory
        for i in range(0, len(points), chunk_size):
            chunk = points[i : i + chunk_size]

            # Convert normalized coordinates to pixel positions
            x_centers = (chunk[:, 0] * target_width).astype(np.int32)
            y_centers = (chunk[:, 1] * target_height).astype(np.int32)

            # Calculate cell boundaries
            half_width = int(cell_width / 2)
            half_height = int(cell_height / 2)

            for j, (x, y, r, g, b) in enumerate(
                zip(x_centers, y_centers, chunk[:, 2], chunk[:, 3], chunk[:, 4])
            ):
                x_start = max(0, x - half_width)
                x_end = min(target_width, x + half_width)
                y_start = max(0, y - half_height)
                y_end = min(target_height, y + half_height)

                output[y_start:y_end, x_start:x_end] = [r, g, b]

        # Convert to tensor
        output = torch.from_numpy(output).unsqueeze(0)
        return (output,)


# # Register nodes
# NODE_CLASS_MAPPINGS = {"GridSampler": GridSampler, "GridPreview": GridPreview}

# NODE_DISPLAY_NAME_MAPPINGS = {
#     "GridSampler": "Grid Sampler",
#     "GridPreview": "Grid Preview",
# }
