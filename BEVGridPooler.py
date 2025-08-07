# You are given a list of 3D lidar points with feature vectors, a 2D grid in bird’s eye view, and a grid cell size as input. Produce a 2D BEV grid with the pooled 3D point features.

# Coding Requirement: Tensorflow or Pytorch with vectorized operations.
# Inputs:
# point_cloud - (N, 3), where N is the number of lidar points, and 3 are the x, y and z coordinates.
# point_feature - (N, C), where N is the number of lidar points, and C is the number of features. 
# grid_size - (2, ), which is the size of the 2D BEV grid in meter, e.g. (64, 64); note that this is not the dimension of the pooled_bev_grid
# grid_cell_size - (2, ), which is the size of each grid cell in meter, e.g. (0.1, 0.1)
# pooling_method - optional; it can be “max”, “mean”, “sum”, etc; this parameter can be passed into the __init__ function. Implementing “max” and “sum” should be very easy while implementing “mean” needs more lines. 

# Outputs:
# pooled_bev_grid - (H, W, C’), where H and W are defined by grid_size and grid_cell_size, and C is the number of features. Note that C’ can be different from C in the above, 

import torch
import torch_scatter


class BEVGridPooler:
    def __init__(self, grid_size, grid_cell_size, pooling_method='max'):
        self.grid_size = torch.tensor(grid_size, dtype=torch.float32)  # (width, height) in meters
        self.grid_cell_size = torch.tensor(grid_cell_size, dtype=torch.float32)  # (dx, dy) in meters
        self.pooling_method = pooling_method

        # Compute grid resolution (H, W)
        self.grid_resolution = (self.grid_size / self.grid_cell_size).long()
        self.H, self.W = self.grid_resolution[1].item(), self.grid_resolution[0].item()

    def __call__(self, point_cloud, point_features):
        """
        point_cloud: (N, 3) - x, y, z
        point_features: (N, C)
        Returns:
            pooled_bev_grid: (H, W, C)
        """
        assert point_cloud.shape[0] == point_features.shape[0]

        # Compute 2D BEV grid indices (ix, iy)
        coords_xy = point_cloud[:, :2]  # (N, 2)
        grid_indices = (coords_xy / self.grid_cell_size).floor().long()  # (N, 2)

        # Filter points that fall outside the grid
        mask = (
            (grid_indices[:, 0] >= 0) & (grid_indices[:, 0] < self.W) &
            (grid_indices[:, 1] >= 0) & (grid_indices[:, 1] < self.H)
        )
        grid_indices = grid_indices[mask]
        point_features = point_features[mask]

        # Convert (ix, iy) to flat 1D index
        flat_indices = grid_indices[:, 1] * self.W + grid_indices[:, 0]  # (N, )

        # Apply pooling using torch_scatter
        if self.pooling_method == 'max':
            pooled_feats, _ = torch_scatter.scatter_max(point_features, flat_indices, dim=0)
        elif self.pooling_method == 'sum':
            pooled_feats = torch_scatter.scatter_add(point_features, flat_indices, dim=0)
        elif self.pooling_method == 'mean':
            pooled_feats = torch_scatter.scatter_mean(point_features, flat_indices, dim=0)
        else:
            raise ValueError(f"Unsupported pooling method: {self.pooling_method}")

        # Create output tensor and fill it
        C = point_features.shape[1]
        bev_grid = torch.zeros((self.H * self.W, C), dtype=pooled_feats.dtype, device=pooled_feats.device)
        bev_grid[flat_indices.unique()] = pooled_feats  # fill only computed cells

        # Reshape to (H, W, C)
        pooled_bev_grid = bev_grid.view(self.H, self.W, C)

        return pooled_bev_grid

N, C = 10000, 8
point_cloud = torch.rand(N, 3) * 64  # random points in 64x64m area
point_features = torch.rand(N, C)
grid_size = (64, 64)
grid_cell_size = (0.2, 0.2)

pooler = BEVGridPooler(grid_size, grid_cell_size, pooling_method='max')
bev_feature_map = pooler(point_cloud, point_features)  # shape (H, W, C)
