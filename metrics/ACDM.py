import torch
from math import exp
import torch.nn as nn
from .color_utils import rgb_to_lab


def gaussian_kernel(window_size, sigma=1.5):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


def create_gaussian_kernel(window_size, sigma=1.5, channel=3):
    _1D_kernel = gaussian_kernel(window_size, sigma).unsqueeze(0)
    _2D_kernel = _1D_kernel.t().mm(_1D_kernel)
    kernel = _2D_kernel.expand(channel, 1, window_size, window_size).contiguous()
    return kernel


class GaussianConv(nn.Module):
    def __init__(self, window_size, sigma=1.5, in_channels=3):
        super(GaussianConv, self).__init__()
        self.padding = window_size // 2
        self.in_channels = in_channels

        kernel = create_gaussian_kernel(window_size, sigma, in_channels)
        self.register_buffer('kernel', kernel)

    def forward(self, x):
        return nn.functional.conv2d(x, self.kernel, padding=self.padding // 2, groups=self.in_channels)


def torch_cdf_loss(tensor_a, tensor_b, p=1, average=True, eps=1e-14):
    # last-dimension is weight distribution
    # p is the norm of the distance, p=1 --> First Wasserstein Distance
    # to get a positive weight with our normalized distribution
    # we recommend combining this loss with other difference-based losses like L1

    # normalize distribution, add 1e-14 to divisor to avoid 0/0
    tensor_a = tensor_a / (torch.sum(tensor_a, dim=1, keepdim=True) + eps)
    tensor_b = tensor_b / (torch.sum(tensor_b, dim=1, keepdim=True) + eps)
    # make cdf with cumsum
    cdf_tensor_a = torch.cumsum(tensor_a, dim=1)
    cdf_tensor_b = torch.cumsum(tensor_b, dim=1)

    # choose different formulas for different norm situations
    if p == 1:
        cdf_distance = torch.sum(torch.abs((cdf_tensor_a - cdf_tensor_b)), dim=1)
    elif p == 2:
        cdf_distance = torch.sqrt(torch.sum(torch.pow((cdf_tensor_a - cdf_tensor_b), 2), dim=1) + eps)
    else:
        cdf_distance = torch.pow(torch.sum(torch.pow(torch.abs(cdf_tensor_a - cdf_tensor_b), p), dim=1), 1 / p)
    if average:
        cdf_loss = cdf_distance.mean()
    else:
        cdf_loss = cdf_distance
    return cdf_loss


def histc_bc(images, bins=10, min_val=0, max_val=1):
    # B, 1, H, W
    B, C, H, W = images.shape
    flatten_images = images.view(B, C, -1)
    bin_centers = torch.linspace(min_val, max_val, bins, dtype=torch.float32, device=images.device)
    distances = torch.abs(flatten_images[..., None] - bin_centers)
    nearest_indices = torch.argmin(distances, dim=-1)
    one_hot = torch.zeros(B, C, H * W, bins, dtype=torch.float32, device=images.device)
    one_hot.scatter_(-1, nearest_indices.unsqueeze(-1), 1)
    histogram = one_hot.sum(dim=2)
    # B, 1, N
    return histogram


def norm_tensor(tensor, min_val=0, max_val=1):
    return (tensor - min_val) / (max_val - min_val)


def compute_diff(img1, img2, bins, min_max):
    """
    compute a single channel, shape should be B, 1, H, W
    """
    hist1 = histc_bc(img1, bins=bins, min_val=min_max[0], max_val=min_max[1]).squeeze(1)
    hist2 = histc_bc(img2, bins=bins, min_val=min_max[0], max_val=min_max[1]).squeeze(1)

    norm = hist1.sum(dim=1, keepdim=True)
    assert torch.allclose(norm, hist2.sum(dim=1, keepdim=True))
    hist1 = hist1 / norm
    hist2 = hist2 / norm
    # distance = torch.mean(torch.abs(hist1-hist2),dim=-1)
    distance = torch_cdf_loss(hist1, hist2, p=1, average=False) / (bins - 1)
    return distance


class AdversarialColorDistanceMetric:
    def __init__(self,
                 window_size=11,
                 bins=(10, 16, 16),
                 value_range=((0, 100), (-128, 127), (-128, 127))):
        self.bins = bins
        self.value_range = value_range
        self.gaussian = GaussianConv(window_size, sigma=1.5, in_channels=3)

    def __call__(self, img1: torch.Tensor, img2: torch.Tensor, size_average=True):
        """
        the input image should be in [0, 1] distribution.
        """
        assert img1.shape == img2.shape
        assert img1.size(1) == 3
        self.gaussian.to(img1.get_device())
        # B,C,H,W
        img1, img2 = self.gaussian(img1), self.gaussian(img2)
        img1, img2 = rgb_to_lab(img1), rgb_to_lab(img2)
        diffs = []
        for i in range(3):
            channel_diff = compute_diff(img1[:, i, :, :].unsqueeze(1),
                                        img2[:, i, :, :].unsqueeze(1),
                                        self.bins[i],
                                        self.value_range[i])
            if size_average:
                channel_diff = torch.mean(channel_diff)
                diffs.append(channel_diff)
            else:
                diffs.append(channel_diff)
        return diffs
