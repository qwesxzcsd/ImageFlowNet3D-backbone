import math
import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F

"""
3D UNet implementation 
- replaced external nn.utils imports with direct implementations to isolate (not sure if this is needed, but I wanted the code to be self-contained)
- follows orginal UNet architecture with downsampling and upsampling blocks
"""

# utilities from original backbone
def conv_nd(dims, *args, **kwargs):
    if dims == 1:
        return nn.Conv1d(*args, **kwargs)
    elif dims == 2:
        return nn.Conv2d(*args, **kwargs)
    elif dims == 3:
        return nn.Conv3d(*args, **kwargs)
    else:
        raise ValueError(f"unsupported dimensions: {dims}")

def normalization(channels):
    return nn.GroupNorm(32, channels)


class Downsample(nn.Module):

    # downsampling layer with an optional convolution.

    def __init__(self, channels, use_conv, dims=2, out_channels=None, padding=1):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        stride = 2 if dims != 3 else (2, 2, 2)  # downsample in all 3 dimensions for 3D
        if use_conv:
            self.op = conv_nd(
                dims, self.channels, self.out_channels, 3, stride=stride, padding=padding
            )
        else:
            assert self.channels == self.out_channels
            if dims == 3:
                self.op = nn.AvgPool3d(kernel_size=2, stride=2)
            else:
                self.op = nn.AvgPool2d(kernel_size=stride, stride=stride)

    def forward(self, x):
        assert x.shape[1] == self.channels
        return self.op(x)


class Upsample(nn.Module):

    # an upsampling layer with an optional convolution.

    def __init__(self, channels, use_conv, dims=2, out_channels=None, padding=1):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        if use_conv:
            self.conv = conv_nd(dims, self.channels, self.out_channels, 3, padding=padding)

    def forward(self, x):
        assert x.shape[1] == self.channels
        if self.dims == 3:
            # upsamples in all 3 dimensions for 3D (i changed the scale factor to 2 for all dims)
            x = F.interpolate(x, scale_factor=(2, 2, 2), mode="nearest")
        else:
            x = F.interpolate(x, scale_factor=2, mode="nearest")
        if self.use_conv:
            x = self.conv(x)
        return x


# two consecutive 3D conv layers with normalization and SiLU activation
def double_conv(in_channels, out_channels, mid_channels=None):
    if mid_channels is None:
        mid_channels = out_channels
    return nn.Sequential(
        # first convolution
        conv_nd(3, in_channels, mid_channels, 3, padding=1),
        normalization(mid_channels),
        nn.SiLU(),
        # second convolution
        conv_nd(3, mid_channels, out_channels, 3, padding=1),
        normalization(out_channels),
        nn.SiLU(),
    )


# encoder block: downsampling followed by feature extraction
class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.op = nn.Sequential(
            Downsample(in_channels, use_conv=False, dims=3),
            # extract features after downsampling
            double_conv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.op(x)


# decoder block: upsampling followed by skip-concatenation and feature extraction
class Up(nn.Module):
    def __init__(self, in_channels, out_channels, trilinear=True):
        super().__init__()
        if trilinear:
            self.up = Upsample(in_channels // 2, use_conv=False, dims=3)
            conv_in = in_channels  # channels after concatenation (skip + upsampled)
        else:
            # transpose conv increases spatial dims by factor 2
            self.up = conv_nd(3, in_channels // 2, in_channels // 2, 2, stride=2)
            conv_in = in_channels
        self.conv = double_conv(conv_in, out_channels)

    def forward(self, x1, x2):
        # upsample the coarse feature map
        x1 = self.up(x1)
        # pad if needed to exactly match encoder feature map size
        dz = x2.size(2) - x1.size(2)
        dy = x2.size(3) - x1.size(3)
        dx = x2.size(4) - x1.size(4)
        x1 = F.pad(
            x1,
            [dx // 2, dx - dx // 2,
             dy // 2, dy - dy // 2,
             dz // 2, dz - dz // 2]
        )
        # skip connection
        x = th.cat([x2, x1], dim=1)
        return self.conv(x)


# final 1×1×1 convolution to map to desired output channels (segmentation classes)
class OutConv(nn.Module):
    # 1×1×1 convolution to map to the desired number of classes
    
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = conv_nd(3, in_channels, out_channels, 1)

    def forward(self, x):
        return self.conv(x)


# full UNet3D architecture: encoder path, bottleneck, decoder path
class UNet3D(nn.Module):
    """
    3D UNet backbone with symmetric downsampling and upsampling paths.
    - encoder path has 4 levels of downsampling
    - decoder with 4 levels of upsampling
    """
    def __init__(self, in_channels, out_channels, base_filters=64, trilinear=True):
        super().__init__()
        # --Encoder--
        # feature extraction
        self.inc = double_conv(in_channels, base_filters)
        # downsample levels, each reduces spatial dims and doubles channels
        self.down1 = Down(base_filters, base_filters * 2)
        self.down2 = Down(base_filters * 2, base_filters * 4)
        self.down3 = Down(base_filters * 4, base_filters * 8)
        # keep flexibility for non-learned vs learned upsampling
        factor = 2 if trilinear else 1
        self.down4  = Down(base_filters * 8, (base_filters * 16) // factor)

        # --Decoder--
        # upsample levels, halves channels and concatenates skip features
        self.up1 = Up(base_filters * 16, base_filters * 8 // factor, trilinear)
        self.up2 = Up(base_filters *  8, base_filters * 4 // factor, trilinear)
        self.up3 = Up(base_filters *  4, base_filters * 2 // factor, trilinear)
        self.up4 = Up(base_filters *  2, base_filters, trilinear)
        # final 1×1×1 conv to produce output volume
        self.outc   = OutConv(base_filters, out_channels)

    def forward(self, x):
        # encoder forward pass: save skip features
        x1 = self.inc(x)      # level 0
        x2 = self.down1(x1)   # level 1
        x3 = self.down2(x2)   # level 2
        x4 = self.down3(x3)   # level 3
        x5 = self.down4(x4)   # bottleneck
        # decoder forward pass: use skip connections
        x  = self.up1(x5, x4)  # combine bottleneck and level 3
        x  = self.up2(x,  x3)  # combine and upsample
        x  = self.up3(x,  x2)
        x  = self.up4(x,  x1)
        # output projection
        return self.outc(x)


# smoke tests to verify the model can handle a random 3D tensor
if __name__ == "__main__":
    # single tensor
    model = UNet3D(in_channels=1, out_channels=2, base_filters=32, trilinear=True)
    x = th.randn(1, 1, 16, 32, 32)
    y = model(x)
    print("single tensor test input shape:", x.shape, "output shape:", y.shape)
    assert y.shape == (1, 2, 16, 32, 32), "shape mismatch"
    print("single tensor passed")

    # random 3D dataset check
    from torch.utils.data import Dataset, DataLoader

    class Random3DDataset(Dataset):
        """
        random 3D volumes datasets
        each sample is an x tensor of shape (1, D, H, W) and y mask of shape (D, H, W).
        """
        def __init__(self, length=4, depth=16, height=32, width=32, num_classes=2):
            self.length = length
            self.depth = depth
            self.height = height
            self.width = width
            self.num_classes = num_classes

        def __len__(self):
            return self.length

        def __getitem__(self, idx):
            # random volume and integer mask
            x = th.randn(1, self.depth, self.height, self.width)
            y = th.randint(0, self.num_classes, (self.depth, self.height, self.width))
            return x, y

    # instantiate DataLoader
    dataset = Random3DDataset(length=4)
    loader = DataLoader(dataset, batch_size=2)

    # iterate one batch through the model
    x_batch, y_batch = next(iter(loader))
    print("dataset batch x:", x_batch.shape, "y:", y_batch.shape)
    pred_batch = model(x_batch)
    print("model output on batch:", pred_batch.shape)
    assert (pred_batch.shape[0] == x_batch.shape[0] and pred_batch.shape[2:] == x_batch.shape[2:]), "Batch shape mismatch!"
    print("random dataset passed")


    # backward-pass check
    # creates a dummy loss
    loss_fn = nn.CrossEntropyLoss()
    # y_batch shape is (N, D, H, W), pred_batch is (N, C, D, H, W)
    loss = loss_fn(pred_batch, y_batch)
    print(f"computed loss: {loss.item()}")
    loss.backward()
    # verifies that at least one parameter has non-zero gradient
    grads = [p.grad.abs().sum().item() for p in model.parameters() if p.grad is not None]
    assert any(g > 0 for g in grads), "no gradients"
    print("backward pass check passed")

    # regression backward check (MSE) (used to better understand MSE vs crossentropy, don't need to sanity check two different loss fucnctions)
    # instantiate model for regression
    model_reg = UNet3D(in_channels=1, out_channels=1, base_filters=32, trilinear=True)
    # generate dummy continuous target of same shape as predicted
    x_reg = th.randn(2, 1, 16, 32, 32)
    y_true = th.randn(2, 1, 16, 32, 32)
    pred_reg = model_reg(x_reg)
    # MSE loss
    mse_loss = nn.MSELoss()(pred_reg, y_true)
    print(f"computed MSE loss: {mse_loss.item()}")
    mse_loss.backward()
    grads_reg = [p.grad.abs().sum().item() for p in model_reg.parameters() if p.grad is not None]
    assert any(g > 0 for g in grads_reg), "fails check"
    print("MSE check passed")
