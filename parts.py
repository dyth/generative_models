import torch


class NormalizationLayerFactory:
    def __init__(self, normalization_layer=torch.nn.BatchNorm2d, count=1):
        self.normalization_layer = normalization_layer
        self.count = count

    def create(self, capacity, affine=True, normalization_layer=None, count=None):
        if normalization_layer is None:
            normalization_layer = self.normalization_layer
        if count is None:
            count = self.count
        return normalization_layer(capacity, affine=affine)


class DoubleConvolution(torch.nn.Module):
    def __init__(self, capacity, normalization_layer_factory, kernel_size=3, padding=None, bias=False,
                 first_capacity=None, last_capacity=None, post_normalization=False, pre_normalization=False):
        super(DoubleConvolution, self).__init__()

        if padding is None:
            padding = kernel_size // 2
        if first_capacity is None:
            first_capacity = capacity
        if last_capacity is None:
            last_capacity = capacity

        layers = []
        if pre_normalization:
            layers.append(normalization_layer_factory.create(last_capacity))
        layers.extend([torch.nn.Conv2d(first_capacity, capacity, kernel_size, 1, padding=padding, bias=bias),
                       torch.nn.ReLU(),
                       normalization_layer_factory.create(capacity),
                       torch.nn.Conv2d(capacity, last_capacity, kernel_size, 1, padding=padding, bias=bias),
                       torch.nn.ReLU()])
        if post_normalization:
            layers.append(normalization_layer_factory.create(last_capacity))

        self.convolve = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.convolve(x)


class DownDoubleConvolution(DoubleConvolution):
    def __init__(self, *args, **kwargs):
        super(DownDoubleConvolution, self).__init__(*args, post_normalization=True, pre_normalization=False, **kwargs)


class UpDoubleConvolution(DoubleConvolution):
    def __init__(self, *args, **kwargs):
        super(UpDoubleConvolution, self).__init__(*args, post_normalization=False, pre_normalization=True, **kwargs)


class DownSampling(torch.nn.Module):
    def __init__(self, capacity, normalization_layer_factory, kernel_size=5, bias=False):
        super(DownSampling, self).__init__()
        self.subsample = torch.nn.Sequential(
            torch.nn.Conv2d(capacity, 2 * capacity, kernel_size, 2, bias=bias),
            torch.nn.ReLU(),
            normalization_layer_factory.create(2 * capacity)
        )

    def forward(self, x):
        return self.subsample(x)


class UpSampling(torch.nn.Module):
    def __init__(self, capacity, normalization_layer_factory, kernel_size=5, bias=False, output_padding=1):
        super(UpSampling, self).__init__()

        self.upsample = torch.nn.Sequential(
            normalization_layer_factory.create(capacity),
            torch.nn.ConvTranspose2d(capacity, capacity // 2, kernel_size, 2, bias=bias, output_padding=output_padding),
            torch.nn.ReLU(),
        )

    def forward(self, x):
        return self.upsample(x)
