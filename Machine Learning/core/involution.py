from typing import Union, Tuple, Optional
import torch
import torch.nn as nn


class Involution2d(nn.Module):
    def __init__(self,
                in_channels: int,
                out_channels: int,
                sigma_mapping: Optional[nn.Module] = None,
                kernel_size: Union[int, Tuple[int, int]] = (7, 7),
                stride: Union[int, Tuple[int, int]] = (1, 1),
                groups: int = 1,
                reduce_ratio: int = 1,
                dilation: Union[int, Tuple[int, int]] = (1, 1),
                padding: Union[int, Tuple[int, int]] = (3, 3),
                bias: bool = False,
                activate_type='leaky',
                **kwargs) -> None:
        """
        Constructor method
        :param in_channels: (int) Number of input channels
        :param out_channels: (int) Number of output channels
        :param sigma_mapping: (nn.Module) Non-linear mapping as introduced in the paper. If none BN + ReLU is utilized
        :param kernel_size: (Union[int, Tuple[int, int]]) Kernel size to be used
        :param stride: (Union[int, Tuple[int, int]]) Stride factor to be utilized
        :param groups: (int) Number of groups to be employed
        :param reduce_ratio: (int) Reduce ration of involution channels
        :param dilation: (Union[int, Tuple[int, int]]) Dilation in unfold to be employed
        :param padding: (Union[int, Tuple[int, int]]) Padding to be used in unfold operation
        :param bias: (bool) If true bias is utilized in each convolution layer
        :param **kwargs: Unused additional key word arguments
        """
        # Call super constructor
        super(Involution2d, self).__init__()
        # Check parameters
        assert in_channels % groups == 0, "out_channels must be divisible by groups"
        assert out_channels % groups == 0, "out_channels must be divisible by groups"
        assert isinstance(sigma_mapping, nn.Module) or sigma_mapping is None, \
            "Sigma mapping must be an nn.Module or None to utilize the default mapping (BN + ReLU)."

        # Save parameters
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.stride = stride if isinstance(stride, tuple) else (stride, stride)
        self.groups = groups
        self.reduce_ratio = reduce_ratio
        self.dilation = dilation if isinstance(dilation, tuple) else (dilation, dilation)
        self.padding = padding if isinstance(padding, tuple) else (padding, padding)
        self.bias = bias

        if activate_type == "leaky":
            activate = nn.LeakyReLU(0.1)
        elif activate_type == "silu":
            activate = nn.SiLU()
        elif activate_type == "mish":
            activate = nn.Mish()
        else:
            raise ValueError("activate_type is no exis !")
        
        # Init modules
        self.sigma_mapping = sigma_mapping if sigma_mapping is not None else nn.Sequential(
            nn.BatchNorm2d(num_features=self.out_channels // self.reduce_ratio, momentum=0.3), activate)
        
        self.reduce_mapping = nn.Conv2d(in_channels=self.in_channels,
                                        out_channels=self.out_channels // self.reduce_ratio, kernel_size=(1, 1),
                                        stride=(1, 1), padding=(0, 0), bias=bias)
        self.span_mapping = nn.Conv2d(in_channels=self.out_channels // self.reduce_ratio,
                                    out_channels=self.kernel_size[0] * self.kernel_size[1] * self.groups,
                                    kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), bias=bias)
        self.unfold = nn.Unfold(kernel_size=self.kernel_size, dilation=dilation, padding=padding, stride=stride)


    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        :param input: (torch.Tensor) Input tensor of the shape [batch size, in channels, height, width]
        :return: (torch.Tensor) Output tensor of the shape [batch size, out channels, height, width] (w/ same padding)
        """
        # Check input dimension of input tensor
        assert input.ndimension() == 4, \
            "Input tensor to involution must be 4d but {}d tensor is given".format(input.ndimension())
    
        # Save input shape and compute output shapes
        batch_size, _, in_height, in_width = input.shape
        out_height = (in_height + 2 * self.padding[0] - self.dilation[0] * (self.kernel_size[0] - 1) - 1) \
                     // self.stride[0] + 1
        out_width = (in_width + 2 * self.padding[1] - self.dilation[1] * (self.kernel_size[1] - 1) - 1) \
                    // self.stride[1] + 1
        
        # Unfold and reshape input tensor
        input_unfolded = self.unfold(input)
        input_unfolded = input_unfolded.view(batch_size, self.groups, self.out_channels // self.groups,
                                            self.kernel_size[0] * self.kernel_size[1],
                                            out_height, out_width)
        # Generate kernel
        kernel = self.span_mapping(self.sigma_mapping(self.reduce_mapping(input)))
        kernel = kernel.view(batch_size, self.groups, self.kernel_size[0] * self.kernel_size[1],
                            kernel.shape[-2], kernel.shape[-1]).unsqueeze(dim=2)
        # Apply kernel to produce output
        output = (kernel * input_unfolded).sum(dim=3)
        # Reshape output
        output = output.view(batch_size, -1, output.shape[-2], output.shape[-1])
        return output


