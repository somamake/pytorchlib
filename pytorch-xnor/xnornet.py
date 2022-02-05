import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn.parameter import Parameter, UninitializedParameter
import torch.nn.modules.conv as nnconv
from torch.nn.common_types import _size_1_t, _size_2_t, _size_3_t
from torch.nn.modules.utils import _pair, _triple, _reverse_repeat_tuple
from typing import Optional, List, Tuple, Union

class BinActive(torch.autograd.Function):
    '''
    Binarize the input activations and calculate the mean across channel dimension.
    '''
    @staticmethod
    def forward(ctx, input,kernel_size):
        ctx.save_for_backward(input)
        mean = torch.mean(input.abs(),1, keepdim=True)
        k = torch.full((1,1,kernel_size,kernel_size),fill_value=1/(kernel_size**2))
        # in: (minibatch,in_channels,iH,iW) weight (out_channels,in_channels/groups,kH,kW)
        K = nn.functional.conv2d(mean, k, bias=None, stride=1, padding=0, dilation=1, groups=1) 
        input = input.sign()
        return input, K
    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[input.ge(1)] = 0
        grad_input[input.le(-1)] = 0
        return grad_input

class BinConv2d(torch.nn.modules.conv._ConvNd):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _size_2_t,
        stride: _size_2_t = 1,
        padding: Union[str, _size_2_t] = 0,
        dilation: _size_2_t = 1,
        groups: int = 1,
        bias: bool = False,
        padding_mode: str = 'zeros',  # TODO: refine this type
        device=None,
        dtype=None
    ) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        kernel_size_ = _pair(kernel_size)
        stride_ = _pair(stride)
        padding_ = padding if isinstance(padding, str) else _pair(padding)
        dilation_ = _pair(dilation)
        self.alpha = None
        super(BinConv2d, self).__init__(
            in_channels, out_channels, kernel_size_, stride_, padding_, dilation_,
            False, _pair(0), groups, bias, padding_mode, **factory_kwargs)

    def _conv_forward(self, input: Tensor, weight: Tensor, bias: Optional[Tensor]):
        if self.padding_mode != 'zeros':
            return F.conv2d(F.pad(input, self._reversed_padding_repeated_twice, mode=self.padding_mode),
                            weight, bias, self.stride,
                            _pair(0), self.dilation, self.groups)
        return F.conv2d(input, weight, bias, self.stride,
                        self.padding, self.dilation, self.groups)

    def forward(self, input: Tensor,K: Tensor) -> Tensor:
        _alpha = torch.mean(self.weight.abs(),(2,3))
        self.alpha = _alpha.reshape((1,len(_alpha),1,1))
        self.weight = Parameter(self.weight.sign())
        return self._conv_forward(input, self.weight, self.bias) * K * self.alpha

class BnBinActiveConv2d(nn.Module):
    def __init__(self, input_channels, output_channels,
            kernel_size=-1, stride=-1, padding=-1, dropout=0):
        super(BnBinActiveConv2d, self).__init__()
        self.layer_type = 'BnBinActiveConv2d'
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dropout_ratio = dropout

        self.bn = nn.BatchNorm2d(input_channels, eps=1e-4, momentum=0.1, affine=True)
        self.bn.weight.data = self.bn.weight.data.zero_().add(1.0)
        if dropout!=0:
            self.dropout = nn.Dropout(dropout)
        self.binconv = BinConv2d(input_channels, output_channels,
                kernel_size=kernel_size, stride=stride, padding=padding)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        x = self.bn(x)
        x, K = BinActive()(x,self.kernel_size)
        if self.dropout_ratio!=0:
            x = self.dropout(x)
        x = self.binconv(x,K)
        x = self.relu(x)
        return x

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.xnor = nn.Sequential(
                nn.Conv2d(3, 192, kernel_size=5, stride=1, padding=2),
                nn.BatchNorm2d(192, eps=1e-4, momentum=0.1, affine=False),
                nn.ReLU(inplace=True),
                BnBinActiveConv2d(192, 160, kernel_size=1, stride=1, padding=0),
                BnBinActiveConv2d(160,  96, kernel_size=1, stride=1, padding=0),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1),

                BnBinActiveConv2d( 96, 192, kernel_size=5, stride=1, padding=2, dropout=0.5),
                BnBinActiveConv2d(192, 192, kernel_size=1, stride=1, padding=0),
                BnBinActiveConv2d(192, 192, kernel_size=1, stride=1, padding=0),
                nn.AvgPool2d(kernel_size=3, stride=2, padding=1),

                BnBinActiveConv2d(192, 192, kernel_size=3, stride=1, padding=1, dropout=0.5),
                BnBinActiveConv2d(192, 192, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm2d(192, eps=1e-4, momentum=0.1, affine=False),
                nn.Conv2d(192,  10, kernel_size=1, stride=1, padding=0),
                nn.ReLU(inplace=True),
                nn.AvgPool2d(kernel_size=8, stride=1, padding=0),
                )

    def forward(self, x):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                if hasattr(m.weight, 'data'):
                    m.weight.data.clamp_(min=0.01)
        x = self.xnor(x)
        x = x.view(x.size(0), 10)
        return x