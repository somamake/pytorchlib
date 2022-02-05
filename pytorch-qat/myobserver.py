import re
import warnings
from abc import ABCMeta, abstractmethod
from collections import OrderedDict
from functools import partial
from typing import Any, List, Tuple, Optional, Dict, Union

import torch
import torch.nn as nn
from torch.ao.quantization.utils import check_min_max_valid, calculate_qmin_qmax

class Pow2ObserverBase(torch.quantization.observer.ObserverBase):
    r"""Internal common base for all qint/quint8 observers.

    This base is for commonly used parameters used internally.
    Users should use `~torch.quantization.observer.ObserverBase` as a base class
    for custom observers.

    Args:
        dtype: Quantized data type.
        qscheme: Quantization scheme to be used.
        reduce_range: Reduces the range of the quantized data type by 1 bit.
                      This is sometimes required to avoid instruction overflow.
        quant_min: Minimum quantization value. If unspecified, it will follow the 8-bit setup.
        quant_max: Maximum quantization value. If unspecified, it will follow the 8-bit setup.

    .. warning::

        :attr:`dtype` can only take ``torch.qint8`` or ``torch.quint8``.

    .. warning::

        :attr:`qscheme` can only take one of the following options:

        - ``torch.per_tensor_affine``
        - ``torch.per_tensor_symmetric``
        - ``torch.per_channel_affine``
        - ``torch.per_channel_symmetric``
    """

    # Note: the version is shared by all observer types
    #
    # Version 1/None
    #   self
    #
    # Version 2 (base class only, does not include child class buffers)
    #   self
    #   |--- eps : Tensor
    #
    # Version 3
    #   for HistogramObserver only, changed the shape of uninitialized
    #   min_val and max_val buffers from torch.Size([0]) to torch.Size([])
    #   for PerChannelObservers, changed the name of the buffers from min_vals
    #   to min_val and from max_vals to max_val.
    _version = 3

    eps: torch.Tensor

    def __init__(
        self,
        dtype=torch.quint8,
        qscheme=torch.per_tensor_affine,
        reduce_range=False,
        quant_min=None,
        quant_max=None,
        factory_kwargs=None,
    )-> None:
        factory_kwargs = torch.nn.factory_kwargs(factory_kwargs)
        super(Pow2ObserverBase, self).__init__(dtype=dtype)
        self.qscheme = qscheme
        if reduce_range:
            warnings.warn(
                "Please use quant_min and quant_max to specify the range for observers. \
                    reduce_range will be deprecated in a future release of PyTorch."
            )
        self.reduce_range = reduce_range
        self.register_buffer(
            "eps", torch.tensor([torch.finfo(torch.float32).eps], **factory_kwargs)
        )
        assert self.qscheme in (
            torch.per_tensor_affine,
            torch.per_tensor_symmetric,
            torch.per_channel_affine,
            torch.per_channel_symmetric,
            torch.per_channel_affine_float_qparams,
        ), "Default Observer only works for per_tensor_affine, \
                per_tensor_symmetric, per_channel_affine, \
                per_channel_symmetric and per_channel_float_qparams quantization scheme"
        assert self.dtype in (
            torch.qint8,
            torch.quint8,
            torch.quint4x2,
        ), "Default Observer only works for qint8, quint8 and quint4x2 data type"
        self.has_customized_qrange = (quant_min is not None) and (quant_max is not None)
        if self.has_customized_qrange:
            self._validate_qmin_qmax(quant_min, quant_max)
        self.quant_min, self.quant_max = \
            calculate_qmin_qmax(quant_min, quant_max, self.has_customized_qrange, self.dtype, self.reduce_range)

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):

        version = local_metadata.get("version", None)

        if version is None or version == 1:
            # eps was moved to a buffer in version 2
            eps = torch.tensor([torch.finfo(torch.float32).eps])
            state_dict[prefix + "eps"] = eps

        super(Pow2ObserverBase, self)._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    @torch.jit.export
    def _validate_qmin_qmax(self, quant_min: int, quant_max: int) -> None:
        r"""Validates that the user-specified quantization range is properly initialized
        and within the given bound supported by the observer dtype.

        To accommodate lower-bit quantization with respect to the existing torch.qint8 and
        torch.quint8 datatypes, the user can choose to use dynamic quantization range by passing
        in a tuple of initial qmin and qmax values. One use case is these customized qmin and qmax
        values are used to calculate static estimates of the scale and zero point for aggressive lower-bit
        fake quantization. These estimates are compared against parameters learned through backpropagation.
        The related literatures for scale and zero point via backpropagation are as follows:

        Learned Step Size Quantization: https://openreview.net/pdf?id=rkgO66VKDS
        Trained Quantization Thresholds: https://arxiv.org/pdf/1903.08066.pdf
        """
        # The variable names are prefixed with "initial" because their values (qmin and qmax) might be adjusted
        # based on whether quantization range is reduced and the datatype (signed/unsigned) used by the observer.
        assert (
            quant_min <= 0 <= quant_max
        ), "Used-specified quantization range must include 0."
        assert (
            quant_min < quant_max
        ), "qmin must be strictly less than qmax for user-specified quantization range."

    @torch.jit.export
    def _calculate_qparams(
        self, min_val: torch.Tensor, max_val: torch.Tensor
    )-> Tuple[torch.Tensor, torch.Tensor]:
        r"""Calculates the quantization parameters, given min and max
        value tensors. Works for both per tensor and per channel cases

        Args:
            min_val: Minimum values per channel
            max_val: Maximum values per channel

        Returns:
            scales: Scales tensor of shape (#channels,)
            zero_points: Zero points tensor of shape (#channels,)
        """
        if not check_min_max_valid(min_val, max_val):
            return torch.tensor([1.0], device=min_val.device.type), torch.tensor([0], device=min_val.device.type)

        quant_min, quant_max = self.quant_min, self.quant_max
        min_val_neg = torch.min(min_val, torch.zeros_like(min_val))
        max_val_pos = torch.max(max_val, torch.zeros_like(max_val))

        device = min_val_neg.device
        scale = torch.ones(min_val_neg.size(), dtype=torch.float32, device=device)
        zero_point = torch.zeros(min_val_neg.size(), dtype=torch.int64, device=device)

        if (
            self.qscheme == torch.per_tensor_symmetric
            or self.qscheme == torch.per_channel_symmetric
        ):
            max_val_pos = torch.max(-min_val_neg, max_val_pos)
            scale = max_val_pos / (float(quant_max - quant_min) / 2)
            scale = torch.max(scale, self.eps)
            # scale = torch.pow(2,torch.log2(scale).ceil())    # power of 2
            if self.dtype == torch.quint8:
                if self.has_customized_qrange:
                    # When customized quantization range is used, down-rounded midpoint of the range is chosen.
                    zero_point = zero_point.new_full(
                        zero_point.size(), (quant_min + quant_max) // 2
                    )
                else:
                    zero_point = zero_point.new_full(zero_point.size(), 128)
        elif self.qscheme == torch.per_channel_affine_float_qparams:
            scale = (max_val - min_val) / float(quant_max - quant_min)
            scale = torch.where(scale > self.eps, scale, torch.ones_like(scale))
            # scale = torch.pow(2,torch.log2(scale).ceil())    # power of 2
            # We use the quantize function
            # xq = Round(Xf * inv_scale + zero_point),
            # setting zero_point to (-1 * min *inv_scale) we get
            # Xq = Round((Xf - min) * inv_scale)
            zero_point = -1 * min_val / scale
        else:
            scale = (max_val_pos - min_val_neg) / float(quant_max - quant_min)
            scale = torch.max(scale, self.eps)
            # scale = torch.pow(2,torch.log2(scale).ceil())    # power of 2
            zero_point = quant_min - torch.round(min_val_neg / scale).to(torch.int)
            zero_point = torch.clamp(zero_point, quant_min, quant_max)

        # For scalar values, cast them to Tensors of size 1 to keep the shape
        # consistent with default values in FakeQuantize.
        if len(scale.shape) == 0:
            # TODO: switch to scale.item() after adding JIT support
            scale = torch.tensor([float(scale)], dtype=scale.dtype, device=device)
        if len(zero_point.shape) == 0:
            # TODO: switch to zero_point.item() after adding JIT support
            zero_point = torch.tensor(
                [int(zero_point)], dtype=zero_point.dtype, device=device
            )
            if self.qscheme == torch.per_channel_affine_float_qparams:
                zero_point = torch.tensor(
                    [float(zero_point)], dtype=zero_point.dtype, device=device
                )
        scale = torch.pow(2,torch.log2(scale).ceil())
        return scale, zero_point

    @torch.jit.export
    def reset_min_max_vals(self):
        raise NotImplementedError("Cannot reset min/max values in the given observer.")

class Pow2MinMaxObserver(Pow2ObserverBase):
    min_val: torch.Tensor
    max_val: torch.Tensor
    def __init__(
        self,
        dtype=torch.quint8,
        qscheme=torch.per_tensor_affine,
        reduce_range=False,
        quant_min=None,
        quant_max=None,
        factory_kwargs=None,
    )-> None:
        super(Pow2MinMaxObserver, self).__init__(
            dtype=dtype,
            qscheme=qscheme,
            reduce_range=reduce_range,
            quant_min=quant_min,
            quant_max=quant_max,
            factory_kwargs=factory_kwargs,
        )
        factory_kwargs = torch.nn.factory_kwargs(factory_kwargs)
        self.register_buffer("min_val", torch.tensor(float("inf"), **factory_kwargs))
        self.register_buffer("max_val", torch.tensor(float("-inf"), **factory_kwargs))
        if (
            self.qscheme == torch.per_tensor_symmetric
            and self.reduce_range
            and self.dtype == torch.quint8
        ):
            raise NotImplementedError(
                "Cannot reduce range for symmetric \
                                       quantization for quint8"
            )
    def forward(self, x_orig):
        r"""Records the running minimum and maximum of ``x``."""
        if x_orig.numel() == 0:
            return x_orig
        x = x_orig.detach()  # avoid keeping autograd tape
        x = x.to(self.min_val.dtype)
        min_val_cur, max_val_cur = torch._aminmax(x)
        min_val = torch.min(min_val_cur, self.min_val)
#         min_val = 2 ** torch.log2(min_val).floor()
        
        max_val = torch.max(max_val_cur, self.max_val)
#         max_val = 2 ** torch.log2(max_val).ceil()
        self.min_val.copy_(min_val)
        self.max_val.copy_(max_val)
        return x_orig

    @torch.jit.export
    def calculate_qparams(self):
        r"""Calculates the quantization parameters."""
        return self._calculate_qparams(self.min_val, self.max_val)

    @torch.jit.export
    def extra_repr(self):
        return "min_val={}, max_val={}".format(self.min_val, self.max_val)

    @torch.jit.export
    def reset_min_max_vals(self):
        """Resets the min/max values."""
        self.min_val = torch.tensor(float("inf"))
        self.max_val = torch.tensor(float("-inf"))
        
class Pow2MovingAverageMinMaxObserver(Pow2MinMaxObserver):
    r"""Observer module for computing the quantization parameters based on the
    moving average of the min and max values.

    This observer computes the quantization parameters based on the moving
    averages of minimums and maximums of the incoming tensors. The module
    records the average minimum and maximum of incoming tensors, and uses this
    statistic to compute the quantization parameters.

    Args:
        averaging_constant: Averaging constant for min/max.
        dtype: Quantized data type
        qscheme: Quantization scheme to be used
        reduce_range: Reduces the range of the quantized data type by 1 bit
        quant_min: Minimum quantization value. If unspecified, it will follow the 8-bit setup.
        quant_max: Maximum quantization value. If unspecified, it will follow the 8-bit setup.

    The moving average min/max is computed as follows

    .. math::

        \begin{array}{ll}
                x_\text{min} = \begin{cases}
                    \min(X) & \text{if~}x_\text{min} = \text{None} \\
                    (1 - c) x_\text{min} + c \min(X) & \text{otherwise}
                \end{cases}\\
                x_\text{max} = \begin{cases}
                    \max(X) & \text{if~}x_\text{max} = \text{None} \\
                    (1 - c) x_\text{max} + c \max(X) & \text{otherwise}
                \end{cases}\\
        \end{array}

    where :math:`x_\text{min/max}` is the running average min/max, :math:`X` is
    is the incoming tensor, and :math:`c` is the ``averaging_constant``.

    The scale and zero point are then computed as in
    :class:`~torch.quantization.observer.MinMaxObserver`.

    .. note:: Only works with ``torch.per_tensor_affine`` quantization scheme.

    .. note:: If the running minimum equals to the running maximum, the scale
              and zero_point are set to 1.0 and 0.
    """

    def __init__(
        self,
        averaging_constant=0.01,
        dtype=torch.quint8,
        qscheme=torch.per_tensor_affine,
        reduce_range=False,
        quant_min=None,
        quant_max=None,
        **kwargs
    ) -> None:
        self.averaging_constant = averaging_constant
        super(Pow2MovingAverageMinMaxObserver, self).__init__(
            dtype=dtype,
            qscheme=qscheme,
            reduce_range=reduce_range,
            quant_min=quant_min,
            quant_max=quant_max,
            **kwargs
        )

    def forward(self, x_orig):
        if x_orig.numel() == 0:
            return x_orig
        x = x_orig.detach()  # avoid keeping autograd tape
        x = x.to(self.min_val.dtype)
        min_val = self.min_val
        max_val = self.max_val
        if min_val == float("inf") and max_val == float("-inf"):
            min_val, max_val = torch._aminmax(x)
        else:
            min_val_cur, max_val_cur = torch._aminmax(x)
            min_val = min_val + self.averaging_constant * (min_val_cur - min_val)
            max_val = max_val + self.averaging_constant * (max_val_cur - max_val)
        self.min_val.copy_(min_val)
        self.max_val.copy_(max_val)
        return x_orig

class Pow2PerChannelMinMaxObserver(Pow2ObserverBase):
    r"""Observer module for computing the quantization parameters based on the
    running per channel min and max values.

    This observer uses the tensor min/max statistics to compute the per channel
    quantization parameters. The module records the running minimum and maximum
    of incoming tensors, and uses this statistic to compute the quantization
    parameters.

    Args:
        ch_axis: Channel axis
        dtype: Quantized data type
        qscheme: Quantization scheme to be used
        reduce_range: Reduces the range of the quantized data type by 1 bit
        quant_min: Minimum quantization value. If unspecified, it will follow the 8-bit setup.
        quant_max: Maximum quantization value. If unspecified, it will follow the 8-bit setup.

    The quantization parameters are computed the same way as in
    :class:`~torch.quantization.observer.MinMaxObserver`, with the difference
    that the running min/max values are stored per channel.
    Scales and zero points are thus computed per channel as well.

    .. note:: If the running minimum equals to the running maximum, the scales
              and zero_points are set to 1.0 and 0.
    """
    min_val: torch.Tensor
    max_val: torch.Tensor

    def __init__(
        self,
        ch_axis=0,
        dtype=torch.quint8,
        qscheme=torch.per_channel_affine,
        reduce_range=False,
        quant_min=None,
        quant_max=None,
        factory_kwargs=None,
    ) -> None:
        super(Pow2PerChannelMinMaxObserver, self).__init__(
            dtype=dtype,
            qscheme=qscheme,
            reduce_range=reduce_range,
            quant_min=quant_min,
            quant_max=quant_max,
            factory_kwargs=factory_kwargs,
        )
        factory_kwargs = torch.nn.factory_kwargs(factory_kwargs)
        self.ch_axis = ch_axis
        self.register_buffer("min_val", torch.tensor([], **factory_kwargs))
        self.register_buffer("max_val", torch.tensor([], **factory_kwargs))
        if (
            self.qscheme == torch.per_channel_symmetric
            and self.reduce_range
            and self.dtype == torch.quint8
        ):
            raise NotImplementedError(
                "Cannot reduce range for symmetric quantization for quint8"
            )

    def forward(self, x_orig):
        return self._forward(x_orig)

    def _forward(self, x_orig):
        if x_orig.numel() == 0:
            return x_orig
        x = x_orig.detach()  # avoid keeping autograd tape
        min_val = self.min_val
        max_val = self.max_val
        x_dim = x.size()

        new_axis_list = [i for i in range(len(x_dim))]  # noqa: C416
        new_axis_list[self.ch_axis] = 0
        new_axis_list[0] = self.ch_axis
        y = x.permute(new_axis_list)
        # Need to match dtype of min/max because the updates to buffers
        # are done in place and types need to match for comparisons
        y = y.to(self.min_val.dtype)
        y = torch.flatten(y, start_dim=1)
        if min_val.numel() == 0 or max_val.numel() == 0:
            min_val, max_val = torch._aminmax(y, 1)
        else:
            min_val_cur, max_val_cur = torch._aminmax(y, 1)
            min_val = torch.min(min_val_cur, min_val)
            max_val = torch.max(max_val_cur, max_val)
        self.min_val.resize_(min_val.shape)
        self.max_val.resize_(max_val.shape)
        self.min_val.copy_(min_val)
        self.max_val.copy_(max_val)
        return x_orig

    @torch.jit.export
    def calculate_qparams(self):
        return self._calculate_qparams(self.min_val, self.max_val)

    def extra_repr(self):
        return "min_val={}, max_val={}".format(self.min_val, self.max_val)

    def _load_from_state_dict(
        self,
        state_dict: Union[Dict[str, torch.Tensor], Dict[str, torch.Tensor]],
        prefix: str,
        local_metadata: Dict[str, torch.Tensor],
        strict: bool,
        missing_keys: List[str],
        unexpected_keys: List[str],
        error_msgs: List[str],
    ):
        version = local_metadata.get("version", None)
        if version is None or version < 3:
            local_state = ["min_vals", "max_vals"]
            expected_min_name = "min_vals"
            expected_max_name = "max_vals"
        else:
            local_state = ["min_val", "max_val"]
            expected_min_name = "min_val"
            expected_max_name = "max_val"
        for name in local_state:
            key = prefix + name
            if key in state_dict:
                val = state_dict[key]
                # Custom handling to allow loading min_val or max_val
                # of size N into uninitialized buffers of size 0. The
                # buffers are resized here, and the values are copied in
                # the default state_dict loading code of the parent.
                if name == expected_min_name:
                    self.min_val.resize_(val.shape)
                elif name == expected_max_name:
                    self.max_val.resize_(val.shape)
                else:
                    warnings.warn("Observer load_from_state_dict got unexpected name {}".format(name))
                # For torchscript module we need to update the attributes here since we do not
                # call the `_load_from_state_dict` function defined module.py
                if torch.jit.is_scripting():
                    if name == expected_min_name:
                        self.min_val.copy_(val)
                    elif name == expected_max_name:
                        self.max_val.copy_(val)
                    else:
                        warnings.warn("Observer load_from_state_dict got unexpected name {}".format(name))
            elif strict:
                missing_keys.append(key)

        if not torch.jit.is_scripting():
            super(Pow2PerChannelMinMaxObserver, self)._load_from_state_dict(
                state_dict,
                prefix,
                local_metadata,
                False,
                missing_keys,
                unexpected_keys,
                error_msgs,
            )

    def _load_from_state_dict_script(
        self,
        state_dict: Union[Dict[str, torch.Tensor], Dict[str, torch.Tensor]],
        prefix: str,
        local_metadata: Dict[str, torch.Tensor],
        strict: bool,
        missing_keys: List[str],
        unexpected_keys: List[str],
        error_msgs: List[str],
    ):

        self._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    @torch.jit.export
    def reset_min_max_vals(self):
        """Resets the min/max values."""
        self.min_val = torch.tensor([])
        self.max_val = torch.tensor([])

class Pow2MovingAveragePerChannelMinMaxObserver(Pow2PerChannelMinMaxObserver):
    r"""Observer module for computing the quantization parameters based on the
    running per channel min and max values.
    This observer uses the tensor min/max statistics to compute the per channel
    quantization parameters. The module records the running minimum and maximum
    of incoming tensors, and uses this statistic to compute the quantization
    parameters.
    Args:
        averaging_constant: Averaging constant for min/max.
        ch_axis: Channel axis
        dtype: Quantized data type
        qscheme: Quantization scheme to be used
        reduce_range: Reduces the range of the quantized data type by 1 bit
        quant_min: Minimum quantization value. If unspecified, it will follow the 8-bit setup.
        quant_max: Maximum quantization value. If unspecified, it will follow the 8-bit setup.
    The quantization parameters are computed the same way as in
    :class:`~torch.ao.quantization.observer.MovingAverageMinMaxObserver`, with the
    difference that the running min/max values are stored per channel.
    Scales and zero points are thus computed per channel as well.
    .. note:: If the running minimum equals to the running maximum, the scales
              and zero_points are set to 1.0 and 0.
    """

    def __init__(
        self,
        averaging_constant=0.01,
        ch_axis=0,
        dtype=torch.quint8,
        qscheme=torch.per_channel_affine,
        reduce_range=False,
        quant_min=None,
        quant_max=None,
        **kwargs
    ) -> None:
        super(Pow2MovingAveragePerChannelMinMaxObserver, self).__init__(
            ch_axis=ch_axis,
            dtype=dtype,
            qscheme=qscheme,
            reduce_range=reduce_range,
            quant_min=quant_min,
            quant_max=quant_max,
            **kwargs
        )
        self.averaging_constant = averaging_constant

    def forward(self, x_orig):
        if x_orig.numel() == 0:
            return x_orig
        x = x_orig.detach()  # avoid keeping autograd tape
        x = x.to(self.min_val.dtype)
        min_val = self.min_val
        max_val = self.max_val
        x_dim = x.size()

        new_axis_list = [i for i in range(len(x_dim))]  # noqa: C416
        new_axis_list[self.ch_axis] = 0
        new_axis_list[0] = self.ch_axis
        y = x.permute(new_axis_list)
        y = torch.flatten(y, start_dim=1)
        if min_val.numel() == 0 or max_val.numel() == 0:
            min_val, max_val = torch._aminmax(y, 1)
        else:
            min_val_cur, max_val_cur = torch._aminmax(y, 1)
            min_val = min_val + self.averaging_constant * (min_val_cur - min_val)
            max_val = max_val + self.averaging_constant * (max_val_cur - max_val)
        self.min_val.resize_(min_val.shape)
        self.max_val.resize_(max_val.shape)
        self.min_val.copy_(min_val)
        self.max_val.copy_(max_val)
        return x_orig
