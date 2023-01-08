import torch.nn as nn
from torch.nn.init import _calculate_correct_fan, calculate_gain
import math


def kaiming_uniform_WIL(tensor, a=0, mode="fan_in", nonlinearity="relu"):
    if torch.overrides.has_torch_function_variadic(tensor):
        return torch.overrides.handle_torch_function(
            kaiming_uniform_,
            (tensor,),
            tensor=tensor,
            a=a,
            mode=mode,
            nonlinearity=nonlinearity,
        )

    if 0 in tensor.shape:
        warnings.warn("Initializing zero-element tensors is a no-op")
        return tensor
    gain = calculate_gain(nonlinearity, a)
    bound = math.sqrt(3.0) * gain  # Calculate uniform bounds from gain
    with torch.no_grad():
        return tensor.uniform_(-bound, bound)


class LinearWIL(nn.Linear):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        device=None,
        dtype=None,
    ) -> None:
        super(LinearWIL, self).__init__(in_features, out_features, bias, device, dtype)

    def reset_parameters(self) -> None:
        # The same as PyTorch's regular kaiming_uniform without scaling
        kaiming_uniform_WIL(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            init.uniform_(self.bias, -1, 1)

    def forward(self, x):
        # Scaling the pre-activation response of the Linear layer
        fan_in = _calculate_correct_fan(input, "fan_in")
        return torch.div(F.linear(input, self.weight, self.bias), math.sqrt(fan_in))


class Conv2DWIL(nn.Conv2D):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _size_2_t,
        stride: _size_2_t = 1,
        padding: Union[str, _size_2_t] = 0,
        dilation: _size_2_t = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = "zeros",  # TODO: refine this type
        device=None,
        dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        kernel_size_ = _pair(kernel_size)
        stride_ = _pair(stride)
        padding_ = padding if isinstance(padding, str) else _pair(padding)
        dilation_ = _pair(dilation)
        super(Conv2DWIL, self).__init__(
            in_channels,
            out_channels,
            kernel_size_,
            stride_,
            padding_,
            dilation_,
            False,
            _pair(0),
            groups,
            bias,
            padding_mode,
            **factory_kwargs
        )

    def reset_parameters(self) -> None:
        # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
        # uniform(-1/sqrt(k), 1/sqrt(k)), where k = weight.size(1) * prod(*kernel_size)
        # For more details see: https://github.com/pytorch/pytorch/issues/15314#issuecomment-477448573
        kaiming_uniform_WIL(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            init.uniform_(self.bias, -1, 1)

    def forward(self, input: Tensor) -> Tensor:
        fan_in = _calculate_correct_fan(input, "fan_in")
        return torch.div(
            self._conv_forward(input, self.weight, self.bias), math.sqrt(fan_in)
        )
