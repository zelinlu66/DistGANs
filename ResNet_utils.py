import torch
from torch import nn
import torch.nn.functional as F
import numpy as np


class Initializer(object):
    """Initialization class that handles multiple kinds of parameter initializations."""

    def __init__(
        self, init, init_type='default', gain_sq_base=2.0, equalized_lr=False
    ):
        super(Initializer, self).__init__()

        self.init = init.casefold()
        self.init_type = init_type.casefold()
        self.gain_sq_base = gain_sq_base
        self.equalized_lr = equalized_lr

    @torch.no_grad()
    def get_init_bound_layer(self, tensor,        distribution_type, stride=1):
        distribution_type = distribution_type.casefold()
        if distribution_type not in ('uniform', 'normal',):
            raise ValueError(
                'Only uniform and normal distributions are supported.'
            )

        if self.init_type == 'default':
            fan_in, fan_out = self._calculate_fan_in_fan_out(
                tensor=tensor, stride=stride
            )
            std = self._calculate_init_weight_std(
                fan_in=fan_in, fan_out=fan_out
            )

        return np.sqrt(3) * std if distribution_type == 'uniform' else std

    @torch.no_grad()
    def _calculate_fan_in_fan_out(self, tensor, stride=1):
        dimensions = tensor.dim()
        if dimensions < 2:
            raise ValueError(
                'Fan in and fan out cannot be computed for tensor with fewer than 2 dimensions.'
            )

        fan_out = None
        if dimensions == 2:  # Linear
            fan_in = tensor.size(1)
            fan_out = tensor.size(0)
        else:
            receptive_field_size = 1
            if dimensions > 2:
                receptive_field_size = tensor[0][0].numel()
            fan_in = tensor.size(1) * receptive_field_size

        return fan_in, fan_out

    @torch.no_grad()
    def _calculate_init_weight_std(self, fan_in=None, fan_out=None):
        gain_sq = self.gain_sq_base / 2.0
        if fan_out is not None and fan_in is not None:
            fan = fan_in + fan_out
            gain_sq *= 2
        elif fan_in is not None:
            fan = fan_in
        elif fan_out is not None:
            fan = fan_out

        if self.init == 'he':
            gain_sq = 2.0 * gain_sq
        elif self.init == 'xavier':
            gain_sq = 1.0 * gain_sq

        std = np.sqrt(gain_sq / fan)

        return std


class Lambda(nn.Module):
    """Converts any function into a PyTorch Module."""

    def __init__(self, func, **kwargs):
        super(Lambda, self).__init__()
        self.func = func
        if kwargs:
            self.kwargs = kwargs
        else:
            self.kwargs = {}

    def forward(self, x):
        return self.func(x, **self.kwargs)


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
# Blur (Low-pass Filtering):
# --------------------------


def get_blur_op(blur_type, num_channels):
    """Options for low-pass filter operations. Only 3x3 kernels supported currently."""
    if blur_type.casefold() == 'box':
        blur_filter = torch.FloatTensor([1.0 / 9]).expand(
            num_channels, 1, 3, 3
        )
        stride = 3
    elif blur_type.casefold() == 'binomial':
        blur_filter = torch.FloatTensor(
            [
                [1.0 / 16, 2.0 / 16, 1.0 / 16],
                [2.0 / 16, 4.0 / 16, 2.0 / 16],
                [1.0 / 16, 2.0 / 16, 1.0 / 16],
            ]
        ).expand(num_channels, 1, 3, 3)
        stride = 1
    elif blur_type.casefold() == 'gaussian':
        raise NotImplementedError('Gaussian blur not yet implemented.')

    # meant to work as a PyTorch Module
    blur_op = Lambda(
        lambda x: F.conv2d(
            x,
            blur_filter.to(x.device),
            stride=stride,
            padding=1,
            groups=num_channels,
        )
    )

    return blur_op


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
# Pooling:
# --------


class NearestPool2d(nn.Module):
    def __init__(self):
        super(NearestPool2d, self).__init__()

    def forward(self, x):
        if x.dim() == 3:
            x = x.view(-1, *x.shape)
        return F.interpolate(x, scale_factor=0.5, mode='nearest')


class BilinearPool2d(nn.Module):
    def __init__(self, align_corners):
        super(BilinearPool2d, self).__init__()
        self.align_corners = align_corners

    def forward(self, x):
        if x.dim() == 3:
            x = x.view(-1, *x.shape)
        return F.interpolate(
            x,
            scale_factor=0.5,
            mode='bilinear',
            align_corners=self.align_corners,
        )


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
# Normalization:
# --------------


class PixelNorm2d(nn.Module):
    def __init__(self):
        super(PixelNorm2d, self).__init__()

    def forward(self, x, eps=1.0e-8):
        return x * (((x ** 2).mean(dim=1, keepdim=True) + eps).rsqrt())


class NormalizeLayer(nn.Module):
    """All normalization methods in one place."""

    def __init__(self, norm_type, ni=None, res=None):
        super(NormalizeLayer, self).__init__()

        norm_type = norm_type.lower()

        if norm_type in ('pixelnorm', 'pixel norm',):
            self.norm = PixelNorm2d()
        elif norm_type in ('instancenorm', 'instance norm',):
            self.norm = nn.InstanceNorm2d(ni, eps=1.0e-8)
        elif norm_type in ('batchnorm', 'batch norm',):
            assert isinstance(ni, int)
            self.norm = nn.BatchNorm2d(ni)
        elif norm_type in ('layernorm', 'layer norm',):
            # self.norm = F.layer_norm
            assert isinstance(ni, int)
            assert isinstance(res, int)
            self.norm = nn.LayerNorm([ni, res, res])
        else:
            raise Exception(f'`norm_type` == "{norm_type}" not supported.')

    def forward(self, x):
        return self.norm(x)


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
# Standard Linear Operations, but with features such as
# equalized LR, custom initialization, etc:
# ------------------------------------------------------------
class Conv2dEx(nn.Module):
    def __init__(
        self,
        ni,
        nf,
        ks,
        stride=1,
        padding=0,
        groups=1,
        init='he',
        init_type='default',
        gain_sq_base=2.0,
        equalized_lr=False,
        lrmul=1.0,
        include_bias=True,
    ):
        super(Conv2dEx, self).__init__()

        self.ni = ni
        self.nf = nf

        init = init.casefold()
        init_type = init_type.casefold()
        self.equalized_lr = equalized_lr
        self.use_lrmul = True if lrmul != 1.0 else False
        self.lrmul = lrmul

        self.initializer = None
        if init_type != 'standard normal':
            self.initializer = Initializer(
                init=init,
                init_type=init_type,
                gain_sq_base=gain_sq_base,
                equalized_lr=equalized_lr,
            )

        self.conv2d = nn.Conv2d(
            ni,
            nf,
            kernel_size=ks,
            stride=stride,
            padding=padding,
            groups=groups,
            bias=include_bias,
        )

        # Weights:
        self.wscale = None
        if init_type == ('default', 'resnet',) and init is not None:
            bound = self.initializer.get_init_bound_layer(
                tensor=self.conv2d.weight,
                distribution_type='Uniform',
                stride=stride,
            )
            if equalized_lr:
                self.wscale = bound
                self.conv2d.weight.data.uniform_(-1.0 / lrmul, 1.0 / lrmul)
            else:
                self.conv2d.weight.data.uniform_(-bound / lrmul, bound / lrmul)
        elif (
            init_type == 'standard normal'
            and init is None
            and not equalized_lr
            and not self.use_lrmul
        ):
            self.conv2d.weight.data.normal_(0.0, 1.0)

        # Biases:
        self.bias = None
        if include_bias:
            # init biases as 0, according to official implementations.
            self.conv2d.bias.data.fill_(0)

    def forward(self, x):
        if self.equalized_lr:
            x = self.conv2d(x.mul(self.wscale))
        else:
            x = self.conv2d(x)

        if self.use_lrmul:
            x *= self.lrmul

        return x


class Conv2dBias(nn.Module):
    def __init__(
        self,
        nf,
        lrmul=1.0,
        device='cuda' if torch.cuda.is_available() else 'cpu',
    ):
        super(Conv2dBias, self).__init__()

        self.use_lrmul = True if lrmul != 1.0 else False
        self.lrmul = lrmul

        self.bias = nn.Parameter(
            torch.FloatTensor(1, nf, 1, 1).to(device).fill_(0)
        )

    def forward(self, x):
        if self.use_lrmul:
            return x + self.bias.mul(self.lrmul)
        else:
            return x + self.bias


# ............................................................................ #


class LinearEx(nn.Module):
    def __init__(
        self,
        nin_feat,
        nout_feat,
        init='xavier',
        init_type='default',
        gain_sq_base=2.0,
        equalized_lr=False,
        lrmul=1.0,
        include_bias=True,
    ):
        super(LinearEx, self).__init__()

        self.nin_feat = nin_feat
        self.nout_feat = nout_feat

        init = init.casefold()
        init_type = init_type.casefold()
        self.equalized_lr = equalized_lr
        self.use_lrmul = True if lrmul != 1.0 else False
        self.lrmul = lrmul

        self.initializer = None
        if init_type != 'standard normal':
            self.initializer = Initializer(
                init=init,
                init_type=init_type,
                gain_sq_base=gain_sq_base,
                equalized_lr=equalized_lr,
            )

        self.linear = nn.Linear(nin_feat, nout_feat, bias=include_bias)

        # Weights:
        self.wscale = None
        if init_type in ('default', 'resnet',) and init is not None:
            bound = self.initializer.get_init_bound_layer(
                tensor=self.linear.weight, distribution_type='Uniform'
            )
            if equalized_lr:
                self.wscale = bound
                self.linear.weight.data.uniform_(-1.0 / lrmul, 1.0 / lrmul)
            else:
                self.linear.weight.data.uniform_(-bound / lrmul, bound / lrmul)
        elif (
            init_type == 'standard normal'
            and init is None
            and not equalized_lr
            and not self.use_lrmul
        ):
            self.linear.weight.data.normal_(0.0, 1.0)

        # Biases:
        self.bias = None
        if include_bias:
            # init biases as 0, according to official implementations.
            self.linear.bias.data.fill_(0)

    def forward(self, x):
        if self.equalized_lr:
            x = self.linear(x.mul(self.wscale))
        else:
            x = self.linear(x)

        if self.use_lrmul:
            x *= self.lrmul

        return x


class LinearBias(nn.Module):
    def __init__(
        self,
        nout_feat,
        lrmul=1.0,
        device='cuda' if torch.cuda.is_available() else 'cpu',
    ):
        super(LinearBias, self).__init__()

        self.use_lrmul = True if lrmul != 1.0 else False
        self.lrmul = lrmul

        self.bias = nn.Parameter(
            torch.FloatTensor(1, nout_feat).to(device).fill_(0)
        )

    def forward(self, x):
        if self.use_lrmul:
            return x + self.bias.mul(self.lrmul)
        else:
            return x + self.bias


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #


class ResBlock2d(nn.Module):
    def __init__(
        self,
        ni,
        nf,
        ks,
        norm_type,
        upsampler=None,
        pooler=None,
        init='He',
        nl=nn.ReLU(),
        res=None,
        flip_sampling=False,
        equalized_lr=False,
        blur_type=None,
    ):
        super(ResBlock2d, self).__init__()

        assert not (upsampler is not None and pooler is not None)

        padding = (ks - 1) // 2  # 'SAME' padding for stride 1 conv

        if not flip_sampling:
            self.nif = nf if (upsampler is not None and pooler is None) else ni
        else:
            self.nif = ni if (upsampler is None and pooler is not None) else nf
        self.convs = (
            Conv2dEx(
                ni,
                self.nif,
                ks=ks,
                stride=1,
                padding=padding,
                init=init,
                equalized_lr=equalized_lr,
            ),
            Conv2dEx(
                self.nif,
                nf,
                ks=ks,
                stride=1,
                padding=padding,
                init=init,
                equalized_lr=equalized_lr,
            ),
            Conv2dEx(
                ni,
                nf,
                ks=1,
                stride=1,
                padding=0,
                init='Xavier',
                equalized_lr=equalized_lr,
            ),  # this is same as a FC layer
        )

        blur_op = (
            get_blur_op(blur_type=blur_type, num_channels=self.convs[0].nf)
            if blur_type is not None
            else None
        )

        _norm_nls = (
            [NormalizeLayer(norm_type, ni=ni, res=res), nl],
            [NormalizeLayer(norm_type, ni=self.convs[0].nf, res=res), nl],
        )

        if upsampler is not None:
            _mostly_linear_op_1 = (
                [upsampler, self.convs[0], blur_op]
                if blur_type is not None
                else [upsampler, self.convs[0]]
            )
            _mostly_linear_op_2 = (
                [upsampler, self.convs[2], blur_op]
                if blur_type is not None
                else [upsampler, self.convs[2]]
            )
            _ops = (
                _mostly_linear_op_1,
                [self.convs[1]],
                _mostly_linear_op_2,
            )
        elif pooler is not None:
            _mostly_linear_op_1 = (
                [blur_op, self.convs[1], pooler]
                if blur_type is not None
                else [self.convs[1], pooler]
            )
            _mostly_linear_op_2 = (
                [blur_op, pooler, self.convs[2]]
                if blur_type is not None
                else [pooler, self.convs[2]]
            )
            _ops = (
                [self.convs[0]],
                _mostly_linear_op_1,
                _mostly_linear_op_2,
            )
        else:
            _ops = (
                [self.convs[0]],
                [self.convs[1]],
                [self.convs[2]],
            )

        self.conv_layer_1 = nn.Sequential(*(_norm_nls[0] + _ops[0]))
        self.conv_layer_2 = nn.Sequential(*(_norm_nls[1] + _ops[1]))

        if (upsampler is not None or pooler is not None) or ni != nf:
            self.skip_connection = nn.Sequential(*(_ops[2]))
        else:
            self.skip_connection = Lambda(lambda x: x)

    def forward(self, x):
        return self.skip_connection(x) + self.conv_layer_2(
            self.conv_layer_1(x)
        )
