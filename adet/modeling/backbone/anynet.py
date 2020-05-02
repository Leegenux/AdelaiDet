#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""AnyNet models."""
import logging
import math
import torch.nn as nn

from detectron2.layers.batch_norm import get_norm
from detectron2.modeling.backbone import Backbone


logger = logging.getLogger(__name__)


def init_weights(m):
    """Performs ResNet-style weight initialization."""
    if isinstance(m, nn.Conv2d):
        # Note that there is no bias due to BN
        fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(mean=0.0, std=math.sqrt(2.0 / fan_out))
    elif isinstance(m, nn.BatchNorm2d):
        zero_init_gamma = (
            hasattr(m, "final_bn") and m.final_bn and True
        )
        m.weight.data.fill_(0.0 if zero_init_gamma else 1.0)
        m.bias.data.zero_()
    elif isinstance(m, nn.Linear):
        m.weight.data.normal_(mean=0.0, std=0.01)
        m.bias.data.zero_()


def get_stem_fun(stem_type):
    """Retrives the stem function by name."""
    stem_funs = {
        "res_stem_cifar": ResStemCifar,
        "res_stem_in": ResStemIN,
        "simple_stem_in": SimpleStemIN,
    }
    assert stem_type in stem_funs.keys(), "Stem type '{}' not supported".format(
        stem_type
    )
    return stem_funs[stem_type]


def get_block_fun(block_type):
    """Retrieves the block function by name."""
    block_funs = {
        "vanilla_block": VanillaBlock,
        "res_basic_block": ResBasicBlock,
        "res_bottleneck_block": ResBottleneckBlock,
    }
    assert block_type in block_funs.keys(), "Block type '{}' not supported".format(
        block_type
    )
    return block_funs[block_type]


class AnyHead(nn.Module):
    """AnyNet head."""

    def __init__(self, w_in, nc):
        super(AnyHead, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(w_in, nc, bias=True)

    def forward(self, x):
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class VanillaBlock(nn.Module):
    """Vanilla block: [3x3 conv, BN, Relu] x2"""

    def __init__(self, w_in, w_out, stride, bm=None, gw=None, se_r=None, norm=""):
        assert (
            bm is None and gw is None and se_r is None
        ), "Vanilla block does not support bm, gw, and se_r options"
        super(VanillaBlock, self).__init__()
        self._construct(w_in, w_out, stride, norm)

    def _construct(self, w_in, w_out, stride, norm):
        # 3x3, BN, ReLU
        self.a = nn.Conv2d(
            w_in, w_out, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.a_bn = get_norm(norm, w_out)
        self.a_relu = nn.ReLU(inplace=True)
        # 3x3, BN, ReLU
        self.b = nn.Conv2d(w_out, w_out, kernel_size=3, stride=1, padding=1, bias=False)
        self.b_bn = get_norm(norm, w_out)
        self.b_relu = nn.ReLU(inplace=True)

    def forward(self, x):
        for layer in self.children():
            x = layer(x)
        return x


class BasicTransform(nn.Module):
    """Basic transformation: [3x3 conv, BN, Relu] x2"""

    def __init__(self, w_in, w_out, stride, norm):
        super(BasicTransform, self).__init__()
        self._construct(w_in, w_out, stride, norm)

    def _construct(self, w_in, w_out, stride, norm):
        # 3x3, BN, ReLU
        self.a = nn.Conv2d(
            w_in, w_out, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.a_bn = get_norm(norm, w_out)
        self.a_relu = nn.ReLU(inplace=True)
        # 3x3, BN
        self.b = nn.Conv2d(w_out, w_out, kernel_size=3, stride=1, padding=1, bias=False)
        self.b_bn = get_norm(norm, w_out)
        self.b_bn.final_bn = True

    def forward(self, x):
        for layer in self.children():
            x = layer(x)
        return x


class ResBasicBlock(nn.Module):
    """Residual basic block: x + F(x), F = basic transform"""

    def __init__(self, w_in, w_out, stride, bm=None, gw=None, se_r=None, norm=""):
        assert (
            bm is None and gw is None and se_r is None
        ), "Basic transform does not support bm, gw, and se_r options"
        super(ResBasicBlock, self).__init__()
        self._construct(w_in, w_out, stride, norm)

    def _add_skip_proj(self, w_in, w_out, stride, norm):
        self.proj = nn.Conv2d(
            w_in, w_out, kernel_size=1, stride=stride, padding=0, bias=False
        )
        self.bn = get_norm(w_out, norm)

    def _construct(self, w_in, w_out, stride, norm):
        # Use skip connection with projection if shape changes
        self.proj_block = (w_in != w_out) or (stride != 1)
        if self.proj_block:
            self._add_skip_proj(w_in, w_out, stride, norm)
        self.f = BasicTransform(w_in, w_out, stride, norm)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        if self.proj_block:
            x = self.bn(self.proj(x)) + self.f(x)
        else:
            x = x + self.f(x)
        x = self.relu(x)
        return x


class SE(nn.Module):
    """Squeeze-and-Excitation (SE) block"""

    def __init__(self, w_in, w_se):
        super(SE, self).__init__()
        self._construct(w_in, w_se)

    def _construct(self, w_in, w_se):
        # AvgPool
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        # FC, Activation, FC, Sigmoid
        self.f_ex = nn.Sequential(
            nn.Conv2d(w_in, w_se, kernel_size=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(w_se, w_in, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.f_ex(self.avg_pool(x))


class BottleneckTransform(nn.Module):
    """Bottlenect transformation: 1x1, 3x3, 1x1"""

    def __init__(self, w_in, w_out, stride, bm, gw, se_r, norm):
        super(BottleneckTransform, self).__init__()
        self._construct(w_in, w_out, stride, bm, gw, se_r, norm)

    def _construct(self, w_in, w_out, stride, bm, gw, se_r, norm):
        # Compute the bottleneck width
        w_b = int(round(w_out * bm))
        # Compute the number of groups
        num_gs = w_b // gw
        # 1x1, BN, ReLU
        self.a = nn.Conv2d(w_in, w_b, kernel_size=1, stride=1, padding=0, bias=False)
        self.a_bn = get_norm(norm, w_b)
        self.a_relu = nn.ReLU(inplace=True)
        # 3x3, BN, ReLU
        self.b = nn.Conv2d(
            w_b, w_b, kernel_size=3, stride=stride, padding=1, groups=num_gs, bias=False
        )
        self.b_bn = get_norm(norm, w_b)
        self.b_relu = nn.ReLU(inplace=True)
        # Squeeze-and-Excitation (SE)
        if se_r:
            w_se = int(round(w_in * se_r))
            self.se = SE(w_b, w_se)
        # 1x1, BN
        self.c = nn.Conv2d(w_b, w_out, kernel_size=1, stride=1, padding=0, bias=False)
        self.c_bn = get_norm(norm, w_out)
        self.c_bn.final_bn = True

    def forward(self, x):
        for layer in self.children():
            x = layer(x)
        return x


class ResBottleneckBlock(nn.Module):
    """Residual bottleneck block: x + F(x), F = bottleneck transform"""

    def __init__(self, w_in, w_out, stride, bm=1.0, gw=1, se_r=None, norm=""):
        super(ResBottleneckBlock, self).__init__()
        self._construct(w_in, w_out, stride, bm, gw, se_r, norm=norm)

    def _add_skip_proj(self, w_in, w_out, stride, norm=""):
        self.proj = nn.Conv2d(
            w_in, w_out, kernel_size=1, stride=stride, padding=0, bias=False
        )
        self.bn = get_norm(norm, w_out)

    def _construct(self, w_in, w_out, stride, bm, gw, se_r, norm=""):
        # Use skip connection with projection if shape changes
        self.proj_block = (w_in != w_out) or (stride != 1)
        if self.proj_block:
            self._add_skip_proj(w_in, w_out, stride, norm=norm)
        self.f = BottleneckTransform(w_in, w_out, stride, bm, gw, se_r, norm=norm)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        if self.proj_block:
            x = self.bn(self.proj(x)) + self.f(x)
        else:
            x = x + self.f(x)
        x = self.relu(x)
        return x


class ResStemCifar(nn.Module):
    """ResNet stem for CIFAR."""

    def __init__(self, w_in, w_out, norm=""):
        super(ResStemCifar, self).__init__()
        self._construct(w_in, w_out, norm=norm)

    def _construct(self, w_in, w_out, norm=""):
        # 3x3, BN, ReLU
        self.conv = nn.Conv2d(
            w_in, w_out, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn = get_norm(norm, w_out)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        for layer in self.children():
            x = layer(x)
        return x


class ResStemIN(nn.Module):
    """ResNet stem for ImageNet."""

    def __init__(self, w_in, w_out, norm=""):
        super(ResStemIN, self).__init__()
        self._construct(w_in, w_out, norm=norm)

    def _construct(self, w_in, w_out, norm=""):
        # 7x7, BN, ReLU, maxpool
        self.conv = nn.Conv2d(
            w_in, w_out, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.bn = get_norm(norm, w_out)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        for layer in self.children():
            x = layer(x)
        return x


class SimpleStemIN(nn.Module):
    """Simple stem for ImageNet."""

    def __init__(self, in_w, out_w, norm):
        super(SimpleStemIN, self).__init__()
        self._construct(in_w, out_w, norm)

    def _construct(self, in_w, out_w, norm):
        # 3x3, BN, ReLU
        self.conv = nn.Conv2d(
            in_w, out_w, kernel_size=3, stride=2, padding=1, bias=False
        )
        self.bn = get_norm(norm, out_w)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        for layer in self.children():
            x = layer(x)
        return x


class AnyStage(nn.Module):
    """AnyNet stage (sequence of blocks w/ the same output shape)."""

    def __init__(self, w_in, w_out, stride, d, block_fun, bm, gw, se_r, norm=""):
        super(AnyStage, self).__init__()
        self._construct(w_in, w_out, stride, d, block_fun, bm, gw, se_r, norm=norm)

    def _construct(self, w_in, w_out, stride, d, block_fun, bm, gw, se_r, norm=""):
        # Construct the blocks
        for i in range(d):
            # Stride and w_in apply to the first block of the stage
            b_stride = stride if i == 0 else 1
            b_w_in = w_in if i == 0 else w_out
            # Construct the block
            self.add_module(
                "b{}".format(i + 1), block_fun(b_w_in, w_out, b_stride, bm, gw, se_r, norm=norm)
            )

    def forward(self, x):
        for block in self.children():
            x = block(x)
        return x


class AnyNet(Backbone):
    """AnyNet model."""

    def __init__(self, **kwargs):
        super(AnyNet, self).__init__()
        self._construct(
            stem_type=kwargs["stem_type"],
            stem_w=kwargs["stem_w"],
            block_type=kwargs["block_type"],
            ds=kwargs["ds"],
            ws=kwargs["ws"],
            ss=kwargs["ss"],
            bms=kwargs["bms"],
            gws=kwargs["gws"],
            se_r=kwargs["se_r"],
            norm=kwargs["norm"]
        )
        self.apply(init_weights)
        
        self._out_features = ["stage{}".format(i) for i in range(2, 6)]
        self._out_feature_channels = {k: kwargs["ws"][i] for i, k in enumerate(self._out_features)}
        self._out_feature_strides = {k: 2 ** (i + 2) for i, k in enumerate(self._out_features)}

    def _construct(self, stem_type, stem_w, block_type, ds, ws, ss, bms, gws, se_r, norm=""):
        logger.info("Constructing AnyNet: ds={}, ws={}".format(ds, ws))
        # Generate dummy bot muls and gs for models that do not use them
        bms = bms if bms else [1.0 for _d in ds]
        gws = gws if gws else [1 for _d in ds]
        # Group params by stage
        stage_params = list(zip(ds, ws, ss, bms, gws))
        # Construct the stem
        stem_fun = get_stem_fun(stem_type)
        self.stem = stem_fun(3, stem_w, norm=norm)
        # Construct the stages
        block_fun = get_block_fun(block_type)
        prev_w = stem_w
        for i, (d, w, s, bm, gw) in enumerate(stage_params):
            self.add_module(
                "s{}".format(i + 1), AnyStage(prev_w, w, s, d, block_fun, bm, gw, se_r, norm=norm)
            )
            prev_w = w

    def forward(self, x):
        results = []
        for i, module in enumerate(self.children()):
            x = module(x)
            if i > 0:
                results.append(x)

        assert len(self._out_features) == len(results)
        return dict(zip(self._out_features, results))
