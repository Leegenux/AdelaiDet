import math
import torch
from torch import nn
import torch.nn.functional as F
import fvcore.nn.weight_init as weight_init
import copy

from detectron2.modeling.backbone import build_resnet_backbone
from detectron2.modeling.backbone.backbone import Backbone
from detectron2.modeling.backbone.fpn import _assert_strides_are_log2_contiguous
from detectron2.modeling.backbone.build import BACKBONE_REGISTRY
from detectron2.layers import Conv2d, ShapeSpec, get_norm

from .resnet_lpf import build_resnet_lpf_backbone
from .resnet_interval import build_resnet_interval_backbone
from .mobilenet import build_mnv2_backbone


class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class MemoryEfficientSwish(nn.Module):
    def forward(self, x):
        return SwishImplementation.apply(x)


class SwishImplementation(torch.autograd.Function):
    @staticmethod
    def forward(ctx, i):
        result = i * torch.sigmoid(i)
        ctx.save_for_backward(i)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        i = ctx.saved_variables[0]
        sigmoid_i = torch.sigmoid(i)
        return grad_output * (sigmoid_i * (1 + i * (1 - sigmoid_i)))


class BiFPN(Backbone):  # todo check initialization
    """
    This module implements :paper:`BiFPN`.
    It creates pyramid features built on top of some input feature maps.
    """

    def __init__(
            self, bottom_up, in_features, out_channels, norm="", top_block=None, epsilon=1E-4
    ):
        """
        Args:
            bottom_up (Backbone): module representing the bottom up subnetwork.
                Must be a subclass of :class:`Backbone`. The multi-scale feature
                maps generated by the bottom up network, and listed in `in_features`,
                are used to generate BiFPN levels.
            in_features (list[str]): names of the input feature maps coming
                from the backbone to which BiFPN is attached. For example, if the
                backbone produces ["res2", "res3", "res4"], any *contiguous* sublist
                of these may be used; order must be from high to low resolution.
            out_channels (int): number of channels in the output feature maps.
            norm (str): the normalization to use.
            top_block (nn.Module or None): if provided, an extra operation will
                be performed on the output of the last (smallest resolution)
                BiFPN output, and the result will extend the result list. The top_block
                further downsamples the feature map. It must have an attribute
                "num_levels", meaning the number of extra BiFPN levels added by
                this block, and "in_feature", which is a string representing
                its input feature (e.g., p5).
            fuse_type (str): types for fusing the top down features and the lateral
                ones. It can be "sum" (default), which sums up element-wise; or "avg",
                which takes the element-wise mean of the two; or "attention", which
                utilize trainable weights to conduct the fusion.
        """
        super(BiFPN, self).__init__()
        assert isinstance(bottom_up, Backbone), "A bottom_up have to be a Backbone instance!"

        # get the information of inputs from bottom-up
        input_shapes = bottom_up.output_shape()  # high resolution to low
        in_strides = [input_shapes[f].stride for f in in_features]  # strides against the original picture
        _assert_strides_are_log2_contiguous(in_strides)
        in_channels = [input_shapes[f].channels for f in in_features]  # channel numbers
        ## Return feature names are "p<stage>", like ["p2", "p3", ..., "p6"]
        self._out_feature_strides = {"p{}".format(int(math.log2(s))): s for s in in_strides}
        ## top block output feature maps.
        self.top_block = top_block
        if self.top_block is not None:
            stage = int(math.log2(in_strides[-1]))
            for s in range(stage, stage + self.top_block.num_levels):
                self._out_feature_strides["p{}".format(s + 1)] = 2 ** (s + 1)
        self._out_features = list(self._out_feature_strides.keys())  # out feature names
        self._out_feature_channels = {k: out_channels for k in self._out_features}  # same channels

        # other stuff
        self.in_features = in_features
        self.bottom_up = bottom_up
        self.swish = MemoryEfficientSwish()  # activation function
        self._size_divisibility = in_strides[-1]
        self.epsilon = epsilon

        # weights for the 2 and 3 features fusion respectively
        self.levels_num = len(self.in_features)
        self.weights_for_2_fusion = nn.Parameter(torch.Tensor(2, self.levels_num), requires_grad=True)
        self.weights_for_3_fusion = nn.Parameter(torch.Tensor(3, self.levels_num - 2), requires_grad=True)

        # 1x1 convs that resizes the input features to fit out channels
        self.is_in_and_out_channels_same = True
        for num_channel in in_channels:
            if num_channel != out_channels:
                self.is_in_and_out_channels_same = False
                break
        use_bias = norm == ""
        if not self.is_in_and_out_channels_same:  # resize layers if not all same
            self.resize_convs = []
            for num_in_channel in in_channels:
                self.resize_convs.append(Conv2d(
                    num_in_channel, out_channels, kernel_size=1, bias=use_bias, norm=get_norm(norm, out_channels)
                ))
            weight_init.c2_xavier_fill(self.resize_convs[-1])

        # bifpn blocks todo use add_module to add conv layers
        self.bifpn_convs = []
        for i in range(2 * self.levels_num - 1):
            self.bifpn_convs.append(
                Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=use_bias,
                       norm=get_norm(norm, out_channels))
            )
            weight_init.c2_xavier_fill(self.bifpn_convs[-1])

    @property
    def size_divisibility(self):
        return self._size_divisibility

    def forward(self, inputs):
        """
        Args:
            input (dict[str->Tensor]): mapping feature map name (e.g., "res5") to
                feature map tensor for each feature level in high to low resolution order.

        Returns:
            dict[str->Tensor]:
                mapping from feature map name to BiFPN feature map tensor
                in high to low resolution order. Returned feature names follow the BiFPN
                paper convention: "p<stage>", where stage has stride = 2 ** stage e.g.,
                ["p2", "p3", ..., "p6"].
        """
        # get features from bottom up
        bottom_up_features = self.bottom_up(inputs)
        inputs_from_bottom_up = [bottom_up_features[f] for f in self.in_features[::-1]]  #

        # fusion weights normalization
        weights_for_2_fusion = F.relu(self.weights_for_2_fusion) / (
                torch.sum(self.weights_for_2_fusion, dim=0) + self.epsilon)
        weights_for_3_fusion = F.relu(self.weights_for_3_fusion) / (
                torch.sum(self.weights_for_3_fusion, dim=0) + self.epsilon)

        # resize all the inputs channel-wise
        if not self.is_in_and_out_channels_same:
            for i, in_feat in enumerate(inputs_from_bottom_up):
                inputs_from_bottom_up[i] = self.resize_convs[i](in_feat)

        # preparation to build
        inter_features_result = copy.deepcopy(inputs_from_bottom_up)
        index_bifpn_convs = 0

        # build top-down structure
        for i in range(self.levels_num - 1, 0, -1):  # (levels_num - 1) fusions here
            inter_features_result[i - 1] = self.bifpn_convs[index_bifpn_convs]( \
                weights_for_2_fusion[0, i - 1] * inputs_from_bottom_up[i - 1] + \
                weights_for_2_fusion[1, i - 1] * F.interpolate(inter_features_result[i], scale_factor=2, mode='nearest')
            )
            index_bifpn_convs += 1

        # build down-up structure
        for i in range(0, self.levels_num - 2):  # (levels_num - 2) fusions here
            inter_features_result[i + 1] = self.bifpn_convs[index_bifpn_convs]( \
                weights_for_3_fusion[0, i] * inter_features_result[i + 1] + \
                weights_for_3_fusion[1, i] * inputs_from_bottom_up[i + 1] + \
                weights_for_3_fusion[2, i] * F.max_pool2d(inter_features_result[i], kernel_size=2)
            )
            index_bifpn_convs += 1

        # the last fusion of two nodes
        inter_features_result[self.levels_num - 1] = self.bifpn_convs[index_bifpn_convs]( \
            weights_for_2_fusion[0, self.levels_num - 1] * inter_features_result[self.levels_num - 1] + \
            weights_for_2_fusion[1, self.levels_num - 1] * F.max_pool2d(inter_features_result[self.levels_num - 2],
                                                                        kernel_size=2)
        )

        # add top-block results
        if self.top_block is not None:
            top_block_in_feature = bottom_up_features.get(self.top_block.in_feature, None)  # in feature is a
            if top_block_in_feature is None:
                top_block_in_feature = inter_features_result[self._out_features.index(self.top_block.in_feature)]
            inter_features_result.extend(self.top_block(top_block_in_feature))

        assert len(self._out_features) == len(inter_features_result)
        return dict(zip(self._out_features, inter_features_result))

    def output_shape(self):
        return {
            name: ShapeSpec(
                channels=self._out_feature_channels[name], stride=self._out_feature_strides[name]
            )
            for name in self._out_features
        }


@BACKBONE_REGISTRY.register()
def build_fcos_resnet_bifpn_backbone(cfg, input_shape: ShapeSpec):
    """
    Args:
        cfg: a detectron2 CfgNode

    Returns:
        backbone (Backbone): backbone module, must be a subclass of :class:`Backbone`.
    """
    if cfg.MODEL.BACKBONE.ANTI_ALIAS:
        bottom_up = build_resnet_lpf_backbone(cfg, input_shape)
    elif cfg.MODEL.RESNETS.DEFORM_INTERVAL > 1:
        bottom_up = build_resnet_interval_backbone(cfg, input_shape)
    elif cfg.MODEL.MOBILENET:
        bottom_up = build_mnv2_backbone(cfg, input_shape)
    else:
        bottom_up = build_resnet_backbone(cfg, input_shape)
    in_features = cfg.MODEL.BIFPN.IN_FEATURES
    out_channels = cfg.MODEL.BIFPN.OUT_CHANNELS

    backbone = BiFPN(
        bottom_up=bottom_up,
        in_features=in_features,
        out_channels=out_channels,
        norm=cfg.MODEL.BIFPN.NORM,
        # fuse_type=cfg.MODEL.BIFPN.FUSE_TYPE, # not supported
    )
    return backbone
