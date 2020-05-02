#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""RegNet models."""

import logging
import numpy as np

from detectron2.modeling.backbone.build import BACKBONE_REGISTRY
from detectron2.modeling.backbone import FPN
from detectron2.layers import ShapeSpec

from .fpn import LastLevelP6, LastLevelP6P7
from .anynet import AnyNet


logger = logging.getLogger(__name__)


def quantize_float(f, q):
    """Converts a float to closest non-zero int divisible by q."""
    return int(round(f / q) * q)


def adjust_ws_gs_comp(ws, bms, gs):
    """Adjusts the compatibility of widths and groups."""
    ws_bot = [int(w * b) for w, b in zip(ws, bms)]
    gs = [min(g, w_bot) for g, w_bot in zip(gs, ws_bot)]
    ws_bot = [quantize_float(w_bot, g) for w_bot, g in zip(ws_bot, gs)]
    ws = [int(w_bot / b) for w_bot, b in zip(ws_bot, bms)]
    return ws, gs


def get_stages_from_blocks(ws, rs):
    """Gets ws/ds of network at each stage from per block values."""
    ts_temp = zip(ws + [0], [0] + ws, rs + [0], [0] + rs)
    ts = [w != wp or r != rp for w, wp, r, rp in ts_temp]
    s_ws = [w for w, t in zip(ws, ts[:-1]) if t]
    s_ds = np.diff([d for d, t in zip(range(len(ts)), ts) if t]).tolist()
    return s_ws, s_ds


def generate_regnet(w_a, w_0, w_m, d, q=8):
    """Generates per block ws from RegNet parameters."""
    assert w_a >= 0 and w_0 > 0 and w_m > 1 and w_0 % q == 0
    ws_cont = np.arange(d) * w_a + w_0
    ks = np.round(np.log(ws_cont / w_0) / np.log(w_m))
    ws = w_0 * np.power(w_m, ks)
    ws = np.round(np.divide(ws, q)) * q
    num_stages, max_stage = len(np.unique(ws)), ks.max() + 1
    ws, ws_cont = ws.astype(int).tolist(), ws_cont.tolist()
    return ws, num_stages, max_stage, ws_cont


class RegNet(AnyNet):
    """RegNet model."""

    def __init__(self, cfg):
        # Generate RegNet ws per block
        b_ws, num_s, _, _ = generate_regnet(
            cfg.MODEL.REGNET.WA, cfg.MODEL.REGNET.W0,
            cfg.MODEL.REGNET.WM, cfg.MODEL.REGNET.DEPTH
        )
        # Convert to per stage format
        ws, ds = get_stages_from_blocks(b_ws, b_ws)
        # Generate group widths and bot muls
        gws = [cfg.MODEL.REGNET.GROUP_W for _ in range(num_s)]
        bms = [cfg.MODEL.REGNET.BOT_MUL for _ in range(num_s)]
        # Adjust the compatibility of ws and gws
        ws, gws = adjust_ws_gs_comp(ws, bms, gws)
        # Use the same stride for each stage
        ss = [cfg.MODEL.REGNET.STRIDE for _ in range(num_s)]
        # Use SE for RegNetY
        se_r = cfg.MODEL.REGNET.SE_R if cfg.MODEL.REGNET.SE_ON else None
        # Construct the model
        kwargs = {
            "stem_type": cfg.MODEL.REGNET.STEM_TYPE,
            "stem_w": cfg.MODEL.REGNET.STEM_W,
            "block_type": cfg.MODEL.REGNET.BLOCK_TYPE,
            "ss": ss,
            "ds": ds,
            "ws": ws,
            "bms": bms,
            "gws": gws,
            "se_r": se_r,
            "norm": cfg.MODEL.REGNET.NORM
        }
        super(RegNet, self).__init__(**kwargs)


@BACKBONE_REGISTRY.register()
def build_fcos_regnet_fpn_backbone(cfg, input_shape: ShapeSpec):
    """
    Args:
        cfg: a detectron2 CfgNode

    Returns:
        backbone (Backbone): backbone module, must be a subclass of :class:`Backbone`.
    """
    assert cfg.MODEL.BACKBONE.FREEZE_AT == -1, "Freezing layers does not be supported for RegNet"

    bottom_up = RegNet(cfg)
    in_features = cfg.MODEL.FPN.IN_FEATURES
    out_channels = cfg.MODEL.FPN.OUT_CHANNELS
    top_levels = cfg.MODEL.FCOS.TOP_LEVELS
    in_channels_top = out_channels

    if top_levels == 2:
        top_block = LastLevelP6P7(in_channels_top, out_channels, "p5")
    elif top_levels == 1:
        top_block = LastLevelP6(in_channels_top, out_channels, "p5")
    elif top_levels == 0:
        top_block = None
    else:
        raise NotImplementedError()

    backbone = FPN(
        bottom_up=bottom_up,
        in_features=in_features,
        out_channels=out_channels,
        norm=cfg.MODEL.FPN.NORM,
        top_block=top_block,
        fuse_type=cfg.MODEL.FPN.FUSE_TYPE,
    )

    return backbone
