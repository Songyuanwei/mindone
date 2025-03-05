import argparse
import json
import os
import random
import re
import subprocess
import sys
import time
from collections import OrderedDict
from typing import Optional, Union

import numpy as np
from tap import Tap

import mindspore as ms


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ms_mode", type=int, default=1, help="Running in GRAPH_MODE(0) or PYNATIVE_MODE(1) (default=1)"
    )
    parser.add_argument(
        "--jit_level",
        default="O0",
        type=str,
        choices=["O0", "O1", "O2"],
        help="Used to control the compilation optimization level. Supports ['O0', 'O1', 'O2']."
        "O0: Except for optimizations that may affect functionality, all other optimizations are turned off, adopt KernelByKernel execution mode."
        "O1: Using commonly used optimizations and automatic operator fusion optimizations, adopt KernelByKernel execution mode."
        "O2: Ultimate performance optimization, adopt Sink execution mode.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="the seed (for reproducible sampling)",
    )
    parser.add_argument(
        "--vae_checkpoint",
        type=str,
        default="model/vae-2972c247.ckpt",
        help="VAE checkpoint file path which is used to load vae weight.",
    )
    parser.add_argument(
        "--var_checkpoint",
        type=str,
        default="model/var-d16.ckpt",
        help="VAR checkpoint file path which is used to load var weight.",
    )
    parser.add_argument(
        "--dtype",
        default="fp16",
        type=str,
        choices=["bf16", "fp16", "fp32"],
        help="what data type to use for latte. Default is `fp32`, which corresponds to ms.float16",
    )
    # data
    parser.add_argument(
        "--data_path",
        type=str,
        default="/path/to/imagenet",
        help="path to imagenet.",
    )
    parser.add_argument("--pn", default="1_2_3_4_5_6_8_10_13_16", type=str, help="patch_nums")
    parser.add_argument("--patch_size", default="16", type=int, help="patch size")

    # VAR
    parser.add_argument("--depth", type=int, default=16, help="VAR depth")
    parser.add_argument("--ini", type=float, default=-1.0, help="automated model parameter initialization")
    parser.add_argument("--hd", type=float, default=0.02, help="head.w *= hd")
    parser.add_argument("--aln", type=float, default=0.5, help="the multiplier of ada_lin.w's initialization")
    parser.add_argument(
        "--alng", type=float, default=1e-5, help="the multiplier of ada_lin.w[gamma channels]'s initialization"
    )

    parser.add_argument("--batch_size", default=96, type=int, help="batch size")

    # optimizer
    parser.add_argument("--optim", default="adamw", type=str, help="optimizer")
    parser.add_argument(
        "--betas", type=float, default=[0.9, 0.999], help="Specify the [beta1, beta2] parameter for the Adam optimizer."
    )
    parser.add_argument("--weight_decay", default=1e-6, type=float, help="Weight decay.")
    parser.add_argument("--afuse", default=True, type=bool, help="fused adamw.")

    # learning rate
    parser.add_argument("--epoch", default=250, type=int, help="epoch")
    parser.add_argument("--wp", default=0, type=int, help="warm up epoch")
    parser.add_argument("--wp0", default=0.005, type=float, help="initial lr ratio at the begging of lr warm up")
    parser.add_argument("--wpe", default=0.01, type=float, help="final lr ratio at the end of training")

    # other hps
    parser.add_argument("--saln", default=False, type=bool, help="whether to use shared adaln")
    parser.add_argument("--anorm", default=True, type=bool, help="whether to use L2 normalized attention")
    parser.add_argument(
        "--fused",
        default=True,
        type=bool,
        help="whether to use fused op like flash attn, xformers, fused MLP, fused LayerNorm, etc.",
    )

    # progressive training
    parser.add_argument(
        "--pg", default=0.0, type=float, help=">0 for use progressive training during [0%, this] of training"
    )
    parser.add_argument(
        "--pg0",
        default=4,
        type=int,
        help="progressive initial stage, 0: from the 1st token map, 1: from the 2nd token map, etc",
    )
    parser.add_argument("--pgwp", default=0, type=float, help="num of warmup epochs at each progressive stage")

    return parser


def parse_args():
    parser = argparse.ArgumentParser()
    parser = parse_args(parser)
    args = parser.parse_args()
    if args.pn == "256":
        args.pn = "1_2_3_4_5_6_8_10_13_16"
    elif args.pn == "512":
        args.pn = "1_2_3_4_6_9_13_18_24_32"
    elif args.pn == "1024":
        args.pn = "1_2_3_4_5_7_9_12_16_21_27_36_48_64"
    args.patch_nums = tuple(map(int, args.pn.replace("-", "_").split("_")))
    args.resos = tuple(pn * args.patch_size for pn in args.patch_nums)
    args.data_load_reso = max(args.resos)

    if args.wp == 0:
        args.wp = args.ep * 1 / 50

    # update args: progressive training
    if args.pgwp == 0:
        args.pgwp = args.ep * 1 / 300
    if args.pg > 0:
        args.sche = f"lin{args.pg:g}"
