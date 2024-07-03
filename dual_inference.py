#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Power by Zongsheng Yue 2023-03-11 17:17:41

import os, sys
import argparse
from pathlib import Path

from omegaconf import OmegaConf
from dual_sampler import ResShiftSampler

from utils.util_opts import str2bool
from basicsr.utils.download_util import load_file_from_url

def get_parser(**parser_kwargs):
    parser = argparse.ArgumentParser(**parser_kwargs)
    parser.add_argument("-o", "--out_path", type=str, default="results2/ias", help="Output path.")
    parser.add_argument("-m", "--model_path", type=str, \
                        default=r"weights/model_100000.pth", \
                        help="Model path.")
    parser.add_argument("-s", "--steps", type=int, default=15, help="Diffusion length.")
    parser.add_argument("--scale", type=int, default=4, help="Scale factor for SR.")
    parser.add_argument("--seed", type=int, default=12345, help="Random seed.")
    args = parser.parse_args()

    return args

def get_configs(args):
    configs = OmegaConf.load('./configs/train.yaml')
    configs.model.ckpt_path = str(args.model_path)
    configs.diffusion.params.steps = args.steps
    configs.diffusion.params.sf = args.scale

    autoencoder_scale = 2 ** (len(configs.autoencoder.params.ddconfig.ch_mult) - 1)
    desired_min_size = 64 * (autoencoder_scale // args.scale)

    return configs, desired_min_size

def main():
    args = get_parser()

    configs, desired_min_size = get_configs(args)

    resshift_sampler = ResShiftSampler(
            configs,
            sf = 4,
            use_fp16=False,
            bs = 1,
            seed=args.seed,
            )

    resshift_sampler.inference(args.out_path,  noise_repeat=False)

if __name__ == '__main__':
    main()
