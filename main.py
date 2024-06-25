#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Power by Zongsheng Yue 2022-05-19 15:15:17
import warnings

'''
terms["loss"] =  (terms["disent"] + terms["mse"] + terms["ms_ssim_val"] + terms["l1_loss"]) * weights
E:\experiment\Daul_resshift2\saved_logs\2024-05-12-15-49

terms["loss"] =  (terms["mse"]) * weights
E:\experiment\Daul_resshift2\saved_logs\2024-05-30-14-47


terms["loss"] =  (terms["disent"] + terms["mse"] + terms["ms_ssim_val"] + terms["l1_loss"]) * weights
E:\experiment\Daul_resshift2\saved_logs\2024-06-02-16-27
'''

warnings.filterwarnings('ignore')
import argparse
from omegaconf import OmegaConf
from trainer import TrainerDifIR as Trainer

def get_parser(**parser_kwargs):
    parser = argparse.ArgumentParser(**parser_kwargs)
    parser.add_argument(
            "--save_dir",
            type=str,
            default="./saved_logs",
            help="Folder to save the checkpoints and training log",
            )
    parser.add_argument(
            "--resume",
            type=str,
            const=True,
            default="",
            nargs="?",
            help="Resume from the save_dir or checkpoint",
            )
    parser.add_argument(
            "--cfg_path",
            type=str,
            # default="./configs/realsr_swinunet_realesrgan256.yaml",
            default="./configs/train.yaml",
            help="Configs of yaml file",
            )
    parser.add_argument(
            "--steps",
            type=int,
            default=15,
            help="Hyper-paLrameters of diffusion steps",
            )
    args = parser.parse_args()

    return args

if __name__ == "__main__":
    args = get_parser()
    import warnings

    warnings.filterwarnings('ignore')

    configs = OmegaConf.load(args.cfg_path)
    configs.diffusion.params.steps = args.steps

    # merge args to config
    for key in vars(args):
        if key in ['cfg_path', 'save_dir', 'resume', ]:
            configs[key] = getattr(args, key)

    trainer = Trainer(configs)
    trainer.train()
'''
save img tensor

from  torchvision import utils as vutils
vutils.save_image(_z, 'E:/code/ResShift-unet/temp_img/z_0.jpg', normalize=True)

'''