#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Power by Zongsheng Yue 2022-07-13 16:59:27
import json
import os, sys, math, random

import cv2
import numpy as np
from pathlib import Path

import pandas as pd
from loguru import logger
from omegaconf import OmegaConf
from torchvision import utils as vutils
from utils import util_net
from utils import util_image
from utils import util_common
from torchvision import utils as vutils
import torch
import torch.nn.functional as F
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.utils.data as udata
from datapipe.datasets import create_dataset
from utils.util_image import ImageSpliterTh
from gbvs.saliency_models import gbvs
from datapipe.image_datasets import load_data
import torchvision.transforms.functional as TF
from skimage.metrics import peak_signal_noise_ratio
from skimage.metrics import structural_similarity
import cv2
import lpips, re
import torch

def sort_key(filename):
    # 提取文件名中的数字部分
    match = re.match(r'(\d+)\.jpg', filename)
    if match:
        return int(match.group(1))
    return filename


def compute_matrix(result_dir):
    # 存储列表
    result_dir_gt = os.path.join(result_dir, 'gt')
    result_dir_sr = os.path.join(result_dir, 'sr')
    result_images = []  # 结果图片名字列表
    GT_image = []  # GT图片名字列表
    image_number = []  # 图片读取求指标的id
    image_name = []  # 图片读取求指标的id 对应的图片名
    psnr_number = []  # psnr值列表
    ssim_number = []  # ssim值列表
    lpips_number = []  # lpips值列表
    for root, _, fnames in sorted(os.walk(result_dir_sr)):
        for fname in fnames:
            result_images.append(fname)  ##结果图片名字列表
    result_images = sorted(result_images, key=sort_key)                       #GT图片名字列表
    loss_fn_vgg = lpips.LPIPS(net='alex').to("cuda:0")
    for i in range(len(result_images)):
        gt_name = result_images[i].replace(".png", ".jpg")
        gt_path = result_dir_gt + "/" + gt_name
        result_path = result_dir_sr + "/" + result_images[i]

        # 开始计算指标
        result = cv2.imread(result_path)
        GT = cv2.imread(gt_path)
        psnr = peak_signal_noise_ratio(GT, result)
        ssim = structural_similarity(GT, result, channel_axis=2, data_range=255)

        restored = result.transpose(2, 0, 1)
        target = GT.transpose(2, 0, 1)

        restored = torch.tensor(restored).cuda()
        target = torch.tensor(target).cuda()

        _lpips_value = (loss_fn_vgg(restored, target)).detach().cpu()
        lpips_value = _lpips_value[0, 0, 0, 0].item()
        # 计算结果存入列表
        image_number.append(str(i))
        image_name.append(gt_name)
        psnr_number.append(psnr)
        ssim_number.append(ssim)
        lpips_number.append(lpips_value)
        print(psnr, ssim, lpips_value)

    # 计算列表中指标值的平均值函数
    def ave(lis):
        s = 0
        total_num = len(lis)
        for i in lis:
            s = s + i
        return s / total_num

    # 计算列表中指标值的平均值，并加入列表
    total = 'total(' + str(len(image_number)) + ')'
    image_number.append(total)
    image_name.append('average')
    psnr_ave = ave(psnr_number)
    ssim_ave = ave(ssim_number)
    lpips_ave = ave(lpips_number)
    psnr_number.append(psnr_ave)
    ssim_number.append(ssim_ave)
    lpips_number.append(lpips_ave)
    dit = {'image_number': image_number, 'result_name': image_name, 'psnr': psnr_number, 'ssim': ssim_number,
           'lpips': lpips_number}
    df = pd.DataFrame(dit)
    csv_path = os.path.join(result_dir, "ssim&psnr.csv")  # 拼接csv名字
    df.to_csv(csv_path, columns=['image_number', 'result_name', 'psnr', 'ssim', 'lpips'], index=False, sep=',')

    print('————————————————————————————finish————————————————————————————')
    print('csv_save_path:', csv_path)  ##csv全路径
    print('result_photos_num:', len(result_images))  # result_photos_num
    print('GT_photos_num:', len(GT_image))  # GT_photos_num
    print('psnr_ave:', psnr_ave)  # psnr_ave
    print('ssim_ave:', ssim_ave)  # ssim_ave
    print('lpip_ave:', lpips_ave)  # ssim_ave
class BaseSampler:
    def __init__(
            self,
            configs,
            sf=None,
            use_fp16=False,
            bs = 1,
            seed=10000,
            ):
        '''
        Input:
            configs: config, see the yaml file in folder ./configs/
            sf: int, super-resolution scale
            seed: int, random seed
        '''
        self.configs = configs
        self.seed = seed
        self.use_fp16 = use_fp16
        if sf is None:
            sf = configs.diffusion.params.sf
        self.sf = sf
        self.bs = bs

        self.setup_dist()  # setup distributed training: self.num_gpus, self.rank

        self.setup_seed()

        self.build_model()

        self.build_dataloader()

    def setup_seed(self, seed=None):
        seed = self.seed if seed is None else seed
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    def setup_dist(self, gpu_id=None):
        num_gpus = torch.cuda.device_count()
        assert num_gpus == 1, 'Please assign one available GPU using CUDA_VISIBLE_DEVICES!'

        self.num_gpus = num_gpus
        self.rank = int(os.environ['LOCAL_RANK']) if num_gpus > 1 else 0

    def write_log(self, log_str):
        if self.rank == 0:
            print(log_str)

    def build_model(self):
        # diffusion model
        log_str = f'Building the diffusion model with length: {self.configs.diffusion.params.steps}...'
        self.write_log(log_str)
        self.base_diffusion = util_common.instantiate_from_config(self.configs.diffusion)
        model = util_common.instantiate_from_config(self.configs.model).cuda()
        ckpt_path =self.configs.model.ckpt_path
        assert ckpt_path is not None
        self.write_log(f'Loading Diffusion model from {ckpt_path}...')
        self.load_model(model, ckpt_path)
        if self.use_fp16:
            model.dtype = torch.float16
            model.convert_to_fp16()
        self.model = model.eval()

        # autoencoder
        if self.configs.autoencoder.ckpt_path is not None:
            ckpt = torch.load(self.configs.autoencoder.ckpt_path, map_location=f"cuda:{self.rank}")
            params = self.configs.autoencoder.get('params', dict)
            autoencoder = util_common.get_obj_from_str(self.configs.autoencoder.target)(**params)
            autoencoder.load_state_dict(ckpt, True)
            for params in autoencoder.parameters():
                params.requires_grad_(False)
            autoencoder.eval()
            if self.configs.autoencoder.use_fp16:
                self.autoencoder = autoencoder.half().cuda()
            else:
                self.autoencoder = autoencoder.cuda()
        else:
            self.autoencoder = None

    def load_model(self, model, ckpt_path=None):
        state = torch.load(ckpt_path, map_location=f"cuda:{self.rank}")
        if 'state_dict' in state:
            state = state['state_dict']
        util_net.reload_model(model, state)

    def build_dataloader(self):
        def _wrap_loader(loader):
            while True: yield from loader

        # make datasets {'lq':im_lq.contiguous(), 'gt':im_gt}
        # datasets = {'train': create_dataset(self.configs.data.get('train', dict)), }
        datasets = {'test': load_data(
            hr_data_dir=self.configs.data.test.gt_data_dir,
            lr_data_dir=self.configs.data.test.lq_data_dir,
            other_data_dir=self.configs.data.test.gbvs_data_dir), }
        # make dataloaders
        if self.num_gpus > 1:
            sampler = udata.distributed.DistributedSampler(
                    datasets['test'],
                    num_replicas=self.num_gpus,
                    rank=self.rank,
                    )
        else:
            sampler = None
        dataloaders ={'test': udata.DataLoader(datasets['test'],
                        batch_size=self.bs,
                        shuffle=False,
                        drop_last=False,
                        num_workers=0,
                        # pin_memory=True,
        )}


        self.datasets = datasets
        self.dataloaders = dataloaders
        self.sampler = sampler

def my_worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)

def ave(lis):
    s = 0
    total_num = len(lis)
    for i in lis:
        s = s + i
    return s / total_num
class ResShiftSampler(BaseSampler):

    def sample_func(self, y0, model_kwargs, noise_repeat=False):
        '''
        Input:
            y0: n x c x h x w torch tensor, low-quality image, [-1, 1], RGB
        Output:
            sample: n x c x h x w, torch tensor, [-1, 1], RGB
        '''
        if noise_repeat:
            self.setup_seed()

        # print(y0.shape, model_kwargs['other'].shape)
        results = self.base_diffusion.p_sample_loop(
                y=y0,
                model=self.model,
                first_stage_model=self.autoencoder,
                noise=None,
                noise_repeat=noise_repeat,
                clip_denoised=(self.autoencoder is None),
                denoised_fn=None,
                model_kwargs=model_kwargs,
                progress=False,
                )    # This has included the decoding for latent space

        return results.clamp_(-1.0, 1.0)

    def encode_first_stage(self, y, first_stage_model, up_sample=False):
        ori_dtype = y.dtype
        if up_sample:
            y = F.interpolate(y, scale_factor=self.sf, mode='bicubic')
        if first_stage_model is None:
            return y
        else:
            with torch.no_grad():
                y = y.type(dtype=next(first_stage_model.parameters()).dtype)
                z_y = first_stage_model.encode(y)
                out = z_y * self.sf
                return out.type(ori_dtype)

    def get_normalize(self, data):
        # 使用torch.clamp将数据限制在[-inf, inf]范围内
        data = torch.clamp(torch.abs(data), float('-inf'), float('inf'))

        # 计算张量的最小值和最大值
        min_val = torch.min(data)
        max_val = torch.max(data)

        # 将最小值从张量中减去，并将最大值用于归一化
        data -= min_val
        data /= max_val

        return data

    def inference(self, out_path, noise_repeat=False):
        '''
        Inference demo.
        Input:
            in_path: str, folder or image path for LQ image
            out_path: str, folder save the results
            bs: int, default bs=1, bs % num_gpus == 0
        '''
        image_number = []  # 图片读取求指标的id
        image_name = []  # 图片读取求指标的id 对应的图片名
        psnr_number = []  # psnr值列表
        ssim_number = []  # ssim值列表
        lpips_number = []  # lpips值列表

        loss_fn_vgg = lpips.LPIPS(net='vgg').to("cuda:0")
        out_path = Path(out_path) if not isinstance(out_path, Path) else out_path
        gt_path = out_path.joinpath("gt")
        lq_path = out_path.joinpath("lq")
        sr_path = out_path.joinpath("sr")
        if not gt_path.exists():
            gt_path.mkdir(parents=True)
        if not lq_path.exists() :
            lq_path.mkdir(parents=True)
        if not sr_path.exists():
            sr_path.mkdir(parents=True)

        for ii, data in enumerate(self.dataloaders['test']):
            print(f"process/all:{ii}/{len(self.dataloaders['test'])}")
            data = {"gt": data[0], "lq": data[1], 'other': data[2]}
            if 'gt' in data:
                im_lq, im_gt, im_other = data['lq'].cuda(), data['gt'].cuda(), data['other'].cuda()
                im_lq, im_gt, im_other = self.get_normalize(im_lq), self.get_normalize(
                    im_gt), self.get_normalize(im_other)
                im_gt = (im_gt - 0.5) / 0.5  # [0, 1] to [-1, 1]
                im_lq = (im_lq - 0.5) / 0.5  # [0, 1] to [-1, 1]
                im_other = (im_other - 0.5) / 0.5  # [0, 1] to [-1, 1]
                vutils.save_image(im_gt, gt_path/ f"{str(ii)}.jpg",
                                  normalize=True)
            else:
                im_lq = data['lq'].cuda()
                im_other = data['other'].cuda()
                im_lq, im_other = self.get_normalize(im_lq), self.get_normalize(im_other)
                im_lq = (im_lq - 0.5) / 0.5  # [0, 1] to [-1, 1]
                im_other = (im_other - 0.5) / 0.5  # [0, 1] to [-1, 1]
            model_kwargs = {'lq': im_lq, 'other': im_other}
            model_kwargs['lq'] = self.encode_first_stage(model_kwargs['lq'], self.autoencoder, up_sample=True)
            model_kwargs['other'] = self.encode_first_stage(model_kwargs['other'], self.autoencoder, up_sample=True)
            im_sr_tensor = self.sample_func(
                        im_lq,
                        model_kwargs,
                        noise_repeat=noise_repeat,
                        )     # 1 x c x h x w, [-1, 1]

            im_sr = im_sr_tensor * 0.5 + 0.5

            vutils.save_image(im_sr, sr_path / f"{str(ii)}.jpg",
                                  normalize=True)
            vutils.save_image(im_lq, lq_path/ f"{str(ii)}.jpg",
                                  normalize=True)
        self.write_log(f"Processing done, enjoy the results in {str(out_path)}")

        #计算指标
        compute_matrix(out_path)


if __name__ == '__main__':
    pass

