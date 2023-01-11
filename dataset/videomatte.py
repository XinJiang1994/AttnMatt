import torch
import os
import random
from torch.utils.data import Dataset
from PIL import Image
import cv2
from .augmentation import MotionAugmentation
import time
import numpy as np

import torchvision.transforms as transforms


random_erasing = transforms.RandomErasing(
    p=1.0,  # 概率值，执行该操作的概率，默认为 0.5
    scale=(0.02, 0.33),  # 按均匀分布概率抽样，遮挡区域的面积 = image * scale
    ratio=(0.3, 3.3),  # 遮挡区域的宽高比，按均匀分布概率抽样
    value=255,  # 遮挡区域的像素值，(R, G, B) or (Gray)；传入字符串表示用随机彩色像素填充遮挡区域
    inplace=False
)
random_erasing_0 = transforms.RandomErasing(
    p=1.0,  # 概率值，执行该操作的概率，默认为 0.5
    scale=(0.02, 0.33),  # 按均匀分布概率抽样，遮挡区域的面积 = image * scale
    ratio=(0.3, 3.3),  # 遮挡区域的宽高比，按均匀分布概率抽样
    value=0,  # 遮挡区域的像素值，(R, G, B) or (Gray)；传入字符串表示用随机彩色像素填充遮挡区域
    inplace=False
)


class VideoMatteDatasetForMH(Dataset):
    def __init__(self,
                 videomatte_dir,
                 background_image_dir,
                 background_video_dir,
                 size,
                 seq_length,
                 seq_sampler,
                 bgs,
                 transform=None):
        self.background_image_dir = background_image_dir
        self.background_image_files = bgs
        self.background_video_dir = background_video_dir
        self.background_video_clips = sorted(os.listdir(background_video_dir))
        self.background_video_frames = [sorted(os.listdir(os.path.join(background_video_dir, clip)))
                                        for clip in self.background_video_clips]

        self.videomatte_dir = videomatte_dir
        self.videomatte_clips = sorted(
            os.listdir(os.path.join(videomatte_dir, 'fgr_com')))
        self.videomatte_frames = [sorted(os.listdir(os.path.join(videomatte_dir, 'fgr_com', clip)))
                                  for clip in self.videomatte_clips]
        self.videomatte_idx = [(clip_idx, frame_idx)
                               for clip_idx in range(len(self.videomatte_clips))
                               for frame_idx in range(0, len(self.videomatte_frames[clip_idx]), seq_length)]
        self.size = size
        self.seq_length = seq_length
        self.seq_sampler = seq_sampler
        self.transform = transform

    def __len__(self):
        return len(self.videomatte_idx)

    def __getitem__(self, idx):
        if random.random() < 0.5:
            bgrs = self._get_random_image_background()
        else:
            bgrs = self._get_random_video_background()

        fgrs, phas, pha_coms,target = self._get_videomatte(idx)

        if self.transform is not None:
            fgrs, phas, pha_coms, bgrs = self.transform(fgrs, phas, pha_coms, bgrs)
            
            
        target=transforms.ToTensor()(target)
        print(f'fgrs {fgrs.shape}, phas {phas.shape}, pha_coms {pha_coms.shape}, bgrs {bgrs.shape}')
#         target=transforms.Resize(size=fgrs.shape[-2:])(target)

        return fgrs, phas, pha_coms, bgrs,target

    def _get_random_image_background(self):
        with Image.open(os.path.join(self.background_image_dir, random.choice(self.background_image_files))) as bgr:
            bgr = self._downsample_if_needed(bgr.convert('RGB'))
        bgrs = [bgr] * self.seq_length
        return bgrs

    def _get_random_video_background(self):
        clip_idx = random.choice(range(len(self.background_video_clips)))
        frame_count = len(self.background_video_frames[clip_idx])
        frame_idx = random.choice(range(max(1, frame_count - self.seq_length)))
        clip = self.background_video_clips[clip_idx]
        bgrs = []
        for i in self.seq_sampler(self.seq_length):
            frame_idx_t = frame_idx + i
            frame = self.background_video_frames[clip_idx][frame_idx_t %
                                                           frame_count]
            with Image.open(os.path.join(self.background_video_dir, clip, frame)) as bgr:
                bgr = self._downsample_if_needed(bgr.convert('RGB'))
            bgrs.append(bgr)
        return bgrs

    def _get_videomatte(self, idx):
        clip_idx, frame_idx = self.videomatte_idx[idx]
        clip = self.videomatte_clips[clip_idx]
        frame_count = len(self.videomatte_frames[clip_idx])
        fgrs, phas, pha_coms = [], [], []
        for i in self.seq_sampler(self.seq_length):
            frame = self.videomatte_frames[clip_idx][(
                frame_idx + i) % frame_count]
            with Image.open(os.path.join(self.videomatte_dir, 'fgr_com', clip, frame)) as fgr, \
                  Image.open(os.path.join(self.videomatte_dir, 'pha', clip, frame)) as pha, \
                  Image.open(os.path.join(self.videomatte_dir, 'pha_com', clip, frame)) as pha_com:
                fgr = self._downsample_if_needed(fgr.convert('RGB'))
                pha = self._downsample_if_needed(pha.convert('L'))
                pha_com = self._downsample_if_needed(pha_com.convert('L'))
            fgrs.append(fgr)
            phas.append(pha)
            pha_coms.append(pha_com)
        with Image.open(os.path.join(self.videomatte_dir, 'target', clip+'.jpg')) as target:
                target = self._downsample_if_needed(target.convert('RGB'))
        return fgrs, phas, pha_coms,target

    def _downsample_if_needed(self, img):
        w, h = img.size
        if min(w, h) > self.size:
            scale = self.size / min(w, h)
            w = int(scale * w)
            h = int(scale * h)
            img = img.resize((w, h))
        return img


class VideoMatteTrainAugmentation(MotionAugmentation):
    def __init__(self, size):
        super().__init__(
            size=size,
            prob_fgr_affine=0.3,
            prob_bgr_affine=0.3,
            prob_noise=0.1,
            prob_color_jitter=0.3,
            prob_grayscale=0.02,
            prob_sharpness=0.1,
            prob_blur=0.02,
            prob_hflip=0.5,
            prob_pause=0.03,
        )


class VideoMatteValidAugmentation(MotionAugmentation):
    def __init__(self, size):
        super().__init__(
            size=size,
            prob_fgr_affine=0,
            prob_bgr_affine=0,
            prob_noise=0,
            prob_color_jitter=0,
            prob_grayscale=0,
            prob_sharpness=0,
            prob_blur=0,
            prob_hflip=0,
            prob_pause=0,
        )
