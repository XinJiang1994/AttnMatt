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
                 vname,
                 size,
                 seq_length,
                 seq_sampler,
                 bgs,
                 transform=None,
                 # a tuple (partation_id, total_partation_nums)
                 ):
        self.background_image_dir = background_image_dir
        # self.background_image_files = os.listdir(background_image_dir)
        self.background_image_files = bgs
        self.background_video_dir = background_video_dir
        self.background_video_clips = sorted(os.listdir(background_video_dir))
        self.background_video_frames = [sorted(os.listdir(os.path.join(background_video_dir, clip)))
                                        for clip in self.background_video_clips]

        self.videomatte_dir = videomatte_dir
        v_path = os.path.join(videomatte_dir, 'fgr_com', vname)
        pha_path = os.path.join(videomatte_dir, 'pha', vname)
        pha_com_path = os.path.join(videomatte_dir, 'pha_com', vname)
        print(v_path)

        self.cap_v = cv2.VideoCapture(v_path)
        self.cap_pha = cv2.VideoCapture(pha_path)
        self.cap_pha_com = cv2.VideoCapture(pha_com_path)

        self.frame_count = int(self.cap_v.get(cv2.CAP_PROP_FRAME_COUNT))
        # random sample

        self.size = size
        self.seq_length = seq_length
        self.seq_sampler = seq_sampler
        self.transform = transform

    def __len__(self):
        return self.frame_count

    def __getitem__(self, idx):
        # st=time.time()

        if random.random() < 0.5:
            bgrs = self._get_random_image_background()
        else:
            bgrs = self._get_random_video_background()
        # bg_time=time.time()
        # print('Load bg time: ',bg_time-st)

        fgrs, phas, pha_coms = self._get_videomatte(idx)

        # get_vide_oframe_time=time.time()
        # print('get_vide_oframe_time: ',get_vide_oframe_time-bg_time)

        if self.transform is not None:
            fgrs, phas, pha_coms, bgrs = self.transform(
                fgrs, phas, pha_coms, bgrs)
            # print('transform time: ',time.time()-get_vide_oframe_time)
            # return fgrs, phas, bgrs
        return fgrs, phas, pha_coms, bgrs

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
        # print('>>>>>>>>>>>>>>>>> reading ', idx)
        fgrs, phas, pha_coms = [], [], []
        for i in self.seq_sampler(self.seq_length):
            frame_idx = (idx+i) % self.frame_count
            self.cap_v.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            self.cap_pha.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            self.cap_pha_com.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, fgr = self.cap_v.read()
            if not ret:
                raise IndexError(
                    f'Idx: {frame_idx} out of video length: {len(self)}')
            ret, pha = self.cap_pha.read()
            if not ret:
                raise IndexError(
                    f'Idx: {frame_idx} out of pha length: {len(self)}')
            ret, pha_com = self.cap_pha_com.read()
            if not ret:
                raise IndexError(
                    f'Idx: {frame_idx} out of pha_com length: {len(self)}')

            fgr = cv2.cvtColor(fgr, cv2.COLOR_BGR2RGB)
            pha = cv2.cvtColor(pha, cv2.COLOR_BGR2RGB)
            pha_com = cv2.cvtColor(pha_com, cv2.COLOR_BGR2RGB)

            fgr = Image.fromarray(fgr)
            pha = Image.fromarray(pha)
            pha_com = Image.fromarray(pha_com)

            fgr = self._downsample_if_needed(fgr.convert('RGB'))
            pha = self._downsample_if_needed(pha.convert('L'))
            pha_com = self._downsample_if_needed(pha_com.convert('L'))
            fgrs.append(fgr)
            phas.append(pha)
            pha_coms.append(pha_com)
        return fgrs, phas, pha_coms

    def _downsample_if_needed(self, img):
        w, h = img.size
        if min(w, h) > self.size:
            scale = self.size / min(w, h)
            w = int(scale * w)
            h = int(scale * h)
            img = img.resize((w, h))
        # img = img.resize((self.size, self.size))
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
