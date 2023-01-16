from collections import OrderedDict
import random
import time

import cv2
import numpy as np
from PIL import Image, ImageEnhance
import copy
import torch
import torch.nn as nn
import torchvision.transforms as transforms

from torchvision.transforms import Compose, ToTensor, Resize, ConvertImageDtype, Normalize

from APPSystem.utils.bg_maintainer import BGMaintainer
from APPSystem.utils.displayer import Displayer
import os
import threading
from model.model import MattingNetwork, MattingNetwork2
from train_config import DATA_PATHS

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device("cuda:0")
# device=torch.device("cpu" )


def convert_state_dict(state_dict):
    state_dict_convert = OrderedDict()
    for k, v in state_dict.items():
        new_k = k.replace('module.', '')
        state_dict_convert[new_k] = v
    return state_dict_convert


torch_transforms = transforms.Compose([
    ToTensor(),
    ConvertImageDtype(torch.float),
    Normalize(0.5, 0.5),
])

mutex = threading.Lock()


class AdaptingThread (threading.Thread):
    def __init__(self, real_bgs, adapter):
        threading.Thread.__init__(self)
        self.adapter = adapter
        self.real_bgs = real_bgs

    def run(self):
        mutex.acquire()
        self.adapter.adapt(self.real_bgs)
        mutex.release()


class Matter():
    def __init__(self, cam, data_root, ckpt, epoch=50, use_gpu=True):
        self.cam = cam
        self.w = cam.width
        self.h = cam.height
        self.displayer = None
        self.data_root = data_root
        self.use_gpu = use_gpu
        self.model = self.init_model()
        self.epoch = epoch
        self.ckpt = ckpt
        self.bg_maintainer = BGMaintainer(self.w, self.h, buffersize=15)
        self.rec = [None] * 4  # Initial recurrent states.
        self.downsample_ratio = 0.5  # Adjust based on your video.
        self.bgs = []
        # self.im_bgs=self.get_im_bgs()

    def get_im_bgs(self):
        f = open('LocalBGs.txt')
        im_bgs = f.readlines()
        im_bgs = [name.replace('\n', '') for name in im_bgs]
        background_image_dir = DATA_PATHS['local_background']['train']
        bgrs = []
        for im in im_bgs:
            bg = cv2.imread(os.path.join(background_image_dir, im))
            bg = cv2.cvtColor(bg, cv2.COLOR_BGR2RGB)
            bg = cv2.resize(bg, (self.w, self.h))
            bgrs.append(bg)

        # bgrs=[cv2.cvtColor(x,cv2.COLOR_BGR2RGB) for x in bgrs]
        return bgrs

    def init_model(self):
        self.model = MattingNetwork2(
            'mobilenetv3').cuda().eval()  # or "resnet50"
        return self.model

    # def init_target(self):
    #     target_img = Image.open('app_data/target/000.jpg')

    def init_displayer(self):
        self.displayer = Displayer('LORO', self.w, self.h, show_info=True)

    def save_bgs(self):
        save_dir = 'Adapt_data/BG/'
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        for i, bg in enumerate(self.bgs):
            filename = os.path.join(save_dir, f'bg{i}.png')
            cv2.imwrite(filename, cv2.cvtColor(
                bg.astype(np.uint8), cv2.COLOR_BGR2RGB))
        print('Local background saved')

    def get_target(self):
        model = MattingNetwork('mobilenetv3').eval().cuda()  # or "resnet50"
        model.load_state_dict(torch.load('checkpoint/rvm_mobilenetv3.pth'))
        bgr = torch.tensor([.47, 1, .6]).view(3, 1, 1).cuda()
        for i in range(300):
            frame_np = self.cam.read()
            frame_np = cv2.cvtColor(frame_np, cv2.COLOR_BGR2RGB)
            frame_PIL = Image.fromarray(frame_np)
            frame_tensor = torch_transforms(frame_PIL)
            frame_tensor = frame_tensor[None, :, :, :]
            if self.use_gpu:
                frame_tensor = frame_tensor.cuda()
            fgr, pha, *rec = model(frame_tensor)
            # matte_np = pha.repeat(1, 3, 1, 1)[
            #     0].data.cpu().numpy().transpose(1, 2, 0)
            # matte_np[matte_np > 150] = 255
            fg_tensor = frame_tensor * pha + bgr * (1 - pha)
            # print('***********fg_tensor:', fg_tensor.shape)
            fg_np = fg_tensor[0].data.cpu().numpy().transpose(1, 2, 0)

            fg_np = cv2.normalize(fg_np, None, 0, 255, cv2.NORM_MINMAX)
            view_np = np.uint8(np.concatenate((frame_np, fg_np), axis=1))

            view_np = cv2.cvtColor(view_np, cv2.COLOR_RGB2BGR)
            key = self.displayer.step(view_np)
            if key == ord('q'):
                break

        return fgr*pha

    def run(self, ckpt, use_adapter=False):
        if not os.path.exists(ckpt):
            ckpt = 'pretrained_ckpt/rvm_mobilenetv3.pth'
        print('Load pre-trained model {} ...'.format(ckpt))

        if self.use_gpu:
            print('Use GPU...')
            self.model = self.model.cuda()
            self.model.load_state_dict(
                convert_state_dict(torch.load(ckpt)), strict=False)
        else:
            print('Use CPU...')
            self.model.load_state_dict(convert_state_dict(torch.load(
                ckpt, map_location=torch.device('cpu'))), strict=False)

        self.init_displayer()
        frame_no = -1
        r = np.zeros([self.h, self.w])
        g = np.ones([self.h, self.w])*255
        b = np.zeros([self.h, self.w])
        bg_fill = np.array([r, g, b]).transpose(1, 2, 0)
        self.model.eval()

        target_tensor = None

        while (True):
            st = time.time()
            frame_np = self.cam.read()
            frame_np = cv2.cvtColor(frame_np, cv2.COLOR_BGR2RGB)
            # frame_np[:, :, 0] -= 30
            # frame_np[frame_np < 0] = 0
            frame_PIL = Image.fromarray(frame_np)
            frame_tensor = torch_transforms(frame_PIL)
            frame_tensor = frame_tensor[None, :, :, :]

            if self.use_gpu:
                frame_tensor = frame_tensor.cuda()

            if target_tensor == None:
                target_tensor = self.get_target()

            get_input_time = time.time()
            # print('######get input time: ',(get_input_time-st))

            with torch.no_grad():
                fgr, pha, *self.rec = self.model(
                    frame_tensor, target_tensor, *self.rec, self.downsample_ratio)
            post_inf_time = time.time()
            matte_np = pha.repeat(1, 3, 1, 1)[
                0].data.cpu().numpy().transpose(1, 2, 0)
            fg_np = frame_np * matte_np + bg_fill * (1-matte_np)

            fg_np = cv2.normalize(fg_np, None, 0, 255, cv2.NORM_MINMAX)

            view_np = np.uint8(np.concatenate((frame_np, fg_np), axis=1))

            view_np = cv2.cvtColor(view_np, cv2.COLOR_RGB2BGR)

            frame_no = (frame_no + 1) % 1000
            post_progress_time = time.time()

            key = self.displayer.step(view_np)

            if key == ord('q'):
                print('Exit Meeting.')
                cv2.destroyAllWindows()
                break
