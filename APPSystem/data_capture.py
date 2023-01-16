import time

import numpy as np

import torch
import cv2
from torchvision.transforms import Compose, ToTensor, Resize
from PIL import Image
import os

from APPSystem.utils.displayer import Displayer


def cv2_frame_to_tensor(frame):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return ToTensor()(Image.fromarray(frame)).unsqueeze_(0)


def checkdir(p):
    if not os.path.exists(p):
        os.makedirs(p)


class DataCapture():
    '''
    DataCapture task:
    1. bg capture
    2. capture frams
    3. get alpha matte by VBM-V2
    4. show results of VBM-V2
    5. Save data
    '''

    def __init__(self, cam, data_root, use_gpu=True):
        self.cam = cam
        self.data_root = data_root
        self.displayer = None
        self.vbm_model = self.init_vbm(use_gpu)
        self.use_gpu = use_gpu
        self.bg_buffer = []
        self.bg = None
        self.mode = 'B'  # B-->BG D-->cap data M--> matting but do not save data
        self.video_writer_fgr, self.video_writer_pha = None, None
        self.completed = False

    def init_vbm(self, use_gpu):
        model = torch.jit.load('BGMV2Model/bgm_mobilenetv2_torchscript.pth')
        model.backbone_scale = 0.25
        model.refine_mode = 'sampling'
        model.refine_sample_pixels = 80_000
        model.model_refine_threshold = 0.7
        model.eval()
        if use_gpu:
            model = model.cuda()
        return model

    def init_video_writer(self):
        fgr_root = os.path.join(self.data_root, 'fgr')
        pha_root = os.path.join(self.data_root, 'pha')
        checkdir(fgr_root)
        checkdir(pha_root)
        tm = time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime())
        save_name_fgr = os.path.join(fgr_root, 'adaption0.mp4')
        save_name_pha = os.path.join(pha_root, 'adaption0.mp4')
        fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
        fps = 30.0
        v_size = (self.cam.width, self.cam.height)
        video_writer_fgr = cv2.VideoWriter(save_name_fgr, fourcc, fps, v_size)
        video_writer_pha = cv2.VideoWriter(save_name_pha, fourcc, fps, v_size)
        print('Fgr saving path: ', save_name_fgr)
        print('pha saving path: ', save_name_pha)
        self.video_writer_fgr, self.video_writer_pha = video_writer_fgr, video_writer_pha

    def init_displayer(self):
        self.displayer = Displayer(
            'Capture Data (Press B to capture BG)', self.cam.width, self.cam.height, show_info=True)

    def save_data(self, fgr, pha):
        # print('saveing ....')
        self.video_writer_fgr.write(np.uint8(fgr))
        self.video_writer_pha.write(np.uint8(pha*255))

    def run(self):
        self.init_video_writer()
        self.init_displayer()

        while not self.completed:
            bgr = None
            while not self.completed:  # grab bgr
                frame = self.cam.read()
                key = self.displayer.step(frame)
                if key == ord('b'):
                    bgr = self.cam.read()
                    break
                elif key == ord('q'):
                    cv2.destroyAllWindows()
                    # self.video_writer_fgr.release()
                    # self.video_writer_pha.release()
                    self.completed = True
                    break

            while not self.completed:  # matting
                frame = self.cam.read()
                # frame=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                src = cv2_frame_to_tensor(frame)
                bgr_tensor = cv2_frame_to_tensor(bgr)
                if self.use_gpu:
                    src, bgr_tensor = src.cuda(), bgr_tensor.cuda()
                pha, fgr = self.vbm_model(src, bgr_tensor)[:2]

                res = pha * fgr + (1 - pha) * torch.ones_like(fgr)
                res = res.mul(255).byte().cpu().permute(0, 2, 3, 1).numpy()[0]
                res = cv2.cvtColor(res, cv2.COLOR_RGB2BGR)

                if self.mode == 'D':
                    fgr_np = np.uint8(frame)
                    pha_np = pha.cpu().permute(
                        0, 2, 3, 1).repeat(1, 1, 1, 3).numpy()[0]
                    self.save_data(fgr_np, pha_np)

                res = np.concatenate((frame, res), axis=1)
                key = self.displayer.step(res)
                if key == ord('b'):
                    break
                if key == ord('d'):
                    print('Capturing data...')
                    self.mode = 'D'
                elif key == ord('q'):
                    print('Complete data collection')
                    cv2.destroyAllWindows()
                    # self.video_writer_fgr.release()
                    # self.video_writer_pha.release()
                    self.completed = True
                    break
        self.video_writer_fgr.release()
        self.video_writer_pha.release()
    # def __exit__(self, exec_type, exc_value, traceback):
    #     print('Release video writers....')
    #     self.video_writer_fgr.release()
    #     self.video_writer_pha.release()
