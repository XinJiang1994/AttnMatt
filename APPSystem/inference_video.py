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

from model import MattingNetwork

torch_transforms = transforms.Compose([
    ToTensor(),
    ConvertImageDtype(torch.float),
    Normalize(0.5, 0.5),
])


class VideoCapture:
    def __init__(self, filename=0, width=1280, height=720):
        self.capture = cv2.VideoCapture(filename)
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.width = width
        self.height = height
        # self.capture.set(cv2.CAP_PROP_BUFFERSIZE, 2)

    def read(self):
        ret, frame = self.capture.read()
        if ret:
            frame = cv2.resize(frame, (self.width, self.height),
                               interpolation=cv2.INTER_AREA)
        return ret, frame

    def __exit__(self, exec_type, exc_value, traceback):
        self.capture.release()


class VedioMatter():
    def __init__(self, data_root, ckpt, w, h, epoch=10, use_gpu=True):
        self.displayer = None
        self.data_root = data_root
        self.w = w
        self.h = h
        self.use_gpu = use_gpu
        self.model = self.init_model()
        self.epoch = epoch
        self.ckpt = ckpt
        # self.bg_maintainer=BGMaintainer(self.w,self.h,buffersize=30)
        self.rec = [None] * 4  # Initial recurrent states.
        self.downsample_ratio = 0.5  # Adjust based on your video.
        # self.adapter=OnlineAdapter(copy.deepcopy(self.model),lr=0.01,data_root=self.data_root,use_gpu=use_gpu)

    def init_model(self):
        pretrained_ckpt = 'APPSystem/app_ckpt/rvm_mobilenetv3_pretrain.pkl'
        if not os.path.exists(pretrained_ckpt):
            pretrained_ckpt = 'checkpoint/rvm_mobilenetv3.pth'
        print('Load pre-trained model {} ...'.format(pretrained_ckpt))
        # modnet = MODNet(backbone_pretrained=False)
        model = MattingNetwork('mobilenetv3').eval()  # or "resnet50"

        if self.use_gpu:
            print('Use GPU...')
            model = model.cuda()
            model.load_state_dict(torch.load(pretrained_ckpt))
        else:
            print('Use CPU...')
            model.load_state_dict(torch.load(
                pretrained_ckpt, map_location=torch.device('cpu')))
        model.eval()
        return model

    def init_displayer(self):
        self.displayer = Displayer(
            'Online Optimization', self.w, self.h, show_info=True)

    def run(self, filename):
        cap = VideoCapture(filename)
        self.init_displayer()
        frame_no = -1
        while (True):
            ret, frame_np = cap.read()
            if not ret:
                print('Read frame {} failed...'.format(frame_no+1))
                break
            frame_np = cv2.cvtColor(frame_np, cv2.COLOR_BGR2RGB)

            frame_PIL = Image.fromarray(frame_np)
            frame_tensor = torch_transforms(frame_PIL)
            frame_tensor = frame_tensor[None, :, :, :]
            if self.use_gpu:
                frame_tensor = frame_tensor.cuda()
            self.model.eval()
            # st=time.time()
            with torch.no_grad():
                fgr, pha, * \
                    self.rec = self.model(
                        frame_tensor, *self.rec, self.downsample_ratio)

            matte_tensor = pha.repeat(1, 3, 1, 1)

            matte_np = matte_tensor[0].data.cpu().numpy().transpose(1, 2, 0)
            r = np.zeros(frame_np.shape[0:2])
            g = np.ones(frame_np.shape[0:2])*255
            b = r
            bg_fill = np.array([r, g, b]).transpose([1, 2, 0])
            fg_np = matte_np * frame_np + (1 - matte_np) * bg_fill
            # print('Model FPS: ', 1 / (post_inf_time - st))
            # if frame_no == -1:
            #     bg = (frame_np * (1 - matte_np)).astype(np.uint8)
            #     self.bg_maintainer.init(bg)
            # else:
            #     self.bg_maintainer.update_bg_buffer(frame_np, matte_np)

            bg = self.bg_maintainer.get_bg()
            frame_no = (frame_no + 1) % 1000

            view_np = np.uint8(np.concatenate((frame_np, fg_np), axis=1))
            # view_np2 = np.uint8(np.concatenate((bg, matte_np*255), axis=1))

            bg_diff = cv2.absdiff(
                bg, (frame_np * (1 - matte_np)).astype(np.uint8))

            # view_np2 = np.uint8(np.concatenate((bg, bg_diff), axis=1))

            view_np = np.uint8(np.concatenate((view_np), axis=0))
            view_np = cv2.cvtColor(view_np, cv2.COLOR_RGB2BGR)

            key = self.displayer.step(view_np)

            if key == ord('q'):
                # if training_thread != None:
                #     training_thread.join()
                # video_writer.release()
                print('Exit Meeting.')
                cv2.destroyAllWindows()
                break
