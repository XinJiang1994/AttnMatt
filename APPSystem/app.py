import os
import time
from tkinter.filedialog import askopenfilename

import cv2
from threading import Thread, Lock

# from APPSystem.adaptionTrainer import pretrain
from APPSystem.data_capture import DataCapture
from APPSystem.inference_video import VedioMatter
from APPSystem.matter import Matter
import tkinter.messagebox as messagebox
from tkinter import Frame, Entry, Button

from APPSystem.post_processing import PostProcessor


def checkdir(p):
    if not os.path.exists(p):
        os.makedirs(p)


class Camera:
    def __init__(self, device_id=0, width=1280, height=720):
        self.capture = cv2.VideoCapture(device_id)
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.width = int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        # self.capture.set(cv2.CAP_PROP_BUFFERSIZE, 2)

        self.success_reading, self.frame = self.capture.read()
        self.read_lock = Lock()
        self.thread = Thread(target=self.__update, args=())
        self.thread.daemon = True
        self.thread.start()

    def __update(self):
        while self.success_reading:
            grabbed, frame = self.capture.read()
            with self.read_lock:
                self.success_reading = grabbed
                self.frame = frame

    def read(self):
        with self.read_lock:
            frame = self.frame.copy()
        return frame

    def release(self):
        self.capture.release()

    def __exit__(self, exec_type, exc_value, traceback):
        self.capture.release()


MODE = {'M': 'MATTING', 'D': 'DATA_CAP'}


class Application(Frame):
    def __init__(self, cam_id, width, height, data_root, ckpt, vnames, master=None):
        Frame.__init__(self, master)
        self.pack()
        self.cam_id, self.width, self.height = cam_id, width, height
        self.createWidgets()
        self.vnames = vnames
        self.data_root = data_root
        self.ckpt = ckpt
        self.postProcessor = PostProcessor()
        self.mode = MODE['D']  # MATTING OR BG_CAP
        self.video_matter = VedioMatter(data_root, ckpt, width, height)

    def createWidgets(self):

        self.clear_data_btn = Button(
            self, text='Clear workspace', fg='black', bg='white', command=self.clear_workspace)
        self.clear_data_btn.pack()
        self.cap_data_btn = Button(
            self, text='Prepare Data', fg='black', bg='white', command=self.cap_data)
        self.cap_data_btn.pack()

        self.post_process_btn = Button(
            self, text='Postprocess', fg='black', bg='white', command=self.postprocess_data)
        self.post_process_btn.pack()

        self.load_default_model_btn = Button(
            self, text='Use Default Model', fg='black', bg='white', command=self.load_defaut_model)
        self.load_default_model_btn.pack()

        self.update_model_btn = Button(
            self, text='Use MAML_KAL Model', fg='black', bg='white', command=self.load_fpa_finetuned_model)
        self.update_model_btn.pack()

        self.update_model_btn = Button(
            self, text='Use Adapted Model', fg='black', bg='white', command=self.use_adapted_model)
        self.update_model_btn.pack()

        self.videoButton = Button(
            self, text='Inference Video', fg='black', bg='white', command=self.infer_video)
        self.videoButton.pack()

        self.startButton = Button(
            self, text='Meeting', fg='black', bg='white', command=self.meeting)
        self.startButton.pack()
        self.exitButton = Button(
            self, text='Exit', fg='black', bg='white', command=self.exit)
        self.exitButton.pack()

    def init_camera(self, cam_id, w, h):
        print('Init WebCam...')
        return Camera(width=w, height=h, device_id=cam_id)

    def clear_workspace(self):
        fgr_root = os.path.join(self.data_root, 'fgr/')
        pha_root = os.path.join(self.data_root, 'pha')
        os.system('rm -rf {}'.format(fgr_root))
        os.system('rm -rf {}'.format(pha_root))
        checkdir(fgr_root)
        checkdir(pha_root)

    def cap_data(self):
        cam = self.init_camera(self.cam_id, w=self.width, h=self.height)
        dataCap = DataCapture(cam, self.data_root)
        dataCap.run()
        cam.release()

    def postprocess_data(self):
        self.postProcessor.run()

    def meeting(self):
        cam = self.init_camera(self.cam_id, w=self.width, h=self.height)
        matter = Matter(cam, self.data_root, self.ckpt)
        matter.run(self.ckpt, use_adapter=False)
        cam.release()

    def load_defaut_model(self):
        print('>>>>Using model: '+'pretrained_ckpt/rvm_mobilenetv3.pth')
        self.ckpt = 'pretrained_ckpt/rvm_mobilenetv3.pth'

    def load_fpa_finetuned_model(self):
        print('>>>>Using model: '+'APPSystem/app_ckpt/epoch-800-v3.pth')
        self.ckpt = 'APPSystem/app_ckpt/epoch-800-v3.pth'

    def use_adapted_model(self):

        self.ckpt = 'APPSystem/app_ckpt/epoch-4.pth'
        print('>>>>Using model: ' + self.ckpt)

    def infer_video(self):
        vname = self.vnames[0]
        vname = askopenfilename(
            initialdir='/home/xinjiang/Documents/VBHOI/Datasets/HOISyn')
        self.video_matter.run(vname)

    def exit(self):
        cv2.destroyAllWindows()
        exit(0)

    # def pretrain(self):
    #     pretrain()
