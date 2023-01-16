import time

import numpy as np

import torch
import cv2
from torchvision.transforms import Compose, ToTensor, Resize
from PIL import Image
import os

class FPSTracker:
    def __init__(self, ratio=0.5):
        self._last_tick = None
        self._avg_fps = None
        self.ratio = ratio
    def tick(self):
        if self._last_tick is None:
            self._last_tick = time.time()
            return None
        t_new = time.time()
        fps_sample = 1.0 / (t_new - self._last_tick)
        self._avg_fps = self.ratio * fps_sample + (1 - self.ratio) * self._avg_fps if self._avg_fps is not None else fps_sample
        self._last_tick = t_new
        return self.get()
    def get(self):
        return self._avg_fps

class Displayer:
    def __init__(self, title, width=None, height=None,frame_count=0, show_info=True):
        self.title, self.width, self.height = title, width, height
        self.frame_count=frame_count
        self.show_info = show_info
        self.fps_tracker = FPSTracker()
        print('FPSTracker builed')
        cv2.namedWindow(self.title, cv2.WINDOW_NORMAL)
        print('Build named window')
        if width is not None and height is not None:
            cv2.resizeWindow(self.title, width, height)
            print('Init camera success')
        self.frame_no=0
    # Update the currently showing frame and return key press char code
    def step(self, image):
        fps_estimate = self.fps_tracker.tick()
        if self.show_info and fps_estimate is not None:
            message = f"{int(fps_estimate)} fps | {self.width}x{self.height} | frame_no={self.frame_no}/{self.frame_count}"
            cv2.putText(image, message, (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (255, 255, 0),thickness=2)
        cv2.imshow(self.title, image)
        self.frame_no+=1
        return cv2.waitKey(100000) & 0xFF


def cv2_frame_to_tensor(frame):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return ToTensor()(Image.fromarray(frame)).unsqueeze_(0)

def checkdir(p):
    if not os.path.exists(p):
        os.makedirs(p)

class PostProcessor():
    '''
    Postprocessing for local collected data
    '''
    def __init__(self,v_src='Adapt_data/fgr/adaption0.mp4',
                    pha_src='Adapt_data/pha/adaption0.mp4',
                    v_save_name='Adapt_data/fgr/adaption.mp4',
                    pha_save_name='Adapt_data/pha/adaption.mp4'):
        self.v_src=v_src
        self.pha_src=pha_src

        self.frame_flag={} # label for each frame, 0 or 1 (selected or not)
        self.pha_save_name=pha_save_name
        self.v_save_name=v_save_name
        self.displayer=None
        self.video_writer_v,self.video_writer_pha=None,None
        self.completed=False

    def init_cap(self):
        self.cap=cv2.VideoCapture(self.v_src)
        self.cap_pha= cv2.VideoCapture(self.pha_src)
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # self.width=1920
        # self.height=1080
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

    def init_video_writer(self):
        fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
        fps = 30.0
        v_size = (self.width, self.height)
        video_writer_v = cv2.VideoWriter(self.v_save_name, fourcc, fps, v_size)
        video_writer_pha = cv2.VideoWriter(self.pha_save_name, fourcc, fps, v_size)
        print('vdieo saving path: ', self.v_save_name)
        print('pha saving path: ', self.pha_save_name)
        self.video_writer_v,self.video_writer_pha = video_writer_v,video_writer_pha

    def init_displayer(self):
        self.displayer=Displayer('Capture Data (Press B to capture BG)', self.width, self.height,self.frame_count, show_info=True)


    def save_data(self,frame,pha):
        # print('saveing ....')
        self.video_writer_v.write(frame)
        self.video_writer_pha.write(cv2.normalize(pha,None,0,255,cv2.NORM_MINMAX,dtype=0))

    def save_selected_frames(self):
        print('Rendering frames...')
        for frame_no, flag in  self.frame_flag.items():
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
            self.cap_pha.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
            if flag==1:
                ret,frame = self.cap.read()
                if not ret:
                    print('Read frame failed!')
                    break
                ret,pha = self.cap_pha.read()
                if not ret:
                    print('Read frame failed!')
                    break
                self.save_data(frame,pha)


    def run(self):
        self.init_cap()
        self.init_video_writer()
        self.init_displayer()
        frame_no=0
        

        while not self.completed:  # matting
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
            self.cap_pha.set(cv2.CAP_PROP_POS_FRAMES, frame_no)

            ret,frame = self.cap.read()
            if not ret:
                print('Read frame failed!')
                break
            ret,pha = self.cap_pha.read()
            if not ret:
                print('Read frame failed!')
                break

            res=np.concatenate((frame,pha),axis=1)
            key = self.displayer.step(res)
            if key == ord('s'):
                self.frame_flag[frame_no]=1
                frame_no=min(frame_no+1,self.frame_count)
                # self.save_data(frame,pha) # save founction with normalize pha_np to 0-255
            elif key == ord('d'):
                self.frame_flag[frame_no]=0
                frame_no=min(frame_no+1,self.frame_count)
            elif key==81: # left
                frame_no=max(frame_no-1,0)
            elif key==83: # right
                frame_no=min(frame_no+1,self.frame_count)
            elif key == ord('q'):
                print('Complete data collection')
                cv2.destroyAllWindows()
                self.completed=True
            else:
                frame_no=frame_no
            
        self.save_selected_frames()
        print('Saving video:',self.v_save_name)
        self.video_writer_v.release()
        print('Saving pha:',self.pha_save_name)
        self.video_writer_pha.release()

        print('Release video writers....')
        self.video_writer_v.release()
        self.video_writer_pha.release()
        cv2.destroyAllWindows()
        
    def __exit__(self, exec_type, exc_value, traceback):
        print('Release video writers....')
        self.video_writer_v.release()
        self.video_writer_pha.release()

if __name__ == '__main__':
    processor=PostProcessor()
    processor.run()