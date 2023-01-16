import numpy as np
import cv2

class BGMaintainer:
    def __init__(self, width, height, buffersize=20):
        self.buffersize=buffersize
        self.bg_buffer = [np.zeros((height, width, 3))]*buffersize
        self.bg=np.zeros((height, width, 3)).astype(np.uint8)
        self.block=np.full((height, width),False)
        self.pre_pha=np.zeros((height, width, 3))
        self.kernel_dilate=cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

    def init(self,bg):
        self.bg=bg.astype(np.uint8)
        self.bg_buffer.append(self.bg)
        self.bg_buffer.pop(0)

    def update_bg_buffer(self, cur_frame, cur_pha):
        bg= (1-cur_pha)*cur_frame
        preserve_area=cur_pha>0.05
        preserve_area = cv2.dilate(preserve_area.astype(np.uint8), self.kernel_dilate, iterations=1)
        preserve_area=preserve_area>0
        bg[preserve_area]=self.bg[preserve_area]
        bg[~preserve_area] = cur_frame[~preserve_area]
        self.bg_buffer.append(bg)
        self.bg_buffer.pop(0)
        bg_from_buffer = np.mean(self.bg_buffer, axis=0)
        self.bg = bg_from_buffer.astype(np.uint8)

    def isBGMoved(self,cur_frame, cur_pha):
        pass

    def isBGStable(self):
        var=np.var(self.bg_buffer,axis=0)
        print('bgmantainer out:',np.mean(var))
        return True if np.mean(var)<40 else False

    def get_bg(self):
        return self.bg

class BGMaintainer2:
    def __init__(self, width, height, buffersize=20):
        self.buffersize=buffersize
        self.bg_buffer = [np.zeros((height, width, 3))]*buffersize
        self.bg=np.zeros((height, width, 3)).astype(np.uint8)
        self.block=np.full((height, width),False)
        self.pre_pha=np.zeros((height, width, 3))
        self.kernel_dilate=cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        self.lock_bg=False

    def init(self,bg):
        self.bg=bg.astype(np.uint8)
        self.bg_buffer.append(self.bg)
        self.bg_buffer.pop(0)

    def update_bg_buffer(self, cur_frame):
        # bg_from_buffer = np.mean(self.bg_buffer, axis=0)
        # self.bg = bg_from_buffer.astype(np.uint8)
        # cv2.absdiff(cur_frame,self.bg)
        if not self.lock_bg:
            self.bg_buffer.append(cur_frame)
            self.bg_buffer.pop(0)

    def isBGMoved(self,cur_frame, cur_pha):
        pass

    def isBGStable(self):
        var=np.var(self.bg_buffer,axis=0)
        print('bgmantainer out:',np.mean(var))
        return True if np.mean(var)<40 else False

    def get_bg(self):
        return self.bg