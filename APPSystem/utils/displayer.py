import time
import cv2

# An FPS tracker that computes exponentialy moving average FPS


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
        self._avg_fps = self.ratio * fps_sample + \
            (1 - self.ratio) * \
            self._avg_fps if self._avg_fps is not None else fps_sample
        self._last_tick = t_new
        return self.get()

    def get(self):
        return self._avg_fps


class Displayer:
    def __init__(self, title, width=None, height=None, show_info=True):
        print('Init camera...')
        self.title, self.width, self.height = title, width, height
        self.show_info = show_info
        self.fps_tracker = FPSTracker()
        print('FPSTracker builed')
        cv2.namedWindow(self.title, cv2.WINDOW_NORMAL)
        print('Build named window')
        if width is not None and height is not None:
            cv2.resizeWindow(self.title, width, height)
            print('Init camera success')
    # Update the currently showing frame and return key press char code

    def step(self, image):
        fps_estimate = self.fps_tracker.tick()
        if self.show_info and fps_estimate is not None:
            message = f"{self.width}x{self.height}"
            cv2.putText(image, message, (30, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 0, 0), thickness=2)
        cv2.imshow(self.title, image)
        return cv2.waitKey(1) & 0xFF
