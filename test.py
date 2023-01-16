import numpy as np
from PIL import Image
import os
import cv2

# from aim import aim


# REPOSITORY_ROOT_PATH = '/home/xinjiang/Documents/VBMH/Code/AttnMatt/'
# SAMPLES_RESULT_COLOR_PATH = REPOSITORY_ROOT_PATH+'output/'

# if __name__ == '__main__':
#     img_path = '/home/xinjiang/Documents/VBMH/Code/AIM/samples/original/girl.jpg'
#     try:
#         img = np.array(Image.open(img_path))[:, :, :3]
#     except Exception as e:
#         print(f'Error: {str(e)} | img_path: {img_path}')
#     com, pha = aim(img)
#     cv2.imwrite(os.path.join(
#         SAMPLES_RESULT_COLOR_PATH, 'girl.jpg'), com.astype(np.uint8))
#     print('Saved to ', os.path.join(
#         SAMPLES_RESULT_COLOR_PATH, 'girl.jpg'))

from APPSystem.test import test
test()
