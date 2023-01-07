"""
Expected directory format:

VideoMatte Train/Valid:
    ├──fgr/
      ├── 0001/
        ├── 00000.jpg
        ├── 00001.jpg
    ├── pha/
      ├── 0001/
        ├── 00000.jpg
        ├── 00001.jpg
        
ImageMatte Train/Valid:
    ├── fgr/
      ├── sample1.jpg
      ├── sample2.jpg
    ├── pha/
      ├── sample1.jpg
      ├── sample2.jpg

Background Image Train/Valid
    ├── sample1.png
    ├── sample2.png

Background Video Train/Valid
    ├── 0000/
      ├── 0000.jpg/
      ├── 0001.jpg/

"""


DATA_PATHS = {

    'videomatte': {
        'train': '/home/xinjiang/Documents/VBMH/Dataset/VideoMatte240K_JPEG/train',
        'valid': '/home/xinjiang/Documents/VBMH/Dataset/VideoMatte240K_JPEG/test',
    },
    'local_background': {
        # 'train': '/home/xinjiang/Documents/VBHOI/Datasets/VBHOI_COCO_BG',
        'train': '/home/xinjiang/Documents/VBHOI/VBHOI-Code/LORO/Adapt_data/BG',
    },
    'background_images': {
        'train': '/home/xinjiang/Documents/VBHOI/Datasets/VBHOI_COCO_BG',
        # 'train': '/home/xinjiang/Documents/VBHOI/VBHOI-Code/LORO/Adapt_data/BG',
        'valid': '/home/xinjiang/Documents/VBHOI/Datasets/VBHOI_COCO_BG',
    },
    'background_videos': {
        'train': '/home/xinjiang/Documents/VBHOI/Datasets/BG_Dynamic_JPEG',
        'valid': '/home/xinjiang/Documents/VBHOI/Datasets/BG_Dynamic_JPEG',
    },


}
