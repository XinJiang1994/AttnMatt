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
        'train': '/root/userfolder/VBMH/Dataset/VideoMatte240K_JPEG/train',
        'valid': '/root/userfolder/VBMH/Dataset/VideoMatte240K_JPEG/train',
    },
    'background_images': {
        'train': '/root/commonfiles/Datasets/MSCOCO/train2014',
        'valid': '/root/commonfiles/Datasets/MSCOCO/train2014',
    },
    'background_videos': {
        'train': '/root/userfolder/Datasets/VBHOI/BackgroundVideos/train',
        'valid': '/root/userfolder/Datasets/VBHOI/BackgroundVideos/test',
    },


}
