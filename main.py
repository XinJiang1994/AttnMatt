from APPSystem.app import Application
import os


def checkdir(p):
    if not os.path.exists(p):
        os.makedirs(p)


def main():
    w, h = 1280, 720
    data_root = 'Adapt_data/'
    ckpt = 'APPSystem/app_ckpt/'
    cam_id = 0
    checkdir(data_root)
    checkdir(ckpt)
    vnames = [
        '/home/xinjiang/Documents/VBHOI/Datasets/VideoMatteSyn/HOI/video/0015.mp4']

    app = Application(cam_id=cam_id, width=w, height=h,
                      data_root=data_root, ckpt=ckpt, vnames=vnames)
    app.master.title('Video Matting System')
    app.mainloop()


if __name__ == '__main__':
    main()
