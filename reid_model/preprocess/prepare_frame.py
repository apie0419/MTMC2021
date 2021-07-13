from path import args
import sys

sys.path.insert(0,str(args.root_path))
sys.path.insert(0,"../")

from options import opt
from pathlib import Path
import os
import cv2


def strip_txt_file(txt_file, cam_dict):
    _file = open(txt_file,'r')
    lines = _file.readlines()
    for line in lines:
        line = line.strip()
        ele = line.split(' ',1)
        cam_dict[ele[0]]=int(ele[1])
    return cam_dict


def read_cam_frame(train_S_list, valid_S_list, test_S_list):
    cam_framenum = Path(opt.raw_data_path).joinpath('cam_framenum')
    train_cam_framenum = {}
    valid_cam_framenum={}
    test_cam_framenum={}
    for S_txt in os.listdir(cam_framenum):
        S_txt_path = cam_framenum.joinpath(S_txt)
        if S_txt[:-4] in train_S_list:
            train_cam_framenum = strip_txt_file(S_txt_path, train_cam_framenum)
        elif S_txt[:-4] in valid_S_list:
            valid_cam_framenum = strip_txt_file(S_txt_path, valid_cam_framenum)
        elif S_txt[:-4] in test_S_list:
            test_cam_framenum = strip_txt_file(S_txt_path, test_cam_framenum)
    return train_cam_framenum, valid_cam_framenum, test_cam_framenum

def gen_frame(_folder):
    print(_folder + " start")
    frame_dir = Path(opt.data_root).joinpath('frames').joinpath(_folder+'_frame')
    frame_dir.mkdir(parents=True, exist_ok=True)
    if _folder == 'valid':
        _folder = 'validation'
    for S_folder in Path(opt.raw_data_path).joinpath(_folder).glob('*'):
        if not S_folder.is_dir():
            continue
        if not str(S_folder).split("/")[-1].startswith("S0"):
            continue
        for camid in os.listdir(S_folder):
            cam_vid = S_folder.joinpath(camid).joinpath('vdo.avi')
            cap = cv2.VideoCapture(str(cam_vid))
            success,image = cap.read()
            cnt = 0
            while success:
                frame_name = camid + '_' + str(cnt).zfill(4)
                img_path = Path(frame_dir).joinpath(frame_name+'.jpg')
                cv2.imwrite(str(img_path),image)
                success,image = cap.read()
                cnt += 1
            print(camid + ' finished')
    print(_folder + " complete\n")
                    


if __name__ == '__main__':
    train_S_list = ['S01', 'S03', 'S04']
    valid_S_list = ['S02', 'S05']
    test_S_list = ['S06']
    train_cam_framenum, valid_cam_framenum, test_cam_framenum = read_cam_frame(train_S_list, valid_S_list, test_S_list)
    gen_frame('train')
    gen_frame('valid')
    gen_frame('test')

























# gt: [frame, ID, left, top, width, height, 1, -1, -1, -1]
