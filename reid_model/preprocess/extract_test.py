import sys
from path import args
sys.path.insert(0,str(args.root_path))
sys.path.insert(0,"../")
from options import opt
from pathlib import Path
import cv2
import os


def extract_im(image_name, camid, frame_id, _id, left, top, width, height, car_exist, query_path, gallery_path):
    image =  cv2.imread(str(image_name))
    left = int(left)
    top = int(top)
    width = int(width)
    height = int(height) 
    crop_image = image[top:top+height, left:left+width]
    crop_image_name = camid + '_' + _id.zfill(5) + '_' + frame_id.zfill(4) + '.jpg'
    if car_exist == True:
        crop_path = Path(gallery_path).joinpath(crop_image_name)
        cv2.imwrite(str(crop_path),crop_image)
    else:
        crop_path = Path(gallery_path).joinpath(crop_image_name)
        cv2.imwrite(str(crop_path),crop_image)
        crop_path = Path(query_path).joinpath(crop_image_name)
        cv2.imwrite(str(crop_path),crop_image)


def read_detection():
    for camid in camid_list:
        camid_path = Path(test_data_path).joinpath(camid)
        os.mkdir(str(camid_path))
        query_path = Path(camid_path).joinpath('query_data')
        os.mkdir(str(query_path))
        gallery_path = Path(camid_path).joinpath('gallery_data')
        os.mkdir(gallery_path)
        ids = []
        mtsc_path = Path(test_path).joinpath(camid, 'mtsc', 'mtsc_tnt_mask_rcnn.txt')
        mtsc = open(str(mtsc_path),'r')
        lines = mtsc.readlines()
        for line in lines:
            frame_id, _id, left, top, width, height, _, _, _, _ = line.split(',')
            frame_name = Path(test_frame_path).joinpath(camid + '_' + frame_id.zfill(4) + '.jpg')
            # print(frame_name)
            if frame_name.exists():
                if _id in ids:
                    car_exist = True
                else:
                    car_exist = False
                    ids.append(_id)
                extract_im(str(frame_name), camid, frame_id, _id, left, top, width, height, car_exist, query_path, gallery_path)
        print(camid+' complete')
                
                     
if __name__ == '__main__':

    test_path = Path(opt.raw_data_path).joinpath('test','S06')
    test_frame_path = Path(opt.data_root).joinpath('frames','test_frame')
    test_data_path = Path(opt.data_root).joinpath('test_data')
    camid_list = ['c041', 'c042', 'c043', 'c044', 'c045', 'c046']
    read_detection()

        