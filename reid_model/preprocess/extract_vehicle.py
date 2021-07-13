from path import args
import sys
sys.path.insert(0,str(args.root_path))
sys.path.insert(0,"../")
from options import opt
from pathlib import Path
import os
import cv2


# [frame, ID, left, top, width, height, 1, -1, -1, -1]


def sort_gt_id(gt_path):
    sorted_gt=[]
    gt_file = open(str(gt_path), 'r')
    lines = gt_file.readlines()
    for line in lines:
        line = line.strip()
        frame_id, vehicle_id, left, top, width, height, _, _, _, _ = line.split(',')
        if vehicle_id in car_list:
            sorted_gt.append([False, frame_id, vehicle_id, int(left), int(top), int(width), int(height)])
        else:
            sorted_gt.append([True, frame_id, vehicle_id, int(left), int(top), int(width), int(height)])
            car_list.append(vehicle_id)
    return sorted_gt


def extract_im(sorted_gt, mode, cam_id):
    for gt in sorted_gt:
        is_query, frame_id, vehicle_id, left, top, width, height = gt

        if mode == 'train':
            image_name = cam_id + '_' + str(int(frame_id)-1).zfill(4) + '.jpg'
            if Path(frames_path).joinpath('train_frame').joinpath(image_name).exists():
                image_path = str(Path(frames_path).joinpath('train_frame').joinpath(image_name))
            else:
                continue

        elif mode == 'valid':
            image_name = cam_id + '_' + str(int(frame_id)-1).zfill(4) + '.jpg'
            if Path(frames_path).joinpath('valid_frame').joinpath(image_name).exists():
                image_path = str(Path(frames_path).joinpath('valid_frame').joinpath(image_name))
            else:
                continue

        img = cv2.imread(image_path)
        crop_img = img[top:top+height, left:left+width]
        crop_img_name = vehicle_id.zfill(5) + '_' + cam_id + '_' + frame_id.zfill(4) + '.jpg'
        if is_query == True:
            crop_img_path = Path(opt.data_root).joinpath('reid_data/query_data').joinpath(crop_img_name)
        elif mode == 'train':
            crop_img_path = Path(opt.data_root).joinpath('reid_data/train_data').joinpath(crop_img_name)
        elif mode == 'valid':
            crop_img_path = Path(opt.data_root).joinpath('reid_data/gallery_data').joinpath(crop_img_name)

        cv2.imwrite(str(crop_img_path), crop_img)


def get_frame(frames_dir):
    frame_dir_path = Path(frames_path).joinpath(frames_dir)
    print(frames_dir + ' start')

    if frames_dir == 'train_frame':
        for i in range(len(train_S_dirname)):
            S_dir = Path(gt_root_path).joinpath('train').joinpath(train_S_dirname[i])
            for cam_id in train_S[i]:
                gt_path = Path(S_dir).joinpath(cam_id).joinpath('gt/gt.txt')
                sorted_gt = sort_gt_id(gt_path)
                extract_im(sorted_gt, 'train', cam_id)
                print(cam_id + ' complete')
            
    elif frames_dir == 'valid_frame':
        for i in range(len(valid_S_dirname)):
            S_dir = Path(gt_root_path).joinpath('validation').joinpath(valid_S_dirname[i])
            for cam_id in valid_S[i]:
                gt_path = Path(S_dir).joinpath(cam_id).joinpath('gt/gt.txt')
                sorted_gt = sort_gt_id(gt_path)
                extract_im(sorted_gt, 'valid', cam_id)
                print(cam_id + ' complete')

    print(frames_dir + ' complete\n')


if __name__ == '__main__':
    S01 = ['c001', 'c002', 'c003', 'c004', 'c005']
    S02 = ['c006', 'c007', 'c008', 'c009']
    S03 = ['c010', 'c011', 'c012', 'c013', 'c014', 'c015']
    S04 = ['c016', 'c017', 'c018', 'c019', 'c020', 'c021', 'c022', 'c023', 'c024', 'c025', 'c026', 'c027',
           'c028', 'c029', 'c030', 'c031', 'c032', 'c033', 'c034', 'c035', 'c036', 'c037', 'c038', 'c039',
           'c040']
    S05 = ['c010', 'c016', 'c017', 'c018', 'c019', 'c020', 'c021', 'c022', 'c023', 'c024', 'c025', 'c026',
           'c027', 'c028', 'c029', 'c033', 'c034', 'c035', 'c036']
    S06 = ['c041', 'c042', 'c043', 'c044', 'c045', 'c046']

    train_S = [S01, S03, S04]
    valid_S = [S02, S05]

    train_S_dirname = ['S01', 'S03', 'S04']
    valid_S_dirname = ['S02', 'S05']
    
    frames_path = Path(opt.data_root).joinpath('frames')
    gt_root_path = opt.raw_data_path
    Path(opt.data_root).joinpath('reid_data/query_data').mkdir(parents=True, exist_ok=True)
    Path(opt.data_root).joinpath('reid_data/train_data').mkdir(parents=True, exist_ok=True)
    Path(opt.data_root).joinpath('reid_data/gallery_data').mkdir(parents=True, exist_ok=True)

    car_list = []

    get_frame('train_frame')
    get_frame('valid_frame')
    
