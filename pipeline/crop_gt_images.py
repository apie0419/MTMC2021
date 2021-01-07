import numpy as np
import cv2, os, json

from utils import init_path

init_path()

from config import cfg

def crop(fnum, frame, boxes, ids, img_path, data):
    idx = 0
    for box in boxes:
        track_id = int(ids[idx])
        if track_id not in data:
            data[track_id] = [[], [], [], [], []]
        data[track_id][0].append(int(fnum))
        for i in range(1, 5):
            data[track_id][i].append(box[i-1])
        cropped_img = frame[box[0]:box[2], box[1]:box[3]]
        img_name = f"{track_id}_{int(fnum)}_crop.jpg"
        
        if not os.path.exists(img_path):
            os.mkdir(img_path)

        cv2.imwrite(os.path.join(img_path, img_name), cropped_img)
        print ("Cropped Image: " + os.path.join(img_path, img_name))
        idx += 1

    return data

def preprocess_boxes(src_boxes, src_ids):
    boxes, ids = list(), list()
    for idx in range(len(src_boxes)):
        src_box = src_boxes[idx]
        src_id  = src_ids[idx]
        boxes.append(src_box)
        ids.append(src_id)

    return boxes, ids

def main(TRAINSET_PATH):

    track_dict = {}
    output_file = os.path.join(TRAINSET_PATH, "gt_track_dict.json")
    for scene_dir in os.listdir(TRAINSET_PATH):

        if not scene_dir.startswith("S0"):
            continue
        for camera_dir in os.listdir(os.path.join(TRAINSET_PATH, scene_dir)):
            if not camera_dir.startswith("c0"):
                continue
            video_path  = os.path.join(TRAINSET_PATH, scene_dir, camera_dir, "vdo.avi")
            gt_path     = os.path.join(TRAINSET_PATH, scene_dir, camera_dir, "gt/gt.txt")
            image_path  = os.path.join(TRAINSET_PATH, scene_dir, camera_dir, "gt_images")
            
            cap = cv2.VideoCapture(video_path)
            gt = np.loadtxt(gt_path, delimiter=",")
            frame_count = 0
            i = 0
            (fnum, id, left, top, width, height) = gt[i][:6]
            
            if os.path.exists(output_file):
                os.remove(output_file)
            
            data = {}
            while cap.isOpened():
                try :
                    ret, frame = cap.read()
                    frame_count += 1
                    boxes = list()
                    ids = list()
                    if not ret:
                        break
                    if frame_count != fnum:
                        continue
                    
                except:
                    break
                
                while fnum == frame_count:
                    boxes.append((int(top), int(left), int(top+height), int(left+width)))
                    
                    ids.append(int(id))
                    i += 1
                    if i >= len(gt):
                        break
                    (fnum, id, left, top, width, height) = gt[i][:6]
                
                boxes, ids = preprocess_boxes(boxes, ids)
                data = crop(fnum, frame, boxes, ids, image_path, data)
            
            track_dict[f"{scene_dir}_{camera_dir}"] = data

            cap.release()

    with open(output_file, 'w+') as f:
        json.dump(track_dict, f, ensure_ascii=False)

if __name__ == "__main__":
    for path in ["train", "validation"]:

        main(os.path.join(cfg.PATH.ROOT_PATH, path))
    
    print ("Finish")