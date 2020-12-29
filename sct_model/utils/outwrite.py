import cv2
import os
import numpy as np

def take_elem(result):
    return result[0]

def write_to_result(visual_dict, output_name):
    """write visual dict into the result file in a standard format.
    Args:
        visual_dict{
            label:{
                frame_id: {
                    track_id: {
                        loc: [xmin, ymin, xmax, ymax],
                        label: label_name
                    }
                }
            }
        } 

    Result format:
        frame_id(from 0), track_id(from 0), class_name, -1, -1, -1, xmin, ymin, xmax, ymax, -1, -1, -1, -1, -1, -1, -1, 1(score)
    """
    result_list = []
    track_id_dict = {}
    count = 0
    for label in visual_dict.keys():
        track_base = count
        if label == 'DontCare':
            continue
        for frame_id in visual_dict[label].keys():
            for track_id in visual_dict[label][frame_id].keys():
                count += 1
                class_name = visual_dict[label][frame_id][track_id]['class_id']
                xmin, ymin, xmax, ymax = visual_dict[label][frame_id][track_id]['loc']
                track_id = int(track_id) + track_base
                result_list.append([int(frame_id), track_id, class_name, round(float(xmin), 2), round(float(ymin), 2), round(float(xmax), 2), round(float(ymax), 2)])
    f = open(output_name, 'w')
    result_list.sort(key=take_elem)
    for i in range(len(result_list)):
        frame_id, track_id, class_name, xmin, ymin, xmax, ymax = result_list[i]
        if track_id not in track_id_dict.keys():
            track_id_new = len(track_id_dict)
            track_id_dict[track_id] = track_id_new
        track_id = track_id_dict[track_id]
        f.writelines(f'{frame_id} {track_id} {class_name} {-1} {-1} {-1} {xmin} {ymin} {xmax} {ymax} {-1} {-1} {-1} {-1} {-1} {-1} {-1} {1}\n')
    f.close()

    return len(track_id_dict)

def getDistinguishableColors(numColors, bgColors=[(1, 1, 1)]):
    """Pick a set of `numColors` colors that are maximally perceptually distinct.

    When plotting a set of lines/curves/points, you might want to distinguish them
    by color. This module generates a set of colors that are ``maximally perceptually
    distinguishable`` in the RGB colorspace. Given an initial seed list of candidate colors,
    it iteratively picks the color from the list that is the farthest (in the RGB colorspace)
    from all previously chosen entries. This is a ``greedy`` method and does not yield
    a global maximum.

    Inspired by the MATLAB implementation from Timothy E. Holy.

    Args:
            numColors (int): number of distinguishable colors to generate
            bgColors (:obj:`list`, optional): list of background colors

    Returns:
            colors (:obj:`list`): list of `numColors` distinguishable colors

    Examples:
            >>> colors = getDistinguishableColors(25)
    """

    # Start out by generating a sizeable number of RGB triples. This represents our space
    # of possible choices. By starting out in the RGB space, we ensure that all of the colors
    # can be generated by the monitor.

    # Number of grid divisions along each axis in RGB space
    numGrid = 30
    x = np.linspace(0, 1, numGrid)
    [R, G, B] = np.meshgrid(x, x, x)
    rgb = np.concatenate((R.T.reshape((numGrid * numGrid * numGrid, 1)),
                          G.T.reshape((numGrid * numGrid * numGrid, 1)),
                          B.T.reshape((numGrid * numGrid * numGrid, 1))), axis=1)
    if numColors > rgb.shape[0] / 3:
        raise ValueError(
            'You cannot really distinguish that many colors! At most 9000 colors')

    # If the user specified multiple bgColors, compute distance from the candidate colors
    # to the background colors.
    mindist = np.full(rgb.shape[0], np.inf)
    for c in bgColors:
        col = np.full(rgb.shape, 1)
        col[:, 0] = c[0]
        col[:, 1] = c[1]
        col[:, 2] = c[2]
        dx = np.sum(np.abs(rgb - col), axis=1)
        mindist = np.minimum(mindist, dx)

    # Initialize a list of colors
    colors = []
    lastColor = bgColors[-1]
    for i in range(numColors):
        col = np.full(rgb.shape, 1)
        col[:, 0] = lastColor[0]
        col[:, 1] = lastColor[1]
        col[:, 2] = lastColor[2]
        dx = np.sum(np.abs(rgb - lastColor), axis=1)
        mindist = np.minimum(mindist, dx)
        index = np.argmax(mindist)
        chosenColor = (rgb[index, 0], rgb[index, 1], rgb[index, 2])
        colors.append(chosenColor)
        lastColor = chosenColor
    return colors


def write_to_img(result_file, frame_root, test_output_dir, track_num):
    """write visual dict into the frame_img and output the video.
    Args:
        Result format:
            frame_id(from 0), track_id(from 0), class_name, -1, -1, -1, xmin, ymin, xmax, ymax, -1, -1, -1, -1, -1, -1, -1, 1(score)
    """
    if not os.path.exists(test_output_dir):
        os.mkdir(test_output_dir)
    f = open(result_file, 'r')
    lines = f.readlines()
    d = {}
    path = {}
    colors = getDistinguishableColors(5*track_num)
    for line in lines:
        info = line.split(' ')
        frame_id = info[0]
        track_id = int(info[1])
        class_name = info[2]
        xmin = int(float(info[6]))
        ymin = int(float(info[7]))
        xmax = int(float(info[8]))
        ymax = int(float(info[9]))
        # color_pic
        if frame_id not in d.keys():
            d[frame_id] = {}
            path[frame_id] = os.path.join(frame_root, frame_id.zfill(6)+'.png') 
        r, g, b = colors[5*track_id]
        d[frame_id][track_id] = dict(
            track_id = track_id,
            class_name = str(track_id) + ': ' + class_name,
            loc = (xmin, ymin, xmax, ymax),
            color = (int(255*r), int(255*g), int(255*b))
        )

    # write img
    for frame_id in d.keys():
        orig_image = cv2.imread(path[frame_id], cv2.IMREAD_COLOR)
        for track_id in d[frame_id].keys():
            box = d[frame_id][track_id]['loc']
            cv2.rectangle(orig_image, (box[0], box[1]),
                         (box[2], box[3]), d[frame_id][track_id]['color'], 4)
            cv2.putText(
                orig_image,
                d[frame_id][track_id]['class_name'],
                (box[0] + 10, box[1] + 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,  # font scale
                (255, 0, 0),
                2)  # line type
        cv2.imwrite(os.path.join(test_output_dir, f'track_{frame_id.zfill(6)}.jpg'), orig_image)

if __name__ == "__main__":
    import json
    cluster_dict = {}
    with open('data/visualize_0010.json', 'r') as f:
        visual_dict = json.load(f)  
    
    track_num = write_to_result(visual_dict, '0010_pred.txt')
    write_to_img('0010_pred.txt', 'data/training/left_02/images/0010', 'test_out', track_num)
    
