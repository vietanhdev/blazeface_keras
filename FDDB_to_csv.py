import sys, os
import numpy as np
from math import *
import math
import cv2
import os
import csv

# Core codes come from https://github.com/ankanbansal/fddb-for-yolo/blob/master/convertEllipseToRect.py, 
# Thank the author for his/her efforts.

def filterCoordinate(c,m):
    if c < 0:
    	return 0
    elif c > m:
    	return m
    else:
    	return c


def train_validate_test_split(data, train_percent=.6, validate_percent=.2, seed=None):
    np.random.seed(seed)
    perm = np.random.permutation(np.arange(len(data)))
    m = len(data)
    train_end = int(train_percent * m)
    validate_end = int(validate_percent * m) + train_end
    train = np.array(data)[perm[:train_end]].copy()
    validate = np.array(data)[perm[train_end:validate_end]].copy()
    test = np.array(data)[perm[validate_end:]].copy()
    return train.tolist(), validate.tolist(), test.tolist()


def write_csv(data, file_path):
    csv_file = open(file_path, 'w')

    # Write csv header
    fieldnames = ['frame', 'xmin', 'xmax', 'ymin', 'ymax', 'class_id']
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    writer.writeheader()


    # Write csv content
    for img_file in data:
        annotations = img_file[1]
        for anno in annotations:
            writer.writerow(anno)

    csv_file.close()

def FDDB_to_csv(annotation_dir, image_dir, train_anno_file, val_anno_file, test_anno_file, train_ratio=0.6, val_ratio=0.3, seed=42):

    images = {}
    anno_files = [f for f in os.listdir(annotation_dir) if os.path.isfile(os.path.join(annotation_dir, f)) and f.endswith("ellipseList.txt")]

    annotations = {}

    # Process ellipse files
    for anno_file in anno_files:
        
        # Open annotation file and read all lines
        with open(os.path.join(annotation_dir, anno_file)) as f:
            lines = [line.rstrip('\n') for line in f]

        i = 0
        while i < len(lines):
            img_file = os.path.join(image_dir, lines[i] + '.jpg')
            img = cv2.imread(img_file)
            h = img.shape[0]
            w = img.shape[1]
            num_faces = int(lines[i+1])
            for j in range(num_faces):
                ellipse = lines[i+2+j].split()[0:5]
                a = float(ellipse[0])
                b = float(ellipse[1])
                angle = float(ellipse[2])
                centre_x = float(ellipse[3])
                centre_y = float(ellipse[4])
                
                tan_t = -(b/a)*tan(angle)
                t = atan(tan_t)
                x1 = centre_x + (a*cos(t)*cos(angle) - b*sin(t)*sin(angle))
                x2 = centre_x + (a*cos(t+pi)*cos(angle) - b*sin(t+pi)*sin(angle))
                x_max = filterCoordinate(max(x1,x2),w)
                x_min = filterCoordinate(min(x1,x2),w)
                
                if tan(angle) != 0:
                    tan_t = (b/a)*(1/tan(angle))
                else:
                    tan_t = (b/a)*(1/(tan(angle)+0.0001))
                t = atan(tan_t)
                y1 = centre_y + (b*sin(t)*cos(angle) + a*cos(t)*sin(angle))
                y2 = centre_y + (b*sin(t+pi)*cos(angle) + a*cos(t+pi)*sin(angle))
                y_max = filterCoordinate(max(y1,y2),h)
                y_min = filterCoordinate(min(y1,y2),h)

                if img_file not in annotations.keys():
                    annotations[img_file] = []
                
                annotations[img_file].append({'frame': lines[i] + '.jpg', 'xmin': int(x_min), 'xmax': int(x_max), 'ymin': int(y_min), 'ymax': int(y_max), 'class_id': 1 })

            i = i + num_faces + 2

    annotations_list = []
    for key, value in annotations.items():
        temp = [key,value]
        annotations_list.append(temp)
    
    # Split data
    train_files, val_files, test_files = train_validate_test_split(annotations_list, train_ratio, val_ratio, seed=seed)

    write_csv(train_files, train_anno_file)
    write_csv(val_files, val_anno_file)
    write_csv(test_files, test_anno_file)





        