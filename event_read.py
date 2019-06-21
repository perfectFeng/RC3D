#!/usr/bin/python
#-*- coding: utf-8 -*-

import cv2
import numpy as np
import random
import c3d_model


event_dic = {'background': 0,
             'shot': 1,
             'corner': 2,
             'free-kick': 3,
             'yellow-card': 4,
             'foul': 5,
             'goal': 6,
             'offside': 7,
             'overhead-kick': 8,
             'solo-drive': 9,
             'penalty-kick': 10,
             'red-card': 11,
             }
             

def readTestFile(batch_size, num_frames):
    f = open("./dataset/64_test.txt", 'r')
    lines = list(f)
    random.shuffle(lines)
    events = []
    labels = []
    for l in range(batch_size):
        frames = []
        event = lines[l].strip('\n').split(' ')
        event_type = event[0]
        start = int(event[1])
        video = event[3]
        labels.append(int(event_dic[str(event_type)]))
        cap = cv2.VideoCapture("../video/" + video)
        first_frame = random.randint(start, end - num_frames)
        for n in range(num_frames):
            cap.set(cv2.CAP_PROP_POS_FRAMES, first_frame + n)  # 设置要获取的帧号
            a, b = cap.read()  # read方法返回一个布尔值和一个视频帧。若帧读取成功，则返回True
            b = cv2.resize(b, (c3d_model.width, c3d_model.height), interpolation=cv2.INTER_CUBIC)
            b = per_image_standard(b, c3d_model.width*c3d_model.height)
            frames.append(b)

        events.append(frames)
    f.close()
    events = np.array(events).astype(np.float32)
    labels = np.array(labels).astype(np.int64)
    return events, labels


def readFile():
    f = open("./dataset/64_train.txt", 'r')
    lines = list(f)
    random.shuffle(lines)
    f.close()

    return lines


def readTrainData(batch, lines, batch_size, num_frames):

    events = []
    labels = []
    ious = []

    for b in range(batch*batch_size, batch*batch_size + batch_size):

        frames = []

        event = lines[b].strip('\n').split(' ')
        event_type = event[0]
        start = int(event[1])
        end = int(event[2])
        video = event[3]
        labels.append(int(event_dic[str(event_type)]))
        iou = float(event[4])
        ious.append(iou)
        cap = cv2.VideoCapture("../video/" + video)

        first_frame = random.randint(start, end-num_frames)

        for n in range(num_frames):

            cap.set(cv2.CAP_PROP_POS_FRAMES, first_frame + n)  # 设置要获取的帧号
            a, b = cap.read()  # read方法返回一个布尔值和一个视频帧。若帧读取成功，则返回True
            b = cv2.resize(b, (c3d_model.width, c3d_model.height), interpolation=cv2.INTER_CUBIC)
            b = per_image_standard(b, c3d_model.width * c3d_model.height)
            frames.append(b)

        events.append(frames)
    events = np.array(events).astype(np.float32)
    labels = np.array(labels).astype(np.int64)
    ious = np.array(ious).astype(np.float32)
    return events, labels, ious


def per_image_standard(image, num_rgb):
    mean = np.mean(image)
    stddev = np.std(image)
    image = (image - mean)/(max(stddev, 1.0/np.sqrt(num_rgb)))
    return image
