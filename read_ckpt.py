import tensorflow as tf
import os
from tensorflow.python import pywrap_tensorflow
import cv2
import os
import numpy as np
import random



checkpoint_path = "chckPts/save2693.ckpt"
reader = pywrap_tensorflow.NewCheckpointReader(checkpoint_path)
# checkpoint_path = os.path.join(model_dir, "model.ckpt-9999")
var_to_shape_map = reader.get_variable_to_shape_map()
# print(var_to_shape_map)
for key in var_to_shape_map:
    if key == 'fc/fcw2':
        print("tensor_name: ", key)
        print(reader.get_tensor(key))

'''
data_dict = np.load("./vgg16.npy", encoding='latin1').item()
# print(data_dict['fc6'][0])
f = open('D:/冯咩咩的文件夹/soccer_detection/dataset/event_train.txt', 'r')
line = list(f)
f.close()
event = []
print(len(line))
s = 0
for l in line:
    if l.split(' ')[0] == 'background':
        event.append(l)
        s = s+1
        if s == 2600:
            break
    else:
        event.append(l)
print(len(event))
f = open('D:/冯咩咩的文件夹/soccer_detection/dataset/event_train1.txt', 'w')
for e in event:
    f.write(e)
f.close()

'''
'''

# 原始数据筛选

txts = []

num=1
segment = 32
for p, d, files in os.walk('D:/冯咩咩的文件夹/soccer_detection/C3D/prepredict/original_result/32'):
    txts = files
    break
for file in txts:
    flag = 0
    event = []
    f = open('D:/冯咩咩的文件夹/soccer_detection/C3D/prepredict/original_result/32/' + file, 'r')
    lines = list(f)
    f.close()
    for l in lines:
        p = float(l.strip('\n').split(' ')[4])
        # if label != '6' and float(p) > 0.5:
        if p>0.5:
            event.append(l)
    f = open('D:/冯咩咩的文件夹/soccer_detection/C3D/prepredict/original_result/32/'+file, 'w')
    for e in event:
        f.write(e)
    f.close()

txts = []

num=1
segment = 64
for p, d, files in os.walk('D:/冯咩咩的文件夹/soccer_detection/C3D/prepredict/original_result/64'):
    txts = files
    break
for file in txts:
    flag = 0
    event = []
    f = open('D:/冯咩咩的文件夹/soccer_detection/C3D/prepredict/original_result/64/' + file, 'r')
    lines = list(f)
    f.close()
    for l in lines:
        p = float(l.strip('\n').split(' ')[4])
        # if label != '6' and float(p) > 0.5:
        if p>0.5:
            event.append(l)
    f = open('D:/冯咩咩的文件夹/soccer_detection/C3D/prepredict/original_result/64/'+file, 'w')
    for e in event:
        f.write(e)
    f.close()

# 原始数据合并

txts = []

num=1
segment = 32
for p, d, files in os.walk('D:/冯咩咩的文件夹/soccer_detection/C3D/prepredict/original_result/32'):
    txts = files
    break
for file in txts:
    flag = 0
    event = []
    f = open('D:/冯咩咩的文件夹/soccer_detection/C3D/prepredict/original_result/32/' + file, 'r')
    lines = list(f)
    f.close()
    pv = lines[0].strip('\n').split(' ')[0]
    ps = int(lines[0].strip('\n').split(' ')[1])
    pe = lines[0].strip('\n').split(' ')[2]
    plabel = lines[0].strip('\n').split(' ')[3]
    pp = float(lines[0].strip('\n').split(' ')[4])
    for l in lines:
        v = l.strip('\n').split(' ')[0]
        s = int(l.strip('\n').split(' ')[1])
        e = int(l.strip('\n').split(' ')[2])
        label = l.strip('\n').split(' ')[3]
        p = float(l.strip('\n').split(' ')[4])
        # if label != '6' and float(p) > 0.5:
        if s-ps<=num*segment*0.5 and label == plabel:
            pe = l.strip('\n').split(' ')[2]
            num = num+1
            if p > pp:
                pp = p
        else:
            event.append(pv+' '+str(ps)+' '+pe+' '+plabel+' '+str(pp)+'\n')
            if lines.index(l) == len(lines)-1:
                flag = 1
            pv = l.strip('\n').split(' ')[0]
            ps = int(l.strip('\n').split(' ')[1])
            pe = l.strip('\n').split(' ')[2]
            plabel = l.strip('\n').split(' ')[3]
            pp = float(l.strip('\n').split(' ')[4])
            num=1
        if flag:
            event.append(l)
    f = open('D:/冯咩咩的文件夹/soccer_detection/C3D/prepredict/32/'+file, 'w')
    for e in event:
        f.write(e)
    f.close()

txts = []

num=1
segment = 64
for p, d, files in os.walk('D:/冯咩咩的文件夹/soccer_detection/C3D/prepredict/original_result/64'):
    txts = files
    break
for file in txts:
    flag = 0
    event = []
    f = open('D:/冯咩咩的文件夹/soccer_detection/C3D/prepredict/original_result/64/' + file, 'r')
    lines = list(f)
    f.close()
    pv = lines[0].strip('\n').split(' ')[0]
    ps = int(lines[0].strip('\n').split(' ')[1])
    pe = lines[0].strip('\n').split(' ')[2]
    plabel = lines[0].strip('\n').split(' ')[3]
    pp = float(lines[0].strip('\n').split(' ')[4])
    for l in lines:
        v = l.strip('\n').split(' ')[0]
        s = int(l.strip('\n').split(' ')[1])
        e = int(l.strip('\n').split(' ')[2])
        label = l.strip('\n').split(' ')[3]
        p = float(l.strip('\n').split(' ')[4])
        # if label != '6' and float(p) > 0.5:
        if s-ps<=num*segment*0.5 and label == plabel:
            pe = l.strip('\n').split(' ')[2]
            num = num+1
            if p > pp:
                pp = p
        else:
            event.append(pv+' '+str(ps)+' '+pe+' '+plabel+' '+str(pp)+'\n')
            if lines.index(l) == len(lines)-1:
                flag = 1
            pv = l.strip('\n').split(' ')[0]
            ps = int(l.strip('\n').split(' ')[1])
            pe = l.strip('\n').split(' ')[2]
            plabel = l.strip('\n').split(' ')[3]
            pp = float(l.strip('\n').split(' ')[4])
            num=1
        if flag:
            event.append(l)
    f = open('D:/冯咩咩的文件夹/soccer_detection/C3D/prepredict/64/'+file, 'w')
    for e in event:
        f.write(e)
    f.close()
'''
'''
t='foul'

# 筛选
# 32
txts = []

for p, d, files in os.walk('D:/冯咩咩的文件夹/soccer_detection/C3D/prepredict/32'):
    txts = files
    break
for file in txts:
    event = []
    f = open('D:/冯咩咩的文件夹/soccer_detection/C3D/prepredict/32/' + file, 'r')
    lines = list(f)
    f.close()
    for l in lines:
        v = l.strip('\n').split(' ')[0]
        label = l.strip('\n').split(' ')[3]
        p = float(l.strip('\n').split(' ')[4])
        if label=='4' and p>0.995:
            event.append(l)
    f = open('D:/冯咩咩的文件夹/soccer_detection/C3D/prepredict/final_result/'+t+'/32/'+v.split('.')[0]+'.txt', 'w')
    for e in event:
        f.write(e)
    f.close()
# 64
txts = []

for p, d, files in os.walk('D:/冯咩咩的文件夹/soccer_detection/C3D/prepredict/64'):
    txts = files
    break
for file in txts:
    event = []
    f = open('D:/冯咩咩的文件夹/soccer_detection/C3D/prepredict/64/' + file, 'r')
    lines = list(f)
    f.close()
    for l in lines:
        v = l.strip('\n').split(' ')[0]
        label = l.strip('\n').split(' ')[3]
        p = float(l.strip('\n').split(' ')[4])
        if label=='4' and p>0.995:
            event.append(l)
    f = open('D:/冯咩咩的文件夹/soccer_detection/C3D/prepredict/final_result/'+t+'/64/'+v.split('.')[0]+'.txt', 'w')
    for e in event:
        f.write(e)
    f.close()

# 再次合并（中间被其他事件阻断的）
# 32
txts = []

for p, d, files in os.walk('D:/冯咩咩的文件夹/soccer_detection/C3D/prepredict/final_result/'+t+'/32'):
    txts = files
    break
for file in txts:
    flag = 0
    event = []
    f = open('D:/冯咩咩的文件夹/soccer_detection/C3D/prepredict/final_result/'+t+'/32/' + file, 'r')
    lines = list(f)
    f.close()
    if len(lines) == 0:
        continue
    pv = lines[0].strip('\n').split(' ')[0]
    ps = int(lines[0].strip('\n').split(' ')[1])
    pe = int(lines[0].strip('\n').split(' ')[2])
    plabel = lines[0].strip('\n').split(' ')[3]
    pp = float(lines[0].strip('\n').split(' ')[4])
    for l in lines:
        v = l.strip('\n').split(' ')[0]
        s = int(l.strip('\n').split(' ')[1])
        e = int(l.strip('\n').split(' ')[2])
        label = l.strip('\n').split(' ')[3]
        p = float(l.strip('\n').split(' ')[4])
        if s < int(pe):
            if e > int(pe):
                pe = e
            if p > pp:
                pp = p
        else:
            event.append(pv+' '+str(ps)+' '+str(pe)+' '+plabel+' '+str(pp)+'\n')
            if lines.index(l) == len(lines) - 1:
                flag = 1
            pv = l.strip('\n').split(' ')[0]
            ps = int(l.strip('\n').split(' ')[1])
            pe = l.strip('\n').split(' ')[2]
            plabel = l.strip('\n').split(' ')[3]
            pp = float(l.strip('\n').split(' ')[4])
        if flag:
            event.append(l)
    f = open('D:/冯咩咩的文件夹/soccer_detection/C3D/prepredict/final_result/'+t+'/32/'+file, 'w')
    for e in event:
        f.write(e)
    f.close()


# 64
txts = []

for p, d, files in os.walk('D:/冯咩咩的文件夹/soccer_detection/C3D/prepredict/final_result/'+t+'/64'):
    txts = files
    break
for file in txts:
    flag = 0
    event = []
    f = open('D:/冯咩咩的文件夹/soccer_detection/C3D/prepredict/final_result/'+t+'/64/' + file, 'r')
    lines = list(f)
    f.close()
    if len(lines) == 0:
        continue
    pv = lines[0].strip('\n').split(' ')[0]
    ps = int(lines[0].strip('\n').split(' ')[1])
    pe = int(lines[0].strip('\n').split(' ')[2])
    plabel = lines[0].strip('\n').split(' ')[3]
    pp = float(lines[0].strip('\n').split(' ')[4])
    for l in lines:
        v = l.strip('\n').split(' ')[0]
        s = int(l.strip('\n').split(' ')[1])
        e = int(l.strip('\n').split(' ')[2])
        label = l.strip('\n').split(' ')[3]
        p = float(l.strip('\n').split(' ')[4])
        if s<int(pe):
            if e>int(pe):
                pe = e
            if p > pp:
                pp = p
        else:
            event.append(pv+' '+str(ps)+' '+str(pe)+' '+plabel+' '+str(pp)+'\n')
            if lines.index(l) == len(lines) - 1:
                flag = 1
            pv = l.strip('\n').split(' ')[0]
            ps = int(l.strip('\n').split(' ')[1])
            pe = l.strip('\n').split(' ')[2]
            plabel = l.strip('\n').split(' ')[3]
            pp = float(l.strip('\n').split(' ')[4])
        if flag:
            event.append(l)
    f = open('D:/冯咩咩的文件夹/soccer_detection/C3D/prepredict/final_result/'+t+'/64/'+file, 'w')
    for e in event:
        f.write(e)
    f.close()

# 不同长度合并
dir_32 = []
dir_64 = []
flag = 0
num = []
for p, d, files in os.walk('D:/冯咩咩的文件夹/soccer_detection/C3D/prepredict/final_result/'+t+'/32'):
    dir_32 = files
    break
for p, d, files in os.walk('D:/冯咩咩的文件夹/soccer_detection/C3D/prepredict/final_result/'+t+'/64'):
    dir_64 = files
    break
for i in range(len(dir_64)):
    event = []
    file_32 = dir_32[i]
    file_64 = dir_64[i]
    f_32 = open('D:/冯咩咩的文件夹/soccer_detection/C3D/prepredict/final_result/'+t+'/32/' + file_32)
    f_64 = open('D:/冯咩咩的文件夹/soccer_detection/C3D/prepredict/final_result/'+t+'/64/' + file_64)
    l_32 = list(f_32)
    l_64 = list(f_64)
    f_32.close()
    f_64.close()
    if len(l_32) == 0 and len(l_64) == 0:
        continue
    if len(l_32) == 0:
        for p in range(len(l_64)):
            event.append(l_64[p])
            continue
    if len(l_64) == 0:
        for p in range(len(l_32)):
            event.append(l_32[p])
            continue
    for j in range(len(l_32)):
            s_32 = int(l_32[j].split(' ')[1])
            e_32 = int(l_32[j].split(' ')[2])
            p_32 = float(l_32[j].split(' ')[4])
            for k in range(len(l_64)):
                    s_64 = int(l_64[k].split(' ')[1])
                    e_64 = int(l_64[k].split(' ')[2])
                    p_64 = float(l_64[k].split(' ')[4])
                    if e_32<=s_64 or e_64<=s_32:
                        if k<len(l_64)-1:
                            continue
                    else:
                        if p_32>p_64:
                            event.append(l_32[j])
                            num.append(k)
                            break
                        else:
                            event.append(l_64[k])
                            num.append(k)
                            break
                    event.append(l_32[j])
    for p in range(len(l_64)):
        if p not in num:
            event.append(l_64[p])

    f = open('D:/冯咩咩的文件夹/soccer_detection/C3D/prepredict/final_result/final_result/'+t+'/'+file_32, 'w')
    for e in event:
        f.write(e)


# 计算P、R
file = []
num = 0
predict_num = 0
label_num = 0

for p, d, files in os.walk('D:/冯咩咩的文件夹/soccer_detection/C3D/prepredict/final_result/final_result/'+ t):
    file = files
    break
for name in file:
    f = open('D:/冯咩咩的文件夹/soccer_detection/C3D/prepredict/final_result/final_result/'+t+'/' + name)
    rf = open('D:/冯咩咩的文件夹/soccer_detection/dataset/test/' + name.split('.')[0] + '_event.txt')
    lf = list(f)
    lrf = list(rf)
    f.close()
    rf.close()
    for j in range(len(lf)):
        s_f = int(lf[j].split(' ')[1])
        e_f = int(lf[j].split(' ')[2])
        for k in range(len(lrf)):
            s_r = int(lrf[k].split(' ')[1])
            e_r = int(lrf[k].split(' ')[2])
            l_r = lrf[k].split(' ')[0]
            replay = lrf[k].split(' ')[3]
            if replay != '1' and l_r in [t]:
                if s_f<s_r<e_f or s_f<e_r<e_f or (s_r<s_f and e_r>e_f):
                    num = num+1
                    print(lf[j])
                    break

for name in file:
    f = open('D:/冯咩咩的文件夹/soccer_detection/C3D/prepredict/final_result/final_result/'+t+'/' + name)
    lf = list(f)
    f.close()
    predict_num = predict_num + len(lf)

file = []
for p, d, files in os.walk('D:/冯咩咩的文件夹/soccer_detection/C3D/prepredict/original_result/32/'):
    file = files
    break
for name in file:
    rf = open('D:/冯咩咩的文件夹/soccer_detection/dataset/test/' + name.split('_')[0] + '_event.txt')
    lrf = list(rf)
    rf.close()
    for k in range(len(lrf)):
        l_r = lrf[k].split(' ')[0]
        replay = lrf[k].strip('\n').split(' ')[3]
        if replay != '1' and l_r in [t]:
            label_num = label_num+1
            print(name, lrf[k])
print(num)


print('R:', float(num)/float(label_num))
print('P:', float(num)/float(predict_num))
'''