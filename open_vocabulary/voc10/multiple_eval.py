import os
import pandas as pd
import numpy as np
from PIL import Image
import multiprocessing
import scipy.io as sio
import argparse
import torch
import torch.nn.functional as F
from typing import List
import json

import sys
sys.path.insert(0, sys.path[0]+"/../..")

from local_config import context_path

def load_embedings():
    embs = [None] # class ids start from 1
    labels_path = os.path.join(context_path, 'context', 'labels.txt')
    with open(labels_path) as f:
        labels_file_rows = f.read().split('\n')
        if labels_file_rows[-1] == '': # fix when empty last line in file
            labels_file_rows = labels_file_rows[:-1]
    for id_and_label in labels_file_rows:
        _, label = id_and_label.split(': ')
        if label == 'tvmonitor':
            label = 'monitor'
        elif label == 'aeroplane':
            label = 'airplane'
        embs.append(label)
    return embs

embs = load_embedings()
categories = embs

def convert_from_mat(mat_file_path, num_cls, threshold):
    data_as_dict = sio.loadmat(mat_file_path)
    h, w = list(data_as_dict.values())[-1].shape
    tensor_data = np.zeros((num_cls, h, w), np.float32)
    for key in list(data_as_dict.keys())[3:]:
        tensor_data[int(key)] = data_as_dict[key] / 255
    tensor_data[0, :, :] = threshold
    data_as_classes = np.argmax(tensor_data, axis=0).astype(np.int16)
    return data_as_classes

def read_gt_file(mat_file_path):
    data = sio.loadmat(mat_file_path)["LabelMap"]
    h, w = data.shape
    tensor_data = np.zeros((h, w), np.int16) + data
    return tensor_data

def compare(start,step,TP,P,T,input_type,threshold, name_list, predict_folder, num_cls, gt_folder, predict_classes):
    for idx in range(start,len(name_list),step):
        name = name_list[idx]
        if input_type == 'png':
            predict_file = os.path.join(predict_folder,'%s.png'%name)
            predict = np.array(Image.open(predict_file)) #cv2.imread(predict_file)
            if num_cls == 81:
                predict = predict - 91
        elif input_type == 'npy':
            predict_file = os.path.join(predict_folder,'%s.mat'%name)
            predict = convert_from_mat(predict_file, num_cls, threshold)

        gt_file = os.path.join(gt_folder,'%s.mat'%name)
        gt = read_gt_file(gt_file)
        mask = (predict == gt)
        
        training_classes_str = predict_classes[f'{name}.jpg']
        training_classes_ids = [embs.index(x) for x in training_classes_str]
        for i in training_classes_ids:
            P[i].acquire()
            P[i].value += np.sum((predict==i))
            P[i].release()
            T[i].acquire()
            T[i].value += np.sum((gt==i))
            T[i].release()
            TP[i].acquire()
            TP[i].value += np.sum((gt==i)*mask)
            TP[i].release()

def read_pred_classes(pred_dict_path):
    choice = choice=['image_similarity','text_similarity']
    with open(pred_dict_path,"r") as tmp:
        data=json.load(tmp)
    re=data[choice[0]]
    if len(choice)==1:
        return re
    for key in re.keys():
        for i in range(1,len(choice)):
            re[key].extend(data[choice[i]][key])
        re[key]=list(set(re[key]))
    return re 

def do_python_eval(predict_folder, gt_folder, name_list, num_cls=21, input_type='png', threshold=1.0, printlog=False, pred_dict_path=None):
    TP = []
    P = []
    T = []
    # count=0
    txt = f'{predict_folder}/text.txt'
    predict_classes = read_pred_classes(pred_dict_path)

    # f=open(txt,"a+")
    for i in range(num_cls):
        TP.append(multiprocessing.Value('i', 0, lock=True))
        P.append(multiprocessing.Value('i', 0, lock=True))
        T.append(multiprocessing.Value('i', 0, lock=True))

    p_list = []
    for i in range(8):
        p = multiprocessing.Process(target=compare, args=(i,8,TP,P,T,input_type,threshold, name_list, predict_folder, num_cls, gt_folder, predict_classes))
        p.start()
        p_list.append(p)
    for p in p_list:
        p.join()
    IoU = []
    T_TP = []
    P_TP = []
    FP_ALL = []
    FN_ALL = [] 
    for i in range(num_cls):
        IoU.append(TP[i].value/(T[i].value+P[i].value-TP[i].value+1e-10))
        T_TP.append(T[i].value/(TP[i].value+1e-10))
        P_TP.append(P[i].value/(TP[i].value+1e-10))
        FP_ALL.append((P[i].value-TP[i].value)/(T[i].value + P[i].value - TP[i].value + 1e-10))
        FN_ALL.append((T[i].value-TP[i].value)/(T[i].value + P[i].value - TP[i].value + 1e-10))
    loglist = {}
    for i in range(num_cls):
        loglist[categories[i]] = IoU[i] * 100
               
    miou = np.mean(np.array(IoU))
    loglist['mIoU'] = miou * 100
    loglist['std'] = np.std(np.array(IoU)) * 100
    fp = np.mean(np.array(FP_ALL))
    loglist['FP'] = fp * 100
    fn = np.mean(np.array(FN_ALL))
    loglist['FN'] = fn * 100
    if printlog:
        for i in range(num_cls):
            if i%2 != 1:
                print('%11s:%7.3f%%'%(categories[i],IoU[i]*100),end='\t')
            else:
                print('%11s:%7.3f%%'%(categories[i],IoU[i]*100))
        print('\n======================================================')
        print('%11s:%7.3f%%'%('mIoU',miou*100))
        print('\n')
        print(f'FP = {fp*100}, FN = {fn*100}')
    return loglist

def writedict(file, dictionary):
    s = ''
    for key in dictionary.keys():
        sub = '%s:%s  '%(key, dictionary[key])
        s += sub
    s += '\n'
    file.write(s)

def writelog(filepath, metric, comment):
    filepath = filepath
    logfile = open(filepath,'a')
    import time
    logfile.write(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    logfile.write('\t%s\n'%comment)
    writedict(logfile, metric)
    logfile.write('=====================================\n')
    logfile.close()

import cv2
import torch.nn.functional as F
# cam visual_code
def show_cam_on_image(img, mask):
    mask = F.interpolate(mask.unsqueeze(0).unsqueeze(0), size=(img.size[1],img.size[0]), mode='bilinear', align_corners=False).squeeze().squeeze()
    img = np.float32(img) / 255.
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    cam = heatmap + img
    cam = cam / np.max(cam)
    cam = np.uint8(255 * cam)
    return cam

def save_segmentation(orig_image: Image, mask: torch.Tensor, file_id: str, args):
    if not os.path.exists(args.save_samples_path):
        os.makedirs(args.save_samples_path)
    save_path = os.path.join(args.save_samples_path, file_id + '.jpg')
    
    cam = show_cam_on_image(orig_image, mask)
    segmented_image = Image.fromarray(cam[:,:,::-1])
    segmented_image.save(save_path)

def save_sample_segmentations(args):
    predict_folder = args.cam_npy_dir
    masks_paths = [os.path.join(predict_folder, x) for x in os.listdir(predict_folder)[:args.save_count]]
    masks_ids = [x.split(os.sep)[-1].replace('.mat', '') for x in masks_paths]
    for mask_path, mask_id in zip(masks_paths, masks_ids):
        orig_image_path = os.path.join(voc_path, 'JPEGImages', mask_id + '.jpg')
        orig_image = Image.open(orig_image_path).convert("RGB")
        saved_dict = sio.loadmat(mask_path)
        for class_id in list(saved_dict.keys())[3:]:
            mask_tensor = torch.Tensor(saved_dict[class_id] / 255)
            save_segmentation(orig_image, mask_tensor, mask_id + str(class_id), args)

class Args:
    val_split = 0.5
    results='./results.csv'
    comment='train1464'
    curve=True
    end=70
    base_dir = './tmp_2/context/all_results/res_cm/voc10_wo_sy_norm'
    image_dir = os.path.join(base_dir,'images')
    cam_npy_dir = os.path.join(base_dir, 'images')
    gt_dir=os.path.join(context_path, 'context', 'trainval')
    list='dataset/voc10/val_id.txt'
    pred_json = 'dataset/voc10/predict_syn_txt_0.8_img_0.97.json'
    # list='/root/autodl-tmp/wjl/ptp_diffusion/voc12/train_aug_id.txt'
    logfile=os.path.join(base_dir,'eval.txt')
    num_classes=len(embs)
    start=30
    t=None
    type='npy'
    sample_images=True
    save_samples_path = os.path.join('output', 'samples')
    save_count=10

def test():
    args = Args()

    result_to_str = lambda x: str(round(x, 2))
    output_paths = os.listdir(args.cam_npy_dir)
    result_rows = []

    if args.sample_images:
        save_sample_segmentations(args)

    for path in output_paths:
        current_results_path = os.path.join(args.cam_npy_dir, path)
        if args.type == 'npy':
            assert args.t is not None or args.curve
        df = pd.read_csv(args.list, names=['filename'])
        prediction_ids = [x.split('.')[0] for x in os.listdir(current_results_path)]
        name_list = [x for x in df['filename'].values if x in prediction_ids]

        print(f"Len prediction labels {len(df['filename'].values)}, Len computed labels {len(prediction_ids)}")

        val_split_size = int(args.val_split * len(name_list))
        val_name_list = name_list[:val_split_size]
        test_name_list = name_list[val_split_size:]

        if not args.curve:
            loglist = do_python_eval(current_results_path, args.gt_dir, name_list, args.num_classes, args.type, args.t, printlog=True, pred_dict_path=args.pred_json)
            writelog(args.logfile, loglist, args.comment)
            best_thr = None
        else:
            l = []
            max_mIoU = 0.0
            best_thr = 0.0
            print(f"Finding threshold on a data split of {len(val_name_list)}")
            for i in range(args.start, args.end):
                t = i/100.0
                loglist = do_python_eval(current_results_path, args.gt_dir, val_name_list, args.num_classes, args.type, t, pred_dict_path=args.pred_json)
                l.append(loglist['mIoU'])
                # print('%d/%d background score: %.3f\tmIoU: %.3f%%'%(i, args.end, t, loglist['mIoU']))
                if loglist['mIoU'] > max_mIoU:
                    max_mIoU = loglist['mIoU']
                    best_thr = t
                # else:
                #     break
            print('Best background score: %.3f\tmIoU: %.3f%%' % (best_thr, max_mIoU))
            writelog(args.logfile, {'mIoU':l, 'Best mIoU': max_mIoU, 'Best threshold': best_thr}, args.comment)
        
        print(f"Using best threshold of {best_thr} for test split of size {len(test_name_list)}")
        loglist_test = do_python_eval(current_results_path, args.gt_dir, test_name_list, args.num_classes, args.type, best_thr, pred_dict_path=args.pred_json)
        print(f"Final score: {loglist_test['mIoU']}, std: {loglist_test['std']}")
        print('-' * 50 + '\n')
        result_rows.append(','.join([
            path,
            result_to_str(best_thr),
            result_to_str(max_mIoU),
            result_to_str(loglist_test['mIoU']),
            result_to_str(loglist_test['std'])
        ]))

    result_rows = sorted(result_rows, key=lambda x: int(x.split(',')[0]))
    result_rows = [",".join(["timestamp", "threshold", "val mIoU", "test mIoU", "test std(mIoU)"])] + result_rows
    with open(args.results, 'w') as f:
        f.write('\n'.join(result_rows))

    
if __name__ == '__main__':
    test()