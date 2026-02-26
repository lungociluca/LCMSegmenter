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

from local_config import context_path, device


def load_embedings():
    embs = [None] # class ids start from 1
    labels_path = os.path.join(context_path, 'context', 'labels.txt')
    with open(labels_path) as f:
        labels_file_rows = f.read().split('\n')
        if labels_file_rows[-1] == '': # fix when empty last line in file
            labels_file_rows = labels_file_rows[:-1]
    for id_and_label in labels_file_rows:
        _, label = id_and_label.split(': ')
        embs.append(label)
    return embs

def get_min_sizes(imgs: List[torch.LongTensor]):
    img: torch.LongTensor
    height = 10**5
    width = 10**5
    for img in imgs:
        current_height, current_width = img.shape
        if current_height < height or current_width < width:
            height = current_height
            width = current_width
    return height, width

embs = load_embedings()
categories = embs

def convert_from_mat(mat_file_path, target_classes_ids, threshold):
    data_as_dict = sio.loadmat(mat_file_path)
    h, w = list(data_as_dict.values())[-1].shape
    # tensor_data = np.zeros((len(embs), h, w), np.float32)
    tensor_data = torch.zeros((len(embs), h, w)).to(device)

    for key in target_classes_ids:
        class_mask = torch.Tensor(data_as_dict[str(key)]).to(device) / 255
        tensor_data[key] += class_mask

    tensor_data[0, :, :] += threshold
    data_as_classes = torch.argmax(tensor_data, axis=0).to(torch.float32)
    return data_as_classes


def read_gt_file(mat_file_path):
    data = sio.loadmat(mat_file_path)["LabelMap"]
    h, w = data.shape
    tensor_data = np.zeros((h, w), np.int16) + data
    return torch.Tensor(tensor_data)

def compare(start,step,TP,P,T,input_type,threshold, name_list, predict_folder, num_cls, gt_folder, predict_classes, bs=16):
    for idx in range(start,len(name_list), bs):
        names = name_list[idx: idx + bs]
        training_classes_str = [predict_classes[name] for name in names]
        training_classes_ids = [[embs.index(x) for x in classes] for classes in training_classes_str]

        predict_files = [os.path.join(predict_folder,'%s.mat'%name) for name in names]
        
        predictions_list = []
        for path, classes_ids in zip(predict_files, training_classes_ids):
            predict = convert_from_mat(path, classes_ids, threshold)
            predictions_list.append(predict)
        
        height, width = get_min_sizes(predictions_list)

        gt_list = []
        for name in names:
            gt_file = os.path.join(gt_folder,'%s.mat'%name)
            gt = read_gt_file(gt_file)
            gt_list.append(gt)
        
        predictions_list_resized = []
        gt_list_resized = []
        interpolate_func = lambda x: F.interpolate(x.unsqueeze(0).unsqueeze(0), size=(height, width), mode='bilinear', align_corners=False).squeeze().squeeze()
        for predict, gt in zip(predictions_list, gt_list):
            predictions_list_resized.append(
                interpolate_func(predict).to(torch.int16)
            )
            gt_list_resized.append(
                interpolate_func(gt).to(torch.int16)
            )

        predict_batch = torch.stack(predictions_list_resized).to(device)
        gt_batch = torch.stack(gt_list_resized).to(device)
        mask = (predict_batch == gt_batch)

        classes_set = set()
        for classes_ids in training_classes_ids:
            for _id in classes_ids:
                classes_set.add(_id)
        for i in list(classes_set):
            P[i] += torch.sum((predict_batch == i)).detach().cpu()
            T[i] += torch.sum((gt_batch==i)).detach().cpu()
            TP[i] += torch.sum((gt_batch==i)*mask).detach().cpu()

    return TP, P, T

def read_pred_classes():
    predictions = {}
    with open(os.path.join("dataset", "voc10", "val_id.txt")) as f:
        content = f.read()
    ids = content.split('\n')[:-1]

    for current_id in ids:
        gt_path = os.path.join(context_path, "context", "trainval", f"{current_id}.mat")
        gt_dict = sio.loadmat(gt_path)
        unique_labels = np.unique(gt_dict["LabelMap"])
        predictions[current_id] = [embs[x] for x in unique_labels]

    return predictions

def do_python_eval(predict_folder, gt_folder, name_list, num_cls=21, input_type='png', threshold=1.0, printlog=False, pred_dict_path=None, bs=16):
    TP = []
    P = []
    T = []
    # count=0
    txt = f'{predict_folder}/text.txt'
    predict_classes = read_pred_classes()

    # f=open(txt,"a+")
    for i in range(num_cls):
        TP.append(0)
        P.append(0)
        T.append(0)

    TP, P, T = compare(0,1,TP,P,T,input_type,threshold, name_list, predict_folder, num_cls, gt_folder, predict_classes, bs)
    IoU = []
    T_TP = []
    P_TP = []
    FP_ALL = []
    FN_ALL = [] 
    for i in range(num_cls):
        IoU.append(TP[i]/(T[i]+P[i]-TP[i]+1e-10))
        T_TP.append(T[i]/(TP[i]+1e-10))
        P_TP.append(P[i]/(TP[i]+1e-10))
        FP_ALL.append((P[i]-TP[i])/(T[i] + P[i] - TP[i] + 1e-10))
        FN_ALL.append((T[i]-TP[i])/(T[i] + P[i] - TP[i] + 1e-10))
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
    # mask = F.interpolate(mask.unsqueeze(0).unsqueeze(0), size=(img.size[1],img.size[0]), mode='bilinear', align_corners=False).squeeze().squeeze()
    img = np.float32(img) / 255.
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    cam = heatmap + img
    cam = cam / np.max(cam)
    cam = np.uint8(255 * cam)
    return cam

def save_segmentation(orig_image: Image, mask: torch.Tensor, file_id: str, args, subfolder):
    save_dir = os.path.join(args.save_samples_path, subfolder)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_path = os.path.join(save_dir, file_id + '.jpg')
    
    cam = show_cam_on_image(orig_image, mask)
    segmented_image = Image.fromarray(cam[:,:,::-1])
    segmented_image.save(save_path)

def save_sample_segmentations(args):
    for timestamp_subfolder in os.listdir(args.cam_npy_dir):
        predict_folder = os.path.join(args.cam_npy_dir, timestamp_subfolder)
        print('LEN FOLDER', predict_folder, len(os.listdir(predict_folder)))
        masks_paths = [os.path.join(predict_folder, x) for x in os.listdir(predict_folder)[:args.save_count]]
        masks_ids = [x.split(os.sep)[-1].replace('.mat', '') for x in masks_paths]
        for mask_path, mask_id in zip(masks_paths, masks_ids):
            orig_image_path = os.path.join(context_path, 'JPEGImages', mask_id + '.jpg')
            orig_image = Image.open(orig_image_path).convert("RGB")
            saved_dict = sio.loadmat(mask_path)
            for class_id in list(saved_dict.keys())[3:]:
                mask_tensor = torch.Tensor(saved_dict[class_id] / 255)
                save_segmentation(orig_image, mask_tensor, mask_id + f'_{embs[int(class_id)]}', args, timestamp_subfolder)

class Args:
    bs=4096
    val_split = 0.2
    results='./results.csv'
    comment='train1464'
    curve=True
    base_dir = 'output/voc10_wo_sy_norm/' if len(sys.argv) < 2 else sys.argv[1]
    image_dir = os.path.join(base_dir,'images')
    cam_npy_dir = os.path.join(base_dir, 'images')
    gt_dir=os.path.join(context_path, 'context', 'trainval')
    list='dataset/voc10/val_id.txt'
    pred_json = 'dataset/voc10/predict_syn_txt_0.8_img_0.97.json'
    # list='/root/autodl-tmp/wjl/ptp_diffusion/voc12/train_aug_id.txt'
    logfile=os.path.join(base_dir,'eval.txt')
    num_classes=len(embs)
    start=20
    end=41
    step=10
    t=None
    type='npy'
    sample_images=False
    save_samples_path = os.path.join('output', 'samples')
    save_count=10

def test():
    args = Args()
    print(args.base_dir)

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

        print(f"Len prediction labels {len(df['filename'].values)}, Len computed labels {len(name_list)}")

        val_split_size = int(args.val_split * len(name_list))
        val_name_list = name_list[:val_split_size]
        test_name_list = name_list[val_split_size:]

        if not args.curve:
            loglist = do_python_eval(current_results_path, args.gt_dir, name_list, args.num_classes, args.type, args.t, printlog=True, pred_dict_path=args.pred_json, bs=args.bs)
            writelog(args.logfile, loglist, args.comment)
            best_thr = None
        else:
            l = []
            max_mIoU = 0.0
            best_thr = 0.0
            print(f"Finding threshold on a data split of {len(val_name_list)}")
            for i in range(args.start, args.end, args.step):
                print(i)
                t = i/100.0
                loglist = do_python_eval(current_results_path, args.gt_dir, val_name_list, args.num_classes, args.type, t, pred_dict_path=args.pred_json, bs=args.bs)
                l.append(loglist['mIoU'])
                # print('%d/%d background score: %.3f\tmIoU: %.3f%%'%(i, args.end, t, loglist['mIoU']))
                if loglist['mIoU'] > max_mIoU:
                    max_mIoU = loglist['mIoU']
                    best_thr = t
                # else:
                #     break
            print('Best background score: %.3f\tmIoU: %.3f%%' % (best_thr, max_mIoU))
            # writelog(args.logfile, {'mIoU':l, 'Best mIoU': max_mIoU, 'Best threshold': best_thr}, args.comment)
        
        print(f"Using best threshold of {best_thr} for test split of size {len(test_name_list)}")
        loglist_test = do_python_eval(current_results_path, args.gt_dir, test_name_list, args.num_classes, args.type, best_thr, pred_dict_path=args.pred_json, bs=args.bs)
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