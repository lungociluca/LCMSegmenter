import os
import json
from torchvision import transforms

from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.data import create_transform

import numpy as np
import torch
from torch.utils.data import Dataset
import PIL.Image
import scipy.io as sio
import torch.nn.functional as F
from local_config import context_path


def load_img_name_list_100(dataset_path):

    img_gt_name_list = open(dataset_path).readlines()
    img_name_list=[]
    img_id=[]
    for img_gt_name in img_gt_name_list:
        tmp=img_gt_name.strip().split(' ')
        img_name_list.append(tmp[0])
        img_id.append(tmp[1])
    return img_name_list,img_id
def load_img_name_list(dataset_path):

    img_gt_name_list = open(dataset_path).readlines()
    img_name_list = [img_gt_name.strip() for img_gt_name in img_gt_name_list]

    return img_name_list

def load_image_label_list_from_npy(img_name_list, label_file_path=None):
    if label_file_path is None:
        return None
    cls_labels_dict = np.load(label_file_path, allow_pickle=True).item()
    # print(cls_labels_dict)
    label_list = []
    for id in img_name_list:
        if id not in cls_labels_dict.keys():
            img_name = id + '.jpg'
        else:
            img_name = id
        label_list.append(cls_labels_dict[img_name])
    return label_list

class COCOClsDataset(Dataset):
    def __init__(self, img_name_list_path, coco_root, label_file_path, train=True, transform=None, gen_attn=False):
        img_name_list_path = os.path.join(img_name_list_path, f'{"train" if train or gen_attn else "val"}_id.txt')
        self.img_name_list = load_img_name_list(img_name_list_path)
        # self.img_name_list=sorted(self.img_name_list)
        self.label_list = load_image_label_list_from_npy(self.img_name_list, label_file_path)
        self.coco_root = coco_root
        self.transform = transform
        self.train = train
        self.gen_attn = gen_attn

    def __getitem__(self, idx):
        name = self.img_name_list[idx]

        if self.train or self.gen_attn :
            img = PIL.Image.open(os.path.join(self.coco_root, 'train2014', name + '.jpg')).convert("RGB")
            img_path=os.path.join(self.coco_root, 'train2014', name + '.jpg')
            
        else:
            img = PIL.Image.open(os.path.join(self.coco_root, 'val2014', name + '.jpg')).convert("RGB")
            img_path=os.path.join(self.coco_root, 'val2014', name + '.jpg')
        
        if not self.label_list:
            label=-1
        else:
            label = torch.from_numpy(self.label_list[idx])
        if self.transform:
            img = self.transform(img)

        return img, label,img_path

    def __len__(self):
        return len(self.img_name_list)



####分割重写数据集
class VOC12Dataset_seg(Dataset):
    def __init__(self, img_name_list_path, label_file_path,voc12_root, train=True, transform=None, gen_attn=False,is_big_data=True):
        if not train :
            img_name_list_path = os.path.join(img_name_list_path, f'val_id.txt')
        elif train and is_big_data:
            img_name_list_path = os.path.join(img_name_list_path, f'train_aug_id.txt')
        else:
            img_name_list_path = os.path.join(img_name_list_path, f'train_id.txt')
        img_name_list = load_img_name_list(img_name_list_path)
        # else:
        #     img_name_list_path = os.path.join(img_name_list_path, f'{"train_100" if train or gen_attn else "val"}_id.txt')
        #     self.img_name_list= load_img_name_list(img_name_list_path)
        self.label_list = load_image_label_list_from_npy(img_name_list,label_file_path)
        self.voc12_root = voc12_root
        self.transform = transform

        json_path = 'dataset/voc12/predict_syn_txt_0.8_img_0.97.json'
        pred_classes= self.read_json(json_path,choice=['image_similarity','text_similarity'])

        self.data_pairs = []
        for img in img_name_list:
            class_key = f'/root/autodl-tmp/dataset/VOCdevkit/VOC2012/JPEGImages/{img}.jpg'
            labels_list = pred_classes[class_key]
            labels_count = len(labels_list)
            for idx, label in enumerate(labels_list):
                self.data_pairs.append(
                    (img, label, labels_count - idx - 1)
                )

    @staticmethod
    def read_json(json_path, choice):
        with open(json_path,"r") as tmp:
            data=json.load(tmp)
        re=data[choice[0]]
        if len(choice)==1:
            return re
        for key in re.keys():
            for i in range(1,len(choice)):
                re[key].extend(data[choice[i]][key])
            re[key]=list(set(re[key]))
        return re 

    def __getitem__(self, idx):
        name, _label, label_idx = self.data_pairs[idx]
        img = PIL.Image.open(os.path.join(self.voc12_root, 'JPEGImages', name + '.jpg')).convert("RGB")
        path = os.path.join(self.voc12_root, 'JPEGImages', name + '.jpg')
        if not self.label_list:
            label=-1
        else:
            label = torch.from_numpy(self.label_list[idx])
        if self.transform:
            img = self.transform(img)
        
        img_temp = img.unsqueeze(0).permute(0, 2, 3, 1)
        orig_images = torch.zeros_like(img_temp)
        orig_images[:, :, :, 0] = img_temp[:, :, :, 0] #* 0.229 + 0.485 R
        orig_images[:, :, :, 1] = img_temp[:, :, :, 1] #* 0.224 + 0.456 G
        orig_images[:, :, :, 2] = img_temp[:, :, :, 2] #* 0.225 + 0.406 B
        orig_images[orig_images>1]=1
        orig_images[orig_images<0]=0

        rgb_512 = F.interpolate(orig_images.permute(0,3,1,2), (512, 512), mode='bilinear', align_corners=False)

        orig_image_shapes = torch.LongTensor([orig_images.shape[1], orig_images.shape[2]])
        return rgb_512[0], label, None, path, _label, label_idx, name, orig_image_shapes

    def __len__(self):
        return len(self.data_pairs)

class VOC10Dataset_seg(Dataset):
    def __init__(self, img_name_list_path, label_file_path,voc12_root, train=True, transform=None, gen_attn=False,is_big_data=True):
        if not train :
            img_name_list_path = os.path.join(img_name_list_path, f'val_id.txt')
        elif train and is_big_data:
            img_name_list_path = os.path.join(img_name_list_path, f'train_aug_id.txt')
        else:
            img_name_list_path = os.path.join(img_name_list_path, f'train_id.txt')
        img_name_list = load_img_name_list(img_name_list_path)
        # else:
        #     img_name_list_path = os.path.join(img_name_list_path, f'{"train_100" if train or gen_attn else "val"}_id.txt')
        #     self.img_name_list= load_img_name_list(img_name_list_path)
        self.label_list = load_image_label_list_from_npy(img_name_list,label_file_path)
        self.voc12_root = voc12_root
        self.transform = transform

        json_path = 'dataset/voc10/predict_syn_txt_0.8_img_0.97.json'
        pred_classes= self.read_json()

        self.data_pairs = []
        for img in img_name_list:
            class_key = f'{img}'
            labels_list = pred_classes[class_key]
            labels_count = len(labels_list)
            for idx, label in enumerate(labels_list):
                self.data_pairs.append(
                    (img, label, labels_count - idx - 1)
                )
    @staticmethod
    def load_embedings():
        embs = [None] # class ids start from 1
        labels_path = os.path.join(context_path, 'context', 'labels.txt')
        with open(labels_path) as f:
            labels_file_rows = f.read().split('\n')
            if labels_file_rows[-1] == '': # fix when empty last line in file
                labels_file_rows = labels_file_rows[:-1]
        for id_and_label in labels_file_rows:
            _, label = id_and_label.split(': ')
            # if label == 'tvmonitor':
            #     label = 'monitor'
            # elif label == 'aeroplane':
            #     label = 'airplane'
            embs.append(label)
        return embs
    
    @staticmethod
    def read_json():
        predictions = {}
        embs = VOC10Dataset_seg.load_embedings()
        with open(os.path.join("dataset", "voc10", "val_id.txt")) as f:
            content = f.read()
        ids = content.split('\n')[:-1]

        for current_id in ids:
            gt_path = os.path.join(context_path, "context", "trainval", f"{current_id}.mat")
            gt_dict = sio.loadmat(gt_path)
            unique_labels = np.unique(gt_dict["LabelMap"])
            predictions[current_id] = [embs[x] for x in unique_labels]

        return predictions

    def __getitem__(self, idx):
        name, _label, label_idx = self.data_pairs[idx]
        img = PIL.Image.open(os.path.join(self.voc12_root, 'JPEGImages', name + '.jpg')).convert("RGB")
        # TODO: remove
        mask = torch.zeros((1,1))
        path = os.path.join(self.voc12_root, 'JPEGImages', name + '.jpg')
        if not self.label_list:
            label=-1
        else:
            label = torch.from_numpy(self.label_list[idx])
        if self.transform:
            img = self.transform(img)
            # tmp=[]
            # tmp.append(transforms.Resize([224,224]))
            # tmp_=transforms.Compose(tmp)
            mask=torch.tensor(np.array(mask))
        
        img_temp = img.unsqueeze(0).permute(0, 2, 3, 1)
        orig_images = torch.zeros_like(img_temp)
        orig_images[:, :, :, 0] = img_temp[:, :, :, 0] #* 0.229 + 0.485 R
        orig_images[:, :, :, 1] = img_temp[:, :, :, 1] #* 0.224 + 0.456 G
        orig_images[:, :, :, 2] = img_temp[:, :, :, 2] #* 0.225 + 0.406 B
        orig_images[orig_images>1]=1
        orig_images[orig_images<0]=0

        rgb_512 = F.interpolate(orig_images.permute(0,3,1,2), (512, 512), mode='bilinear', align_corners=False)

        orig_image_shapes = torch.LongTensor([orig_images.shape[1], orig_images.shape[2]])
        return rgb_512[0], label, mask, path, _label, label_idx, name, orig_image_shapes

    def __len__(self):
        return len(self.data_pairs)

def build_dataset(is_train, args, gen_attn=False, is_big_data=True):
    transform = build_transform(False, args)
    # transform = None
    dataset = None
    nb_classes = None

    if args.data_set == 'VOC12':
        dataset = VOC12Dataset(img_name_list_path=args.img_list, voc12_root=args.data_path,
                               train=is_train, gen_attn=gen_attn, transform=transform)
        nb_classes = 20
    elif args.data_set == 'VOC12seg':
        dataset = VOC12Dataset_seg(img_name_list_path=args.img_list, label_file_path=args.label_file_path,voc12_root=args.data_path,
                               train=is_train, gen_attn=gen_attn, transform=transform,is_big_data=is_big_data)
        nb_classes = 20
    elif args.data_set == 'COCO':
        dataset = COCOClsDataset(img_name_list_path=args.img_list, coco_root=args.data_path, label_file_path=args.label_file_path,
                               train=is_train, gen_attn=gen_attn, transform=transform)
        nb_classes = 80
    elif args.data_set == 'COCOMS':
        dataset = COCOClsDatasetMS(img_name_list_path=args.img_list, coco_root=args.data_path, scales=None, label_file_path=args.label_file_path,
                               train=is_train, gen_attn=gen_attn, transform=transform)
        nb_classes = 80
    elif args.data_set == 'VOC10seg':
        dataset = VOC10Dataset_seg(img_name_list_path=args.img_list, label_file_path=args.label_file_path, voc12_root=args.data_path,
                               train=is_train, gen_attn=gen_attn, transform=transform,is_big_data=is_big_data)

    return dataset, nb_classes


def build_transform(is_train, args):
    resize_im = args.input_size > 32
    if is_train:
        # this should always dispatch to transforms_imagenet_train
        transform = create_transform(
            input_size=args.input_size,
            is_training=True,
            # color_jitter=args.color_jitter,
            # auto_augment=args.aa,
            interpolation=args.train_interpolation,
            # re_prob=args.reprob,
            # re_mode=args.remode,
            # re_count=args.recount,
        )
        if not resize_im:
            # replace RandomResizedCropAndInterpolation with
            # RandomCrop
            transform.transforms[0] = transforms.RandomCrop(
                args.input_size, padding=4)
        return transform

    t = []
    # if resize_im and not args.gen_attention_maps:
    #     size = int((256 / 224) * args.input_size)
    #     t.append(
    #         transforms.Resize(size, interpolation=3),  # to maintain same ratio w.r.t. 224 images
    #     )
    #     t.append(transforms.CenterCrop(args.input_size))
    if False:
        t.append(transforms.Resize([args.input_size,args.input_size], interpolation=3))
    else:
        # t.append(transforms.Resize([args.input_size,args.input_size], interpolation=3))
        t.append(transforms.ToTensor())
        # t.append(transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD))
    return transforms.Compose(t)
