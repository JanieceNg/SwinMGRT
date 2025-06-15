import random

import cv2
import torch
from torchvision import datasets, models, transforms
from torch.utils.data import Dataset, DataLoader
import json
from PIL import Image
import os
from numpy.random import choice
from tqdm import tqdm
from PIL import PngImagePlugin
import albumentations as A
from albumentations.pytorch import ToTensorV2

MaximumDecompressedSize = 1024
MegaByte = 2**20
PngImagePlugin.MAX_TEXT_CHUNK = MaximumDecompressedSize * MegaByte


def get_transforms0(config,tag):
    img_size=config.DATA.IMG_SIZE
    if tag=='train':
        tf = transforms.Compose([transforms.RandomResizedCrop(img_size),
                                         transforms.RandomHorizontalFlip(),
                                         transforms.ToTensor(),
                                         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        return tf
    elif tag == "val":
        tf = transforms.Compose([transforms.Resize(int(img_size * 1.143)),
                                       transforms.CenterCrop(img_size),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        return tf

def train_augmentation(config):
    tfs = []

    # 简单几何变换
    if config.DATA.AUGMENT.LEVEL1 > 0:
        tf1 = A.OneOf([
            A.Transpose(p=0.5),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(0.5),
            A.RandomRotate90(0.5),
            A.Rotate(limit=45,p=0.5),
            A.RandomScale(scale_limit=0.1,p=0.5),
            A.ShiftScaleRotate(shift_limit=0.1,scale_limit=0.1,rotate_limit=20,p=0.5)
        ],
            p=config.DATA.AUGMENT.LEVEL1,
        )
        tfs.append(tf1)

    # 复杂形态变换
    if config.DATA.AUGMENT.LEVEL2 > 0:
        tf2 = A.OneOf([
            A.OpticalDistortion(distort_limit=0.05, shift_limit=0.05,p=0.5),
            A.GridDistortion(num_steps=5, distort_limit=0.3,p=0.5),
            A.ElasticTransform(alpha=1,sigma=50,p=0.5),
            A.RandomGridShuffle(grid=(3, 3), p=0.5),
            A.PadIfNeeded(min_height=1024, min_width=1024,p=0.5)
        ],
            p=config.DATA.AUGMENT.LEVEL2,
        )
        tfs.append(tf2)

    # 颜色变换
    if config.DATA.AUGMENT.LEVEL3 > 0:
        tf3 = A.OneOf([
            A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.5),
            A.RGBShift(r_shift_limit=10, g_shift_limit=10, b_shift_limit=10,  p=0.5),
            A.RandomBrightnessContrast(brightness_limit = 0.2,contrast_limit = 0.2,p = 0.5),
            A.RandomGamma(gamma_limit=(80, 120), eps=1e-07, p=0.5),
        ],
            p=config.DATA.AUGMENT.LEVEL3,
        )
        tfs.append(tf3)

    # 噪声模糊
    if config.DATA.AUGMENT.LEVEL4 > 0:
        tf4 = A.OneOf([
            A.Blur(blur_limit=(3,7),p=0.5),
            A.MedianBlur(blur_limit=7,p=0.5),
            A.GaussianBlur(blur_limit=(3,7), p=0.5),
            A.GaussNoise(var_limit=(10.0, 30.0),p=0.5),
            A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=0.5),
        ],
            p=config.DATA.AUGMENT.LEVEL4,
        )
        tfs.append(tf4)

    # 随机擦除
    if config.DATA.AUGMENT.LEVEL5 > 0:
        tf5 = A.CoarseDropout(min_holes=3, max_holes=8,min_width=5,min_height=5,max_width=20,max_height=20,fill_value=0,
            p=config.DATA.AUGMENT.LEVEL5)
        tfs.append(tf5)

    return tfs

def get_transforms(config, tag):
    img_size = config.DATA.IMG_SIZE
    if tag == 'train':
        tfs = train_augmentation(config)
        tfs.extend([A.RandomResizedCrop(height=img_size, width=img_size, scale=(0.8, 1.0), ratio=(0.75, 1.3333333333333333),
                                interpolation=1, p=1.0),
                    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], p=1.0),
        ToTensorV2(p=1.0)])
        tf = A.Compose(tfs,p=1.0)
        return tf

    elif tag in {'test', "val"}:
        tf = A.Compose([
            A.SmallestMaxSize(max_size=int(img_size * 1.143),p=1.0),
            A.CenterCrop(height=img_size, width=img_size,p=1.0),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], p=1.0),
            ToTensorV2(p=1.0)
        ], p=1.0)
        return tf



class ClassifyDataset(Dataset):
    def __init__(self, config,tag):
        """
        Args:
            dataset_dir (string): Path to the dataset directory.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.dataset_dir = os.path.join(config.DATA.DATASET_ROOT, config.DATA.DATASET)
        self.img_size = config.DATA.IMG_SIZE
        self.tag = tag
        if tag == 'train':
            self.single_class_num = config.TRAIN.SINGLE_CLASS_NUM
        elif tag in {'val','test'}:
            self.single_class_num = config.VAL.SINGLE_CLASS_NUM
        self.transform = get_transforms(config,self.tag)
        self.class_indices = self._get_class_indices()
        self.class_paths = self._get_class_paths()
        self.paths,self.labels = self._load_folder()

    def _get_class_indices(self,dir='class_indices.json'):
        selected_class_url = os.path.join(self.dataset_dir, dir)
        if os.path.exists(selected_class_url):
            with open(selected_class_url, 'r') as f:
                class_indices = json.load(f)
        else:
            print(os.path.join(self.dataset_dir, self.tag))
            selected_class = os.listdir(os.path.join(self.dataset_dir, self.tag))
            selected_class.sort()
            class_indices = dict((k, v) for v, k in enumerate(selected_class))
            with open(selected_class_url, 'w') as f:
                json.dump(class_indices,f,indent=4)
        return class_indices

    def _get_class_paths(self):
        class_paths = dict()
        for c in self.class_indices.keys():
            class_img_path = os.path.join(self.dataset_dir, self.tag, c)
            img_paths = [os.path.join(class_img_path,img) for img in os.listdir(class_img_path)]
            class_paths[c]=img_paths
        return class_paths

    def _load_folder(self,load=False):
        cache_url = os.path.join(self.dataset_dir,self.tag+'_Eff_Swin.cache')
        if os.path.exists(cache_url) and load:
            cache = torch.load(cache_url)
            paths = cache['paths']
            labels = cache['labels']
            print('load from %s' % cache_url)
            return paths, labels
        else:
            paths = []
            labels = []
            for c in self.class_indices.keys():
                img_paths = self.class_paths[c]
                if self.tag == 'train':
                    sample = self._get_sample(img_paths,self.single_class_num)
                else:
                    # 验证集不随机采样
                    sample = img_paths[:self.single_class_num] if 0<self.single_class_num<len(img_paths) else img_paths

                paths.extend(sample)
                labels.extend([self.class_indices[c]]*len(sample))

            cache = {'paths':paths,'labels':labels}
            torch.save(cache,cache_url)
            print('create %s'%cache_url)
            return paths,labels

    def _get_sample(self,ds,sample_num):
        n = len(ds)
        if sample_num<=0 or sample_num==n:
            return ds
        elif sample_num>n:
            return choice(ds, size=sample_num,replace=True)
        else:
            return choice(ds, size=sample_num, replace=False)

    def update_paths(self):
        assert self.tag=='train',f'{self.tag} set is not necessary to update'
        # k=random.gauss(1,0.1)
        # k=0.6 if k<0.6 else k
        # k=1.4 if k>1.4 else k
        # nums = int(self.single_class_num*k)
        nums = self.single_class_num
        paths = []
        labels = []
        for c in self.class_indices.keys():
            img_paths = self.class_paths[c]
            sample = self._get_sample(img_paths, nums)
            paths.extend(sample)
            labels.extend([self.class_indices[c]] * len(sample))
        self.paths = paths
        self.labels = labels

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        # print(self.paths[idx])
        # img = Image.open(self.paths[idx])
        # # RGB为彩色图片，L为灰度图片
        # if img.mode != 'RGB':
        #     img = img.convert('RGB')
        img = cv2.imread(self.paths[idx])
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        label = self.labels[idx]
        if self.transform is not None:
            img = self.transform(image=img)['image']
        return img, label

class ScaleClassifyDataset(Dataset):
    def __init__(self, config,tag):
        """
        Args:
            dataset_dir (string): Path to the dataset directory.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.dataset_dir = os.path.join(config.DATA.DATASET_ROOT, config.DATA.DATASET)
        self.img_size = config.DATA.IMG_SIZE
        self.tag = tag
        if tag == 'train':
            self.single_class_num = config.TRAIN.SINGLE_CLASS_NUM
        elif tag in {'val','test'}:
            self.single_class_num = config.VAL.SINGLE_CLASS_NUM

        self.transform = get_transforms(config,self.tag)
        self.class_indices = self._get_class_indices()
        self.scale_class = self._get_scale_class(config.DATA.SCALE_CLASS)
        self.scale_indices = [
            self._get_scale_class_indices('family'),
            self._get_scale_class_indices('genus')
        ]
        assert config.DATA.LABEL_CLASS,'config.DATA.LABEL_CLASS not exist'
        self.label_class = config.DATA.LABEL_CLASS #决定了生成的标签
        self.class_paths = self._get_class_paths()
        self.paths,self.labels = self._load_folder()

    def _get_scale_class(self,dir):
        class_url = os.path.join(self.dataset_dir, dir)
        assert os.path.exists(class_url), 'scale class not exsits in %s'%class_url
        with open(class_url, 'r') as f:
            scale_class = json.load(f)
        return scale_class
    def _get_scale_class_indices(self,key,dir=None):
        if dir is None:
            dir = '%s_indices.json'%key
        selected_class_url = os.path.join(self.dataset_dir, dir)
        if os.path.exists(selected_class_url):
            with open(selected_class_url, 'r') as f:
                class_indices = json.load(f)
        else:
            selected_class = list(set([c[key] for c in self.scale_class.values()]))
            selected_class.sort()
            class_indices = dict((k, v) for v, k in enumerate(selected_class))
            with open(selected_class_url, 'w') as f:
                json.dump(class_indices,f,indent=4)
        return class_indices


    def _get_class_indices(self,dir='class_indices.json'):
        selected_class_url = os.path.join(self.dataset_dir, dir)
        if os.path.exists(selected_class_url):
            with open(selected_class_url, 'r') as f:
                class_indices = json.load(f)
        else:
            print(os.path.join(self.dataset_dir, self.tag))
            selected_class = os.listdir(os.path.join(self.dataset_dir, self.tag))
            selected_class.sort()
            class_indices = dict((k, v) for v, k in enumerate(selected_class))
            with open(selected_class_url, 'w') as f:
                json.dump(class_indices,f,indent=4)
        return class_indices

    def _get_class_paths(self):
        class_paths = dict()
        for c in self.class_indices.keys():
            class_img_path = os.path.join(self.dataset_dir, self.tag, c)
            img_paths = [os.path.join(class_img_path,img) for img in os.listdir(class_img_path)]
            class_paths[c]=img_paths
        return class_paths

    def _load_folder(self,load=False):
        cache_url = os.path.join(self.dataset_dir,self.tag+'_Swin.cache')
        if os.path.exists(cache_url) and load:
            cache = torch.load(cache_url)
            paths = cache['paths']
            labels = cache['labels']
            print('load from %s' % cache_url)
            return paths, labels
        else:
            paths = []
            labels = []
            for c in self.class_indices.keys():
                img_paths = self.class_paths[c]
                if self.tag == 'train':
                    sample = self._get_sample(img_paths,self.single_class_num)
                else:
                    # 验证集不随机采样
                    sample = img_paths[:self.single_class_num] if 0<self.single_class_num<len(img_paths) else img_paths

                paths.extend(sample)
                labels.extend([self._get_scale_label(c)]*len(sample))

            cache = {'paths':paths,'labels':labels}
            torch.save(cache,cache_url)
            print('create %s'%cache_url)
            return paths,labels

    def _get_scale_label(self,class_name):
        item = self.scale_class[class_name]
        genus,family=item['genus'],item['family']
        gid = self.scale_indices[1][genus]
        fid = self.scale_indices[0][family]
        sid = self.class_indices[class_name]
        label_id = (fid,gid,sid)
        label = torch.tensor([label_id[i] for i in self.label_class])
        return label

    def get_species_map(self):
        spmap = {}
        for sp,sid in self.class_indices.items():
            item = self.scale_class[sp]
            genus, family = item['genus'], item['family']
            gid = self.scale_indices[1][genus]
            fid = self.scale_indices[0][family]
            spmap[sid]=(gid,fid)
        return spmap

    def _get_sample(self,ds,sample_num):
        n = len(ds)
        if sample_num<=0 or sample_num==n:
            return ds
        elif sample_num>n:
            return choice(ds, size=sample_num,replace=True)
        else:
            return choice(ds, size=sample_num, replace=False)

    def update_paths(self):
        assert self.tag=='train',f'{self.tag} set is not necessary to update'
        # k=random.gauss(1,0.1)
        # k=0.6 if k<0.6 else k
        # k=1.4 if k>1.4 else k
        # nums = int(self.single_class_num*k)
        nums = self.single_class_num
        paths = []
        labels = []
        for c in self.class_indices.keys():
            img_paths = self.class_paths[c]
            sample = self._get_sample(img_paths, nums)
            paths.extend(sample)
            c1,c2 = self.scale_class[c]['family'],self.scale_class[c]['genus']
            labels.extend([self._get_scale_label(c)] * len(sample))
        self.paths = paths
        self.labels = labels

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        # print(self.paths[idx])
        # print(self.scale_class['1355868'])

        # img = Image.open(self.paths[idx])
        # # RGB为彩色图片，L为灰度图片
        # if img.mode != 'RGB':
        #     img = img.convert('RGB')
        # label = self.labels[idx]
        # img = self.transform(img)

        img = cv2.imread(self.paths[idx])
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        label = self.labels[idx]
        if self.transform is not None:
            img = self.transform(image=img)['image']
        return img,label,self.paths[idx]

def test_collate(batch):
    b = len(batch)
    # print(batch[0])
    # print(len(batch[0]))
    # print(batch[0][0].shape)
    images = torch.stack([s[0] for s in batch])

    # 将每个标签分别收集到独立的tensor中
    # 假设每个标签是一个包含三个整数的元组
    labels = torch.stack([s[1] for s in batch])
    urls = list([s[2] for s in batch])
    # 返回图像和标签元组
    return images, labels, urls


def collate(batch):
    b = len(batch)
    # print(batch[0])
    # print(len(batch[0]))
    # print(batch[0][0].shape)
    images = torch.stack([s[0] for s in batch])

    # 将每个标签分别收集到独立的tensor中
    # 假设每个标签是一个包含三个整数的元组
    labels = torch.stack([s[1] for s in batch])

    # 返回图像和标签元组
    return images, labels

def get_train_dataloader(config,dataset=None):
    if config.DATA.SCALE:
        dataset = ScaleClassifyDataset(config,'train') if not dataset else dataset
        collate_fn = collate
    else:
        dataset = ClassifyDataset(config,'train') if not dataset else dataset
        collate_fn = None
    train_dataset = dataset
    train_dataloader = DataLoader(train_dataset,
                                  shuffle=True,
                                  drop_last=True,
                                  batch_size=config.TRAIN.BATCH_SIZE,
                                  num_workers=config.MISC.NUM_WORKERS,
                                  collate_fn=collate_fn,
                                  pin_memory=True)
    return train_dataloader

def get_trainset(config):
    if config.DATA.SCALE:
        dataset = ScaleClassifyDataset(config,'train')
    else:
        dataset = ClassifyDataset(config,'train')
    return dataset

def get_val_dataloader(config):
    if config.DATA.SCALE:
        val_dataset = ScaleClassifyDataset(config, 'val')
        collate_fn = collate
    else:
        val_dataset = ClassifyDataset(config, 'val')
        collate_fn = None
    val_dataloader = DataLoader(val_dataset,
                                shuffle=False,
                                drop_last=False,
                                batch_size=config.VAL.BATCH_SIZE,
                                num_workers=config.MISC.NUM_WORKERS,
                                collate_fn=collate_fn,
                                pin_memory=True)
    return val_dataloader

def get_test_dataloader(config,tag='test'):
    if config.DATA.SCALE:
        val_dataset = ScaleClassifyDataset(config, tag)
        collate_fn = test_collate
    else:
        val_dataset = ClassifyDataset(config, tag)
        collate_fn = None
    val_dataloader = DataLoader(val_dataset,
                                shuffle=False,
                                drop_last=False,
                                batch_size=config.VAL.BATCH_SIZE,
                                num_workers=config.MISC.NUM_WORKERS,
                                collate_fn=collate_fn,
                                pin_memory=True)
    return val_dataloader
