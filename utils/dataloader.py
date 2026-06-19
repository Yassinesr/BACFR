import os

import numpy as np
import torch.utils.data as data
import torchvision.transforms as transforms

from PIL import Image

from utils.custom_transforms import *

class PolypDataset(data.Dataset):
    def __init__(self, img_root,mask_root, transform_list,type = 'masks'):

        print(img_root)
        print(mask_root)

        self.images = [os.path.join(img_root, f) for f in os.listdir(img_root) if f.endswith('.jpg') or f.endswith('.png')]
        self.images = sorted(self.images)
        
        self.gts = [os.path.join(mask_root, f) for f in os.listdir(mask_root) if f.endswith('.png')]
        self.gts = sorted(self.gts)
        
        self.filter_files()
        
        self.size = len(self.images)
        self.transform = self.get_transform(transform_list)

    @staticmethod
    def get_transform(transform_list):
        tfs = []
        for key, value in zip(transform_list.keys(), transform_list.values()):
            if value is not None:
                tf = eval(key)(**value)
            else:
                tf = eval(key)()
            tfs.append(tf)
        return transforms.Compose(tfs)

    def __getitem__(self, index):
        image = Image.open(self.images[index]).convert('RGB')
        gt = Image.open(self.gts[index]).convert('L')
        shape = gt.size[::-1]
        name = self.images[index].split('/')[-1]
        if name.endswith('.jpg'):
            name = name.split('.jpg')[0] + '.png'
            
        sample = {'image': image, 'gt': gt, 'name': name, 'shape': shape}

        sample = self.transform(sample)
        return sample

    def filter_files(self):
        assert len(self.images) == len(self.gts)
        images, gts = [], []
        for img_path, gt_path in zip(self.images, self.gts):
            img, gt = Image.open(img_path), Image.open(gt_path)
            if img.size == gt.size:
                images.append(img_path)
                gts.append(gt_path)
        self.images, self.gts = images, gts

    def __len__(self):
        return self.size


class PatchDataset(data.Dataset):
    def __init__(self, root, transform_list, type):
        # `root` is either a single path (original behavior) or a list of
        # paths. When it's a list, file lists are collected from each root
        # in turn and concatenated (per-root pairing, so identical
        # filenames in different roots don't cross-contaminate). Used by
        # the mixed-traindataset recipe; backward-compatible for single-
        # dataset configs.
        if isinstance(root, (list, tuple)):
            roots = list(root)
        else:
            roots = [root]

        all_images, all_masks, all_gts = [], [], []
        for r in roots:
            image_root = os.path.join(r, 'img_dir', type)
            mask_root = os.path.join(r, 'mask_dir', type)
            gt_root = os.path.join(r, 'ann_dir', type)

            imgs = sorted([os.path.join(image_root, f) for f in os.listdir(image_root)
                           if f.endswith('.jpg') or f.endswith('.png')])
            msks = sorted([os.path.join(mask_root, f) for f in os.listdir(mask_root)
                           if f.endswith('.jpg') or f.endswith('.png')])
            gts = sorted([os.path.join(gt_root, f) for f in os.listdir(gt_root)
                          if f.endswith('.png')])

            assert len(imgs) == len(msks) == len(gts), \
                f'len mismatch in {r}: imgs={len(imgs)} masks={len(msks)} gts={len(gts)}'

            # filter at the per-root level so concat doesn't introduce
            # cross-root filename collisions in the basename comparison
            for ip, mp, gp in zip(imgs, msks, gts):
                if os.path.basename(ip) == os.path.basename(mp) == os.path.basename(gp):
                    all_images.append(ip)
                    all_masks.append(mp)
                    all_gts.append(gp)
                else:
                    print(f'mismatch in {r}: {ip}')

        self.images = all_images
        self.masks = all_masks
        self.gts = all_gts
        self.size = len(self.images)
        self.transform = self.get_transform(transform_list)

    @staticmethod
    def get_transform(transform_list):
        tfs = []
        for key, value in zip(transform_list.keys(), transform_list.values()):
            if value is not None:
                tf = eval(key)(**value)
            else:
                tf = eval(key)()
            tfs.append(tf)
        return transforms.Compose(tfs)

    def __getitem__(self, index):


        image = Image.open(self.images[index]).convert('RGB')
        mask = Image.open(self.masks[index]).convert('L')
        gt = Image.open(self.gts[index]).convert('L')
        shape = gt.size[::-1]
        name = self.images[index].split('/')[-1]
        if name.endswith('.jpg'):
            name = name.split('.jpg')[0] + '.png'

        sample = {'image': image,'mask':mask, 'gt': gt, 'name': name, 'shape': shape}

        sample = self.transform(sample)

        return sample

    def __len__(self):
        return self.size
