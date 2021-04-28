import random
import os
import json

import PIL
import torch
from PIL import Image
import numpy as np

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

import torchvision.transforms.functional as TF


def pil_loader(path):
    img = Image.open(path)

    return img


def load_frames(root_dir, filenames):
    video = []
    for fn in filenames:
        image_path = os.path.join(root_dir, fn)
        if os.path.exists(image_path):
            video.append(pil_loader(image_path))
        else:
            raise ValueError('File {} not exists.'.format(image_path))

    return video


class NTU_dataset(Dataset):
    """NTU dataset."""

    def __init__(self,
                 data_root,
                 indexfile_path,
                 window_size=5,
                 transform=None,
                 crop_size=224,
                 apply_crop=True,
                 apply_rotation=False,
                 apply_coloraug=False,
                 apply_horizontal_flip=True,
                 is_testing=False
                 ):
        self.data_root = data_root
        self.indexfile = indexfile_path

        self.window_size = window_size
        self.padding = window_size // 2
        self.is_testing = is_testing

        self.crop_size = crop_size
        self.apply_crop = apply_crop
        self.apply_rotation = apply_rotation
        self.apply_coloraug = apply_coloraug
        self.apply_hflip = apply_horizontal_flip

        self.transform = transform

        if self.is_testing:
            assert not self.apply_crop and \
                   not self.apply_hflip and \
                   not self.apply_coloraug and \
                   not self.apply_rotation, "No transform should be applied during validation / testing."

        self.data = self.create_dataset()

    def create_dataset(self):
        data = {'input': [], 'gt': []}

        instances = json.load(open(self.indexfile))

        for inst in instances:

            if self.is_testing:
                inst = inst[:self.padding] + inst + inst[-self.padding:]

            total_frames = len(inst)

            start = 0
            end = total_frames - self.window_size

            ip_entries, gt_entries = [], []
            for l in range(start, end + 1):
                ip_path = [f['rain'] for f in inst[l: l + self.window_size]]
                gt_path = [f['gt'] for f in inst[l: l + self.window_size]]

                ip_entries.append(ip_path)
                gt_entries.append(gt_path)

            data['input'].extend(ip_entries)
            data['gt'].extend(gt_entries)

        return data

    def __getitem__(self, idx):
        ip_data = self.data['input'][idx]
        gt_data = self.data['gt'][idx]

        ip_frames = load_frames(self.data_root, ip_data)
        gt_frames = load_frames(self.data_root, gt_data)

        ip_tensors, gt_tensors = [], []

        # the whole clip should apply same transform.
        transform_params = self.get_transform_params(ip_frames[0].size[0], ip_frames[0].size[1])
        for ip_frame, gt_frame in zip(ip_frames, gt_frames):
            ip_frame, gt_frame = self.apply_transform(ip_frame, gt_frame, transform_params)

            ip_tensors.append(ip_frame)
            gt_tensors.append(gt_frame)

        ip_tensors = torch.stack(ip_tensors, 0).permute(1, 0, 2, 3)
        gt_tensors = torch.stack(gt_tensors, 0).permute(1, 0, 2, 3)

        # tensors shape: Channel (C) x temporal window size (T) x H x W
        return ip_tensors, gt_tensors

    def __len__(self):
        return len(self.data['input'])

    def get_transform_params(self, w, h):
        x0 = random.randint(0, w - self.crop_size)
        y0 = random.randint(0, h - self.crop_size)

        hflip_rnd = random.uniform(0, 1)
        vflip_rnd = random.uniform(0, 1)
        degree = random.choice([0, 90, 180, 270])

        return {'x0': x0,
                'y0': y0,
                'hflip_rnd': hflip_rnd,
                'vflip_rnd': vflip_rnd,
                'degree': degree
                }

    def apply_transform(self, ip_frame, gt_frame, params):
        x0, y0, hflip_rnd, vflip_rnd, deg = params['x0'], params['y0'], params['hflip_rnd'], params['vflip_rnd'], params['degree']

        # random cropping
        if self.apply_crop:
            x1 = x0 + self.crop_size
            y1 = y0 + self.crop_size

            ip_frame = ip_frame.crop((x0, y0, x1, y1))
            gt_frame = gt_frame.crop((x0, y0, x1, y1))

        # horizontal flip
        if self.apply_hflip:
            if hflip_rnd < 0.5:
                ip_frame = ip_frame.transpose(Image.FLIP_LEFT_RIGHT)
                gt_frame = gt_frame.transpose(Image.FLIP_LEFT_RIGHT)

            if vflip_rnd < 0.5:
                ip_frame = ip_frame.transpose(Image.FLIP_TOP_BOTTOM)
                gt_frame = gt_frame.transpose(Image.FLIP_TOP_BOTTOM)

        # color augmentattion
        if self.apply_coloraug:
            # gamma correction
            # ip_frame = TF.adjust_gamma(ip_frame, 1)
            # gt_frame = TF.adjust_gamma(gt_frame, 1)

            # saturation
            # sat_factor = 1 + (0.2 - 0.4*np.random.rand())
            # ip_frame = TF.adjust_saturation(ip_frame, sat_factor)
            # gt_frame = TF.adjust_saturation(gt_frame, sat_factor)
            pass

        # other transforms, e.g. Normalize, ToTensor
        if self.transform:
            ip_frame = self.transform(ip_frame)
            gt_frame = self.transform(gt_frame)

        return ip_frame, gt_frame


def get_dataloader(dataset, batch_size=8, shuffle=True, num_workers=8):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=False)


if __name__ == '__main__':
    data_root = '/media/hdd/derain/NTU-derain'
    indexfile = '/home/dxli/workspace/derain/proj/data/Dataset_Training_Synthetic.json'
    # indexfile = '/home/dxli/workspace/derain/proj/data/Dataset_Testing_Synthetic.json'

    __imagenet_stats = {'mean': [0.485, 0.456, 0.406],
                        'std': [0.229, 0.224, 0.225]
                        }

    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize(__imagenet_stats['mean'],
                                                         __imagenet_stats['std'])
                                    ]
                                   )

    dataset = NTU_dataset(data_root=data_root,
                          indexfile_path=indexfile,
                          transform=transform,
                          apply_rotation=False,
                          apply_coloraug=False
                          )

    ip, gt = dataset[-2]
    print(ip.shape, gt.shape)
