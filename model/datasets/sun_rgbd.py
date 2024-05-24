# -*- coding: utf-8 -*-

import os
import pickle
import numpy as np
from PIL import Image
from .base import BaseDataset


# SUN-RGBD Base
class SUNRBDBase:
    SPLITS = ['train', 'val']

    # number of classes without void
    NUM_CLASS = 37

    CLASS_NAMES_ENGLISH = ['void', 'wall', 'floor', 'cabinet', 'bed', 'chair',
                           'sofa', 'table', 'door', 'window', 'bookshelf',
                           'picture', 'counter', 'blinds', 'desk', 'shelves',
                           'curtain', 'dresser', 'pillow', 'mirror',
                           'floor mat', 'clothes', 'ceiling', 'books',
                           'fridge', 'tv', 'paper', 'towel', 'shower curtain',
                           'box', 'whiteboard', 'person', 'night stand',
                           'toilet', 'sink', 'lamp', 'bathtub', 'bag']

    CLASS_COLORS = [(0, 0, 0), (119, 119, 119), (244, 243, 131),
                    (137, 28, 157), (150, 255, 255), (54, 114, 113),
                    (0, 0, 176), (255, 69, 0), (87, 112, 255), (0, 163, 33),
                    (255, 150, 255), (255, 180, 10), (101, 70, 86),
                    (38, 230, 0), (255, 120, 70), (117, 41, 121),
                    (150, 255, 0), (132, 0, 255), (24, 209, 255),
                    (191, 130, 35), (219, 200, 109), (154, 62, 86),
                    (255, 190, 190), (255, 0, 255), (192, 79, 212),
                    (152, 163, 55), (230, 230, 230), (53, 130, 64),
                    (155, 249, 152), (87, 64, 34), (214, 209, 175),
                    (170, 0, 59), (255, 0, 0), (193, 195, 234), (70, 72, 115),
                    (255, 255, 0), (52, 57, 131), (12, 83, 45)]

class SUNRGBD(SUNRBDBase, BaseDataset):
    def __init__(self, root, split='train', mode='train', transform=None, dep_transform=None,
                    target_transform=None, depth_mode='refined', with_input_orig=False, **kwargs):
        super(SUNRGBD, self).__init__(root, split, mode, transform, target_transform, **kwargs)
        
        self.BASE_DIR = os.path.join(root, 'sunrgbd')
        
        self.dep_transform = dep_transform
        print('==check dep_transform {}'.format(dep_transform))

        self._cameras = ['realsense', 'kv2', 'kv1', 'xtion']
        assert split in self.SPLITS, f'parameter split must be one of {self.SPLITS}, got {split}'
        self._split = split
        assert depth_mode in ['refined', 'raw']
        self._depth_mode = depth_mode
        self._with_input_orig = with_input_orig

        self.img_dirs, self.depth_dirs, self.label_dirs = self.load_file_lists(split)

        self.classes = self.CLASS_NAMES_ENGLISH
        self.class_colors = np.array(self.CLASS_COLORS, dtype='uint8')

        # note that mean and std differ depending on the selected depth_mode
        # however, the impact is marginal, therefore, we decided to use the
        # stats for refined depth for both cases
        # stats for raw: mean: 18320.348967710495, std: 8898.658819551309
        self._depth_mean = 19025.14930492213
        self._depth_std = 9880.916071806689

    def __getitem__(self, idx):
        _img = self.load_image(idx)
        _dep = self.load_depth(idx)
        _target = self.load_target(idx)

        # synchronized transform
        if self.mode == 'train':
            # return _img (Image), _dep (Image), _target (2D tensor)
            _img, _dep, _target = self._sync_transform(_img, _target, depth=_dep,
                                                       IGNORE_LABEL=0)  # depth need to modify
        elif self.mode == 'val':
            _img, _dep, _target = self._val_sync_transform(_img, _target, depth=_dep)
            # _img: 3 channel image, pixel value 0~255,
            # _dep: 1 channel image, pixel value 1~65528,
            # _target: 1 channel tensor, pixel value 0~37

        # by default ignore class is -1
        _target -= 1

        # general resize, normalize and toTensor
        if self.transform is not None:
            _img = self.transform(_img)  # _img to tensor, normalize
        if self.dep_transform is not None:
            _dep = self.dep_transform(_dep)  # depth to tensor, normalize
        if self.target_transform is not None:
            _target = self.target_transform(_target)
        return _img, _dep, _target  # all tensors

    def load_image(self, idx):
        img_path = os.path.join(self.BASE_DIR, self.img_dirs['list'][idx])
        return Image.open(img_path).convert('RGB')

    def load_depth(self, idx):
        dep_file = self.depth_dirs['list'][idx]
        if self._depth_mode == 'raw':
            dep_file = dep_file.replace('depth_bfx', 'depth')
        dep_path = os.path.join(self.BASE_DIR, dep_file)
        return Image.open(dep_path)

    def load_target(self, idx):
        target_path = os.path.join(self.BASE_DIR, self.label_dirs['list'][idx])
        _target = np.load(target_path).astype(np.int8)
        return Image.fromarray(_target, 'L')

    def load_file_lists(self, split):
        def _get_filepath(filename):
            return os.path.join(self.BASE_DIR, filename)

        img_dir_train_file = _get_filepath('train_rgb.txt')
        depth_dir_train_file = _get_filepath('train_depth.txt')
        label_dir_train_file = _get_filepath('train_label.txt')

        img_dir_test_file = _get_filepath('test_rgb.txt')
        depth_dir_test_file = _get_filepath('test_depth.txt')
        label_dir_test_file = _get_filepath('test_label.txt')

        img_dirs = dict()
        depth_dirs = dict()
        label_dirs = dict()

        if split == 'train':
            img_dirs['list'], img_dirs['dict'] = self.list_and_dict_from_file(img_dir_train_file)
            depth_dirs['list'], depth_dirs['dict'] = self.list_and_dict_from_file(depth_dir_train_file)
            label_dirs['list'], label_dirs['dict'] = self.list_and_dict_from_file(label_dir_train_file)
        else:
            img_dirs['list'], img_dirs['dict'] = self.list_and_dict_from_file(img_dir_test_file)
            depth_dirs['list'], depth_dirs['dict'] = self.list_and_dict_from_file(depth_dir_test_file)
            label_dirs['list'], label_dirs['dict'] = self.list_and_dict_from_file(label_dir_test_file)

        return img_dirs, depth_dirs, label_dirs

    def list_and_dict_from_file(self, filepath):
        with open(filepath, 'r') as f:
            file_list = f.read().splitlines()
        dictionary = dict()
        for cam in self._cameras:
            dictionary[cam] = [i for i in file_list if cam in i]

        return file_list, dictionary

    def __len__(self):
        return len(self.img_dirs['list'])

    def compute_depth_mean_std(self, recompute=False):
        # ensure that mean and std are computed on train set only
        assert self.split == 'train'

        # build filename
        out_path = os.path.join(self.BASE_DIR, 'meta_data')
        if not os.path.exists(out_path):
            os.makedirs(out_path)
        depth_stats_filepath = os.path.join(out_path, f'depth_{self._depth_mode}_mean_std.pickle')
        depth_stats_obs_path = os.path.join(out_path, 'depth_stats_by_obs.txt')

        if not recompute and os.path.exists(depth_stats_filepath):
            depth_stats = pickle.load(open(depth_stats_filepath, 'rb'))
            print(f'Loaded depth mean and std from {depth_stats_filepath}')
            print(depth_stats)
            return depth_stats

        print('Compute mean and std for depth images.')

        pixel_sum = np.float64(0)
        pixel_n = np.uint64(0)
        std_sum = np.float64(0)
        pixel_min, pixel_max = np.float32(10000), np.float32(-1)
        all_obs = dict()

        print('Compute mean')
        for i in range(len(self)):
            depth = self.load_depth(i)
            depth = np.asarray(depth).T
            if self._depth_mode == 'raw':
                depth_valid = depth[depth > 0]
            else:
                depth_valid = depth.flatten()
            pixel_sum += np.sum(depth_valid)
            pixel_n += np.uint64(len(depth_valid))
            pixel_min = min(pixel_min, np.min(depth_valid))
            pixel_max = max(pixel_max, np.max(depth_valid))
            all_obs[i] = {'mean': np.mean(depth_valid), 'std': np.std(depth_valid)}
            print('\n{}/{}, current:{}'.format(i + 1, len(self), all_obs[i]))

        mean = pixel_sum / pixel_n

        print('Compute std')
        for i in range(len(self)):
            depth = self.load_depth(i)
            depth = np.asarray(depth).T
            if self._depth_mode == 'raw':
                depth_valid = depth[depth > 0]
            else:
                depth_valid = depth.flatten()
            std_sum += np.sum(np.square(depth_valid - mean))
            print(f'\r{i + 1}/{len(self)}', end='')

        std = np.sqrt(std_sum / pixel_n)

        depth_stats = {'mean': mean, 'std': std, 'min': pixel_min, 'max': pixel_max}
        print(depth_stats)

        with open(depth_stats_filepath, 'wb') as f:
            pickle.dump(depth_stats, f)
        with open(depth_stats_obs_path, 'w') as f:
            for k, v in all_obs.items():
                f.write(str(k) + '\t'
                        + 'mean' + '\t' + str(v['mean']) + '\t'
                        + 'std' + '\t' + str(v['std']) + '\n')

        return depth_stats