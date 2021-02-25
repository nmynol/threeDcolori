import os
import random

import torch.utils.data as data
import numpy as np

from data.data_utils import is_image_file, get_img, get_xdog, get_diffs, transform_pic


class MyDataset(data.Dataset):
    def __init__(self, img_path, xdog_path, img_size, seq_len, threshold):
        super(MyDataset, self).__init__()
        self.img_path = img_path
        self.xdog_path = xdog_path
        self.img_size = img_size
        self.seq_len = seq_len
        self.threshold = threshold
        self.scene_array = []

        # make up filename list
        self.image_filenames = [x for x in os.listdir(self.img_path) if is_image_file(x)]
        # print(self.image_filenames)
        # make up scene number list
        self.scene_num = list(set(int(x.split('e')[-1].split('_')[0]) for x in self.image_filenames))
        # print(self.scene_num)
        # initialize temporal scene array
        scene_array_temp = [[] for i in range(max(self.scene_num) + 1)]
        # print(scene_array_temp)
        # append pics to each dimension of scene array
        for x in self.image_filenames:
            scene_array_temp[int(x.split('e')[-1].split('_')[0])].append(x)
        # print(scene_array_temp)
        # sort each dimension
        for i in range(len(scene_array_temp)):
            scene_array_temp[i] = sorted(scene_array_temp[i])
        # print(scene_array_temp)
        # remove [] to make final scene array
        for i in scene_array_temp:
            if i:
                self.scene_array.append(i)
        # print(self.scene_array)

    def __getitem__(self, index):
        whole_scene = self.scene_array[index]
        start_index = random.randint(1, len(whole_scene) - self.seq_len)
        target_frames = []
        for i in range(self.seq_len):
            target_frames.append(whole_scene[start_index + i])
        # (seq_len, h, w, channel)
        imgs_now = np.concatenate([get_img(os.path.join(self.img_path, target_frames[i]), self.img_size)[np.newaxis, :]
                                   for i in range(len(target_frames))], 0)
        # (seq_len, h, w)
        xdogs_now = np.concatenate([get_xdog(os.path.join(self.img_path, target_frames[i]), self.img_size)[np.newaxis, :]
                                    for i in range(len(target_frames))], 0)
        # (h, w, channel)
        img_first, xdog_first = self.get_first(whole_scene[start_index - 1])
        # (seq_len, h, w)
        img_diffs, xdog_diffs = get_diffs(imgs_now, img_first, xdogs_now, xdog_first, self.threshold)

        img_diffs, xdog_diffs, imgs_now, img_first, xdogs_now, xdog_first = transform_pic(img_diffs, xdog_diffs,
                                                                                          imgs_now, img_first,
                                                                                          xdogs_now, xdog_first)

        return img_diffs, xdog_diffs, imgs_now, img_first, xdogs_now, xdog_first

    def __len__(self):
        return len(self.scene_array)

    def get_first(self, name):
        # frame_num = int(name.split('.')[0].split('_')[-1])
        fimg = get_img(os.path.join(self.img_path, name.split('_')[0] + '_' + name.split('_')[1] +
                                    '_0001.' + name.split('.')[-1]), self.img_size)
        fxdog = get_xdog(os.path.join(self.xdog_path, name.split('_')[0] + '_' + name.split('_')[1] +
                                      '_0001.' + name.split('.')[-1]), self.img_size)
        # print(os.path.join(self.img_path, name.split('_')[0] + '_' + name.split('_')[1] +
        #                    '_0001.' + name.split('.')[-1]))
        return fimg, fxdog
