import torch
import torch.utils.data as data
import numpy as np
import re, os
import pickle, random
from os.path import join
from sklearn.model_selection import train_test_split



class XY_dataset_5inOne(data.Dataset):
    def __init__(self, tvt = 'train', serial_len = 5, frame_len = 3750, channel_num = 5, datasetName = 'SHHS'):
        super(XY_dataset_5inOne, self).__init__()
        self.serial_len = serial_len
        self.frame_len = frame_len
        self.channel_num = channel_num
        cache_path = './prepared_data/{:}_{:}_cache.pkl'.format(tvt, datasetName)

        #if cache
        if os.path.exists(cache_path):
            with open(cache_path, 'rb') as f:
                cache = pickle.load(f)
            self.items = cache['items']
            self.loc = cache['loc']
            self.len = len(self.items)
            return
        #else
        npz_path ='../CNN-SHHS/prepared_data/{:}/'.format(datasetName)
        person_paths = [join(npz_path, f) for f in os.listdir(npz_path) if os.path.isdir(join(npz_path, f))]

        train_idx, valid_idx = train_test_split(person_paths, train_size = 0.8,  random_state = 0)
        valid_idx, test_idx = train_test_split(valid_idx, train_size = 0.5, random_state = 0)
        # train_idx = valid_idx = test_idx = person_paths
        # train_num = 10
        # train_idx, valid_idx, test_idx = person_paths[0:train_num], [person_paths[train_num]], person_paths[train_num:]
        
        self.items, self.y, self.loc = [], [], []
        person_paths = train_idx if tvt == 'train' else valid_idx if tvt == 'valid' else test_idx
        for person_path in person_paths:
            frame_paths = [join(person_path, f) for f in sorted(os.listdir(person_path))]  # 一个人的所有帧
            for i in range(len(frame_paths) - (serial_len + 1)):
                item = []
                for j in range(i, i + serial_len):
                    item.append(frame_paths[j])
                self.items.append(item)
                self.loc.append(i + int(serial_len / 2))
        self.len = len(self.items)
        # save cache
        cache = {'items': self.items, 'loc': self.loc}
        with open(cache_path, 'wb') as f:
            pickle.dump(cache, f)

    def __getitem__(self, index):
        paths = self.items[index]
        loc = self.loc[index]
        X = torch.zeros(size = [self.serial_len ,self.frame_len, self.channel_num]).float()
        for i in range(self.serial_len):
            npz = np.load(paths[i])
            X[i] = torch.from_numpy(npz['X']).float()
            if i == int(0.5 * self.serial_len):
                y = torch.from_numpy(npz['y']).long()
        return X, y, loc

    def __len__(self):
        return self.len
