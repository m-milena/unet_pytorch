import os
import cv2
import torch
import numpy as np
from torch.utils import data
from skimage import transform


class Dataset(data.Dataset):
    def __init__(self, data_path, transform=None):
        self.data_path = data_path
        self.inputs = [f for f in os.listdir(self.data_path + 'inputs/') if os.path.isdir(self.data_path + 'inputs/'+f)]
        self.outputs = [f for f in os.listdir(self.data_path + 'outputs/') if os.path.isdir(self.data_path + 'outputs/'+f)]
        self.samples = [f for f in os.listdir(self.data_path + 'inputs/'+ self.inputs[0] + '/')
                             if f[-3:] == 'png']
        self.samples.sort()
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = {'inputs': {}, 'outputs': {}}
        try:
            for f in self.inputs:
                img = cv2.imread(self.data_path + 'inputs/' + f + '/' + self.samples[index], 0)
                sample['inputs'][f] = img
                
            for f in self.outputs:
                img = cv2.imread(self.data_path + 'outputs/' + f + '/' + self.samples[index], 0)
                sample['outputs'][f] = img

                if self.transform:
                    sample = self.transform(sample)
        except Exception as E:
            print(E)
        return sample['inputs'], sample['outputs']


class Preprocessing(object):
    def __init__(self, normalize=True, size=(128,128), toTensor=True):
        self.normalize = normalize
        self.size = size
        self.toTensor = toTensor
    
    def __call__(self, sample):
        for f in sample:
            for key in sample[f]:
                if self.normalize:
                    sample[f][key] = sample[f][key]/255.0
                sample[f][key] = cv2.resize(sample[f][key], self.size)
            sample[f] = np.array([sample[f][key] for key in sample[f]])
            if self.toTensor:
                sample[f] = torch.from_numpy(sample[f]).float()
        return sample

