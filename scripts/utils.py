import torch
import numpy as np
import matplotlib.pyplot as plt
import einops
from torch.utils.data import Dataset, DataLoader

from dataprocessing import get_brain_tumor_dataset


def calculate_mean_and_std(dataset: Dataset):
    n_channels = dataset[0][0].shape[0] # assumes images are in the C,H,W format (ToTensorV2/ToTensor have been applied)
    mean_list = [0 for i in range(n_channels)]
    std_list = [0 for i in range(n_channels)]
    n_total_pixels = 0

    for j, (img, _) in enumerate(dataset):
        n_total_pixels += img.shape[1] * img.shape[2]
        for i in range(n_channels):
            mean_list[i] += img[i].sum()

    mean_list = np.array(mean_list) / n_total_pixels

    for j, (img, _) in enumerate(dataset):
        for i in range(n_channels):
            std_list[i] += ((img[i] - mean_list[i])**2).sum()

    std_list = np.sqrt(np.array(std_list) / n_total_pixels)


    return mean_list, std_list


if __name__ == '__main__':
    path_to_dataset_train = '/home/ifeeding99/Downloads/brain-tumor-dataset/Training'
    dataset = get_brain_tumor_dataset(path_to_dataset=path_to_dataset_train, t_train=False)
    m,s = calculate_mean_and_std(dataset)
    print(m)
    print(s)