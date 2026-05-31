import torch
from torch.utils.data import Dataset, DataLoader
import os
import numpy as np
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt


train_transforms = A.Compose([
    A.VerticalFlip(p=0.5),
    A.HorizontalFlip(p=0.5),
    A.RandomSunFlare(p=0.5),
    A.ColorJitter(p=0.5),
    A.Resize(300,300),
    ToTensorV2()
])

val_transforms = A.Compose([
    A.Resize(300,300),
    ToTensorV2()
])


class BrainTumorDataset(Dataset):
    def __init__(self, path_to_dataset, t = None):
        super().__init__()
        self.label_mappings = {'notumor':0, 'pituitary':1, 'meningioma':2, 'glioma':3}
        self.reverse_mappings = {v:k for k,v in self.label_mappings.items()}
        self.images = []
        self.labels = []
        self.t = t
        for label in os.listdir(path_to_dataset):
            current_label = self.label_mappings[label]
            label_path = os.path.join(path_to_dataset, label)
            for img in os.listdir(label_path):
                img_path = os.path.join(label_path, img)
                self.images.append(img_path)
                self.labels.append(current_label)
        self.l = len(self.labels)


    def __getitem__(self, i):
        img = cv2.imread(self.images[i])
        label = self.labels[i]
        img  = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if self.t:
            transformed = self.t(image=img)
            img = transformed['image']
        return img/255, label
    
    def __len__(self):
        return self.l
    

    def show_image(self, i):
        img, label = self.__getitem__(i)
        img = img.permute(1, 2, 0).numpy()
        plt.imshow(img)
        plt.title(f'{self.reverse_mappings[label]}')
        plt.show()


def get_brain_tumor_data_loader(path_to_dataset, batch_size, t_train = True, shuffle=True):
    if t_train:
        dataset = BrainTumorDataset(path_to_dataset, train_transforms)
    else:
        dataset = BrainTumorDataset(path_to_dataset, val_transforms)
    loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle)
    return loader



if __name__ == '__main__':
    path_to_dataset_train = '/home/ifeeding99/Downloads/brain-tumor-dataset/Training'
    train_dataset = BrainTumorDataset(path_to_dataset=path_to_dataset_train, t=train_transforms)
    print(train_dataset[0][0].shape)
    train_dataset.show_image(0)



        