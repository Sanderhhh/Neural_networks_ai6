import torch
from torch.utils.data import random_split
import torchvision
from torchvision.datasets import ImageFolder
from torchvision.transforms import ToTensor
import os

if __name__ == "__main__":
    print("HEllo world!")
    train_folder_path = "subset-data"
    test_folder_path = "subset-data"
    train_folder = os.listdir(train_folder_path)
    test_folder = os.listdir(test_folder_path)

    # make it so that half is for training, half is for validation, can adjust this ratio later
    apple_folder_path = "subset-data/Apple"
    pear_folder_path = "subset-data/Pear"
    apple_folder = os.listdir(apple_folder_path)
    pear_folder = os.listdir(pear_folder_path)
    image_total = len(apple_folder) + len(pear_folder)
    training_size = int(image_total / 2)
    validation_size = image_total - training_size

    dataset = ImageFolder(train_folder_path, transform = ToTensor())
    print(dataset)
    training_dataset, validation_dataset = torch.utils.data.random_split(dataset, [training_size, validation_size])