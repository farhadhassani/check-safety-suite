import sys
from pathlib import Path
import torch
from torchvision import transforms
from src.models.unet_plusplus import UNetPlusPlus
from scripts.train_unet_plusplus import ChequeDataset

def main():
    data_dir = "d:/Antigraph/check-safety-suite/data/IDRBT Cheque Image Dataset/300"
    transform = transforms.Compose([transforms.Resize((256,256)), transforms.ToTensor()])
    ds = ChequeDataset(data_dir, transform=transform)
    print('Dataset length:', len(ds))
    if len(ds)>0:
        img, mask = ds[0]
        print('Sample shapes:', img.shape, mask.shape)

if __name__ == '__main__':
    main()
