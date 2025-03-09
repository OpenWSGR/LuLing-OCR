import torch
import torch.nn as nn
from torch.utils.data import Dataset
from PIL import Image

from torchvision import transforms

class Image2TextDataset(Dataset):
    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        image = Image.open(item['image_path']).convert('L')
        label = item['text_label']
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

