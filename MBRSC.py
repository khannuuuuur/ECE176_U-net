import torch
import PIL
from torch.utils.data import Dataset
import os
import os.path as osp
import torchvision.transforms as transforms
import torchvision.datasets as dset
import torchvision.transforms as T
from PIL import Image
import numpy as np
import cv2



class MBRSC(Dataset):
    def __init__(
        self,
        data_list
    ):
        self.data_list = data_list
        self.means = [0.525091009935856, 0.5349247905585822, 0.5491133125103805]
        self.stds = [0.2972007767501209, 0.30354216287443164, 0.3211262417037121]
        
    

    def __len__(self):
        return len(self.data_list)
             
    def __getitem__(self, i):
        # input and target images
        in_name = self.data_list[i]
        gt_name = self.data_list[i].replace('.jpg','.png').replace('image','mask')

        # process the images
        
        
        
        normalize = transforms.Normalize(mean=self.means,
                                         std=self.stds)
        transf_img = transforms.Compose([
            transforms.ToTensor(),
            normalize
        ])
        
        

        in_image = np.array(Image.open(in_name))
        gt_label = np.array(Image.open(gt_name))
        

        #w, h = in_image.size
        #gt_label = np.frombuffer(gt_label.tobytes(), dtype=np.ubyte)#.reshape((h, w))

        #in_image = cv2.resize(np.array(in_image), None, fx=1, fy=1, interpolation = cv2.INTER_NEAREST)
        #gt_label = cv2.resize(np.array(gt_label), None, fx=1, fy=1, interpolation = cv2.INTER_NEAREST)
        
        in_image = transf_img(in_image)
        
        gt_label = torch.tensor(gt_label)
        gt_label = gt_label.long()
        

        return in_image, gt_label

    def add_label_colors(label):
        result = np.zeros((label.shape[0], label.shape[1], 3))

        colors = np.array([[60, 16, 152], [132,41,246], [155,155,155], [226,169,41], [110,193,228], [254,221,58]])
        total = label.shape[0]*label.shape[1]
        for i in range(len(colors)):
            layer = np.zeros(result.shape)
            for j in range(3):
                layer[:,:,j] = colors[i,j]
            h = label==i
            h = np.repeat(h[:,:,np.newaxis], 3, axis=2)
            result = np.add(result, np.multiply(layer, h))
        return result.astype('uint8')
            
    def revert_input(self, img, label):
        img = np.transpose(img.cpu().numpy(), (1, 2, 0))
        
        std_img = np.array(self.means).reshape((1, 1, -1))
        mean_img = np.array(self.stds).reshape((1, 1, -1))
        img *= std_img
        img += mean_img
        
        
        img = (img-np.min(img))/(np.max(img)-np.min(img))*255
      
        label = MBRSC.add_label_colors(label.cpu().numpy())
        return img.astype('uint8'), label
    
