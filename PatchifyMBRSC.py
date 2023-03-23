import PIL
import os
import os.path as osp
from os import path
from PIL import Image
import numpy as np
from Unet import Unet

#========================================================
#                       Image IO
#========================================================

def openImage(name):
    return Image.open(name)

def saveImage(imageArray, name, overwrite=True, mode='RGB'):
    if overwrite:
        Image.fromarray(imageArray).save(name)
        return name

    count = 1
    while path.exists(name[:name.index('.')] + str(count) + name[name.index('.'):]):
        count += 1
    fileName = name[:name.index('.')] + str(count) + name[name.index('.'):]
    Image.fromarray(imageArray, mode=mode).save(fileName)
    return fileName

#========================================================
#                   Other stuff
#========================================================

"""
0. Building: #3C1098
1. Land (unpaved area): #8429F6
2. Road: #6EC1E4
3. Vegetation: #FEDD3A
4. Water: #E2A929
5. Unlabeled: #9B9B9B
"""

def removeZeros(image):

    means = np.array([[60, 16, 152], [132,41,246], [110,193,228], [254,221,58], [226,169,41], [155,155,155]])


    for row in range(image.shape[0]):
        for col in range(image.shape[1]):
            if np.sum(image[row,col]) < 10:
                image[row,col] = np.array([155,155,155])
    return image.astype('uint8')

def NN(image, classes):

    for row in range(image.shape[0]):
        for col in range(image.shape[1]):
            pixel = image[row,col]
            if pixel not in classes:
                image[row,col] = classes[np.argmin(np.abs(classes-pixel))]
    return image.astype('uint8')

def get_data_list():
    root_dir = 'MBRSC'
    data_list = []
    mask_list = []
    for name in ['Tile 1', 'Tile 2', 'Tile 3', 'Tile 4', 'Tile 5', 'Tile 6', 'Tile 7']:
        data_dir = osp.join(
            root_dir, name, 'images'
        )
        data_list += sorted(
            osp.join(data_dir, img_name) for img_name in
            filter(
                lambda x: x[-4:] == '.jpg',
                os.listdir(data_dir)
            )
        )
        mask_dir = osp.join(
            root_dir, name, 'masks'
        )
        mask_list += sorted(
            osp.join(mask_dir, img_name) for img_name in
            filter(
                lambda x: x[-4:] == '.png',
                os.listdir(mask_dir)
            )
        )
    assert len(data_list) == len(mask_list)
    return data_list, mask_list
   
def get_channel_normalization_params():
    data_list, _ = get_data_list()
    channels = [np.array([]), np.array([]), np.array([])]
    for i in range(len(data_list)):
        image = np.array(openImage(data_list[i]))
        for j in range(3):
            channels[j] = np.append(channels[j], image[:,:,j].flatten()/255)
    means = [np.mean(channel) for channel in channels]
    std = [np.std(channel) for channel in channels]
    return means, std
    
def encodemask(mask): 
    classes = np.array([ 45.,  92., 155., 171., 172., 212.,])
    for j in range(len(classes)):
            mask[mask==classes[j]] = j
    assert np.sum(mask>7) == 0
    return mask

def get_out_size(in_size):
    out_size = in_size
    for _ in range(4):
        out_size -= 4
        if out_size < 1:
            return -1
        out_size = out_size//2
        if out_size < 1:
            return -1
    for _ in range(4):
        out_size = (out_size-4)*2
    return out_size-4
    
def patchify(patch_size=300, stride=None, rotate=True, whole=False):
    if stride is None:
        stride=patch_size

    data_list, mask_list = get_data_list()

    """
    out_size = get_out_size(patch_size)
    while out_size == -1:
        print('Patch size', patch_size, 'is not large enough. Changing patch_size to', patch_size+2)
        patch_size += 2
        out_size = get_out_size(patch_size)
         
    d = (patch_size-out_size)//2
    """

    os.system('rm MBRSC/Patched/*')
    
    count = 0
    for i in range(len(data_list)):
        image = np.array(openImage(data_list[i]))
        mask = np.array(openImage(mask_list[i]).convert('I')) 
        assert patch_size < image.shape[0] and patch_size < image.shape[1]
            
        encodemask(mask)

        for row in range(0, image.shape[0], stride):
            for col in range(0, image.shape[1], stride):
                if row+patch_size < image.shape[0] and col+patch_size < image.shape[1]:
                    image_patch = image[row:row+patch_size, col:col+patch_size, :]
                    #mask_patch = mask[row+d:row+d+out_size, col+d:col+d+out_size]
                    mask_patch = mask[row:row+patch_size, col:col+patch_size]
                elif whole:
                    end_row = min(image.shape[0], row+patch_size)
                    end_col = min(image.shape[1], col+patch_size)
                    image_patch = image[end_row-patch_size:end_row, end_col-patch_size:end_col, :]
                    #mask_patch = mask[end_row-d-out_size:end_row-d, end_col-d-out_size:end_col-d]
                    mask_patch = mask[end_row-patch_size:end_row, end_col-patch_size:end_col]
                else:
                    continue
                
                assert image_patch.shape[0] == image_patch.shape[1] == patch_size
                #assert mask_patch.shape[0] == mask_patch.shape[1] == out_size
              
                saveImage(image_patch, 'MBRSC/Patched/image' + str(count) + '.jpg')
                saveImage(mask_patch, 'MBRSC/Patched/mask' + str(count) + '.png', mode='I')
                count += 1

                if rotate:
                    for _ in range(3):
                        image_patch = np.rot90(image_patch)
                        mask_patch = np.rot90(mask_patch)
                        saveImage(image_patch, 'MBRSC/Patched/image' + str(count) + '.jpg')
                        saveImage(mask_patch, 'MBRSC/Patched/mask' + str(count) + '.png', mode='I')
                        count += 1
                        
                        

if __name__ == '__main__':
    patchify()
