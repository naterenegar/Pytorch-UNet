import logging
from os import listdir
from os.path import splitext
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

import matplotlib.pyplot as plt

class NPZDataset(Dataset):

    def __init__(self, npz_path: str, n_channels: int = 1, X_str: str = 'X',
            mask_str: str = 'y', scale: float = 1.0, transform=None):
        self.file = np.load(npz_path)
        self.images = self.file[X_str]
        self.masks = self.file[mask_str]
        self.transform = transform
        #assert 0 < scale <= 1, 'Scale must be between 0 and 1'
        assert len(self.images) == len(self.masks)
        self.scale = scale

        if self.images.shape[-1] >= 3 and n_channels == 1:
            self.images = self.convert_to_gray(self.images)
        logging.info(f'Creating dataset with {len(self.images)} examples')

    def __len__(self):
        return len(self.images)

    @classmethod
    def preprocess(cls, pil_img, scale, is_mask):
        w, h = pil_img.size
        newW, newH = int(scale * w), int(scale * h)
        assert newW > 0 and newH > 0, 'Scale is too small, resized images would have no pixel'
        pil_img = pil_img.resize((newW, newH))
        img_ndarray = np.asarray(pil_img)

        if img_ndarray.ndim == 2 and not is_mask:
            img_ndarray = img_ndarray[np.newaxis, ...]
        elif not is_mask:
            img_ndarray = img_ndarray.transpose((2, 0, 1))

        if not is_mask:
            img_ndarray = img_ndarray / 255

        return img_ndarray

    @classmethod
    def convert_to_gray(cls, images):
        red = images[:,:,:,0] / 255
        green = images[:,:,:,1] / 255
        blue = images[:,:,:,2] / 255

        # Linear approximation of gamma decompression-recompression
        return (0.299 * red + 0.587 * green + 0.144 * blue)

    @classmethod
    def set_transform(self, transform):
        self.transform = transform

    def __getitem__(self, idx):
        mask = self.masks[idx].copy()
        mask[mask > 0] = 1
        img = self.images[idx]

        assert img.size == mask.size, \
            'Image and mask {name} should be the same size, but are {img.size} and {mask.size}'

        if self.transform:
            # Append our image and mask into one image with more channels, then transform, then split 
            # NOTE: Assumes grayscale
            append = torch.as_tensor(np.stack([np.squeeze(img),np.squeeze(mask)],axis=0).copy())
            append = self.transform(append)
            append = append.detach().numpy()
            img = append[0,:,:]
            mask = append[1,:,:]

        img = Image.fromarray(np.squeeze(img))
        mask = Image.fromarray(np.squeeze(mask))

        img = self.preprocess(img, self.scale, is_mask=False)
        mask = self.preprocess(mask, self.scale, is_mask=True)

        #plt.imshow(np.squeeze(img), cmap='gray')
        #plt.show()

        #plt.imshow(np.squeeze(img), cmap='gray')
        #plt.show()

        return {
            'image': torch.as_tensor(img.copy()).float().contiguous(),
            'mask': torch.as_tensor(mask.copy()).long().contiguous()
        }

class BasicDataset(Dataset):
    def __init__(self, images_dir: str, masks_dir: str, scale: float = 1.0, mask_suffix: str = ''):
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir)
        assert 0 < scale <= 1, 'Scale must be between 0 and 1'
        self.scale = scale
        self.mask_suffix = mask_suffix

        self.ids = [splitext(file)[0] for file in listdir(images_dir) if not file.startswith('.')]
        if not self.ids:
            raise RuntimeError(f'No input file found in {images_dir}, make sure you put your images there')
        logging.info(f'Creating dataset with {len(self.ids)} examples')

    def __len__(self):
        return len(self.ids)

    @classmethod
    def preprocess(cls, pil_img, scale, is_mask):
        w, h = pil_img.size
        newW, newH = int(scale * w), int(scale * h)
        assert newW > 0 and newH > 0, 'Scale is too small, resized images would have no pixel'
        pil_img = pil_img.resize((newW, newH))
        img_ndarray = np.asarray(pil_img)
    
        if img_ndarray.ndim == 2 and not is_mask:
            img_ndarray = img_ndarray[np.newaxis, ...]
        elif not is_mask:
            img_ndarray = img_ndarray.transpose((2, 0, 1))

        if not is_mask:
            img_ndarray = img_ndarray / 255

        return img_ndarray

    @classmethod
    def load(cls, filename):
        ext = splitext(filename)[1]
        if ext in ['.npz', '.npy']:
            return Image.fromarray(np.load(filename))
        elif ext in ['.pt', '.pth']:
            return Image.fromarray(torch.load(filename).numpy())
        else:
            return Image.open(filename)

    def __getitem__(self, idx):
        name = self.ids[idx]
        mask_file = list(self.masks_dir.glob(name + self.mask_suffix + '.*'))
        img_file = list(self.images_dir.glob(name + '.*'))

        assert len(mask_file) == 1, f'Either no mask or multiple masks found for the ID {name}: {mask_file}'
        assert len(img_file) == 1, f'Either no image or multiple images found for the ID {name}: {img_file}'
        mask = self.load(mask_file[0])
        img = self.load(img_file[0])

        assert img.size == mask.size, \
            'Image and mask {name} should be the same size, but are {img.size} and {mask.size}'

        img = self.preprocess(img, self.scale, is_mask=False)
        mask = self.preprocess(mask, self.scale, is_mask=True)

        return {
            'image': torch.as_tensor(img.copy()).float().contiguous(),
            'mask': torch.as_tensor(mask.copy()).long().contiguous()
        }


class CarvanaDataset(BasicDataset):
    def __init__(self, images_dir, masks_dir, scale=1):
        super().__init__(images_dir, masks_dir, scale, mask_suffix='_mask')
