import glob
import os

import torchvision
from PIL import Image
from tqdm import tqdm
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset


class MnistDataset(Dataset):
    def __init__(self, split, im_path, num_train_samples, im_ext='png'):
        r"""
        Init method for initializing the dataset properties
        :param split: train/test to locate the image files
        :param im_path: root folder of images
        :param im_ext: image extension. assumes all
        images would be this type.
        """
        self.split = split
        self.im_ext = im_ext
        self.num_train_samples = num_train_samples
        self.images, self.labels = self.load_images(im_path)
    
    def load_images(self, im_path):
        assert os.path.exists(im_path), "path of images {} does not exist".format(im_path)
        ims = []
        labels = None
        for d_name in tqdm(os.listdir(im_path)[:self.num_train_samples]):
            # for fname in glob.glob(os.path.join(im_path)):
            ims.append(os.path.join(im_path, d_name))
            #labels.append(int(d_name))
        print('Found {} images for split {}'.format(len(ims), self.split))
        return ims, labels
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index):
        im = Image.open(self.images[index])
        im_tensor = torchvision.transforms.ToTensor()(im)
        
        # Convert input to -1 to 1 range.
        im_tensor = (2 * im_tensor) - 1
        return im_tensor[:2,:,:]