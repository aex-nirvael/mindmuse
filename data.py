'''
copyright Alex Whelan 2025
code for data processing
'''

import glob
import os
import cv2
import numpy as np

from torch.utils import data


class MindDataset(data.Dataset):
    def __init__(self, image_size):
        super(MindDataset, self).__init__()
        self.image_size = image_size

        self.path = "C:/Users/Alexf/Pictures/GAN Experiments/VaultGAN/Dataset V1"

        self.images = sorted(glob.glob(os.path.join(self.path, "*.jpg")))

        print(f"[*] {len(self.images)} images found in {self.path}...")


    def __len__(self):
        return len(self.images)
    
    def load_image(self, image):

        image = cv2.imread(image, cv2.IMREAD_UNCHANGED)

        # resize to desired size
        image = cv2.resize(image, (self.image_size, self.image_size), interpolation = cv2.INTER_LINEAR)

        # normalise to [-1,1]
        image = (image / 127.5) - 1.0

        # convert to float32
        image = image.astype(np.float32)

        assert image.ndim == 3
        assert image.shape[2] == 3
        assert np.min(image) >= -1.0
        assert np.max(image) <= 1.0
        assert image.dtype == np.float32

        return image
    
    
    def __getitem__(self, index):

        image_path = self.images[index]

        image = self.load_image(image_path)

        return image


class MindDataLoader:
    def __init__(self, dataset, batch_size, train=True):
        super(MindDataLoader, self).__init__()

        if train:
            train_sampler = data.sampler.RandomSampler(dataset)
        else:
            train_sampler = None

        self.data_loader = data.DataLoader(
                dataset, batch_size=batch_size, shuffle=(train_sampler is None),
                num_workers=0, pin_memory=True, drop_last=True, sampler=train_sampler
        )
        self.dataset = dataset
        self.data_iter = self.data_loader.__iter__()

    def next_batch(self):
        try:
            batch = self.data_iter.__next__()
        except StopIteration:
            self.data_iter = self.data_loader.__iter__()
            batch = self.data_iter.__next__()

        return batch
        

def save_image(image):
    """
    (B,3,H,W) tensor -> converts each image to np image and renormalise to [0,255]
    """
    image_np = image.permute(1,2,0).cpu().detach().numpy()
    # normalise to 0-255
    image_np = (image_np + 1.0) * 127.5

    return image_np.astype(np.uint8)


def save_images(gts, preds, outdir, step):
    """
    (B,3,H,W) tensor -> saves each image in batch to out dir
    """
    for i, (gt, pred) in enumerate(zip(gts, preds)):
        gt_np = save_image(gt)
        pred_np = save_image(pred)

        all_images = np.concatenate((gt_np, pred_np), axis=1)
        cv2.imwrite(f"{outdir}/image_step{step}_batch{i}.png", all_images)

