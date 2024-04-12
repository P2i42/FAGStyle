import torch
from torch import nn
import kornia.augmentation as K
import math
# import ipdb

class SwinImageAugmentations(nn.Module):
    def __init__(self, output_size, aug_prob, patch=False, window_size=128):
        super().__init__()
        self.output_size = output_size
        
        self.aug_prob = aug_prob
        self.patch = patch
        # self.window_size = window_size
        self.augmentations = nn.Sequential(
            K.RandomAffine(degrees=15, translate=0.1, p=aug_prob, padding_mode="border"),  # type: ignore
            K.RandomPerspective(0.7, p=aug_prob),
        )
        # self.random_patch = K.RandomResizedCrop(size=(128,128), scale=(p_min,p_max))
        self.avg_pool = nn.AdaptiveAvgPool2d((self.output_size, self.output_size))

    def forward(self, input, num_patch=None, is_global=False):
        """Extents the input batch with augmentations

        If the input is consists of images [I1, I2] the extended augmented output
        will be [I1_resized, I2_resized, I1_aug1, I2_aug1, I1_aug2, I2_aug2 ...]

        Args:
            input ([type]): input batch of shape [batch, C, H, W]

        Returns:
            updated batch: of shape [batch * augmentations_number, C, H, W]
        """
        #input (1,3,256,256) 原图大小
        _, _, H, W = input.shape
        num_patches_side = math.ceil(math.sqrt(num_patch))

        # Calculate window size. Since stride is half the window size and
        # we want to fit exactly num_patches_side patches across one dimension,
        # we adjust the window size accordingly.
        # total_stride = (W - num_patches_side) / (num_patches_side - 1) if num_patches_side > 1 else W
        total_stride = W / (num_patches_side+1) if num_patches_side > 1 else W
        window_size = int(total_stride * 2)
        stride = window_size // 2
        # input_patches = self.shifted_window(input, stride, self.window_size)


        if is_global:
            input = input.repeat(num_patch,1,1,1)
        else:   # 走这边

            if self.aug_prob > 0.0:
                input = self.augmentations(self.shifted_window(input, stride, window_size, num_patches_side))
            else:
                input = self.shifted_window(input, stride, window_size)

        resized_images = self.avg_pool(input)
        return resized_images


    def shifted_window(self, input, stride, window_size, num_patches_side):
        """Generate shifted window patches."""
        # _, _, H, W = input.shape
        patches = []
        for i in range(num_patches_side):
            for j in range(num_patches_side):
                y = i * stride
                x = j * stride
                patch = input[:, :, y:y + window_size, x:x + window_size]

                patches.append(patch)
        return torch.cat(patches, dim=0)

