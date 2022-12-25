import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt


def plot_lin_transf(fix_img, fix_mask, mov_img, resampled_mov_img, idx):
    fix_img_data = sitk.GetArrayFromImage(fix_img)
    fix_mask_data = sitk.GetArrayFromImage(fix_mask)
    mov_img_data = sitk.GetArrayFromImage(mov_img)
    resampled_mov_img_data = sitk.GetArrayFromImage(resampled_mov_img)

    plt.figure(figsize=(20,60))
    plt.subplot(131)
    plt.imshow(fix_img_data[idx], cmap='Blues') # fixed image
    plt.imshow(fix_mask_data[idx], cmap = 'Reds', alpha = 0.5)
    plt.title('Reference image and its mask')

    plt.subplot(132)
    plt.imshow(fix_img_data[idx], cmap='Blues') # fixed image
    plt.imshow(mov_img_data[idx], cmap = 'Reds', alpha = 0.5)
    plt.title('Reference image and moving image')

    plt.subplot(133)
    plt.imshow(fix_img_data[idx], cmap='Blues') # fixed image
    plt.imshow(resampled_mov_img_data[idx], cmap = 'Reds', alpha = 0.5)
    plt.title('Reference image and moving image')

    plt.show()