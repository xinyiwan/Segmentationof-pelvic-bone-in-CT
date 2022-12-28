import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt


def plot_transf(fix_img, fix_mask, mov_img, resampled_mov_img, x,y,z):
    fix_img_data = sitk.GetArrayFromImage(fix_img)
    fix_mask_data = sitk.GetArrayFromImage(fix_mask)
    mov_img_data = sitk.GetArrayFromImage(mov_img)
    mov_img_resampled_data = sitk.GetArrayFromImage(resampled_mov_img)

    fig, axs = plt.subplots(3, 3, figsize=(15, 10))


    # marker symbol
    axs[0, 0].imshow(fix_img_data[x], cmap='Blues')
    axs[0, 0].imshow(fix_mask_data[x], cmap = 'Reds', alpha = 0.3)
    axs[0, 0].set_title('Reference image and mask')

    axs[0, 1].imshow(fix_img_data[x], cmap='Blues')
    axs[0, 1].imshow(mov_img_data[x], cmap = 'Reds', alpha = 0.3)
    axs[0, 1].set_title('Reference image and moving image')

    axs[0, 2].imshow(fix_img_data[x], cmap='Blues')
    axs[0, 2].imshow(mov_img_resampled_data[x], cmap = 'Reds', alpha = 0.3)
    axs[0, 2].set_title('Reference image and transformed image')

    # marker symbol
    axs[1, 0].imshow(fix_img_data[:,y,:], cmap='Blues')
    axs[1, 0].imshow(fix_mask_data[:,y,:], cmap = 'Reds', alpha = 0.3)
    axs[1, 0].set_title('Reference image and mask')

    axs[1, 1].imshow(fix_img_data[:,y,:], cmap='Blues')
    axs[1, 1].imshow(mov_img_data[:,y,:], cmap = 'Reds', alpha = 0.3)
    axs[1, 1].set_title('Reference image and moving image')

    axs[1, 2].imshow(fix_img_data[:,y,:], cmap='Blues')
    axs[1, 2].imshow(mov_img_resampled_data[:,y,:], cmap = 'Reds', alpha = 0.3)
    axs[1, 2].set_title('Reference image and transformed image')

    axs[2, 0].imshow(fix_img_data[:,:,z], cmap='Blues')
    axs[2, 0].imshow(fix_mask_data[:,:,z], cmap = 'Reds', alpha = 0.3)
    axs[2, 0].set_title('Reference image and mask')

    axs[2, 1].imshow(fix_img_data[:,:,z], cmap='Blues')
    axs[2, 1].imshow(mov_img_data[:,:,z], cmap = 'Reds', alpha = 0.3)
    axs[2, 1].set_title('Reference image and moving image')

    axs[2, 2].imshow(fix_img_data[:,:,z], cmap='Blues')
    axs[2, 2].imshow(mov_img_resampled_data[:,:,z], cmap = 'Reds', alpha = 0.3)
    axs[2, 2].set_title('Reference image and transformed image')

    fig.show()



def plot_atlas_seg(est_lin_mask, fix_mask, lin_imgs, lin_masks, idx):
    reference_segmentation_data = sitk.GetArrayFromImage(est_lin_mask)

    plt.figure(figsize=(20,20))
    plt.subplot(141)
    plt.imshow(sitk.GetArrayFromImage(fix_mask)[idx], cmap = 'Blues')
    plt.imshow(reference_segmentation_data[idx], cmap='Reds', alpha= 0.5) 
    plt.title('Reference mask and the atlas-seg mask from label voting')

    plt.subplot(142)
    plt.imshow(sitk.GetArrayFromImage(lin_imgs[0])[idx], cmap = 'Blues')
    plt.imshow(sitk.GetArrayFromImage(lin_masks[0])[idx], cmap='Reds', alpha= 0.5) 
    plt.title('Transformed image and trasformed mask of g1_53')

    plt.subplot(143)
    plt.imshow(sitk.GetArrayFromImage(lin_imgs[1])[idx], cmap = 'Blues')
    plt.imshow(sitk.GetArrayFromImage(lin_masks[1])[idx], cmap='Reds', alpha= 0.5) 
    plt.title('Transformed image and trasformed mask of g1_54')

    plt.subplot(144)
    plt.imshow(sitk.GetArrayFromImage(lin_imgs[2])[idx], cmap = 'Blues')
    plt.imshow(sitk.GetArrayFromImage(lin_masks[2])[idx], cmap='Reds', alpha= 0.5) 
    plt.title('Transformed image and trasformed mask of g1_55')
    
    plt.show()