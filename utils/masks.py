import numpy as np
import SimpleITK as sitk
from numpy import uint8


def extract_masks(masks_path, mask_labels):

    """
    This function is used to extract new masks with the given labels from raw masks .

    Parameters
    ----------
    masks_path : list
        The list of the paths of raw masks
    mask_labels : list
        The list of targrt labels in the raw masks

    Returns
    -----------

    """

    for path in masks_path:

        mask_image = sitk.ReadImage(path,sitk.sitkFloat32)
        mask_data = sitk.GetArrayFromImage(mask_image)

        new_mask_data = np.zeros_like(mask_data)
        
        info = ".nii.gz"
        for i in mask_labels:
            new_mask_data[np.where(mask_data==i)]=i
            info = "_" + str(i) + info
        
        new_mask_image = sitk.GetImageFromArray(new_mask_data)
        writer = sitk.ImageFileWriter()
        writer.SetImageIO("NiftiImageIO")
        path = path.replace(".nii.gz",info)
        writer.SetFileName(path)
        writer.Execute(new_mask_image)
        print("The new mask is saved as '{}'.".format(path))

def binarize(mask_img):
    """
    This function is used to binarize the mask with multiple labels.

    Parameters
    ----------
    mask_img : Image
        The input mask with multiple labels

    
    Returns
    ----------
    new_mask_img : Image
        Binarized mask.
    """

    mask_data = sitk.GetArrayFromImage(mask_img)
    labels = np.unique(mask_data)[1:]
    new_mask_data = np.zeros_like(mask_data)
    for i in labels:
        new_mask_data[np.where(mask_data==i)]=1
    new_mask_data = new_mask_data.astype(uint8)
    return sitk.GetImageFromArray(new_mask_data)


def resample_mask(input, out_size = [512,512,286]):

    """
    This function is used to resample the input image mask with the given size.

    Parameters
    ----------
    input : Image mask
        The image mask to be resamples
    out_size : list
        The target size of output

    Returns
    -----------
    Image mask which is a result of resampling using the given size.

    """

    out_spacing = [origin_sz * origin_spc / out_sz  for origin_sz, origin_spc, out_sz in zip(input.GetSize(), input.GetSpacing(), out_size)]
    return sitk.Resample(
        input, 
        out_size, 
        sitk.Transform(), 
        sitk.sitkNearestNeighbor, 
        input.GetOrigin(), 
        out_spacing, 
        input.GetDirection(), 
        0.0, 
        input.GetPixelIDValue())