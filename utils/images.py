import numpy as np
import SimpleITK as sitk
from numpy import uint8

def resample_img(input, out_size = [512,512,286]):

    """
    This function is used to resample the input image with the given size.

    Parameters
    ----------
    input : Image
        The image to be resampled
    out_size : list
        The target size of output

    Returns
    -----------
    Resampled Image

    """

    out_spacing = [origin_sz * origin_spc / out_sz  for origin_sz, origin_spc, out_sz in zip(input.GetSize(), input.GetSpacing(), out_size)]
    return sitk.Resample(
        input, 
        out_size, 
        sitk.Transform(), 
        sitk.sitkLinear, 
        input.GetOrigin(), 
        out_spacing, 
        input.GetDirection(), 
        0.0, 
        input.GetPixelIDValue())
    

def normalize(input):
    """
    This function applys normalization to make the data be in the range [-1,1].
    """
    input_data = sitk.GetArrayFromImage(input)
    min = np.min(input_data)
    max = np.max(input_data)
    input_data = (input_data - min) / (max - min) * 256.0
    input_data = input_data.astype(uint8)
    return sitk.GetImageFromArray(input_data)