import numpy as np
import SimpleITK as sitk

def resample_img(input, out_size = [512,512,286]):

    """
    This function is used to resample the input image with the given size.

    Parameters
    ----------
    input : Image
        The image we want to resample
    out_size : list
        The target size of output

    Returns
    -----------
    Image which is a result of resampling using the given size.

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