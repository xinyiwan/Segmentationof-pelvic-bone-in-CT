import numpy as np
import SimpleITK as sitk

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
