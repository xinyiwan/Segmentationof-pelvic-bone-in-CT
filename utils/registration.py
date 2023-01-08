import matplotlib.pyplot as plt
import SimpleITK as sitk

def est_lin_transf(fix_img, fix_mask, mov_img):

    """
    This function is used to estimated the linear transformation from mov_img to fix_img.

    Parameters
    ----------
    fix_img : Image
        The reference image
    mov_img : Image
        The moving image
    fix_mask : Image
        The mask of reference image

    Returns
    -----------
    final_transform : Transform (SITK.Transform)
        The esmated transformation

    """

    # initialize alignment of two volumes
    initial_transform = sitk.CenteredTransformInitializer(fix_img, mov_img, sitk.AffineTransform(3), sitk.CenteredTransformInitializerFilter.GEOMETRY)

    # initialize the registration
    registration_method = sitk.ImageRegistrationMethod()

    # Metric settings
    registration_method.SetMetricAsMeanSquares()
    # registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    registration_method.SetMetricSamplingStrategy(registration_method.RANDOM)
    registration_method.SetMetricSamplingPercentage(0.001)

    # set the mask on which you are going to evaluate the similarity between the two images
    registration_method.SetMetricFixedMask(fix_mask)

    # Set interpolator
    registration_method.SetInterpolator(sitk.sitkLinear)

    # Set gradient descent optimizer
    registration_method.SetOptimizerAsGradientDescent(learningRate=1, 
                                                      numberOfIterations=100, 
                                                      convergenceMinimumValue=1e-6,
                                                      convergenceWindowSize=10)

    registration_method.SetOptimizerScalesFromPhysicalShift()

    # Setup for the multi-resolution framework
    registration_method.SetShrinkFactorsPerLevel(shrinkFactors=[4, 2, 1])
    registration_method.SetSmoothingSigmasPerLevel(smoothingSigmas=[2, 1, 0])
    registration_method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

    # Set the initial transformation 
    registration_method.SetInitialTransform(initial_transform, inPlace=False)

    # perform registration
    final_transform = registration_method.Execute(sitk.Cast(fix_img, sitk.sitkFloat32),
                                                  sitk.Cast(mov_img, sitk.sitkFloat32))

    print("--------")
    print("Linear registration:")
    print('Final mean squares value: {0}'.format(registration_method.GetMetricValue()))
    print("Optimizer stop condition: {0}".format(registration_method.GetOptimizerStopConditionDescription()))
    print("Number of iterations: {0}".format(registration_method.GetOptimizerIteration()))
    print("--------")
    return final_transform


def apply_lin_transf(fix_img, mov_img, lin_transf):

    """
    This function is used to apply the esmated linear transformation to mov_img.

    Parameters
    ----------
    fix_img : Image
        The reference image
    mov_img : Image
        The moving image
    lin_tranf : Transform
        The estmated transform

    Returns
    -----------
    mov_img_resampled : Image
        The resampled moving image from linear transformation

    """
    # # only supports images with sitkFloat32 and sitkFloat64 pixel types
    # fix_img = sitk.Cast(fix_img, sitk.sitkFloat32)
    # mov_img = sitk.Cast(mov_img, sitk.sitkFloat32)

    # # resample moving image
    # resampler = sitk.ResampleImageFilter()

    # # set the reference image
    # resampler.SetReferenceImage(fix_img)

    # # set a linear interpolator
    # resampler.SetInterpolator(sitk.sitkLinear)

    # # set the desired transformation
    # resampler.SetTransform(lin_transf)

    # mov_img_resampled = resampler.Execute(mov_img)

    output = sitk.Resample(mov_img, fix_img, lin_transf, sitk.sitkLinear, 0.0, mov_img.GetPixelID())
    return output



def est_nl_transf(fix_img, mov_img, fix_mask):
    """
    This function is used to estimated the nonlinear transformation from mov_img to fix_img.

    Parameters
    ----------
    fix_img : Image
        The reference image
    mov_img : Image
        The moving image
    fix_mask : Image
        The mask of reference image

    Returns
    -----------
    final_transform : Transform (SITK.Transform)
        The esmated transformation

    """

    # initialize the registration
    reg_method = sitk.ImageRegistrationMethod()

    # create initial identity transformation.
    transform_to_displacement_field_filter = sitk.TransformToDisplacementFieldFilter()
    transform_to_displacement_field_filter.SetReferenceImage(fix_img)
    initial_transform = sitk.DisplacementFieldTransform(transform_to_displacement_field_filter.Execute(sitk.Transform()))

    #  regularization. 
    #  The update field refers to fluid regularization; the total field to elastic regularization.
    initial_transform.SetSmoothingGaussianOnUpdate(varianceForUpdateField=0.0, varianceForTotalField=2.0)

    # set the initial transformation
    reg_method.SetInitialTransform(initial_transform)

    # use the function 'SetMetricAsDemons' to be able to perform Demons registration.
    # Provide a parameter (the intensity difference threshold) as input:
    # during the registration, intensities are considered to be equal if their difference is less than the given threshold.
    reg_method.SetMetricAsDemons(10)

    # evaluate the metrics only in the mask, if provided as an input
    reg_method.SetMetricFixedMask(fix_mask)

    # Multi-resolution framework
    reg_method.SetShrinkFactorsPerLevel(shrinkFactors = [4,2,1])
    reg_method.SetSmoothingSigmasPerLevel(smoothingSigmas=[8,4,0])

    # set a linear interpolator
    reg_method.SetInterpolator(sitk.sitkLinear)

    # set a gradient descent optimizer
    reg_method.SetOptimizerAsGradientDescent(learningRate=0.5, 
                                             numberOfIterations=50, 
                                             convergenceMinimumValue=1e-6,
                                             convergenceWindowSize=10)

    reg_method.SetOptimizerScalesFromPhysicalShift()

    # perform registration
    final_transform = reg_method.Execute(sitk.Cast(fix_img, sitk.sitkFloat32),
                                                  sitk.Cast(mov_img, sitk.sitkFloat32))

    print("--------")
    print("Demons registration:")
    print('Final metric value: {0}'.format(reg_method.GetMetricValue()))
    print("Optimizer stop condition: {0}".format(reg_method.GetOptimizerStopConditionDescription()))
    print("Number of iterations: {0}".format(reg_method.GetOptimizerIteration()))
    print("--------")    
    return final_transform

def apply_nl_transf(fix_img, mov_img, nl_transf):
    """
    This function is used to apply the esmated linear transformation to mov_img.

    Parameters
    ----------
    fix_img : Image
        The reference image
    mov_img : Image
        The moving image
    nl_tranf : Transform
        The estmated nonlinear transform

    Returns
    -----------
    output : Image
        The resampled moving image from nonlinear transformation

    """
    output = sitk.Resample(mov_img, fix_img, nl_transf, sitk.sitkLinear, 0.0, mov_img.GetPixelID())

    return output


def seg_atlas(atlas_seg_list): 
    """
    Apply atlas-based segmentation of `im` using the list of CT images in `atlas_ct_list` 
    and the corresponding segmentation masks in `atlas_seg_list`. 
    Return the resulting segmentation mask after majority voting.
    """
    labelForUndecidedPixels = 10
    reference_segmentation= sitk.LabelVoting(atlas_seg_list, labelForUndecidedPixels)

    return reference_segmentation

