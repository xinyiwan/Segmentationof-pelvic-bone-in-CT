import matplotlib.pyplot as plt
import SimpleITK as sitk

def est_lin_transf(fix_img, mov_img, fix_mask):

    """
    This function is used to estimated the linear transformation from mov_img to fix_img.

    Parameters
    ----------
    fix_img : str
        The Path of the reference image
    mov_img : str
        The path of the moving image
    fix_mask : str
        The path of the mask of reference image

    Returns
    -----------
    final_transform : dics
        The dictonary contains all the transformation parameters

    """

    # initialize alignment of two volumes
    initial_transform = sitk.CenteredTransformInitializer(fix_img, mov_img, sitk.AffineTransform(3), sitk.CenteredTransformInitializerFilter.GEOMETRY)

    # initialize the registration
    registration_method = sitk.ImageRegistrationMethod()

    # Metric settings
    registration_method.SetMetricAsMeanSquares()
    # registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    registration_method.SetMetricSamplingStrategy(registration_method.RANDOM)
    registration_method.SetMetricSamplingPercentage(0.01)

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
    print("Affine registration:")
    print('Final metric value: {0}'.format(registration_method.GetMetricValue()))
    print("Optimizer stop condition: {0}".format(registration_method.GetOptimizerStopConditionDescription()))
    print("Number of iterations: {0}".format(registration_method.GetOptimizerIteration()))
    print("--------")
    return final_transform


def apply_lin_transf(fix_img, mov_img, lin_transf, is_label=False):
    """
    Apply given linear transform `lin_transf` to `mov_img` and return
    the transformed image.
    """
    # only supports images with sitkFloat32 and sitkFloat64 pixel types
    fix_img = sitk.Cast(fix_img, sitk.sitkFloat32)
    mov_img = sitk.Cast(mov_img, sitk.sitkFloat32)

    # resample moving image
    resampler = sitk.ResampleImageFilter()

    # set the reference image
    resampler.SetReferenceImage(fix_img)

    # use a linear interpolator
    if is_label:
        resampler.SetInterpolator(sitk.sitkNearestNeighbor)
    else:
        resampler.SetInterpolator(sitk.sitkLinear)

    # set the desired transformation
    resampler.SetTransform(lin_transf)

    mov_img_resampled = resampler.Execute(mov_img)
    mov_img_resampled_data = sitk.GetArrayFromImage(mov_img_resampled)
    return mov_img_resampled

def est_nl_transf(fix_img, mov_img,fix_mask,print_log=False):
    """
    Estimate non-linear transform to align `im_mov` to `im_ref` and
    return the transform parameters.
    """

    # initialize the registration
    reg_method = sitk.ImageRegistrationMethod()

    # create initial identity transformation.
    transform_to_displacement_field_filter = sitk.TransformToDisplacementFieldFilter()
    transform_to_displacement_field_filter.SetReferenceImage(fix_img)
    initial_transform = sitk.DisplacementFieldTransform(transform_to_displacement_field_filter.Execute(sitk.Transform()))

    #  regularization. The update field refers to fluid regularization; the total field to elastic regularization.
    initial_transform.SetSmoothingGaussianOnUpdate(varianceForUpdateField=0.0, varianceForTotalField=2.0)

    # set the initial transformation
    reg_method.SetInitialTransform(initial_transform)

    # use the function 'SetMetricAsDemons' to be able to perform Demons registration.
    # Be aware that you will need to provide a parameter (the intensity difference threshold) as input:
    # during the registration, intensities are considered to be equal if their difference is less than the given threshold.
    reg_method.SetMetricAsDemons(0.01)

    # evaluate the metrics only in the mask, if provided as an input
    reg_method.SetMetricFixedMask(fix_mask)

    # Multi-resolution framework.
    reg_method.SetShrinkFactorsPerLevel(shrinkFactors = [4,2,1])
    reg_method.SetSmoothingSigmasPerLevel(smoothingSigmas=[8,4,0])

    # set a linear interpolator
    reg_method.SetInterpolator(sitk.sitkLinear)

    # set a gradient descent optimizer
    reg_method.SetOptimizerAsGradientDescent(learningRate=0.5, numberOfIterations=40, convergenceMinimumValue=1e-6,
                                             convergenceWindowSize=10)
    reg_method.SetOptimizerScalesFromPhysicalShift()

    if print_log:
        print("--------")
        print("Demons registration:")
        print('Final metric value: {0}'.format(reg_method.GetMetricValue()))
        print("Optimizer stop condition: {0}".format(reg_method.GetOptimizerStopConditionDescription()))
        print("Number of iterations: {0}".format(reg_method.GetOptimizerIteration()))
        print("--------")
    return reg_method.Execute(sitk.Cast(fix_img, sitk.sitkFloat32), sitk.Cast(mov_img, sitk.sitkFloat32))

def apply_nl_transf(fix_img, mov_img, nl_transf, is_label=False):
    """
    Apply given non-linear transform `nl_xfm` to `im_mov` and return
    the transformed image."""
    if is_label:
        output = sitk.Resample(mov_img, fix_img, nl_transf, sitk.sitkNearestNeighbor, 0.0, mov_img.GetPixelID())
    else:
        output = sitk.Resample(mov_img, fix_img, nl_transf, sitk.sitkLinear, 0.0, mov_img.GetPixelID())
    return output