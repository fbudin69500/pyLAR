# Library: pyLAR
#
# Copyright 2014 Kitware Inc. 28 Corporate Drive,
# Clifton Park, NY, 12065, USA.
#
# All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

""" pyLAR Image Registration Related Functions.

Requires to build the following toolkits first:
    BRAINSTools: http://brainsia.github.io/BRAINSTools
    ANTS: http://stnava.github.io/ANTs
    ITKUtils: https://github.com/XiaoxiaoLiu/ITKUtils

    List of the binaries used in this module:
    * BRAINSFit (BRAINSTools package)
    * ANTS (ANTS package)
    * WarpImageMultiTransform (ANTS package)
    * CreateJacobianDeterminantImage (ANTS package)
    * BRAINSDemonWarp (BRAINSTools package)
    * ComposeMultiTransform (BRAINSTools package)
    * BSplineDeformableRegistration
    * BRAINSResample (BRAINSTools package)
    * AverageImages (ANTS package)
    * InvertDeformationField [1]

    [1] https://github.com/XiaoxiaoLiu/ITKUtils
"""

import subprocess
import os

__status__ = "Development"


def _execute(cmd, verbose, log_file, EXECUTE=True):
    if verbose:
        print cmd
    if EXECUTE:
        tempFile = open(log_file, 'w')
        process = subprocess.Popen(cmd, stdout=tempFile, shell=True)
        process.wait()
        tempFile.close()

def AffineReg(EXE_BRAINSFit, fixedIm, movingIm, outputIm, outputTransform=None, verbose=False):
    """ Computes an affine registration using BRAINSFit

    Parameters
    ----------
    EXE_BRAINSFit: Path to BRAINSFit executable.
    fixedIm: fixed image file name used for the registration.
    movingIm: moving image file name used for the registration.
    outputIm: output image file name.
    outputTransform: output transform file name.
    verbose: displays additional information.

    Returns
    -------
    cmd: returns the command line that has been executed.

    """
    executable = EXE_BRAINSFit
    if not outputTransform:
        outputTransform = outputIm + '.tfm'

    result_folder = os.path.dirname(outputIm)
    arguments = ' --fixedVolume  ' + fixedIm \
                + ' --movingVolume ' + movingIm \
                + ' --outputVolume ' + outputIm \
                + ' --linearTransform ' + outputTransform \
                + ' --initializeTransformMode  useMomentsAlign --useAffine --samplingPercentage 0.1   \
                  --numberOfIterations 1500 --maskProcessingMode NOMASK --outputVolumePixelType float \
                  --backgroundFillValue 0 --maskInferiorCutOffFromCenter 1000 --interpolationMode Linear \
                  --minimumStepLength 0.005 --translationScale 1000 --reproportionScale 1 --skewScale 1 \
                  --numberOfHistogramBins 50 --numberOfMatchPoints 10 --fixedVolumeTimeIndex 0 \
                  --movingVolumeTimeIndex 0 --medianFilterSize 0,0,0 --removeIntensityOutliers 0 \
                  --ROIAutoDilateSize 0 --ROIAutoClosingSize 9 --relaxationFactor 0.5 --maximumStepLength 0.2 \
                  --failureExitCode -1 --numberOfThreads -1 --debugLevel 0 --costFunctionConvergenceFactor 1e+09 \
                  --projectedGradientTolerance 1e-05 --costMetric MMI'
    cmd = executable + ' ' + arguments
    _execute(cmd, verbose, os.path.join(result_folder, 'bspline.log'))
    return cmd


def RigidReg(EXE_BRAINSFit, fixedIm, movingIm, outputIm, outputTransform=None, verbose=False):
    """ Computes a rigid registration using BRAINSFit

    Parameters
    ----------
    EXE_BRAINSFit: Path to BRAINSFit executable.
    fixedIm: fixed image file name used for the registration.
    movingIm: moving image file name used for the registration.
    outputIm: output image file name.
    outputTransform: output transform file name.
    verbose: displays additional information.

    Returns
    -------
    cmd: returns the command line that has been executed.

    """
    executable = EXE_BRAINSFit
    if not outputTransform:
        outputTransform = outputIm + '.tfm'

    result_folder = os.path.dirname(outputIm)
    arguments = ' --fixedVolume  ' + fixedIm \
                + ' --movingVolume ' + movingIm \
                + ' --outputVolume ' + outputIm \
                + ' --linearTransform ' + outputTransform \
                + ' --initializeTransformMode  useMomentsAlign --useRigid --samplingPercentage 0.1   \
                  --numberOfIterations 1500 --maskProcessingMode NOMASK --outputVolumePixelType float \
                  --backgroundFillValue 0 --maskInferiorCutOffFromCenter 1000 --interpolationMode Linear \
                  --minimumStepLength 0.005 --translationScale 1000 --reproportionScale 1 --skewScale 1 \
                  --numberOfHistogramBins 50 --numberOfMatchPoints 10 --fixedVolumeTimeIndex 0 \
                  --movingVolumeTimeIndex 0 --medianFilterSize 0,0,0 --removeIntensityOutliers 0 \
                  --ROIAutoDilateSize 0 --ROIAutoClosingSize 9 --relaxationFactor 0.5 --maximumStepLength 0.2 \
                  --failureExitCode -1 --numberOfThreads -1 --debugLevel 0 --costFunctionConvergenceFactor 1e+09 \
                  --projectedGradientTolerance 1e-05 --costMetric MMI'
    cmd = executable + ' ' + arguments
    _execute(cmd, verbose, os.path.join(result_folder, 'affine_reg.log'))
    return cmd


def ANTS(EXE_ANTS, fixedIm, movingIm, outputTransformPrefix, params, initialTransform=None, EXECUTE=False,
         verbose=False):
    """ Computes a registration using ANTS.

    Parameters
    ----------
    EXE_ANTS: Path to ANTS executable.
    fixedIm: fixed image file name used for the registration.
    movingIm: moving image file name used for the registration.
    outputTransformPrefix: output prefix used to name output image and output transform files.
    params: registration parameters. It should contain:
        * 'Dimension', i.e. 3
        * 'Convergence', i.e. "[100x70x50x20,1e-6,10]"
        * 'ShrinkFactors', i.e. "8x4x2x1"
        * 'SmoothingSigmas', i.e. "3x2x1x0vox"
        * 'Transform', i.e. "SyN[0.25]"
        * 'Metric', i.e. "MeanSquares[fixedIm,movingIm,1,0]"
    initialTransform: initial transform file to use to initialize the registration.
    verbose: display additional information.

    Returns
    -------
    cmd: returns the command line that has been executed.

    """
    executable = EXE_ANTS

    dim = params['Dimension']
    CONVERGENCE = params['Convergence']  # "[100x70x50x20,1e-6,10]"
    SHRINKFACTORS = params['ShrinkFactors']  # "8x4x2x1"
    SMOOTHINGSIGMAS = params['SmoothingSigmas']  # "3x2x1x0vox"
    TRANSFORM = params['Transform']  # "SyN[0.25]"
    METRIC = params['Metric']  # "MI[%s,%s, 1,50]" %(fixedIm,movingIm)
    METRIC = METRIC.replace('fixedIm', fixedIm)
    METRIC = METRIC.replace('movingIm', movingIm)
    # Parameter Notes From ANTS/Eaxmples/antsRegistration.cxx:
    # "CC[fixedImage,movingImage,metricWeight,radius,<samplingStrategy={None,Regular,Random}>,<samplingPercentage=[0,1]>]" );
    # "Mattes[fixedImage,movingImage,metricWeight,numberOfBins,<samplingStrategy={None,Regular,Random}>,<samplingPercentage=[0,1]>]" );
    # "Demons[fixedImage,movingImage,metricWeight,radius=NA,<samplingStrategy={None,Regular,Random}>,<samplingPercentage=[0,1]>]" );
    # option->SetUsageOption( 10, "SyN[gradientStep,updateFieldVarianceInVoxelSpace,totalFieldVarianceInVoxelSpace]" );
    arguments = ' --dimensionality ' + str(dim) \
                + ' --float 1' \
                + ' --interpolation Linear' \
                + ' --output [%s,%sWarped.nrrd]' % (outputTransformPrefix, outputTransformPrefix) \
                + ' --interpolation Linear' \
                + ' --transform ' + TRANSFORM \
                + ' -m ' + METRIC \
                + ' --convergence ' + CONVERGENCE \
                + ' --shrink-factors ' + SHRINKFACTORS \
                + ' --smoothing-sigmas ' + SMOOTHINGSIGMAS
    #            +' --use-histogram-match'
    if initialTransform:
        arguments += ' --initial-moving-transform  %s' % (initialTransform)
    cmd = executable + ' ' + arguments
    _execute(cmd, verbose, outputTransformPrefix + 'ANTS.log', EXECUTE)
    return cmd


def getANTSOutputVelocityNorm(logFile):
    """ Utility function to extract the integrated velocity field norm from the ANTS log file.

    Parameters
    ----------
    logFile: file to extract the velocity norm from.

    Returns
    -------
    vn: velocity norm.

    """
    # spatio-temporal velocity field norm : 2.8212e-02
    STRING = "    spatio-temporal velocity field norm : "
    vn = 0.0
    f = open(logFile, 'r')
    for line in f:
        if line.find(STRING) > -1:
            vn = float(line.split(STRING)[1].split()[0][0:-1])
    return vn


def geodesicDistance3D(EXE_ANTS, inputImage, referenceImage, outputTransformPrefix, verbose):
    """ Computes geodesic distance between input image and reference image.

    Parameters
    ----------
    EXE_ANTS: Path to ANTS executable.
    inputImage: Input image file name.
    referenceImage: Reference image file name.
    outputTransformPrefix: output prefix used for output transform files and output image files.
    verbose: Displays additional information.

    Returns
    -------
    geodesicDis: geodesic distance between input image and reference image.
    """
    geodesicDis = -1
    affineParams = {
        'Dimension': 3,
        'Convergence': '[200,1e-6,10]',
        'ShrinkFactors': '1',
        'SmoothingSigmas': '0vox',
        'Transform': 'affine[0.1]',
        'Metric': 'Mattes[fixedIm,movingIm,1,50,Regular,0.95]'
    }
    antsParams = {
        'Dimension': 3,
        'Convergence': '[200x100x50,1e-6,10]',
        'ShrinkFactors': '4x2x1',
        'SmoothingSigmas': '2x1x0vox',
        'Transform': 'TimeVaryingVelocityField[1.0,4,8,0,0,0]',
        'Metric': 'Mattes[fixedIm,movingIm,1,50,Regular,0.95]'
    }
    ANTS(EXE_ANTS, referenceImage, inputImage, outputTransformPrefix + '_affine', affineParams, None, True, verbose)
    outputIm = outputTransformPrefix + '_affineWarped.nrrd'
    if os.path.isfile(outputIm):
        ANTS(EXE_ANTS, referenceImage, inputImage, outputTransformPrefix, antsParams, None, True, verbose)
        logFile = outputTransformPrefix + 'ANTS.log'
        geodesicDis = getANTSOutputVelocityNorm(logFile)
    else:
        print "affine registraion failed: no affine results are generated"
    return geodesicDis


def ANTSWarpImage(EXE_WarpImageMultiTransform, inputIm, outputIm, referenceIm,
                  transformPrefix, inverse=False, EXECUTE=False, verbose=False
                  ):
    """ Transforms input image using given transform and reference space.

    Parameters
    ----------
    EXE_WarpImageMultiTransform: Path to WarpImageMultiTransform executable.
    inputIm
    outputIm
    referenceIm
    transformPrefix
    inverse
    EXECUTE
    verbose

    Returns
    -------

    """
    dim = 3
    executable = EXE_WarpImageMultiTransform
    result_folder = os.path.dirname(outputIm)
    if not inverse:
        t = transformPrefix + '0Warp.nii.gz'
    else:
        t = transformPrefix + '0InverseWarp.nii.gz'
    arguments = str(dim) + ' %s  %s  -R %s %s ' % (inputIm, outputIm, referenceIm, t)
    cmd = executable + ' ' + arguments
    _execute(cmd, verbose, os.path.join(result_folder, 'ANTSWarpImage.log'), EXECUTE)
    return cmd


def ANTSWarp2DImage(EXE_WarpImageMultiTransform, inputIm, outputIm, referenceIm,
                    transformPrefix, inverse=False, EXECUTE=False, verbose=False
                    ):
    dim = 2
    executable = EXE_WarpImageMultiTransform
    result_folder = os.path.dirname(outputIm)
    if not inverse:
        t = transformPrefix + '0Warp.nii.gz'
    else:
        t = transformPrefix + '0InverseWarp.nii.gz'
    arguments = str(dim) + ' %s  %s  -R %s %s ' % (inputIm, outputIm, referenceIm, t)
    cmd = executable + ' ' + arguments
    _execute(cmd, verbose, os.path.join(result_folder, 'ANTSWarpImage.log'), EXECUTE)
    return cmd


def createJacobianDeterminantImage(EXE_CreateJacobianDeterminantImage, imageDimension, dvfImage,
                                   outputIm, EXECUTE=False, verbose=False):
    executable = EXE_CreateJacobianDeterminantImage
    result_folder = os.path.dirname(outputIm)
    arguments = str(imageDimension) + ' %s  %s ' % (dvfImage, outputIm)
    cmd = executable + ' ' + arguments
    _execute(cmd, verbose, os.path.join(result_folder, 'CreateJacobianDeterminantImage.log'), EXECUTE)
    return cmd


def DemonsReg(EXE_BRAINSDemonWarp, fixedIm, movingIm, outputIm, outputDVF, EXECUTE=False, verbose=False):
    executable = EXE_BRAINSDemonWarp
    result_folder = os.path.dirname(movingIm)
    arguments = '--movingVolume ' + movingIm \
                + ' --fixedVolume ' + fixedIm \
                + ' --inputPixelType float ' \
                + ' --outputVolume ' + outputIm \
                + ' --outputDisplacementFieldVolume ' + outputDVF \
                + ' --outputPixelType float ' \
                + ' --interpolationMode Linear --registrationFilterType Diffeomorphic \
       --smoothDisplacementFieldSigma 1 --numberOfPyramidLevels 3 \
       --minimumFixedPyramid 8,8,8 --minimumMovingPyramid 8,8,8 \
       --arrayOfPyramidLevelIterations 300,50,30,20,15 \
       --numberOfHistogramBins 256 --numberOfMatchPoints 2 --medianFilterSize 0,0,0 --maskProcessingMode NOMASK \
--lowerThresholdForBOBF 0 --upperThresholdForBOBF 70 --backgroundFillValue 0 --seedForBOBF 0,0,0 \
--neighborhoodForBOBF 1,1,1 --outputDisplacementFieldPrefix none --checkerboardPatternSubdivisions 4,4,4 \
--gradient_type 0 --upFieldSmoothing 0 --max_step_length 2 --numberOfBCHApproximationTerms 2 --numberOfThreads -1'
    cmd = executable + ' ' + arguments
    _execute(cmd, verbose, os.path.join(result_folder, 'demons.log'), EXECUTE)
    return cmd


def BSplineReg_BRAINSFit(EXE_BRAINSFit, fixedIm, movingIm, outputIm, outputTransform,
                         gridSize=[5, 5, 5], maxDisp=5.0, EXECUTE=False, verbose=False
                         ):
    result_folder = os.path.dirname(movingIm)
    string_gridSize = ','.join([str(gridSize[0]), str(gridSize[1]), str(gridSize[2])])
    executable = EXE_BRAINSFit
    arguments = ' --fixedVolume  ' + fixedIm \
                + ' --movingVolume ' + movingIm \
                + ' --outputVolume ' + outputIm \
                + ' --outputTransform ' + outputTransform \
                + ' --initializeTransformMode Off --useBSpline \
                  --samplingPercentage 0.1  --splineGridSize ' + string_gridSize \
                + ' --maxBSplineDisplacement  ' + str(maxDisp) \
                + ' --numberOfHistogramBins 50  --numberOfIterations 500 --maskProcessingMode NOMASK\
                 --outputVolumePixelType float --backgroundFillValue 0   --numberOfThreads -1 --costMetric MMI'

    cmd = executable + ' ' + arguments
    _execute(cmd, verbose, os.path.join(result_folder, 'bspline.log'), EXECUTE)
    return cmd


def BSplineReg_Legacy(EXE_BSplineDeformableRegistration, fixedIm, movingIm, outputIm, outputDVF, gridSize=5,
                      iterationNum=20, EXECUTE=False, verbose=False):
    result_folder = os.path.dirname(movingIm)
    executable = EXE_BSplineDeformableRegistration
    arguments = '  --iterations ' + str(iterationNum) \
                + ' --gridSize ' + str(gridSize) \
                + ' --histogrambins 100 --spatialsamples 50000 ' \
                + ' --outputwarp ' + outputDVF \
                + ' --resampledmovingfilename ' + outputIm \
                + ' ' + fixedIm \
                + ' ' + movingIm

    cmd = executable + ' ' + arguments
    _execute(cmd, verbose, os.path.join(result_folder, 'bspline.log'), EXECUTE)
    return cmd


def ConvertTransform(EXE_BSplineToDeformationField, fixedIm, outputTransform, outputDVF, EXECUTE=False, verbose=False):
    result_folder = os.path.dirname(outputDVF)
    cmd = EXE_BSplineToDeformationField \
          + ' --tfm ' + outputTransform \
          + ' --refImage ' + fixedIm \
          + ' --defImage ' + outputDVF
    _execute(cmd, verbose, os.path.join(result_folder, 'convertTransform.log'), EXECUTE)
    return cmd


def WarpImageMultiDVF(EXE_WarpImageMultiTransform, movingImage, refImage,
                      DVFImageList, outputImage, EXECUTE=False, verbose=False):
    result_folder = os.path.dirname(outputImage)
    string_DVFImageList = ' '.join(DVFImageList)

    cmd = EXE_WarpImageMultiTransform \
          + ' 3 ' \
          + '  ' + movingImage \
          + '  ' + outputImage \
          + ' -R  ' + refImage \
          + '  ' + string_DVFImageList
    _execute(cmd, verbose, os.path.join(result_folder, 'warpImageMultiDVF.log'), EXECUTE)
    return cmd


def composeMultipleDVFs(EXE_ComposeMultiTransform, refImage, DVFImageList,
                        outputDVFImage, EXECUTE=False, verbose=False):
    result_folder = os.path.dirname(outputDVFImage)
    # T3(T2(T1(I))) = T3*T2*T1(I), need to reverse the sequence of the DVFs
    string_DVFImageList = ' '.join(DVFImageList[::-1])

    cmd = EXE_ComposeMultiTransform \
          + ' 3 ' \
          + '  ' + outputDVFImage \
          + ' -R  ' + refImage \
          + '  ' + string_DVFImageList
    _execute(cmd, verbose, os.path.join(result_folder, 'composeDVF.log'), EXECUTE)
    return cmd


def applyLinearTransform(EXE_BRAINSResample, inputImage, refImage, transform,
                         newInputImage, EXECUTE=False, verbose=False):
    result_folder = os.path.dirname(newInputImage)
    cmd = EXE_BRAINSResample \
          + ' --inputVolume ' + inputImage \
          + ' --referenceVolume ' + refImage \
          + ' --outputVolume ' + newInputImage \
          + ' --pixelType float ' \
          + ' --warpTransform ' + transform \
          + ' --defaultValue 0 --numberOfThreads -1 '
    _execute(cmd, verbose, os.path.join(result_folder, 'applyLinearTransform.log'), EXECUTE)
    return cmd


def updateInputImageWithDVF(EXE_BRAINSResample, inputImage, refImage, DVFImage,
                            newInputImage, EXECUTE=False, verbose=False):
    result_folder = os.path.dirname(newInputImage)
    cmd = EXE_BRAINSResample \
          + ' --inputVolume ' + inputImage \
          + ' --referenceVolume ' + refImage \
          + ' --outputVolume ' + newInputImage \
          + ' --pixelType float ' \
          + ' --deformationVolume ' + DVFImage \
          + ' --defaultValue 0 --numberOfThreads -1 '
    _execute(cmd, verbose, os.path.join(result_folder, 'applyDVF.log'), EXECUTE)
    return cmd


def updateInputImageWithTFM(EXE_BRAINSResample, inputImage, refImage, transform,
                            newInputImage, EXECUTE=False, verbose=False):
    result_folder = os.path.dirname(movingIm)
    cmd = EXE_BRAINSResample \
          + ' --inputVolume ' + inputImage \
          + ' --referenceVolume ' + refImage \
          + ' --outputVolume ' + newInputImage \
          + ' --pixelType float ' \
          + ' --warpTransform ' + transform \
          + ' --defaultValue 0 --numberOfThreads -1 '
    _execute(cmd, verbose, os.path.join(result_folder, 'applyTransform.log'), EXECUTE)
    return cmd


def AverageImages(EXE_AverageImages, listOfImages, outputIm, verbose=False):
    result_folder = os.path.dirname(outputIm)
    arguments = ' 3 ' + outputIm + '  0  ' + ' '.join(listOfImages)
    cmd = EXE_AverageImages + ' ' + arguments
    _execute(cmd, verbose, os.path.join(result_folder, 'average.log'))
    return cmd


def genInverseDVF(EXE_InvertDeformationField, DVFImage, InverseDVFImage, EXECUTE=False, verbose=False):
    result_folder = os.path.dirname(InverseDVFImage)
    cmd = EXE_InvertDeformationField \
          + DVFImage \
          + ' ' \
          + InverseDVFImage
    _execute(cmd, verbose, os.path.join(result_folder, 'InverseDVF.log'))
    return cmd
