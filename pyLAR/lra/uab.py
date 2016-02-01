#!/usr/bin/env python

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

""" Unbiased atlas creation from a selection of images

Command line arguments (See command line help: -h):
---------------------------------------------------
    Required:
        --configFN (string): Parameter configuration file.
        --configSoftware (string): Software configuration file.

Configuration file must contain:
--------------------------------
    fileListFN (string): File containing path to input images.
    data_dir (string): Folder containing the "fileListFN" file.
    result_dir (string): output directory where outputs will be saved.
    selection (list): select images that are processed in given list [must contain at least 2 values].
    reference_im_fn (string): reference image used for the registration.
    NUM_OF_ITERATIONS_PER_LEVEL (int): Number of iteration per level for the registration [>=0]
    NUM_OF_LEVELS (int): Number of levels (starting the registration at a down-sampled level) for the registration [>=1]
    antsParams (see example and ANTS documentation):
            antsParams = {'Convergence' : '[100x50x25,1e-6,10]',\
                  'Dimension': 3,\
                  'ShrinkFactors' : '4x2x1',\
                  'SmoothingSigmas' : '2x1x0vox',\
                  'Transform' :'SyN[0.5]',\
                  'Metric': 'Mattes[fixedIm,movingIm,1,50,Regular,0.95]'}

Optional for 'set_and_run'/required for 'run_low_rank':
----------------------------------------------------
    verbose (boolean): If not specified or set to False, outputs are written in a log file.

Configuration Software file must contain:
-----------------------------------------
    EXE_BRAINSFit (string): Path to BRAINSFit executable (BRAINSFit package)
    EXE_AverageImages (string): Path to AverageImages executable (ANTS package)
    EXE_ANTS (string): Path to ANTS executable (ANTS package)
    EXE_WarpImageMultiTransform (string): path to WarpImageMultiTransform (ANTS package)
"""

import pyLAR
import shutil
import os
import gc
import subprocess
import time


def _runIteration(level, currentIter, antsParams, result_dir, selection, software, verbose):
    """Iterative Atlas-to-image registration"""

    EXE_AverageImages = software.EXE_AverageImages
    EXE_ANTS = software.EXE_ANTS
    EXE_WarpImageMultiTransform = software.EXE_WarpImageMultiTransform
    # average the images to produce the Atlas
    prefix = 'L' + str(level) + '_Iter'
    prev_prefix = prefix + str(currentIter-1)
    prev_iter_path = os.path.join(result_dir, prev_prefix)
    current_prefix = prefix + str(currentIter)
    current_prefix_path = os.path.join(result_dir, current_prefix)
    atlasIm = prev_iter_path + '_atlas.nrrd'
    listOfImages = []
    num_of_data = len(selection)
    for i in range(num_of_data):
        lrIm = prev_iter_path + '_' + str(i) + '.nrrd'
        listOfImages.append(lrIm)
    pyLAR.AverageImages(EXE_AverageImages, listOfImages, atlasIm, verbose=verbose)

    try:
        import matplotlib.pyplot as plt
        import SimpleITK as sitk
        im = sitk.ReadImage(atlasIm)
        im_array = sitk.GetArrayFromImage(im)
        z_dim, x_dim, y_dim = im_array.shape
        plt.figure()
        implot = plt.imshow(im_array[z_dim/2, :, :], plt.cm.gray)
        plt.title(prev_prefix+ ' atlas')
        plt.savefig(os.path.join(result_dir, 'atlas_' + prev_prefix + '.png'))
    except ImportError:
        pass
    reference_im_fn = atlasIm

    ps = [] # to use multiple processors
    for i in range(num_of_data):
        logFile = open(current_prefix_path + '_RUN_' + str(i) + '.log', 'w')
        cmd = ''
        initialInputImage= os.path.join(result_dir, prefix + '0_' + str(i) + '.nrrd')
        newInputImage = current_prefix_path + '_' + str(i) + '.nrrd'

        # Will generate a warp(DVF) file and an affine file
        outputTransformPrefix = current_prefix_path + '_' + str(i) + '_'
        fixedIm = atlasIm
        movingIm = initialInputImage
        cmd += pyLAR.ANTS(EXE_ANTS, fixedIm, movingIm, outputTransformPrefix, antsParams, verbose=verbose)
        cmd += ";" + pyLAR.ANTSWarpImage(EXE_WarpImageMultiTransform, initialInputImage,\
                                         newInputImage, reference_im_fn, outputTransformPrefix, verbose=verbose)
        print cmd
        process = subprocess.Popen(cmd, stdout=logFile, shell=True)
        ps.append(process)
    for p in ps:
        p.wait()
    return


def run(config, software, im_fns, check=True, verbose=True):
    """Unbiased atlas building - Atlas-to-image registration"""
    if check:
        check_requirements(config, software, verbose=verbose)
    reference_im_fn = config.reference_im_fn
    selection = config.selection
    result_dir = config.result_dir
    antsParams = config.antsParams
    NUM_OF_ITERATIONS_PER_LEVEL = config.NUM_OF_ITERATIONS_PER_LEVEL
    NUM_OF_LEVELS = config.NUM_OF_LEVELS  # multiscale bluring (coarse-to-fine)
    s = time.time()

    pyLAR.affineRegistrationStep(software.EXE_BRAINSFit, im_fns, result_dir, selection, reference_im_fn, verbose)
    #cnormalizeIntensityStep()
    #histogramMatchingStep()

    num_of_data = len(selection)
    iterCount = 0
    for level in range(0, NUM_OF_LEVELS):
        for iterCount in range(1, NUM_OF_ITERATIONS_PER_LEVEL+1):
            print 'Level: ', level
            print 'Iteration ' + str(iterCount)
            _runIteration(level, iterCount, antsParams, result_dir, selection, software, verbose)
            gc.collect()  # garbage collection
        # We need to check if NUM_OF_ITERATIONS_PER_LEVEL is set to 0, which leads
        # to computing an average on the affine registration.
        if level != NUM_OF_LEVELS - 1:
            print 'WARNING: No need for multiple levels! TO BE REMOVED!'
            for i in range(num_of_data):
                current_file_name = 'L' + str(level) + '_Iter' + str(iterCount) + '_' + str(i) + '.nrrd'
                current_file_path = os.path.join(result_dir, current_file_name)
                nextLevelInitIm = os.path.join(result_dir, 'L'+str(level+1)+'_Iter0_' + str(i) + '.nrrd')
                shutil.copyfile(current_file_path, nextLevelInitIm)
        # if NUM_OF_LEVELS > 1:
        #     print 'WARNING: No need for multiple levels! TO BE REMOVED!'
        #     for i in range(num_of_data):
        #         next_prefix = 'L' + str(level+1) + '_Iter0_'
        #         next_path = os.path.join(result_dir, next_prefix)
        #         newLevelInitIm = next_path + str(i) + '.nrrd'
    current_prefix = 'L' + str(NUM_OF_LEVELS-1) + '_Iter' + str(NUM_OF_ITERATIONS_PER_LEVEL)
    current_path = os.path.join(result_dir, current_prefix)
    atlasIm = current_path + '_atlas.nrrd'
    listOfImages = []
    num_of_data = len(selection)
    for i in range(num_of_data):
        lrIm = current_path + '_' + str(i) + '.nrrd'
        listOfImages.append(lrIm)
    pyLAR.AverageImages(software.EXE_AverageImages, listOfImages, atlasIm, verbose)
    try:
        import matplotlib.pyplot as plt
        import SimpleITK as sitk
        import numpy as np
        im = sitk.ReadImage(atlasIm)
        im_array = sitk.GetArrayFromImage(im)
        z_dim, x_dim, y_dim = im_array.shape
        plt.figure()
        plt.imshow(np.flipud(im_array[z_dim/2, :]), plt.cm.gray)
        plt.title(current_prefix + ' atlas')
        plt.savefig(current_path + '.png')
    except ImportError:
        pass

    e = time.time()
    l = e - s
    print 'Total running time:  %f mins' % (l/60.0)

def check_requirements(config, software, configFileName=None, softwareFileName=None, verbose=True):
    """Verifying that all options and software paths are set."""
    result_dir = config.result_dir
    required_field = ['reference_im_fn', 'data_dir',
                      'result_dir', 'fileListFN', 'selection',
                      'NUM_OF_ITERATIONS_PER_LEVEL', 'NUM_OF_LEVELS', 'antsParams']
    if not pyLAR.containsRequirements(config, required_field, configFileName):
        raise Exception('Error in configuration file')
    required_software = ['EXE_BRAINSFit', 'EXE_AverageImages', 'EXE_ANTS', 'EXE_WarpImageMultiTransform']
    if not pyLAR.containsRequirements(software, required_software, softwareFileName):
        raise Exception('Error in configuration file')
    if not config.NUM_OF_ITERATIONS_PER_LEVEL >= 0:
        if verbose:
            print '\'NUM_OF_ITERATIONS_PER_LEVEL\' must be a positive integer (>=0).'
        raise Exception('Error in configuration file')
    if not config.NUM_OF_LEVELS >= 1:
        if verbose:
            print '\'NUM_OF_LEVELS\' must be a strictly positive integer (>=1).'
        raise Exception('Error in configuration file')
    if len(config.selection) < 2:
        if verbose:
            print '\'selection\' must contain at least two values.'
        raise Exception('Error in configuration file')
    if verbose:
        print 'Results will be stored in:', result_dir
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)