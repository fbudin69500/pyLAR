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

"""Low rank decomposition of a set of images

Command line arguments (See command line help: -h):
---------------------------------------------------
    Required:
        --configFN (string): Parameter configuration file.
        --configSoftware (string): Software configuration file.
    Optional:
        --HistogramMatching (boolean) that overwrites configuration file histogram matching parameter.

Configuration file must contain:
--------------------------------
    lamda (float): the tuning parameter that weights between the low-rank component and the sparse component.
    sigma (float): blurring kernel size.
    fileListFN (string): File containing path to input images.
    data_dir (string): Folder containing the "fileListFN" file.
    result_dir (string): output directory where outputs will be saved.
    selection (list): select images that are processed in given list [must contain at least 1 value].
    reference_im_fn (string): reference image used for the registration.
    registration (string): 'affine' or 'rigid'

Optional for 'set_and_run'/required for 'run_low_rank':
----------------------------------------------------
    HistogramMatching (boolean): If not specified or set to False, no histogram matching performed.
    verbose (boolean): If not specified or set to False, outputs are written in a log file.

Configuration Software file must contain:
-----------------------------------------
    EXE_BRAINSFit (string): Path to BRAINSFit executable (BRAINSFit package)
"""

import sys
import os
import shutil
import numpy as np
import SimpleITK as sitk
import pyLAR
import argparse
import time


def run_low_rank(config, software, im_fns, verbose=True):
    """Low-rank decomposition."""
    # Initialize variables
    selection = config.selection
    result_dir = config.result_dir
    sigma = config.sigma
    reference_im_fn = config.reference_im_fn
    num_of_data = len(selection)
    # Pre-processing: registration and histogram matching
    s = time.time()
    if config.registration == 'affine':
        if verbose:
            print 'Affine registration'
        pyLAR.affineRegistrationStep(software.EXE_BRAINSFit, im_fns, result_dir, selection, reference_im_fn, verbose)
    elif config.registration == 'rigid':
        if verbose:
            print 'Rigid registration'
        pyLAR.rigidRegistrationStep(software.EXE_BRAINSFit, im_fns, result_dir, selection, reference_im_fn, verbose)
    else:
        raise Exception('Unknown registration')
    if config.HistogramMatching:
        pyLAR.histogramMatchingStep(selection, result_dir)

    e = time.time()
    l = e - s
    if verbose:
        print 'Preprocessing - total running time:  %f mins' % (l / 60.0)

    # Loading images and blurring them if option selected.
    s = time.time()
    im_ref = sitk.ReadImage(reference_im_fn)
    im_ref_array = sitk.GetArrayFromImage(im_ref)
    z_dim, x_dim, y_dim = im_ref_array.shape
    vector_length = z_dim * x_dim * y_dim
    del im_ref, im_ref_array
    Y = np.zeros((vector_length, num_of_data))
    for i in range(num_of_data):
        im_file = os.path.join(result_dir, 'L0_Iter0_' + str(i) + '.nrrd')
        if verbose:
            print "Input File: " + im_file
        inIm = sitk.ReadImage(im_file)
        tmp = sitk.GetArrayFromImage(inIm)
        if sigma > 0:  # blurring
            if verbose:
                print "Blurring: " + str(sigma)
            outIm = pyLAR.GaussianBlur(inIm, None, sigma)
            tmp = sitk.GetArrayFromImage(outIm)
        Y[:, i] = tmp.reshape(-1)
        del tmp
    # Low-Rank and sparse decomposition
    low_rank, sparse, n_iter, rank, sparsity, sum_sparse = pyLAR.rpca(Y, config.lamda)
    pyLAR.saveImagesFromDM(low_rank, os.path.join(result_dir, 'L' + '_LowRank_'), reference_im_fn)
    pyLAR.saveImagesFromDM(sparse, os.path.join(result_dir, 'L' + '_Sparse_'), reference_im_fn)
    e = time.time()
    l = e - s
    if verbose:
        print "Rank: " + str(rank)
        print "Sparsity: " + str(sparsity)
        print 'Processing - total running time:  %f mins' % (l / 60.0)
    return sparsity, sum_sparse


def setup_and_run(config, software, im_fns, configFN="", configSoftware="", fileListFN=""):
    """Setting up processing:

    -Setting up options.
    -Verifying that all options and software paths are set.
    -Saving parameters in output folders for reproducibility.
    """
    if not pyLAR.containsRequirements(software, ['EXE_BRAINSFit'], configSoftware):
        return 1
    required_field = ['reference_im_fn', 'result_dir', 'selection', 'lamda', 'sigma', 'registration']
    if not pyLAR.containsRequirements(config, required_field, configFN):
        return 1
    result_dir = config.result_dir
    if not hasattr(config, "HistogramMatching"):
        config.HistogramMatching = False
    if config.HistogramMatching:
        print "Script will perform histogram matching."
    if len(config.selection) < 1:
        print '\'selection\' must contain at least one value.'
        return 3
    print 'Results will be stored in: ' + result_dir
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    # For reproducibility: save all parameters into the result dir
    savedFileName = lambda name, default: os.path.basename(name) if name else default
    configFN = savedFileName(configFN, 'Config.txt')
    pyLAR.saveConfiguration(os.path.join(result_dir, configFN), config)
    configSoftware = savedFileName(configSoftware, 'Software.txt')
    pyLAR.saveConfiguration(os.path.join(result_dir, configSoftware), software)
    fileListFN = savedFileName(fileListFN, 'listFiles.txt')
    pyLAR.writeTxtIntoList(os.path.join(result_dir, fileListFN), im_fns)
    currentPyFile = os.path.realpath(__file__)
    shutil.copy(currentPyFile, result_dir)
    # Start processing
    if not(hasattr(config, "verbose") and config.verbose):
        sys.stdout = open(os.path.join(result_dir, 'RUN.log'), "w")
    run_low_rank(config, software, im_fns, True)


def main(argv=None):
    """Parsing command line arguments and reading input files."""
    if argv is None:
        argv = sys.argv
    parser = argparse.ArgumentParser(
            prog=argv[0],
            description=__doc__
    )
    parser.add_argument('-c', "--configFN", required=True, help="Parameter configuration file")
    parser.add_argument('-s', "--configSoftware", required=True, help="Software configuration file")
    parser.add_argument('-m', "--HistogramMatching", action='store_true',
                        help="overwrites configuration file histogram matching parameter")
    args = parser.parse_args(argv[1:])
    # Assign parameters from the input config txt file
    configFN = args.configFN
    config = pyLAR.loadConfiguration(configFN, 'config')

    # Load software paths from file
    configSoftware = args.configSoftware
    software = pyLAR.loadConfiguration(configSoftware, 'software')
    if args.HistogramMatching:
        config.HistogramMatching = True

    if not pyLAR.containsRequirements(config, ['data_dir', 'fileListFN'], configFN):
        return 1
    data_dir = config.data_dir
    fileListFN = config.fileListFN
    im_fns = pyLAR.readTxtIntoList(os.path.join(data_dir, fileListFN))

    setup_and_run(config, software, im_fns, configFN, configSoftware, fileListFN)
    return 0


if __name__ == "__main__":
    sys.exit(main())
