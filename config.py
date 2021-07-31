"""
Project: Synthetic NF1 MRI Images (syn26010238)
Team: DCD (3430042)
Competition: Hack4Rare 2021
Description: module to store configuration
"""


from enum import Enum
from os.path import isdir

import enum_handler


class Folder(Enum):
    """
    Provide constants for directory environment
    ===========================================

    Notes
    -----
        The meaning of nomenclature.
            ROOT : the root directory of data that contains at least two folders
                   (1) for the imaging data and (2) is for the reports
            CASES : directory for patients' cases
            REPORT : directory for .xls report files
            IMG_ROOT : root directory for train images
            IMG_ORIGINAL : directory of original image slcies
            IMG_NORMALIZED : directory of normalized image slcies
            MODEL_ROOT : root directory for model images
            DISCRIMINATOR : discriminator's model directory
            GENERATOR : Generator's model directory
            SAMPLES : directory for image samples
    """

    ROOT = './data'
    CASES = ROOT + '/Imaging Data'
    REPORT = ROOT + '/report'
    SEGMENTATIONS = REPORT + '/segmentation_50cases'
    IMG_ROOT = './img'
    IMG_ORIGINAL = IMG_ROOT + '/original'
    IMG_NORMALIZED = IMG_ROOT + '/normalized'
    MODEL_ROOT = './model'
    DISCRIMINATOR = MODEL_ROOT + '/discriminator'
    GENERATOR = MODEL_ROOT + '/generator'
    SAMPLES = './samples'


class Run(Enum):
    """
    Provide constants for the main behaviour
    ========================================

    Notes
    -----
        The meaning of nomenclature.
            BEGIN_CHECK : whether to run checking method at the beginning
            VERBOSE : whether to print messages or to remain in silent
    """

    BEGIN_CHECK = True
    VERBOSE = True


if __name__ == '__main__':
    if Run.VERBOSE:
        print('[!] This is a module for configuration, not a script.')
else:
    if Run.BEGIN_CHECK:
        # Check the existence of neccessary folders
        for folder, path in enum_handler.get_contents_from_enum(Folder).items():
            if isdir(path):
                if Run.VERBOSE:
                    print('[+] Folder {} is OK at `{}` path.'
                         .format(folder, path))
            else:
                if Run.VERBOSE:
                    print('[!] Folder {} is missing at {} path.'
                         .format(folder, path))
