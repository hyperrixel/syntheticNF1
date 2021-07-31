"""
Project: Synthetic NF1 MRI Images (syn26010238)
Team: DCD (3430042)
Competition: Hack4Rare 2021
Description: module to store global constants
"""


from enum import Enum


class Columns(Enum):
    """
    Provide constants for the columns header contents
    =================================================

    See Also
    --------
        Possible location of the tumor : class Sites(Enum)
        Possible physical form of the tumor : class Types(Enum)


    Notes
    -----
        The meaning of nomenclature.
            SUBJECT : ID of the patient
            SITE : Location of the tumor
            TYPE : Form of the tumor
            VOLUME : Physical size of the tumor
            COMMENT : Any additional information
    """

    SUBJECT = 'Subject'
    SITE = 'Site'
    TYPE = 'Type'
    VOLUME = 'Volume'
    COMMENT = 'Comment'


class Sites(Enum):
    """
    Provide constants forthe tumor's location
    =========================================

    """

    HEAD_NECK = 'head/neck'
    THORAX = 'thorax'
    ABDOMEN = 'abdomen'
    PELVIS = 'pelvis'
    RIGHT_ARM = 'right arm'
    LEFT_ARM = 'left arm'
    RIGHT_LEG = 'right leg'
    LEFT_LEG = 'left leg'
    THORAX_AND_RIGHT_ARM = 'thorax and right arm'
    THORAX_AND_LEFT_ARM = 'thorax and left arm'
    PELVIS_AND_RIGHT_LEG = 'pelvis and right leg'
    PELVIS_AND_LEFT_LEG = 'pelvis and left leg'


class Types(Enum):
    """
    Provide constants for the tumor's location
    ==========================================

    """

    DISCRETE = 'discrete'
    PLEXIFORM = 'plexiform'


class Ranges(Enum):
    """
    Provide constants for the different range categories
    ====================================================

    """

    VOLUME = 2
    MAX_VOLUME = 50
