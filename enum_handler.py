"""
Project: Synthetic NF1 MRI Images (syn26010238)
Team: DCD (3430042)
Competition: Hack4Rare 2021
Description: module to deal with enums a bit more easily
"""


from enum import Enum


def get_contents_from_enum(enum : Enum, apply : str = None) -> dict:
    """
    Get content of an enum as dict
    ==============================

    Parameters
    ----------
    enum : Enum
        Enum to extract data from.
    apply : str, optional (None if omitted)
        If apply is set, an operation is applied. The only implemented operation
        is 'counter'.

    Returns
    -------
    dict
        The dict with the enum's data.

    Notes
    -----
        Supported apply Parameters:
            None        Dict keys are enum names, dict values are enum values.
            'counter'   Create a dict, where enum's valuas have their poistion
                        as the key in the dict.
    """

    if apply == 'counter':
        return {int(index) : x.value for index, x in enumerate(enum)}
    else:
        return {x.name : x.value for x in enum}


def get_names_from_enum(enum : Enum) -> list:
    """
    Get available enum names
    ========================

    Parameters
    ----------
    enum : Enum
        Enum to extract data from.

    Returns
    -------
    list
        List of valid names in the enum.
    """

    return [x.name for x in enum]


def get_values_from_enum(enum : Enum) -> list:
    """
    Get stored values of the enum
    =============================

    Parameters
    ----------
    enum : Enum
        Enum to extract data from.

    Returns
    -------
    list
        List of stored values in the enum.
    """


    return [x.value for x in enum]
