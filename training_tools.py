"""
Project: Synthetic NF1 MRI Images (syn26010238)
Team: DCD (3430042)
Competition: Hack4Rare 2021
Description: this module contains functions to help training of the models
"""


from collections.abc import Iterable
from random import uniform

import torch


def weight_init(m : torch.nn.Module, mean : float = 0.0, std : float = 1.0):
    """
    Intialize weight and bias data of a layer
    =========================================

    Parameters
    ----------
    m : torch.nn.Module
        Layer to initialize.
    mean : float, optional (0.0 if omitted)
        Mean to be applied.
    std : float = (1.0 if omitted)
        Standard deviation to be applied.
    """

    if isinstance(m, torch.nn.Conv2d):
        torch.nn.init.normal_(m.weight.data, mean, std)
        if m.bias is not None:
            torch.nn.init.constant_(m.bias.data, 0)
    elif isinstance(m, torch.nn.BatchNorm2d):
        torch.nn.init.normal_(m.weight.data, mean, std)
        if m.bias is not None:
            torch.nn.init.constant_(m.bias.data, 0)
    elif isinstance(m, torch.nn.Linear):
        torch.nn.init.normal_(m.weight.data, mean, std)
        if m.bias is not None:
            torch.nn.init.constant_(m.bias.data, 0)


def weight_noise(m : torch.nn.Module, min : float = 0.0, max : float = 1.0):
    """
    Add noise on  layer
    ===================

    Parameters
    ----------
    m : torch.nn.Module
        Layer to add noise to.
    min : float, optional (0.0 if omitted)
        Minimum noise value.
    max : float, optional (1.0 if omitted)
        Maximum noise value.
    """

    if isinstance(m, torch.nn.Conv2d):
        m.weight.data += uniform(min, max)
        if m.bias is not None:
            m.bias.data += uniform(min, max)
    elif isinstance(m, torch.nn.BatchNorm2d):
        m.weight.data += uniform(min, max)
        if m.bias is not None:
            m.bias.data += uniform(min, max)
    elif isinstance(m, torch.nn.Linear):
        m.weight.data += uniform(min, max)
        if m.bias is not None:
            m.bias.data += uniform(min, max)


def yield_batches(x : Iterable, y : Iterable, batch_size : int,
                  drop_last : bool =False) -> tuple:
    """
    Yield batches of data
    =====================

    Parameters
    ----------
    x : Iterable
        Inputs data.
    y : Iterable
        Targets data.
    batch_size : int
        Number of data elements in a batch.
    drop_last : bool, optional (False if omitted)
        Whether to drop or not the last element of a batch.

    Yields
    ------
    tuple(Iterable, Iterable)
        Tuple of slices of x and y elements.

    Raises
    ------
    AssertionError
        When the lengths of X and Y are not equal.
    """

    assert len(x) == len(y), 'X and Y must have the same length'
    pos = 0
    total_len = len(x)
    while pos + batch_size < total_len:
        yield x[pos:pos + batch_size], y[pos:pos + batch_size]
        pos += batch_size
    if pos < total_len and not drop_last:
        yield x[pos:], y[pos:]

def yield_x_batches(x : Iterable, batch_size : int,
                    drop_last : bool =False) -> Iterable:
    """
    Yield batches of data
    =====================

    Parameters
    ----------
    x : Iterable
        Inputs data.
    batch_size : int
        Number of data elements in a batch.
    drop_last : bool, optional (False if omitted)
        Whether to drop or not the last element of a batch.

    Yields
    ------
    Iterable
        Slice of x elements.
    """

    pos = 0
    total_len = len(x)
    while pos + batch_size < total_len:
        yield x[pos:pos + batch_size]
        pos += batch_size
    if pos < total_len and not drop_last:
        yield x[pos:]
