# -*- coding: utf-8 -*-

from check_time import validate_time

import numpy as np

import pytest

def test_pass():
    
    ndata = 100
    spacing = 0.5
    time = np.arange(0, ndata)*spacing
    validate_time(time, spacing)

def test_fail1():
    
    ndata = 100
    spacing = 0.5
    time = np.arange(0, ndata)*spacing
    time[-1] = time[-2] - spacing
    with pytest.raises(AssertionError):
        validate_time(time, spacing)

def test_fail2():
    
    ndata = 100
    spacing = 0.5
    time = np.arange(0, ndata)*spacing
    time[-1] = time[-2] + 2*spacing
    with pytest.raises(AssertionError):
        validate_time(time, spacing)


    
