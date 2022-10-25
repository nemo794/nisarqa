import numpy as np
import pytest

import nisarqa

objects_to_skip = nisarqa.get_all(__name__)

# Tolerances for floating point errors
# Format follows order of inputs for 
# numpy.testing.assert_allclose: (<rtol>, <atol>)
tol = (1e-6, 1e-6)

def test_multilook_unequal_nlook1():
    '''
    Tests if the simplest version of multilooking is implemented correctly.
    Dimensions of array are even multiples of the nlooks values.
    Axis 1 nlook value is larger than Axis 0 nlook value.
    '''
    arr = np.arange(1,25, dtype=np.float32).reshape((4,6))

    expected_out = np.array([[5,8],
                                [17,20]])
    actual_out = nisarqa.multilook(arr, (2,3))

    np.testing.assert_allclose(expected_out, actual_out, *tol)


def test_multilook_unequal_nlook2():
    '''
    Tests if basic multilooking is implemented correctly.
    Dimensions of array are even multiples of the nlooks values.
    Axis 0 nlook value is larger than Axis 1 nlook value.
    '''
    arr = np.arange(1,25, dtype=np.float32).reshape((4,6))

    expected_out = np.array([[4,5,6,7,8,9],
                                [16,17,18,19,20,21]])
    actual_out = nisarqa.multilook(arr, (2,1))

    np.testing.assert_allclose(expected_out, actual_out, *tol)


def test_multilook_uneven_nlook_axis1():
    '''
    Tests if basic multilooking is implemented correctly.
    Axis 0 of array is an even multiple of the Axis 0 nlook value.
    Axis 1 of array is less than an even multiple of the Axis 1 nlook value.
    '''
    arr = np.arange(1,25, dtype=np.float32).reshape((4,6))

    expected_out = np.array([[5.5],
                                [17.5]])

    # Catch the RuntimeWarning that should be thrown
    with pytest.warns(RuntimeWarning) as record:
        actual_out = nisarqa.multilook(arr, (2,4))

    np.testing.assert_allclose(expected_out, actual_out, *tol)

    # Assert that a warning was thrown
    assert len(record) == 1


def test_multilook_uneven_nlook_axis0():
    '''
    Tests if basic multilooking is implemented correctly.
    Axis 0 of array is less than an even multiple of the Axis 0 nlook value.
    Axis 1 of array is an even multiple of the Axis 1 nlook value.
    '''
    arr = np.arange(1,25, dtype=np.float32).reshape((4,6))

    expected_out = np.array([[7.5, 9.5, 11.5]], dtype=np.float32)

    # Catch the RuntimeWarning that should be thrown
    with pytest.warns(RuntimeWarning) as record:
        actual_out = nisarqa.multilook(arr, (3,2))

    np.testing.assert_allclose(expected_out, actual_out, *tol)

    # Assert that a warning was thrown
    assert len(record) == 1


def test_multilook_too_big_nlook():
    '''
    Tests if error is raised if invalid nlooks value is provided.
    Axis 0 of array is less than the Axis 0 nlook value.
    '''
    arr = np.arange(1,25, dtype=np.float32).reshape((4,6))

    with pytest.raises(ValueError) as excinfo:
        nisarqa.multilook(arr, (5,2))

    # Assert that the correct error was thrown
    assert 'ValueError' in str(excinfo.type)


def test_multilook_consistent_dtype():
    '''
    Tests if multilooking retains the same datatype as input array.
    '''
    arr = np.arange(1,25, dtype=np.float16).reshape((4,6))

    actual_out = nisarqa.multilook(arr, (2,3))

    assert arr.dtype == actual_out.dtype


def test_multilook_wrong_data_type():
    '''
    Tests if multilook rejects invalid datatypes.
    '''
    arr = np.arange(1,25, dtype=np.int16).reshape((4,6))

    with pytest.raises(TypeError) as excinfo:
        nisarqa.multilook(arr, (5,2))

    # Assert that the correct error was thrown
    assert 'TypeError' in str(excinfo.type)


def test_multilook_zeros():
    '''
    Tests if multilooking handles zeros in the input array correctly.
    '''
    arr = np.arange(1,25, dtype=np.float32).reshape((4,6))
    arr[2,2] = 0.0

    expected_out = np.array([[5, 8],
                                [14.5, 20]])

    actual_out = nisarqa.multilook(arr, (2,3))

    np.testing.assert_allclose(expected_out, actual_out, *tol)


def test_multilook_nans():
    '''
    Tests if multilooking handles zeros in the input array correctly.
    '''
    arr = np.arange(1,25, dtype=np.float32).reshape((4,6))
    arr[2,2] = np.nan

    expected_out = np.array([[5, 8],
                                [np.nan, 20]])

    actual_out = nisarqa.multilook(arr, (2,3))

    np.testing.assert_allclose(expected_out, actual_out, *tol)


def test_multilook_negatives():
    '''
    Tests if multilooking handles values in the input array correctly.
    '''
    # TODO - what should this do??

    arr = np.arange(1,9, dtype=np.float32)

    # Make every-other element negative
    arr[::2] = arr[::2] * -1

    arr = arr.reshape((2,4))

    expected_out = np.array([[0.5, 0.5]])

    actual_out = nisarqa.multilook(arr, (2,2))

    np.testing.assert_allclose(expected_out, actual_out, *tol)

__all__ = nisarqa.get_all(__name__, objects_to_skip)
