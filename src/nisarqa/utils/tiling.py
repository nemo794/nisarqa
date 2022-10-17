import numpy as np
import itertools
import warnings

import nisarqa

###############################################
######      Main Tiling Functions       #######
###############################################

class TileIterator:
    def __init__(self, arr_shape, col_stride=-1, row_stride=-1):
        '''
        Simple iterator class to iterate over a 1D or 2D array by tiles.

        Parameters
        ----------
        arr_shape : tuple
            The shape of the 1D or 2D array that this TileIterator is for.
        col_stride : int
            Number of columns for each tile
            Defaults to -1, meaning entire rows will be processed.
        row_stride : int, optional
            Number of rows for each tile.
            Defaults to -1, meaning entire columns will be processed.
            To process entire columns in an array, set row_stride = arr_shape[0]
            Will be ignored if arr_shape is for a 1D array.
        '''

        self.arr_shape = arr_shape

        if col_stride == -1:
            self.col_stride = arr_shape[1]
        else:
            self.col_stride = col_stride

        self.num_dim = len(arr_shape)
        if self.num_dim not in (1,2):
            raise ValueError(f'Provided array shape has {self.num_dim} dimensions'
                            ' but only 1 or 2 dimensions are currently supported.')

        # If the array is 2D, set row_stride
        if self.num_dim == 2:
            if row_stride == -1:
                self.row_stride = arr_shape[0]
            else:
                self.row_stride = row_stride
        else:
            # There is no row_stride for a 1D array
            self.row_stride = None


    def __iter__(self):
        '''
        Iterator for TileIterator class.

        Yields
        ------
        np_slice : tuple of slice objects
            A tuple of slice objects that can be used for 
            indexing into the next tile of an array_like object.
        '''
        for row_start in range(0,self.arr_shape[0], self.row_stride):
            for col_start in range(0,self.arr_shape[1], self.col_stride):
                if self.num_dim == 2:
                    yield np.s_[row_start:row_start+self.row_stride,
                                col_start:col_start+self.col_stride]

                else:  # 1 dimension array
                    yield np.s_[col_start:col_start+self.col_stride]


def process_arr_by_tiles(in_arr, out_arr, func,
                            input_batches, output_batches, in_arr_2=None):
    '''
    Map a function to tiles of an array.
    
    Apply `func` to the input array sequentially by tiles
    and store the result in `out_arr`.

    Parameters
    ----------
    in_arr : array_like
        Input 1D or 2D array
    out_arr : array_like
        Output 1D or 2D array. Will be populated by this function.
    func : 
        Function to process arrays that has been partially-instantiated 
        via make_partial_func(). For a given input tile shape, `func` must 
        return an array with the same shape as an output tile shape
    input_batches : TileIterator
        Iterator for the input array
    output_batches : TileIterator
        Iterator for the output array
    in_arr_2 : array_like, optional
        Optional 2nd input argument of same shape as `in_arr`.
    '''
    for out_slice, in_slice in zip(output_batches, input_batches):

        # Process this batch
        if in_arr_2 is None:
            tmp_out = func(in_arr[in_slice])
        else:
            tmp_out = func(in_arr[in_slice], in_arr_2[in_slice])

        # Write the batch output to the output array
        out_arr[out_slice] = tmp_out


def make_partial_func(func, num_unfrozen_positional_args, **func_kwargs):
    '''
    Return a partial-instantiation of the function `func`.

    The arguments for `func` will be frozen to those provided 
    as the `func_kwargs`.

    Parameters
    ----------
    func : function
        A function
    num_unfrozen_positional_args : int
        The number of unfrozen positional arguments for `func`.
        Currently only supported for range [1,3].
        TODO: Allow this range to scale.
    **func_kwargs : keyword arguments
        The keyword arguments for `func` with corresponding values that will be 
        'frozen' into the returned partial function.
    
    Returns
    -------
    partial_func : function
        Partial instantiation of `func` with `num_unfrozen_positional_args` 
        unfrozen and keyword arguments in `func_kwargs` frozen
    '''

    if num_unfrozen_positional_args == 1:
        return lambda arg1 : func(arg1, **func_kwargs)
    elif num_unfrozen_positional_args == 2:
        return lambda arg1, arg2 : func(arg1, arg2, **func_kwargs)
    elif num_unfrozen_positional_args == 3:
        return lambda arg1, arg2, arg3 : func(arg1, arg2, arg3, **func_kwargs)
    else:
        raise ValueError('`num_unfrozen_positional_args` is '
                        f'{num_unfrozen_positional_args} but must be an integer'
                        'in range [1,3].')


###############################################
######       RSLC Tiling Functions      #######
###############################################

def compute_multilooked_power_by_tiling(arr,
                                        nlooks,
                                        linear_units=True,
                                        tile_shape=(512,-1)):
    '''
    Compute the multilooked power array by tiling.

    Parameters
    ----------
    arr : array_like
        The input 2D array
    nlooks : tuple of ints
        Number of looks along each axis of the input array to be 
        averaged during multilooking.
        Format: (num_rows, num_cols) 
    linear_units : bool
        True to compute power in linear units, False for decibel units.
        Defaults to True.
    tile_shape : tuple of ints
        Shape of each tile to be processed. If `tile_shape` is
        larger than the shape of `arr`, or if the dimensions of `arr`
        are not integer multiples of the dimensions of `tile_shape`,
        then smaller tiles may be used.
        -1 to use all rows / all columns (respectively).
        Format: (num_rows, num_cols) 
        Defaults to (512, -1) to use all columns (i.e. full rows of data)
        and leverage Python's row-major ordering.
    
    Returns
    -------
    multilook_img : numpy.ndarray
        The multilooked power image.

    Notes
    -----
    If the length of the input array along a given axis is not evenly divisible by the
    specified number of looks, any remainder samples from the end of the array will be
    discarded in the output.

    If a cell in the input array is nan (invalid), then the corresponding cell in the
    output array will also be nan.

    '''
    arr_shape = np.shape(arr)

    if len(arr_shape) != 2:
        raise ValueError(f'Input array has shape {arr_shape}'
                          ' but can only have 2 dimensions.')

    if tile_shape[0] == -1:
        tile_shape = (arr_shape[0], tile_shape[1])
    if tile_shape[1] == -1:
        tile_shape = (tile_shape[0], arr_shape[1])

    # Compute the portion (shape) of the input array 
    # that is integer multiples of nlooks.
    # This will be used to trim off (discard) the 'uneven edges' of the image,
    # i.e. the pixels beyond the largest integer multiples of nlooks.
    in_arr_valid_shape = tuple([(m // n) * n for m, n in zip(arr_shape, nlooks)])

    # Compute the shape of the output multilooked array
    final_out_arr_shape = tuple([m // n for m, n in zip(arr_shape, nlooks)])

    # Reduce the tiling shape to be integer multiples of nlooks
    # Otherwise, the tiling will get messy to book-keep.
    in_tiling_shape = tuple([(m // n) * n for m, n in zip(tile_shape, nlooks)])

    out_tiling_shape = tuple([m // n for m, n in zip(tile_shape, nlooks)])

    # Ensure that the tiling size is big enough for at least one multilook window
    if nlooks[0] > in_tiling_shape[0] or nlooks[1] > in_tiling_shape[1]:
        raise ValueError(f'Given nlooks values {nlooks} must be less than or '
                        f'equal to the adjusted tiling shape {in_tiling_shape}. '
                        f'(Provided, unadjusted tiling shape was {tile_shape}.')

    # Create the Iterators 
    input_iter = TileIterator(in_arr_valid_shape,
                                    row_stride=in_tiling_shape[0],
                                    col_stride=in_tiling_shape[1])
    out_iter = TileIterator(final_out_arr_shape,
                                    row_stride=out_tiling_shape[0],
                                    col_stride=out_tiling_shape[1])

    # Create an inner function for this use case.
    def calc_power_and_multilook(arr, nlooks, linear_units):

        # Calc power in linear of array
        out = nisarqa.arr2pow(arr)

        # Multilook
        out = nisarqa.multilook(out, nlooks)

        if not linear_units:
            # Convert to dB
            out = nisarqa.pow2db(out)

        return out

    # Partially instantiate the function
    partial_func = make_partial_func(func=calc_power_and_multilook,
                                    num_unfrozen_positional_args=1,
                                    nlooks=nlooks,
                                    linear_units=linear_units)

    # Instantiate the output array
    multilook_img = np.zeros(final_out_arr_shape, dtype=np.float32)  # 32 bit precision

    # Ok to pass the full input array; the tiling iterators
    # are constrained such that the 'uneven edges' will be ignored.
    process_arr_by_tiles(arr, multilook_img, partial_func,
                        input_batches=input_iter,
                        output_batches=out_iter)

    return multilook_img
