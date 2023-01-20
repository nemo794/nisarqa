import os
import sys
import warnings
from dataclasses import dataclass, field, fields
from typing import Iterable, Optional, Tuple, Union

import nisarqa
import numpy as np
from nisarqa import BaseParams, Param
from ruamel.yaml import YAML
from ruamel.yaml import CommentedMap as CM

objects_to_skip = nisarqa.get_all(__name__)

@dataclass(frozen=True)
class WorkflowsParams(BaseParams):
    '''
    The parameters specifying which RSLC-Caltools QA workflows should be run.

    Parameters
    ----------
    validate : bool, optional
        True to run the validate workflow. Default: False
    qa_reports : bool, optional
        True to run the QA Reports workflow. Default: False
    absolute_calibration_factor : bool, optional
        True to run the Absolute Calibration Factor (AbsCal) workflow.
        Default: False
    nesz : bool, optional
        True to run the Noise Estimator (NESZ) workflow. Default: False
    point_target_analyzer : bool, optional
        True to run the Point Target Analyzer (PTA) workflow. Default: False
    '''

    # None of the attributes in this class will be saved to the 
    # stats.h5 file, so they can be type bool (not required for them
    # to be of type Param)
    validate: bool = False
    qa_reports: bool = False
    absolute_calibration_factor: bool = False
    nesz: bool = False
    point_target_analyzer: bool = False

    @staticmethod
    def get_path_to_group_in_runconfig():
        return ['runconfig','groups','qa','workflows']

    @staticmethod
    def populate_runcfg(runconfig_cm):
        # Docstring taken from the populate_runcfg() @abstractmethod 
        # in the parent dataclass.

        # Create a default instance of this class
        default = WorkflowsParams()

        # build yaml params group
        params_cm = CM()

        # set indentation for displaying the comments correctly in the yaml
        comment_indent = len(default.get_path_to_group_in_runconfig()) * 4

        # Add all attributes from this dataclass to the group
        for field in fields(WorkflowsParams):
            params_cm[field.name] = field.default
            params_cm.yaml_set_comment_before_after_key(field.name, 
                before=f'\nTrue to run the {field.name} workflow. Default: '
                       f'{str(field.default)}',
                indent=comment_indent)

        # Add the new parameter group to the runconfig
        default.add_param_group_to_runconfig(runconfig_cm, params_cm)

    def write_params_to_h5(self, h5_file=None, bands=None):
        '''
        Populate h5_file HDF5 file with select processing parameters
        of this instance of the dataclass.

        No params will be added for this dataclass. Input parameters
        are included to conform to the API, but will be ignored.
        '''
        # this dataclass has no params to save to h5 file
        pass


@dataclass(frozen=True)
class InputFileGroupParams(BaseParams):
    '''
    Parameters from the Inpu File Group runconfig group.

    Arguments should be passed into this dataclass' parameters with 
    types such as `str` or `int`. During initialization, these arguments
    are processed into, stored as, and later accessible by the calling
    function as attributes with the type Param. The attributes will have
    the same name as the corresponding parameter, but will be of 
    type Param instead of type e.g. `str`.
    
    The original argument is accessible via the `val` attribute 
    of the new Param class; the additional Param attributes
    will be populated with default metadata by this dataclass.

    If no value or `None` is provided as an argument, then the
    Param attribute will be initialized with all default values.


    Parameters
    ----------
    qa_input_file : str, optional
        The input NISAR product file name (with path).
    '''

    # Set attributes to Param type for correct downstream type checking
    qa_input_file: Param

    # For `qa_input_file`, set default to None.
    # In practise, if a user tries to do QA without providing an input file,
    # the QA code will throw an error that it cannot open the file.
    # But, setting it to `None` will create a default-only Param
    # for use in runconfig generation, etc.
    def __init__(self, qa_input_file: Optional[str] = None):

       # For frozen dataclasses, set attributes via the superclass.
        object.__setattr__(self, 'qa_input_file', 
                            self._qa_input_file_2_param(qa_input_file))

    def _qa_input_file_2_param(self, qa_input_file):
        '''Return `qa_input_file` as a Param'''

        if not qa_input_file is None and not isinstance(qa_input_file, str):
            raise TypeError('`qa_input_file` must be a str or None')

        if isinstance(qa_input_file, str):
            if not os.path.isfile(qa_input_file):
                raise TypeError(
                    f'`qa_input_file` is not a valid file: {qa_input_file}')

            if not qa_input_file.endswith('.h5'):
                raise TypeError(
                    f'`qa_input_file` must end with .h5: {qa_input_file}')

        # Construct defaults for the new Param
        out = Param(name='qa_input_file',
                        val=qa_input_file,
                        units=None,
                        short_descr='Input NISAR Product filename',
                        long_descr='''
                Filename of the input file for QA.
                REQUIRED for QA. NOT REQUIRED if only running Product SAS.
                If Product SAS and QA SAS are run back-to-back,
                this field should be identical to `sas_output_file`.
                Otherwise, this field should contain the filename of the single
                NISAR product for QA to process.'''
                )

        return out

    @staticmethod
    def get_path_to_group_in_runconfig():
        return ['runconfig','groups','input_file_group']

    @staticmethod
    def populate_runcfg(runconfig_cm):
        # Docstring taken from the populate_runcfg() @abstractmethod 
        # in the parent dataclass.

        # Create a default instance of this class
        default = InputFileGroupParams()

        # build new yaml params group for this dataclass
        params_cm = CM()

        # Add runconfig parameters
        default.add_param_to_cm(params_cm=params_cm,
                                param_attr=default.qa_input_file)

        # Add the new parameter group to the runconfig
        default.add_param_group_to_runconfig(runconfig_cm, params_cm)


    def write_params_to_h5(self, h5_file=None, bands=None):
        '''
        Populate h5_file HDF5 file with select processing parameters
        of this instance of the dataclass.

        No params will be added for this dataclass. Input parameters
        are included to conform to the API, but will be ignored.
        '''
        # this dataclass has no params to save to h5 file
        pass


@dataclass(frozen=True)
class DynamicAncillaryFileParams(BaseParams):
    '''
    The parameters from the QA Dynamic Ancillary File runconfig group.

    Arguments should be passed into this dataclass' parameters with 
    types such as `str` or `int`. During initialization, these arguments
    are processed into, stored as, and later accessible by the calling
    function as attributes with the type Param. The attributes will have
    the same name as the corresponding parameter, but will be of 
    type Param instead of type e.g. `str`.
    
    The original argument is accessible via the `val` attribute 
    of the new Param class; the additional Param attributes
    will be populated with default metadata by this dataclass.

    If no value or `None` is provided as an argument, then the
    Param attribute will be initialized with all default values.

    Parameters
    ----------
    corner_reflector_file : str, optional
        The input corner reflector file's file name (with path).
    '''

    # Set attributes to Param type for correct downstream type checking
    corner_reflector_file: Param

    # For `corner_reflector_file`, set default to None.
    # In practise, if a user tries to do QA without providing an input file,
    # the QA code will throw an error that it cannot open the file.
    # But, setting it to `None` will create a default-only Param
    # for use in runconfig generation, etc.
    def __init__(self, corner_reflector_file: Optional[str] = None):

       # For frozen dataclasses, set attributes via the superclass.
        object.__setattr__(self, 'corner_reflector_file',
            self._corner_reflector_file_2_param(corner_reflector_file))

    def _corner_reflector_file_2_param(self, corner_reflector_file):
        '''Return `attr1` as a Param'''

        if not corner_reflector_file == None and \
            not isinstance(corner_reflector_file, str):
            raise TypeError('`corner_reflector_file` must be a str or None')

        if isinstance(corner_reflector_file, str):
            if not os.path.isfile(corner_reflector_file):
                raise TypeError(
                    '`corner_reflector_file` is not a valid file: '
                    f'{corner_reflector_file}')

        # Construct defaults for the new Param
        out = \
            Param(name='corner_reflector_file',
                  val=corner_reflector_file,
                  units=None,
                  short_descr='Source file for corner reflector locations',
                  long_descr='''
            Locations of the corner reflectors in the input product.
            Only required if `absolute_calibration_factor` or
            `point_target_analyzer` runconfig params are set to True for QA.'''
            )

        return out


    @staticmethod
    def get_path_to_group_in_runconfig():
        return ['runconfig','groups','dynamic_ancillary_file_group']

    @staticmethod
    def populate_runcfg(runconfig_cm):
        # Docstring taken from the populate_runcfg() @abstractmethod 
        # in the parent dataclass.

        # Create a default instance of this class
        default = DynamicAncillaryFileParams()

        # build new yaml params group for this dataclass
        params_cm = CM()

        # Add runconfig parameters
        default.add_param_to_cm(params_cm=params_cm,
                                param_attr=default.corner_reflector_file)

        # Add the new parameter group to the runconfig
        default.add_param_group_to_runconfig(runconfig_cm, params_cm)


    def write_params_to_h5(self, h5_file, bands=('LSAR')):
        '''
        Populate h5_file HDF5 file with select processing parameters
        of this instance of the dataclass.

        This function will populate the following fields
        in `h5_file` for all bands in `bands`:
            /science/<band>/identification/cornerReflectorFilename

        Parameters
        ----------
        h5_file : h5py.File
            Handle to an h5 file where the parameter metadata
            should be saved
        bands : iterable of str, optional
            Sequence of the band names. Ex: ('SSAR', 'LSAR')
            Defaults to ('LSAR')
        '''
        for band in bands:
            filename = self.corner_reflector_file.val
            if filename is None:
                filename = 'file not provided'
            nisarqa.create_dataset_in_h5group(
                h5_file=h5_file,
                grp_path=f'/science/{band}/identification',
                ds_name='cornerReflectorFilename',
                ds_data=filename,
                ds_description=self.corner_reflector_file.short_descr,
                ds_units=self.corner_reflector_file.units)


@dataclass(frozen=True)
class ProductPathGroupParams(BaseParams):
    '''
    Parameters from the Product Path Group runconfig group.

    Arguments should be passed into this dataclass' parameters with 
    types such as `str` or `int`. During initialization, these arguments
    are processed into, stored as, and later accessible by the calling
    function as attributes with the type Param. The attributes will have
    the same name as the corresponding parameter, but will be of 
    type Param instead of type e.g. `str`.
    
    The original argument is accessible via the `val` attribute 
    of the new Param class; the additional Param attributes
    will be populated with default metadata by this dataclass.

    If no value or `None` is provided as an argument, then the
    Param attribute will be initialized with all default values.

    Parameters
    ----------
    qa_output_dir : str, optional
        Filepath to the output directory to store NISAR QA output files.
        Defaults to './qa'
    '''

    # Set attributes to Param type for correct downstream type checking
    qa_output_dir: Param

    def __init__(self, qa_output_dir: Optional[str] = './qa'):
       # For frozen dataclasses, set attributes via the superclass.
        object.__setattr__(self, 'qa_output_dir',
                            self._qa_output_dir_2_param(qa_output_dir))

    def _qa_output_dir_2_param(self, qa_output_dir):
        '''Return `qa_output_dir` as a Param.'''

        # validate input type
        if not isinstance(qa_output_dir, str):
            raise TypeError('qa_output_dir '
                        f'must be a string: {qa_output_dir}')

        # Construct the new Param with metadata
        out = Param(name='qa_output_dir',
                        val=qa_output_dir,
                        units=None,
                        short_descr='Directory to store NISAR QA output files',
                        long_descr='''
                            Output directory to store all QA output files.
                            Defaults to ./qa'''
                        )

        # If this directory does not exist, make it.
        if not os.path.isdir(out.val):
            print(f'Creating QA output directory: {out.val}')
            os.makedirs(out.val, exist_ok=True)

        return out


    @staticmethod
    def get_path_to_group_in_runconfig():
        return ['runconfig','groups','product_path_group']

    @staticmethod
    def populate_runcfg(runconfig_cm):
        # Docstring taken from the populate_runcfg() @abstractmethod 
        # in the parent dataclass.

        # Create a default instance of this class
        default = ProductPathGroupParams()

        # build new yaml params group for this dataclass
        params_cm = CM()

        # Add runconfig parameters
        default.add_param_to_cm(params_cm=params_cm,
                                param_attr=default.qa_output_dir)

        # Add the new parameter group to the runconfig
        default.add_param_group_to_runconfig(runconfig_cm, params_cm)


    def write_params_to_h5(self, h5_file=None, bands=None):
        '''
        Populate h5_file HDF5 file with select processing parameters
        of this instance of the dataclass.

        No params will be added for this dataclass. Input parameters
        are included to conform to the API, but will be ignored.
        '''
        # this dataclass has no params to save to h5 file
        pass


@dataclass(frozen=True)
class RSLCPowerImageParams(BaseParams):
    '''
    Parameters to generate RSLC Power Images; this corresponds to the
    `qa_reports: power_image` runconfig group.

    Arguments should be passed into this dataclass' parameters with 
    types such as `str` or `int`. During initialization, these arguments
    are processed into, stored as, and later accessible by the calling
    function as attributes with the type Param. The attributes will have
    the same name as the corresponding parameter, but will be of 
    type Param instead of type e.g. `str`.
    
    The original argument is accessible via the `val` attribute 
    of the new Param class; the additional Param attributes
    will be populated with default metadata by this dataclass.

    If no value or `None` is provided as an argument, then the
    Param attribute will be initialized with all default values.

    Parameters
    ----------
    linear_units : bool, optional
        True to compute power in linear units, False for decibel units.
        Defaults to True.
    nlooks_freqa, nlooks_freqb : int, iterable of int, None, optional
        Number of looks along each axis of the input array 
        for the specified frequency. If None, then nlooks will be computed
        on-the-fly based on `num_mpix`.
    num_mpix : float, optional
        The approx. size (in megapixels) for the final multilooked image.
        Superseded by nlooks_freq* parameters. Defaults to 4.0 MPix.
    middle_percentile : float, optional
        Defines the middle percentile range of the image array
        that the colormap covers. Must be in the range [0.0, 100.0].
        Defaults to 95.0.
    gamma : float, None, optional
        The gamma correction parameter.
        Gamma will be applied as follows:
            array_out = normalized_array ^ gamma
        where normalized_array is a copy of the image with values
        scaled to the range [0,1]. 
        The image colorbar will be defined with respect to the input
        image values prior to normalization and gamma correction.
        Defaults to None (no normalization, no gamma correction)
    tile_shape : iterable of int, optional
        Preferred tile shape for processing images by batches.
        Actual tile shape used during processing might
        be smaller due to shape of image.
        Format: (num_rows, num_cols) 
        -1 to indicate all rows / all columns (respectively).
        Defaults to (1024, 1024) to use all columns 
        (i.e. full rows of data).

    Attributes
    ----------
    pow_units : Param
        Units of the power image. The `val` attribute of this Param
        will be of type `str`, accessible via `pow_units.val`
        If `linear_units` is True, this will be set to 'linear'.
        If `linear_units` is False, this will be set to 'dB'.
    '''

    # Set attributes to Param type for correct downstream type checking
    linear_units: Param
    nlooks_freqa: Param
    nlooks_freqb: Param
    num_mpix: Param
    middle_percentile: Param
    gamma: Param
    tile_shape: Param

    # Auto-generated attributes
    pow_units: Param = field(init=False)

    def __init__(self,
                 linear_units: Optional[bool] = True,
                 nlooks_freqa: Optional[Union[int, Iterable[int]]] = None,
                 nlooks_freqb: Optional[Union[int, Iterable[int]]] = None,
                 num_mpix: Optional[float] = 4.0,
                 middle_percentile: Optional[float] = 95.0,
                 gamma: Optional[float] = None,
                 tile_shape: Optional[Iterable[int]] = (1024,1024)
                 ):

       # For frozen dataclasses, set attributes via the superclass.
        object.__setattr__(self, 'linear_units',
                            self._linear_units_2_param(linear_units))

        object.__setattr__(self, 'nlooks_freqa',
                            self._nlooks_2_param(nlooks_freqa, 'A'))

        object.__setattr__(self, 'nlooks_freqb',
                            self._nlooks_2_param(nlooks_freqb, 'B'))

        object.__setattr__(self, 'num_mpix', self._num_mpix_2_param(num_mpix))

        object.__setattr__(self, 'middle_percentile', 
                        self._middle_percentile_2_param(middle_percentile))

        object.__setattr__(self, 'gamma', self._gamma_2_param(gamma))

        object.__setattr__(self, 'tile_shape', 
                            self._tile_shape_2_param(tile_shape))

        object.__setattr__(self, 'pow_units', self._pow_units_2_param())

    def _nlooks_2_param(self, nlooks, freq):
        '''Return the number of looks for given frequency as a Param.
        
        Parameters
        ----------
        nlooks : int or iterable of int
            Number of looks along each axis of the input array 
            for the specified frequency.
        freq : str
            The frequnecy to assign this number of looks to.
            Options: 'A' or 'B'
        
        Returns
        -------
        nlooks_param : Param
            `nlooks` for frequency `freq` as a Param object.
        '''

        if isinstance(nlooks, int):
            if nlooks <= 0:
                raise TypeError(
                    f'nlooks_freq{freq.lower()} must be a positive '
                    'int or sequence of two positive ints: {nlooks}')

        elif isinstance(nlooks, (list, tuple)):
            if all(isinstance(e, int) for e in nlooks):
                if any((e <= 0) for e in nlooks) or not len(nlooks) == 2:
                    raise TypeError(
                        f'nlooks_freq{freq.lower()} must be a positive '
                        'int or sequence of two positive ints: {nlooks}')
        elif nlooks is None:
            pass
        else:
            raise TypeError('`nlooks` must be of type int, iterable of int, '
                            f'or None: {nlooks}')

        out = Param(
            name=f'nlooks_freq{freq.lower()}',
            val=nlooks,
            units='unitless',
            short_descr='Number of looks along each axis of the '
                        f' Frequency {freq.upper()} image arrays'
                        ' for multilooking the power image.',
            long_descr=f'''
                Number of looks along each axis of the Frequency {freq.upper()}
                image arrays for multilooking the power image.
                Format: [<num_rows>, <num_cols>]
                Example: [6,7]
                If not provided, will default to computing the nlooks
                values that would produce an approx. `num_mpix` MPix
                browse image.'''
            )

        return out


    def _linear_units_2_param(self, linear_units):
        '''Return `linear_units` as a Param.'''

        if not isinstance(linear_units, bool):
            raise TypeError(f'linear_units must be a bool: {linear_units}')

        # Construct defaults for the new Param
        out = Param(
            name='linear_units',
            val=linear_units,
            units=None,
            short_descr='True to compute power in linear units for power image',
            long_descr='''
                True to compute power in linear units when generating 
                the power image for the browse images and graphical
                summary PDF. False for decibel units.
                Defaults to True.'''
            )

        return out


    def _num_mpix_2_param(self, num_mpix):
        '''Return `num_mpix` as a Param.'''

        if not isinstance(num_mpix, float):
            raise TypeError(f'num_mpix must be a float: {num_mpix}')

        if num_mpix <= 0.0:
            raise TypeError(f'`num_mpix` must be a positive value: {num_mpix}')

        # Construct defaults for the new Param
        out = Param(
            name='num_mpix',
            val=num_mpix,
            units='megapixels',
            short_descr='Approx. size (in megapixels) for the multilooked '
                        'power image(s)',
            long_descr='''
                The approx. size (in megapixels) for the final
                multilooked browse image(s). Defaults to 4.0 MPix.
                If `nlooks_freq*` parameter(s) is not None, nlooks
                values will take precedence.'''
            )
        
        return out


    def _middle_percentile_2_param(self, middle_percentile):
        '''Return `middle_percentile` as a Param.'''

        if not isinstance(middle_percentile, float):
            raise TypeError(
                f'`middle_percentile` must be a float: {middle_percentile}')

        if middle_percentile < 0.0 or middle_percentile > 100.0:
            raise TypeError('middle_percentile is '
                f'{middle_percentile}, must be in range [0.0, 100.0]')

        # Construct defaults for the new Param
        out = Param(
            name='middle_percentile',
            val=middle_percentile,
            units='unitless',
            short_descr='Middle percentile range of the image array '
                        'that the colormap covers',
            long_descr='''
                Defines the middle percentile range of the image array
                that the colormap covers. Must be in the range [0.0, 100.0].
                Defaults to 95.0.'''            )

        return out


    def _gamma_2_param(self, gamma):
        '''Return `gamma` as a Param.'''

        if not gamma == None and not isinstance(gamma, float):
            raise TypeError('`gamma` must be a float or None: {gamma}')

        if isinstance(gamma, float):
            if gamma < 0.0:
                raise TypeError(f'gamma must be a non-negative value: {gamma}')

        # Construct defaults for the new Param
        out = Param(
            name='gamma',
            val=gamma,
            units='unitless',
            short_descr='Gamma correction applied to power image',
            long_descr='''
                The gamma correction parameter.
                Gamma will be applied as follows:
                    array_out = normalized_array ^ gamma
                where normalized_array is a copy of the image with values
                scaled to the range [0,1]. 
                The image colorbar will be defined with respect to the input
                image values prior to normalization and gamma correction.
                Defaults to None (no normalization, no gamma correction)'''
            )

        return out


    def _tile_shape_2_param(self, tile_shape):
        '''Return `tile_shape` as a Param.
        
        TODO - this is duplicate code to other Params dataclasses. Fix.
        '''

        if not isinstance(tile_shape, (list, tuple)):
            raise TypeError('`tile_shape` must be a list or tuple: '
                                f'{tile_shape}')

        if not len(tile_shape) == 2:
            raise TypeError('`tile_shape` must have a length'
                                f'of two: {tile_shape}')

        if not all(isinstance(e, int) for e in tile_shape):
            raise TypeError('`tile_shape` must contain only '
                                f'integers: {tile_shape}')

        if any(e <= 0 for e in tile_shape):
            raise TypeError('`tile_shape` must contain only '
                                f'positive values: {tile_shape}')

        # Construct defaults for the new Param
        out = Param(
            name='tile_shape',
            val=tile_shape,
            units='unitless',
            short_descr='Preferred tile shape for processing images by batches.',
            long_descr='''
                Preferred tile shape for processing images by batches.
                Actual tile shape used during processing might
                be smaller due to shape of image.
                Format: [<num_rows>, <num_cols>]
                -1 to indicate all rows / all columns (respectively).
                Defaults to [1024, 1024] to use all columns 
                (i.e. full rows of data).'''
            )

        return out


    def _pow_units_2_param(self):
        '''Return `pow_units` as a Param.'''

        # Phase bin edges - allow for either radians or degrees
        if self.linear_units.val:
            val='linear'
        else:
            val='dB'

        template = Param(
                name='pow_units',
                val=val,
                units=None,
                short_descr='Units for the power image',
                long_descr='''
                    Units of the power image.
                    If `linear_units` is True, this will be set to 'linear'.
                    If `linear_units` is False, this will be set to 'dB'.'''
                )

        return template

    @staticmethod
    def get_path_to_group_in_runconfig():
        return ['runconfig','groups','qa','qa_reports','power_image']

    @staticmethod
    def populate_runcfg(runconfig_cm):
        # Docstring taken from the populate_runcfg() @abstractmethod 
        # in the parent dataclass.

        # Create a default instance of this class
        default = RSLCPowerImageParams()

        # build new yaml params group for this dataclass
        params_cm = CM()

        # Add runconfig parameters
        default.add_param_to_cm(params_cm=params_cm,
                                param_attr=default.linear_units)
        default.add_param_to_cm(params_cm=params_cm,
                                param_attr=default.nlooks_freqa)
        default.add_param_to_cm(params_cm=params_cm,
                                param_attr=default.nlooks_freqb)
        default.add_param_to_cm(params_cm=params_cm,
                                param_attr=default.num_mpix)
        default.add_param_to_cm(params_cm=params_cm,
                                param_attr=default.middle_percentile)
        default.add_param_to_cm(params_cm=params_cm,
                                param_attr=default.gamma)
        default.add_param_to_cm(params_cm=params_cm,
                                param_attr=default.tile_shape)

        # Add the new parameter group to the runconfig
        default.add_param_group_to_runconfig(runconfig_cm, params_cm)


    def write_params_to_h5(self, h5_file, bands=('LSAR')):
        '''
        Populate h5_file HDF5 file with select processing parameters
        of this instance of the dataclass.

        This function will populate the following fields
        in `h5_file` for all bands in `bands`:
            /science/<band>/QA/processing/powerImageMiddlePercentile
            /science/<band>/QA/processing/powerImageUnits
            /science/<band>/QA/processing/powerImageGammaCorrection

        Parameters
        ----------
        h5_file : h5py.File
            Handle to an h5 file where the parameter metadata
            should be saved
        bands : iterable of str, optional
            Sequence of the band names. Ex: ('SSAR', 'LSAR')
            Defaults to ('LSAR')
        '''

        for band in bands:
            # Open the group in the file, creating it if it doesn’t exist.
            grp_path = os.path.join('/science', band, 'QA', 'processing')

            nisarqa.create_dataset_in_h5group(
                h5_file=h5_file,
                grp_path=grp_path,
                ds_name='powerImageMiddlePercentile',
                ds_data=self.middle_percentile.val,
                ds_units=self.middle_percentile.units,
                ds_description=self.middle_percentile.short_descr)

            nisarqa.create_dataset_in_h5group(
                h5_file=h5_file,
                grp_path=grp_path,
                ds_name='powerImageUnits',
                ds_data=self.pow_units.val,
                ds_units=self.pow_units.units,
                ds_description=self.pow_units.short_descr)

            # Cannot assign Nonetype to a h5py dataset.
            # Use gamma = 1.0, which is equivalent to no gamma correction.
            gamma_cor = self.gamma.val
            if gamma_cor is None:
                gamma_cor = 1.0
            nisarqa.create_dataset_in_h5group(
                h5_file=h5_file,
                grp_path=grp_path,
                ds_name='powerImageGammaCorrection',
                ds_data=gamma_cor,
                ds_units=self.gamma.units,
                ds_description=self.gamma.short_descr)


@dataclass(frozen=True)
class RSLCHistogramParams(BaseParams):
    '''
    Parameters to generate the RSLC Power and Phase Histograms;
    this corresponds to the `qa_reports: histogram` runconfig group.

    Arguments should be passed into this dataclass' parameters with 
    types such as `str` or `int`. During initialization, these arguments
    are processed into, stored as, and later accessible by the calling
    function as attributes with the type Param. The attributes will have
    the same name as the corresponding parameter, but will be of 
    type Param instead of type e.g. `str`.
    
    The original argument is accessible via the `val` attribute 
    of the new Param class; the additional Param attributes
    will be populated with default metadata by this dataclass.

    If no value or `None` is provided as an argument, then the
    Param attribute will be initialized with all default values.

    Parameters
    ----------
    decimation_ratio : pair of int, optional
        The step size to decimate the input array for computing
        the power and phase histograms.
        For example, (2,3) means every 2nd azimuth line and
        every 3rd range line will be used to compute the histograms.
        Defaults to (10,10).
        Format: (<azimuth>, <range>)
    pow_histogram_start : numeric, optional
        The starting value (in dB) for the range of the power histogram edges.
        Defaults to -80. If `pow_histogram_start` is updated, then 
        `pow_bin_edges` will be updated to match.
    pow_histogram_endpoint : numeric, optional
        The endpoint value (in dB) for the range of the power histogram edges.
        Defaults to 20. If `pow_histogram_endpoint` is updated, then 
        `pow_bin_edges` will be updated to match.
    phs_in_radians : bool, optional
        True to compute phase in radians units, False for degrees units.
        Defaults to True. If `phs_in_radians` is updated, then 
        `phs_bin_edges` will be updated to match.
    tile_shape : iterable of int, optional
        Preferred tile shape for processing images by batches.
        Actual tile shape used during processing might
        be smaller due to shape of image.
        Format: (num_rows, num_cols) 
        -1 to indicate all rows / all columns (respectively).
        Defaults to (1024, 1024) to use all columns 
        (i.e. full rows of data).

    Attributes
    ----------
    pow_bin_edges : Param
        The bin edges (including endpoint) to use when computing
        the power histograms. Will be set to 100 uniformly-spaced bins
        in range [`pow_histogram_start`, `pow_histogram_endpoint`],
        including endpoint. (units are dB)
        This will be stored as a numpy.ndarray in `pow_bin_edges.val`
    phs_bin_edges : Param
        The bin edges (including endpoint) to use when computing
        the phase histograms.
        This will be stored as a numpy.ndarray in `phs_bin_edges.val`
        If `phs_in_radians` is True, this will be set to 100 
        uniformly-spaced bins in range [-pi,pi], including endpoint.
        If `phs_in_radians` is False, this will be set to 100
        uniformly-spaced bins in range [-180,180], including endpoint.
    '''

    # Set attributes to Param type for correct downstream type checking
    # User-Provided attributes:
    decimation_ratio: Param
    pow_histogram_start: Param
    pow_histogram_endpoint: Param
    phs_in_radians: Param
    tile_shape: Param

    # Auto-generated attributes
    # Power Bin Edges (generated based upon
    # `pow_histogram_start` and `pow_histogram_endpoint`)
    pow_bin_edges: Param = field(init=False)

    # Phase bin edges (generated based upon `phs_in_radians`)
    phs_bin_edges: Param = field(init=False)

    def __init__(self,
                decimation_ratio: Optional[Tuple[int, int]] = (10,10),
                pow_histogram_start: Optional[float] = -80.0,
                pow_histogram_endpoint: Optional[float] = 20.0,
                phs_in_radians: Optional[bool] = True,
                tile_shape: Optional[Iterable[int]] = (1024,1024)
                ):

       # For frozen dataclasses, set attributes via the superclass.
        object.__setattr__(self, 'decimation_ratio',
                self._decimation_ratio_2_param(decimation_ratio))

        object.__setattr__(self, 'pow_histogram_start',
                self._pow_histogram_start_2_param(pow_histogram_start))

        object.__setattr__(self, 'pow_histogram_endpoint',
                self._pow_histogram_endpoint_2_param(pow_histogram_endpoint))

        object.__setattr__(self, 'phs_in_radians',
                self._phs_in_radians_2_param(phs_in_radians))

        object.__setattr__(self, 'tile_shape',
                self._tile_shape_2_param(tile_shape))

        object.__setattr__(self, 'pow_bin_edges',
                self._pow_bin_edges_2_param())

        object.__setattr__(self, 'phs_bin_edges',
                self._phs_bin_edges_2_param())


    def _decimation_ratio_2_param(self, decimation_ratio):
        '''Return `decimation_ratio` as a Param.'''

        if not isinstance(decimation_ratio, (list, tuple)):
            raise TypeError('`decimation_ratio` must be a list or tuple '
                            f'{decimation_ratio}')

        if not len(decimation_ratio) == 2:
            raise TypeError('`decimation_ratio` must have a length of '
                            f'two: {decimation_ratio}')

        if not all(isinstance(e, int) for e in decimation_ratio):
            raise TypeError('`decimation_ratio` must contain only '
                            f'integers: {decimation_ratio}')

        if any(e <= 0 for e in decimation_ratio):
            raise TypeError('`decimation_ratio` must contain only positive'
                            f'values: {decimation_ratio}')

        # Construct defaults for the new Param
        out = Param(
            name='decimation_ratio',
            val=decimation_ratio,
            units='unitless',
            short_descr='Decimation ratio for processing power and phase '
                        'histograms.',
            long_descr='''
                The step size to decimate the input array for computing
                the power and phase histograms.
                For example, [2,3] means every 2nd azimuth line and
                every 3rd range line will be used to compute the histograms.
                Defaults to [10,10].
                Format: [<azimuth>, <range>]'''
            )

        return out


    def _pow_histogram_start_2_param(self, pow_histogram_start):
        '''Return `pow_histogram_start` as a Param.'''

        if not isinstance(pow_histogram_start, float):
            raise TypeError(f'pow_histogram_start must be float: '
                            f'{pow_histogram_start}')

        # Construct defaults for the new Param
        out = Param(
            name='pow_histogram_start',
            val=pow_histogram_start,
            units='dB',
            short_descr='Starting value for the range of the '
                        'power histogram edges.',
            long_descr='''
                Starting value (in dB) for the range of the power
                histogram edges. Defaults to -80.0.'''
            )
        
        return out


    def _pow_histogram_endpoint_2_param(self, pow_histogram_endpoint):
        '''Return `pow_histogram_endpoint` as a Param.'''

        if not isinstance(pow_histogram_endpoint, float):
            raise TypeError(f'pow_histogram_endpoint must be float: '
                            f'{pow_histogram_endpoint}')

        # Construct defaults for the new Param
        out = Param(
            name='pow_histogram_endpoint',
            val=pow_histogram_endpoint,
            units='dB',
            short_descr='Endpoint value for the range of the '
                        'power histogram edges.',
            long_descr='''
                Endpoint value (in dB) for the range of the power
                histogram edges. Defaults to 20.0.'''
            )
        
        return out


    def _phs_in_radians_2_param(self, phs_in_radians):
        '''Return `phs_in_radians` as a Param.'''

        if not isinstance(phs_in_radians, bool):
            raise TypeError('phs_in_radians must be bool: '
                                f'{phs_in_radians}')

        # Construct defaults for the new Param
        out = Param(
            name='phs_in_radians',
            val=phs_in_radians,
            units=None,
            short_descr='True to compute phase in radians units, False for degrees units',
            long_descr='''
                True to compute phase in radians units, False for 
                degrees units. Defaults to True.'''
            )

        return out


    def _tile_shape_2_param(self, tile_shape):
        '''Return `tile_shape` as a Param.

        TODO - this is duplicate code to other Params dataclasses. Fix.
        '''

        if not isinstance(tile_shape, (list, tuple)):
            raise TypeError('`tile_shape` must be a list or tuple: '
                                f'{tile_shape}')

        if not len(tile_shape) == 2:
            raise TypeError('`tile_shape` must have a length'
                                f'of two: {tile_shape}')

        if not all(isinstance(e, int) for e in tile_shape):
            raise TypeError('`tile_shape` must contain only '
                                f'integers: {tile_shape}')

        if any(e <= 0 for e in tile_shape):
            raise TypeError('`tile_shape` must contain only '
                                f'positive values: {tile_shape}')

        # Construct defaults for the new Param
        out = Param(
            name='tile_shape',
            val=tile_shape,
            units='unitless',
            short_descr='Preferred tile shape for processing images by batches.',
            long_descr='''
                Preferred tile shape for processing images by batches.
                Actual tile shape used during processing might
                be smaller due to shape of image.
                Format: [<num_rows>, <num_cols>]
                -1 to indicate all rows / all columns (respectively).
                Defaults to [1024, 1024] to use all columns 
                (i.e. full rows of data).'''
            )

        return out


    def _pow_bin_edges_2_param(self):
        '''Return `pow_bin_edges` as a Param.'''

        # Power Bin Edges - hardcode to be in decibels
        # 101 bin edges => 100 bins
        bin_edges = np.linspace(self.pow_histogram_start.val,
                                self.pow_histogram_endpoint.val,
                                num=101,
                                endpoint=True)

        out = Param(
            name='pow_bin_edges',
            val=bin_edges,
            units='dB',
            short_descr='Bin edges (including endpoint) for power histogram',
            long_descr='''
                The bin edges (including endpoint) to use when computing
                the power histograms. Will be set to 100 uniformly-spaced bins
                in range [`pow_histogram_start`, `pow_histogram_endpoint`],
                including endpoint. (units are dB)'''
            )

        return out


    def _phs_bin_edges_2_param(self):
        '''Return `phs_bin_edges` as a Param.'''

        # Phase bin edges - allow for either radians or degrees
        if self.phs_in_radians:
            phs_units = 'radians'
            start = -np.pi
            stop = np.pi
        else:
            phs_units = 'degrees'
            start = -180
            stop = 180

        # 101 bin edges => 100 bins
        bin_edges = np.linspace(start, stop, num=101, endpoint=True)

        out = Param(
            name='phs_bin_edges',
            val=bin_edges,
            units=phs_units,
            short_descr='Bin edges (including endpoint) for phase histogram',
            long_descr='''
            The bin edges (including endpoint) to use when computing
            the phase histograms in units of radians or degrees.
            If `phs_in_radians` is True, this will be set to 100 
            uniformly-spaced bins in range [-pi,pi], including endpoint.
            If `phs_in_radians` is False, this will be set to 100
            uniformly-spaced bins in range [-180,180], including endpoint.'''
            )

        return out


    @staticmethod
    def get_path_to_group_in_runconfig():
        return ['runconfig','groups','qa','qa_reports','histogram']


    @staticmethod
    def populate_runcfg(runconfig_cm):
        # Docstring taken from the populate_runcfg() @abstractmethod 
        # in the parent dataclass.

        # Create a default instance of this class
        default = RSLCHistogramParams()

        # build new yaml params group for this dataclass
        params_cm = CM()

        # Add runconfig parameters
        default.add_param_to_cm(params_cm=params_cm,
                                param_attr=default.decimation_ratio)
        default.add_param_to_cm(params_cm=params_cm,
                                param_attr=default.pow_histogram_start)
        default.add_param_to_cm(params_cm=params_cm,
                                param_attr=default.pow_histogram_endpoint)
        default.add_param_to_cm(params_cm=params_cm,
                                param_attr=default.phs_in_radians)
        default.add_param_to_cm(params_cm=params_cm,
                                param_attr=default.tile_shape)

        # Add the new parameter group to the runconfig
        default.add_param_group_to_runconfig(runconfig_cm, params_cm)


    def write_params_to_h5(self, h5_file, bands=('LSAR')):
        '''
        Populate h5_file HDF5 file with select processing parameters
        of this instance of the dataclass.

        This function will populate the following fields
        in `h5_file` for all bands in `bands`:
            /science/<band>/QA/processing/histogramDecimationAz
            /science/<band>/QA/processing/histogramDecimationRange
            /science/<band>/QA/processing/histogramEdgesPower
            /science/<band>/QA/processing/histogramEdgesPhase

        Parameters
        ----------
        h5_file : h5py.File
            Handle to an h5 file where the parameter metadata
            should be saved
        bands : iterable of str, optional
            Sequence of the band names. Ex: ('SSAR', 'LSAR')
            Defaults to ('LSAR')
        '''

        for band in bands:
            # Open the group in the file, creating it if it doesn’t exist.
            grp_path = os.path.join('/science', band, 'QA', 'processing')

            nisarqa.create_dataset_in_h5group(
                h5_file=h5_file,
                grp_path=grp_path,
                ds_name='histogramDecimationAz',
                ds_data=self.decimation_ratio.val[0],
                ds_units=self.decimation_ratio.units,
                ds_description='Azimuth decimation stride used to compute' \
                               ' power and phase histograms')

            nisarqa.create_dataset_in_h5group(
                h5_file=h5_file,
                grp_path=grp_path,
                ds_name='histogramDecimationRange',
                ds_data=self.decimation_ratio.val[1],
                ds_units=self.decimation_ratio.units,
                ds_description='Range decimation stride used to compute' \
                               ' power and phase histograms')

            nisarqa.create_dataset_in_h5group(
                h5_file=h5_file,
                grp_path=grp_path,
                ds_name='histogramEdgesPower',
                ds_data=self.pow_bin_edges.val,
                ds_units=self.pow_bin_edges.units,
                ds_description=self.pow_bin_edges.short_descr)

            nisarqa.create_dataset_in_h5group(
                h5_file=h5_file,
                grp_path=grp_path,
                ds_name='histogramEdgesPhase',
                ds_data=self.phs_bin_edges.val,
                ds_units=self.phs_bin_edges.units,
                ds_description=self.phs_bin_edges.short_descr)


@dataclass(frozen=True)
class AbsCalParams(BaseParams):
    '''
    Parameters from the QA-CalTools Absolute Calibration Factor
    runconfig group.

    Arguments should be passed into this dataclass' parameters with 
    types such as `str` or `int`. During initialization, these arguments
    are processed into, stored as, and later accessible by the calling
    function as attributes with the type Param. The attributes will have
    the same name as the corresponding parameter, but will be of 
    type Param instead of type e.g. `str`.
    
    The original argument is accessible via the `val` attribute 
    of the new Param class; the additional Param attributes
    will be populated with default metadata by this dataclass.

    If no value or `None` is provided as an argument, then the
    Param attribute will be initialized with all default values.

    Parameters
    ----------
    attr1 : float, optional
        Placeholder Attribute 1.
    '''

    attr1: Param

    def __init__(self, attr1: Optional[float] = 2.3):

       # For frozen dataclasses, set attributes via the superclass.
        object.__setattr__(self, 'attr1', self._attr1_2_param(attr1))

    def _attr1_2_param(self, attr1):
        '''Return `attr1` as a Param'''

        if not isinstance(attr1, float):
            raise TypeError(f'attr1 must be a float: {attr1}')

        if attr1 < 0.0:
            raise TypeError(f'attr1 must be a non-negative value: {attr1}')

        # Construct defaults for the new Param
        out = Param(name='attr1',
                        val=attr1,
                        units='smoot',
                        short_descr='Description of attr1 for stats.h5 file',
                        long_descr='''
            Placeholder: Attribute 1 description for runconfig. Each new line
            of text will be a separate line in the runconfig template.
            `attr1` is a non-negative float value. Default: 2.3'''
            )

        return out

    @staticmethod
    def get_path_to_group_in_runconfig():
        return ['runconfig','groups','qa','absolute_calibration_factor']

    @staticmethod
    def populate_runcfg(runconfig_cm):
        # Docstring taken from the populate_runcfg() @abstractmethod 
        # in the parent dataclass.

        # Create a default instance of this class
        default = AbsCalParams()

        # build new yaml params group for this dataclass
        params_cm = CM()

        # Add runconfig parameters
        default.add_param_to_cm(params_cm=params_cm,
                                param_attr=default.attr1)

        # Add the new parameter group to the runconfig
        default.add_param_group_to_runconfig(runconfig_cm, params_cm)


    def write_params_to_h5(self, h5_file, bands=('LSAR')):
        '''
        Populate h5_file HDF5 file with select processing parameters
        of this instance of the dataclass.

        This function will populate the following fields
        in `h5_file` for all bands in `bands`:
            /science/<band>/absoluteCalibrationFactor/processing/attr1

        Parameters
        ----------
        h5_file : h5py.File
            Handle to an h5 file where the parameter metadata
            should be saved
        bands : iterable of str, optional
            Sequence of the band names. Ex: ('SSAR', 'LSAR')
            Defaults to ('LSAR')
        '''
        for band in bands:
            nisarqa.create_dataset_in_h5group(
                h5_file=h5_file,
                grp_path=f'/science/{band}/absoluteCalibrationFactor/processing',
                ds_name='attr1',
                ds_data=self.attr1.val,
                ds_description=self.attr1.short_descr,
                ds_units=self.attr1.units)


@dataclass(frozen=True)
class NESZParams(BaseParams):
    '''
    Parameters from the QA-CalTools Noise Estimator (NESZ) runconfig group.

    Arguments should be passed into this dataclass' parameters with 
    types such as `str` or `int`. During initialization, these arguments
    are processed into, stored as, and later accessible by the calling
    function as attributes with the type Param. The attributes will have
    the same name as the corresponding parameter, but will be of 
    type Param instead of type e.g. `str`.
    
    The original argument is accessible via the `val` attribute 
    of the new Param class; the additional Param attributes
    will be populated with default metadata by this dataclass.

    If no value or `None` is provided as an argument, then the
    Param attribute will be initialized with all default values.

    Parameters
    ----------
    attr1 : float, optional
        Placeholder Attribute 1.

    Attributes
    ----------
    attr2 : Param
        Placeholder parameter of type bool.
    '''

    # Attributes for running the NESZ workflow
    attr1: Param

    # Auto-generated attributes
    attr2: Param = field(init=False)

    def __init__(self, attr1: Optional[float] = 11.3):
       # For frozen dataclasses, set attributes via the superclass.
        object.__setattr__(self, 'attr1', self._attr1_2_param(attr1))
        object.__setattr__(self, 'attr2', self._attr2_2_param())

    def _attr1_2_param(self, attr1):
        '''Return `attr1` as a Param'''

        if not isinstance(attr1, float):
            raise TypeError(f'attr1 must be a float: {attr1}')

        if attr1 < 0.0:
            raise TypeError('attr1 must be a non-negative value')

        out = Param(name='attr1',
                        val=attr1,
                        units='parsecs',
                        short_descr='score for Kessel Run',
                        long_descr='''
            Placeholder: Attribute 1 description for runconfig. Each new line
            of text will be a separate line in the runconfig template.
            `attr1` is a non-negative float value. Default: 11.9'''
            )

        return out


    def _attr2_2_param(self):
        '''Return `attr2` as a Param.'''

        # Here is where the dependency upon attr1 occurs:
        if self.attr1.val < 12.0:
            val = True
        else:
            val = False

        template = Param(
            name='attr2',
            val=val,
            units=self.attr1.units,
            short_descr='True if it was a good run',
            long_descr='''
            Placeholder: Attribute 2 description for runconfig.
            (attr2 is not in the runconfig, but it can be nice to have the
            docstring-like description in a centralized location.)
            `attr1` is a bool. Default: False'''
            )

        return template


    @staticmethod
    def get_path_to_group_in_runconfig():
        return ['runconfig','groups','qa','nesz']

    @staticmethod
    def populate_runcfg(runconfig_cm):
        # Docstring taken from the populate_runcfg() @abstractmethod 
        # in the parent dataclass.

        # Create a default instance of this class
        default = NESZParams()

        # build new yaml params group for this dataclass
        params_cm = CM()

        # Add runconfig parameters
        # Note: only add `attr1` to the runconfig.
        # `attr2` is not part of the runconfig, so do not add it.
        default.add_param_to_cm(params_cm=params_cm,
                                param_attr=default.attr1)

        # Add the new parameter group to the runconfig
        default.add_param_group_to_runconfig(runconfig_cm, params_cm)


    def write_params_to_h5(self, h5_file, bands=('LSAR')):
        '''
        Populate h5_file HDF5 file with select processing parameters
        of this instance of the dataclass.

        This function will populate the following fields
        in `h5_file` for all bands in `bands`:
            /science/<band>/NESZ/processing/attribute1
            /science/<band>/NESZ/processing/attribute2

        Parameters
        ----------
        h5_file : h5py.File
            Handle to an h5 file where the parameter metadata
            should be saved
        bands : iterable of str, optional
            Sequence of the band names. Ex: ('SSAR', 'LSAR')
            Defaults to ('LSAR')
        '''

        # In this function, you get to decide which of the processing 
        # parameters get to be saved to the STATS.h5 file. Not all
        # parameters need to be stored.

        for band in bands:
            grp_path=f'/science/{band}/NESZ/processing'
            
            # Save attr1
            nisarqa.create_dataset_in_h5group(
                h5_file=h5_file,
                grp_path=grp_path,
                ds_name='attribute1',
                ds_data=self.attr1.val,
                ds_description=self.attr1.short_descr,
                ds_units=self.attr1.units)

            # Save attr2
            nisarqa.create_dataset_in_h5group(
                h5_file=h5_file,
                grp_path=grp_path,
                ds_name='attribute2',
                ds_data=self.attr2.val,
                ds_description=self.attr2.short_descr,
                ds_units=self.attr2.units)


@dataclass(frozen=True)
class PointTargetAnalyzerParams(BaseParams):
    '''
    Parameters from the QA-CalTools Point Target Analyzer runconfig group.

    Arguments should be passed into this dataclass' parameters with 
    types such as `str` or `int`. During initialization, these arguments
    are processed into, stored as, and later accessible by the calling
    function as attributes with the type Param. The attributes will have
    the same name as the corresponding parameter, but will be of 
    type Param instead of type e.g. `str`.
    
    The original argument is accessible via the `val` attribute 
    of the new Param class; the additional Param attributes
    will be populated with default metadata by this dataclass.

    If no value or `None` is provided as an argument, then the
    Param attribute will be initialized with all default values.

    Parameters
    ----------
    attr1 : str, optional
        Placeholder Attribute 1.
    '''

    attr1: Param

    def __init__(self, attr1: Optional[float] = 2300.5):
       # For frozen dataclasses, set attributes via the superclass.
        object.__setattr__(self, 'attr1', self._attr1_2_param(attr1))

    def _attr1_2_param(self, attr1):
        '''Return `attr1` as a Param'''

        if not isinstance(attr1, float):
            raise TypeError(f'attr1 must be a float: {attr1}')

        if attr1 < 0.0:
            raise TypeError(f'attr1 must be a non-negative value: {attr1}')

        # Construct defaults for the new Param
        out = Param(name='attr1',
                        val=attr1,
                        units='beard-second',
                        short_descr='amount of growth at end of November',
                        long_descr='''
            Placeholder: Attribute 1 description for runconfig. Each new line
            of text will be a separate line in the runconfig template.
            `attr1` is a non-negative float value. Default: 2300.5'''
            )

        return out

    @staticmethod
    def get_path_to_group_in_runconfig():
        return ['runconfig','groups','qa','point_target_analyzer']

    @staticmethod
    def populate_runcfg(runconfig_cm):
        # Docstring taken from the populate_runcfg() @abstractmethod 
        # in the parent dataclass.

        # Create a default instance of this class
        default = PointTargetAnalyzerParams()

        # build new yaml params group for this dataclass
        params_cm = CM()

        # Add runconfig parameters
        default.add_param_to_cm(params_cm=params_cm,
                                param_attr=default.attr1)

        # Add the new parameter group to the runconfig
        default.add_param_group_to_runconfig(runconfig_cm, params_cm)


    def write_params_to_h5(self, h5_file, bands=('LSAR')):
        '''
        Populate h5_file HDF5 file with select processing parameters
        of this instance of the dataclass.

        This function will populate the following fields
        in `h5_file` for all bands in `bands`:
            /science/<band>/absoluteCalibrationFactor/processing/attr1

        Parameters
        ----------
        h5_file : h5py.File
            Handle to an h5 file where the parameter metadata
            should be saved
        bands : iterable of str, optional
            Sequence of the band names. Ex: ('SSAR', 'LSAR')
            Defaults to ('LSAR')
        '''
        for band in bands:
            nisarqa.create_dataset_in_h5group(
                h5_file=h5_file,
                grp_path=f'/science/{band}/pointTargetAnalyzer/processing',
                ds_name='attr1',
                ds_data=self.attr1.val,
                ds_description=self.attr1.short_descr,
                ds_units=self.attr1.units)


@dataclass
class RSLCRootParams:
    '''
    Dataclass of all *Params objects to process QA for NISAR RSLC products.

    `workflows` is the only required parameter. Based on the workflows set to
    True in `workflows`, the other RSLCRootParams parameters will be set
    per these rules:
        a) If a *Params object is needed by any workflow, it will be
           set to an instance of that *Params object.

                i) If a *Params object is provided by the caller, the
                corresponding attribute in RSLCRootParams will be set
                to that.

                i) If a *Params object is not provided, one will be 
                instantiated using all default value, and the
                corresponding attribute in RSLCRootParams will be set
                to that.

        b) If a *Params object is not needed by any workflow,
           it will be set to `None`, regardless of the input.
    
    Parameters
    ----------
    workflows : WorkflowsParams
        RSLC QA Workflows parameters
    input_f : InputFileGroupParams, optional
        Input File Group parameters for RSLC QA
    prodpath : ProductPathGroupParams, optional
        Product Path Group parameters for RSLC QA
    power_img : RSLCPowerImageParams
        Power Image Group parameters for RSLC QA
    histogram : RSLCHistogramParams
        Histogram Group parameters for RSLC QA
    anc_files : DynamicAncillaryFileParams, optional
        Dynamic Ancillary File Group parameters for RSLC QA-Caltools
    abs_cal : AbsCalParams, optional
        Absolute Calibration Factor group parameters for RSLC QA-Caltools
    nesz : NESZParams, optional
        NESZ group parameters for RSLC QA-Caltools
    pta : PointTargetAnalyzerParams, optional
        Point Target Analyzer group parameters for RSLC QA-Caltools
    '''

    # Shared parameters
    workflows: WorkflowsParams
    input_f: Optional[InputFileGroupParams] = None
    prodpath: Optional[ProductPathGroupParams] = None

    # QA parameters
    power_img: Optional[RSLCPowerImageParams] = None
    histogram: Optional[RSLCHistogramParams] = None

    # CalTools parameters
    anc_files: Optional[DynamicAncillaryFileParams] = None
    abs_cal: Optional[AbsCalParams] = None
    nesz: Optional[NESZParams] = None
    pta: Optional[PointTargetAnalyzerParams] = None

    def __post_init__(self):

        # Ensure that the minimum parameters were provided

        # If any of the workflows requested, then prodpath must be an
        # instance of ProductPathGroupParams.
        # prodpath is only optional in the case of doing a dumpconfig
        if any([getattr(self.workflows, field.name) \
                            for field in fields(self.workflows)]):
            if not isinstance(self.input_f, InputFileGroupParams):
                raise TypeError('`input_f` parameter of type '
                    'InputFileGroupParams is required to run any of the '
                    'QA workflows.')

        # If any of the workflows requested, then prodpath must be an
        # instance of ProductPathGroupParams.
        # prodpath is only optional in the case of doing a dumpconfig
        if any([getattr(self.workflows, field.name) \
                            for field in fields(self.workflows)]):
            if not isinstance(self.prodpath, ProductPathGroupParams):
                raise TypeError('`prodpath` parameter of type '
                    'ProductPathGroupParams is required to run any of the '
                    'QA workflows.')

        if self.workflows.qa_reports:
            if self.power_img is None or \
                not isinstance(self.power_img, RSLCPowerImageParams):
                raise TypeError('`power_img` parameter of type '
                    'RSLCPowerImageParams is required to run the '
                    'requested qa_reports workflow')

            if self.histogram is None or \
                not isinstance(self.histogram, RSLCHistogramParams):
                raise TypeError('`histogram` parameter of type '
                    'RSLCHistogramParams is required to run the '
                    'requested qa_reports workflow')

        if self.workflows.absolute_calibration_factor:
            if self.abs_cal is None or \
                not isinstance(self.abs_cal, AbsCalParams):
                raise TypeError('`abs_cal` parameter of type '
                    'AbsCalParams is required to run the '
                    'requested absolute_calibration_factor workflow')

            if self.anc_files is None or \
                not isinstance(self.anc_files, DynamicAncillaryFileParams):
                raise TypeError('`anc_files` parameter of type '
                    'DynamicAncillaryFileParams is required to run the '
                    'requested absolute_calibration_factor workflow')

        if self.workflows.nesz:
            if self.nesz is None or \
                not isinstance(self.nesz, NESZParams):
                raise TypeError('`nesz` parameter of type '
                    'NESZParams is required to run the '
                    'requested nesz workflow')

        if self.workflows.point_target_analyzer:
            if self.pta is None or \
                not isinstance(self.pta, PointTargetAnalyzerParams):
                raise TypeError('`pta` parameter of type '
                    'PointTargetAnalyzerParams is required to run the '
                    'requested point_target_analyzer workflow')

            if self.anc_files is None or \
                not isinstance(self.anc_files, DynamicAncillaryFileParams):
                raise TypeError('`anc_files` parameter of type '
                    'DynamicAncillaryFileParams is required to run the '
                    'requested point_target_analyzer workflow')

        # Ensure all provided attributes are a subtype of BaseParams
        for attr_name in self.__annotations__:
            attr = getattr(self, attr_name)
            if attr is not None:
                assert issubclass(type(attr), BaseParams), \
                    f'{attr_name} attribute must be a subclass of BaseParams'

    @staticmethod
    def dump_runconfig_template():
        '''Output the runconfig template (with default values) to stdout.
        '''

        # Build a ruamel yaml object that contains the runconfig structure
        yaml = YAML()
        yaml.indent(mapping=4, offset=4)

        runconfig_cm = CM()

        # Populate the yaml object. This order determines the order
        # the groups will appear in the runconfig.
        InputFileGroupParams.populate_runcfg(runconfig_cm)
        DynamicAncillaryFileParams.populate_runcfg(runconfig_cm)
        ProductPathGroupParams.populate_runcfg(runconfig_cm)
        WorkflowsParams.populate_runcfg(runconfig_cm)
        RSLCPowerImageParams.populate_runcfg(runconfig_cm)
        RSLCHistogramParams.populate_runcfg(runconfig_cm)
        AbsCalParams.populate_runcfg(runconfig_cm)
        NESZParams.populate_runcfg(runconfig_cm)
        PointTargetAnalyzerParams.populate_runcfg(runconfig_cm)

        # output to console. Let user stream that into a file.
        yaml.dump(runconfig_cm, sys.stdout)


    def save_params_to_stats_file(self, h5_file, bands=('LSAR')):
        '''Update the provided HDF5 file handle with select attributes
        (parameters) of this instance of RLSCRootParams.

        Parameters
        ----------
        h5_file : h5py.File
            Handle to an h5 file where the parameter metadata
            should be saved
        bands : iterable of str, optional
            Sequence of the band names. Ex: ('SSAR', 'LSAR')
            Defaults to ('LSAR')
        '''
        for params_obj in fields(self):
            # If a workflow was not requested, its RootParams attribute
            # will be None, so there will be no params to add to the h5 file
            po = getattr(self, params_obj.name)
            if po is not None:
                po.write_params_to_h5(h5_file, bands=bands)


def parse_rslc_runconfig(runconfig_yaml):
    '''
    Parse a QA RSLC Runconfig yaml file into a RSLCRootParams object.
    
    Parameters
    ----------
    runconfig_yaml : str
        Filename (with path) to an RSLC QA runconfig yaml file.
    
    Returns
    -------
    rslc_params : RSLCRootParams
        RSLCRootParams object populated with runconfig values where provided,
        and default values for missing runconfig parameters.
    '''
    # parse runconfig into a dict structure
    parser = YAML(typ='safe')
    with open(runconfig_yaml, 'r') as f:
        user_rncfg = parser.load(f)

    # Construct WorkflowsParams dataclass
    rncfg_path = WorkflowsParams.get_path_to_group_in_runconfig()
    try:
        params_dict = nisarqa.get_nested_element_in_dict(user_rncfg, rncfg_path)
    except KeyError:
        # if group does not exist in runconfig, use defaults
        workflows_params = WorkflowsParams()
    else:
        workflows_params = WorkflowsParams(**params_dict)

    # Construct InputFileGroupParams dataclass (required for all workflows)
    rncfg_path = InputFileGroupParams.get_path_to_group_in_runconfig()
    try:
        params_dict = nisarqa.get_nested_element_in_dict(user_rncfg,
                                                            rncfg_path)
    except KeyError as e:
        raise KeyError('`input_file_group` is a required runconfig group') from e

    try:
        input_file_params = InputFileGroupParams(
                        qa_input_file=params_dict['qa_input_file'])
    except KeyError as e:
        raise KeyError('`qa_input_file` is a required parameter for QA') from e

    # Construct DynamicAncillaryFileParams dataclass
    # Only two of the CalVal workflows use the dynamic_ancillary_file_group
    if workflows_params.absolute_calibration_factor or \
        workflows_params.point_target_analyzer:

        rncfg_path = DynamicAncillaryFileParams.get_path_to_group_in_runconfig()
        try:
            params_dict = nisarqa.get_nested_element_in_dict(user_rncfg, 
                                                                rncfg_path)
        except KeyError as e:
            raise KeyError('`dynamic_ancillary_file_group` is a required '
                           'runconfig group to run Absolute Calibration Factor'
                           ' or Point Target Analyzer workflows.') from e
        try:
            dyn_anc_files = DynamicAncillaryFileParams(
                    corner_reflector_file=params_dict['corner_reflector_file'])
        except KeyError as e:
            raise KeyError('`corner_reflector_file` is a required runconfig '
                           'parameter for Absolute Calibration Factor '
                           'or Point Target Analyzer workflows') from e
    else:
        dyn_anc_files = None

    # Construct ProductPathGroupParams dataclass
    rncfg_path = ProductPathGroupParams.get_path_to_group_in_runconfig()
    try:
        params_dict = nisarqa.get_nested_element_in_dict(
                                user_rncfg, rncfg_path)
    except KeyError:
        # group not found in runconfig. Use defaults.
        warnings.warn('`product_path_group` not found in runconfig. '
                      'Using default output directory.')
        product_path_params = ProductPathGroupParams()
    else:
        try:
            product_path_params = ProductPathGroupParams(
                                    qa_output_dir=params_dict['qa_output_dir'])
        except KeyError:
            # parameter not found in runconfig. Use defaults.
            warnings.warn('`qa_output_dir` not found in runconfig. '
                        'Using default output directory.')
            product_path_params = ProductPathGroupParams()

    # Construct RSLCPowerImageParams dataclass
    if workflows_params.qa_reports:
        rncfg_path = RSLCPowerImageParams.get_path_to_group_in_runconfig()
        try:
            params_dict = nisarqa.get_nested_element_in_dict(user_rncfg, 
                                                                rncfg_path)
        except KeyError:
            pow_img_params = RSLCPowerImageParams()
        else:
            pow_img_params = RSLCPowerImageParams(**params_dict)
    else:
        pow_img_params = None

    # Construct RSLCHistogramParams dataclass
    if workflows_params.qa_reports:
        rncfg_path = RSLCHistogramParams.get_path_to_group_in_runconfig()
        try:
            params_dict = nisarqa.get_nested_element_in_dict(user_rncfg, 
                                                                rncfg_path)
        except KeyError:
            histogram_params = RSLCHistogramParams()
        else:
            histogram_params = RSLCHistogramParams(**params_dict)
    else:
        histogram_params = None

    # Construct AbsCalParams dataclass
    if workflows_params.absolute_calibration_factor:
        rncfg_path = AbsCalParams.get_path_to_group_in_runconfig()
        try:
            params_dict = nisarqa.get_nested_element_in_dict(user_rncfg,
                                                                rncfg_path)
        except KeyError:
            abscal_params = AbsCalParams()
        else:
            abscal_params = AbsCalParams(**params_dict)
    else:
        abscal_params = None

    # Construct NESZ dataclass
    if workflows_params.nesz:
        rncfg_path = NESZParams.get_path_to_group_in_runconfig()
        try:
            params_dict = nisarqa.get_nested_element_in_dict(user_rncfg, 
                                                                rncfg_path)
        except KeyError:
            nesz_params = NESZParams()
        else:
            nesz_params = NESZParams(**params_dict)
    else:
        nesz_params = None

    # Construct PointTargetAnalyzerParams dataclass
    if workflows_params.point_target_analyzer:
        rncfg_path = PointTargetAnalyzerParams.get_path_to_group_in_runconfig()
        try:
            params_dict = nisarqa.get_nested_element_in_dict(user_rncfg,
                                                                rncfg_path)
        except KeyError:
            pta_params = PointTargetAnalyzerParams()
        else:
            pta_params = PointTargetAnalyzerParams(**params_dict)
    else:
        pta_params = None

    # Construct RSLCRootParams
    rslc_params = RSLCRootParams(workflows=workflows_params,
                                 input_f=input_file_params,
                                 anc_files=dyn_anc_files,
                                 prodpath=product_path_params,
                                 power_img=pow_img_params,
                                 histogram=histogram_params,
                                 abs_cal=abscal_params,
                                 nesz=nesz_params,
                                 pta=pta_params
                                 )

    return rslc_params


__all__ = nisarqa.get_all(__name__, objects_to_skip)
