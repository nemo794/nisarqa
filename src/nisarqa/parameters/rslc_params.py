import os
import sys
import warnings
from dataclasses import dataclass, field, fields
from typing import Any, ClassVar, Iterable, Optional, Tuple, Union

import nisarqa
import numpy as np
import numpy.typing as npt
from nisarqa import BaseParams, Param
from ruamel.yaml import YAML
from ruamel.yaml import CommentedMap as CM

objects_to_skip = nisarqa.get_all(__name__)

@dataclass
class WorkflowsParams(BaseParams):

    # None of the attributes in this class will be saved to the 
    # stats.h5 file, so they can be type bool (not required for them
    # to be of type Param)
    validate: bool = True
    qa_reports: bool = True
    absolute_calibration_factor: bool = True
    nesz: bool = True
    point_target_analyzer: bool = True

    @staticmethod
    def get_path_to_group_in_runconfig():
        return ['runconfig','groups','processing','qa', 'workflows']

    @staticmethod
    def populate_runcfg(runconfig_cm):
        '''
        
        Parameters
        ----------
        runconfig_cm : ruamel.yaml.comments.CommentedMap
            The base commented map object; this will be modified to
            include the appropriate dataclass parameters with comments
            and default values.
        '''
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
                       f'{str(field.default)}.',
                indent=comment_indent)

        # Add the new parameter group to the runconfig
        default.add_param_group_to_runconfig(runconfig_cm, params_cm)

    def write_params_to_h5(self, h5_file=None, bands=None):
        # no params to save to h5 file
        pass


@dataclass
class DynamicAncillaryFileParams(BaseParams):
    '''
    Data structure to hold information about a QA Dynamic Ancillary File
    group parameters.

    Note that however data values are passed into this dataclass,
    they will always be processed, stored, and accessible as Param instances.
        - If a non-Param type is provided, default values will be used
          to fill in the additional Param attributes.
        - If a Param type is provided, then the <attribute>.val must contain
          a valid non-Param input for that attribute.
        - If no value or `None` is supplied, then the attribute will be 
          set to its default Param value.

    Parameters
    ----------
    corner_reflector_file : str, Param, optional
        The input corner reflector file file name (with path).
        If a Param is supplied, `corner_reflector_file.val` must contain 
        the filename.
    '''

    corner_reflector_file: Optional[Union[str, Param]] = None


    def __setattr__(self, key, value):
        if key == 'corner_reflector_file':
            super().__setattr__(key, 
                                self.get_corner_reflector_file_param(value))
        else:
            raise KeyError(f'{key}')


    def get_corner_reflector_file_param(self, corner_reflector_file):
        '''Get Param for the `corner_reflector_file` attribute
        based on the input type.'''

        if isinstance(corner_reflector_file, Param):
            if not corner_reflector_file.name == 'corner_reflector_file':
                raise ValueError('corner_reflector_file.name must be "corner_reflector_file"')
            if not os.path.isfile(corner_reflector_file.val):
                raise ValueError('corner_reflector_file.val does not reference'
                                f' a valid file: {corner_reflector_file.val}')
            if not corner_reflector_file.val.endswith('.h5'):
                raise ValueError('corner_reflector_file.val does not reference'
                                f' an .h5 file: {corner_reflector_file.val}')

            return corner_reflector_file

        # Construct defaults for the new Param
        default = \
            Param(name='corner_reflector_file',
                  val=None,
                  units=None,
                  short_descr='Source file for corner reflector locations',
                  long_descr= \
            '''
            Locations of the corner reflectors in the input product.
            Only required if `absolute_calibration_factor` or
            `point_target_analyzer` runconfig params are set to True for QA.'''
            )

        if corner_reflector_file is None:
            warnings.warn('`corner_reflector_file` not provided. Using default.')
            return default

        elif isinstance(corner_reflector_file, str):
            if not os.path.isfile(corner_reflector_file):
                raise ValueError('`corner_reflector_file` is not a valid '
                                 f'file: {corner_reflector_file}')
            
            # update the default with the new value
            default.val = corner_reflector_file
            return default

        else:
            raise ValueError('`corner_reflector_file` must be a str, Param, '
                             'or None')

    @staticmethod
    def get_path_to_group_in_runconfig():
        return ['runconfig','groups','dynamic_ancillary_file_group']

    @staticmethod
    def populate_runcfg(runconfig_cm):
        '''
        Add the desired attributes of this dataclass to a Commented Map
        that will be used to generate a QA runconfig.

        Only default values will be used.

        Parameters
        ----------
        runconfig_cm : ruamel.yaml.comments.CommentedMap
            The base commented map; will be updated with the attributes
            from this dataclass that are in the QA runconfig file
        '''

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


@dataclass
class ProductPathGroupParams(BaseParams):
    '''
    Data structure to hold information about a QA Product Path Group parameters.

    Note that however data values are passed into this dataclass,
    they will always be processed, stored, and accessible as Param instances.
        - If a non-Param type is provided, default values will be used
          to fill in the additional Param attributes.
        - If a Param type is provided, then the <attribute>.val must contain
          a valid non-Param input for that attribute.
        - If no value or `None` is supplied, then the attribute will be 
          set to its default Param value.

    Parameters
    ----------
    qa_input_file : str, Param, optional
        The input NISAR product file name (with path).
    qa_output_dir : str, Param, optional
        Filepath to the output directory to store NISAR QA output files.
    '''

    qa_input_file: Optional[Union[str, Param]] = None
    qa_output_dir: Optional[Union[str, Param]] = None


    def __setattr__(self, key, value):
        if key == 'qa_input_file':
            super().__setattr__(key, self.get_qa_input_file_param(value))
        elif key == 'qa_output_dir':
            super().__setattr__(key, self.get_qa_output_dir_param(value))
        else:
            raise KeyError(f'{key}')


    def get_qa_input_file_param(self, qa_input_file):
        '''Return `qa_input_file` as a Param'''

        if isinstance(qa_input_file, Param):
            if not qa_input_file.name == 'qa_input_file':
                raise ValueError('qa_input_file.name must be "qa_input_file"')
            if not os.path.isfile(qa_input_file.val):
                raise ValueError('qa_input_file.val does not reference'
                                f' a valid file: {qa_input_file}')
            if not qa_input_file.val.endswith('.h5'):
                raise ValueError('qa_input_file.val does not reference'
                                f' an .h5 file: {qa_input_file}')

            return qa_input_file

        # Construct defaults for the new Param
        default = Param(name='qa_input_file',
                        val=None,
                        units=None,
                        short_descr='Input NISAR Product filename',
                        long_descr= \
                        '''
                        Filename of the input file for QA.
                        (This is the same as the output HDF5 from the SAS.)'''
                        )

        if qa_input_file is None:
            # return the default Param
            warnings.warn('`qa_input_file` not provided.')
            return default

        elif isinstance(qa_input_file, str):
            if not os.path.isfile(qa_input_file):
                raise ValueError(
                    f'`qa_input_file` is not a valid file: {qa_input_file}')
            if not qa_input_file.endswith('.h5'):
                raise ValueError(
                    f'`qa_input_file` must end with .h5: {qa_input_file}')
            
            # update the default with the new value and return it
            default.val = qa_input_file
            return default

        else:
            raise ValueError('`qa_input_file` must be a str, Param, or None')

    def get_qa_output_dir_param(self, qa_output_dir):
        '''Return `qa_output_dir` as a Param.'''

        if isinstance(qa_output_dir, Param):
            # validate input type
            if not isinstance(qa_output_dir.val, str):
                raise ValueError('qa_output_dir.val '
                            f'must be a string: {qa_output_dir.val}')

            # If this directory does not exist, make it.
            if not os.path.isdir(qa_output_dir.val):
                print(f'Creating QA output directory: {qa_output_dir.val}')
                os.makedirs(qa_output_dir.val, exist_ok=True)

            return qa_output_dir

        # Construct defaults for the new Param
        default = Param(name='qa_output_dir',
                        val='.',
                        units=None,
                        short_descr='Directory to store NISAR QA output files',
                        long_descr= \
                            '''
                            Output directory to store all QA output files.
                            Defaults to current working directory'''
                        )

        if qa_output_dir is None:
            return default

        elif isinstance(qa_output_dir, str):
            # validate input type
            if not isinstance(qa_output_dir, str):
                raise ValueError('qa_output_dir parameter '
                                 f'must be a string: {qa_output_dir}')

            # If this directory does not exist, make it.
            if not os.path.isdir(qa_output_dir):
                print(f'Creating QA output directory: {qa_output_dir}')
                os.makedirs(qa_output_dir, exist_ok=True)

            # update the default with the new value and return it
            default.val = qa_output_dir
            return default

        else:
            raise ValueError('qa_output_dir must be a str, Param, or None')

    @staticmethod
    def get_path_to_group_in_runconfig():
        return ['runconfig','groups','product_path_group']

    @staticmethod
    def populate_runcfg(runconfig_cm):
        '''
        Parameters
        ----------
        runconfig_cm : ruamel.yaml.comments.CommentedMap
            The base commented map to add the parameters to.
        '''
        # Create a default instance of this class
        default = ProductPathGroupParams()

        # build new yaml params group for this dataclass
        params_cm = CM()

        # Add runconfig parameters
        default.add_param_to_cm(params_cm=params_cm,
                                param_attr=default.qa_input_file)

        default.add_param_to_cm(params_cm=params_cm,
                                param_attr=default.qa_output_dir)

        # Add the new parameter group to the runconfig
        default.add_param_group_to_runconfig(runconfig_cm, params_cm)


    def write_params_to_h5(self, h5_file=None, bands=None):
        # No attributes to save to the .h5 file
        pass


@dataclass
class RSLCPowerImageParams(BaseParams):
    '''
    Data structure to hold the parameters to generate the
    RSLC Power Images.

    Note that however data values are passed into this dataclass,
    they will always be processed, stored, and accessible as Param instances.
        - If a non-Param type is provided, default values will be used
          to fill in the additional Param attributes.
        - If a Param type is provided, then the <attribute>.val must contain
          a valid non-Param input for that attribute.
        - If no value or `None` is supplied, then the attribute will be 
          set to its default Param value.

    Parameters
    ----------
    linear_units : bool, Param, optional
        True to compute power in linear units, False for decibel units.
        Defaults to True.
    nlooks_freqa, nlooks_freqb : int, iterable of int, Param, None, optional
        Number of looks along each axis of the input array 
        for the specified frequency. If None, then nlooks will be computed
        on-the-fly based on `num_mpix`.
    num_mpix : float, Param, None, optional
        The approx. size (in megapixels) for the final multilooked image.
        Superseded by nlooks_freq* parameters. Defaults to 4.0 MPix.
    middle_percentile : float, Param, None, optional
        Defines the middle percentile range of the image array
        that the colormap covers. Must be in the range [0.0, 100.0].
        Defaults to 100.0.
    gamma : float, Param, None, optional
        The gamma correction parameter.
        Gamma will be applied as follows:
            array_out = normalized_array ^ gamma
        where normalized_array is a copy of the image with values
        scaled to the range [0,1]. 
        The image colorbar will be defined with respect to the input
        image values prior to normalization and gamma correction.
        Defaults to None (no normalization, no gamma correction)
    tile_shape : iterable of int, Param, optional
        Preferred tile shape for processing images by batches.
        Actual tile shape used during processing might
        be smaller due to shape of image.
        Format: (num_rows, num_cols) 
        -1 to indicate all rows / all columns (respectively).
        Defaults to (1024, 1024) to use all columns 
        (i.e. full rows of data).

    Attributes
    ----------
    pow_units : str
        Units of the power image.
        If `linear_units` is True, this will be set to 'linear'.
        If `linear_units` is False, this will be set to 'dB'.
    '''

    # Attributes for generating the Power Image(s)
    # (default values will be set during post_init)
    linear_units: Optional[Union[bool, Param]] = None
    nlooks_freqa: Optional[Union[int, Iterable[int], Param]] = None
    nlooks_freqb: Optional[Union[int, Iterable[int], Param]] = None
    num_mpix: Optional[Union[float, Param]] = None
    middle_percentile: Optional[Union[float, Param]] = None
    gamma: Optional[Union[float, Param]] = None
    tile_shape: Optional[Union[Iterable[int], Param]] = None

    # Auto-generated attributes
    pow_units: Param = field(init=False)


    def __setattr__(self, key, value):
        if key == 'linear_units':
            super().__setattr__(key, self._get_linear_units_param(value))
            super().__setattr__('pow_units', self._get_pow_units_param())
        elif key == 'nlooks_freqa':
            super().__setattr__(key, self._get_nlooks_param(value, 'A'))
        elif key == 'nlooks_freqb':
            super().__setattr__(key, self._get_nlooks_param(value, 'B'))
        elif key == 'num_mpix':
            super().__setattr__(key, self._get_num_mpix_param(value))
        elif key == 'middle_percentile':
            super().__setattr__(key, self._get_middle_percentile_param(value))
        elif key == 'gamma':
            super().__setattr__(key, self._get_gamma_param(value))
        elif key == 'tile_shape':
            super().__setattr__(key, self._get_tile_shape_param(value))
        elif key == 'pow_units':
            raise KeyError(f'`pow_units` is only updated when setting '
                            '`linear_units` attribute')
        else:
            raise KeyError(f'{key}')


    def _get_nlooks_param(self, nlooks, freq):
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

        if isinstance(nlooks, Param):
            if not nlooks.name == f'nlooks_freq{freq.lower()}':
                raise ValueError(f'nlooks_freq{freq.lower()}.name must be '
                                 f'"nlooks_freq{freq.lower()}"')
            if not isinstance(nlooks.val, int) or \
                (all(isinstance(e, int) for e in nlooks.val) and \
                    len(nlooks.val) == 2):
                raise ValueError(f'nlooks_freq{freq.lower()}.val must be '
                                'an int, sequence of two integers, '
                                f'or None: {nlooks.val}')

            return nlooks

        # Construct defaults for the new Param
        default = Param(\
            name=f'nlooks_freq{freq.lower()}',
            val=None,
            units='unitless',
            short_descr='Number of looks along each axis of the '
                        f' Frequency {freq.upper()} image arrays'
                        ' for multilooking the power image.',
            long_descr= \
                f'''
                Number of looks along each axis of the Frequency {freq.upper()}
                image arrays for multilooking the power image.
                Format: [<num_rows>, <num_cols>]
                Example: [6,7]
                If not provided, will default to computing the nlooks
                values that would produce an approx. `num_mpix` MPix
                browse image.'''
            )

        if nlooks is None:
            # return the default Param
            warnings.warn(f'`nlooks_freq{freq.lower()}` not provided.')
            return default

        elif isinstance(nlooks, int):
            if nlooks <= 0:
                raise ValueError(
                    f'nlooks_freq{freq.lower()} must be a positive '
                    'int or sequence of two positive ints: {nlooks}')
            
            # update the default with the new value and return it
            default.val = nlooks
            return default
            
        elif all(isinstance(e, int) for e in nlooks):
            if any((e <= 0) for e in nlooks) or not len(nlooks) == 2:
                raise ValueError(
                    f'nlooks_freq{freq.lower()} must be a positive '
                    'int or sequence of two positive ints: {nlooks}')
            
            # update the default with the new value and return it
            default.val = nlooks
            return default

        else:
            raise ValueError(
                f'nlooks_freq{freq.lower()} must be a str, Param, or None')


    def _get_linear_units_param(self, linear_units):
        '''Return linear_units as a Param.
        
        Parameters
        ----------
        linear_units : bool
            True to compute power in linear units, False for decibel units.
            Defaults to True.
        
        Returns
        -------
        linear_units_param : Param
            `linear_units` as a Param object.
        '''

        if isinstance(linear_units, Param):
            if not linear_units.name == 'linear_units':
                raise ValueError('linear_units.name must be `linear_units`')
            if not isinstance(linear_units.val, bool):
                raise ValueError(f'linear_units.val must be bool '
                                f'or None: {linear_units.val}')

            return linear_units

        # Construct defaults for the new Param
        default = Param(\
            name='linear_units',
            val=True,
            units=None,
            short_descr='True to compute power in linear units for power image',
            long_descr= \
                '''True to compute power in linear units when generating 
                the power image for the browse images and graphical
                summary PDF. False for decibel units.
                Defaults to True.'''
            )

        if linear_units is None:
            # return the default Param
            warnings.warn('`linear_units` not provided.')
            return default

        elif isinstance(linear_units, bool):
            # update the default with the new value and return it
            default.val = linear_units
            return default

        else:
            raise ValueError(
                f'linear_units must be a bool or None')

    def _get_num_mpix_param(self, num_mpix):
        '''Return num_mpix as a Param.
        
        Parameters
        ----------
        num_mpix : float
            The approx. size (in megapixels) for the final
            multilooked browse image(s). Defaults to 4.0 MPix.

        Returns
        -------
        num_mpix_param : Param
            `num_mpix` as a Param object.
        '''

        if isinstance(num_mpix, Param):
            if not num_mpix.name == 'num_mpix':
                raise ValueError('num_mpix.name must be `num_mpix`')
            if not isinstance(num_mpix.val, bool):
                raise ValueError(f'num_mpix.val must be bool '
                                f'or None: {num_mpix.val}')
            if num_mpix.val <= 0.0:
                raise ValueError('num_mpix.val must be a positive value')

            return num_mpix

        # Construct defaults for the new Param
        default = Param(\
            name='num_mpix',
            val=4.0,
            units='megapixels',
            short_descr='Approx. size (in megapixels) for the multilooked '
                        'power image(s)',
            long_descr= \
                '''
                The approx. size (in megapixels) for the final
                multilooked browse image(s). Defaults to 4.0 MPix.
                If `nlooks_freq*` parameter(s) is not None, nlooks
                values will take precedence.'''
            )

        if num_mpix is None:
            # return the default Param
            warnings.warn('`num_mpix` not provided.')
            return default

        elif isinstance(num_mpix, float):
            if num_mpix <= 0.0:
                raise ValueError('num_mpix must be a positive value')
            # update the default with the new value and return it
            default.val = num_mpix
            return default

        else:
            raise ValueError(
                f'num_mpix must be a float, Param, or None')


    def _get_middle_percentile_param(self, middle_percentile):
        '''Return middle_percentile as a Param.
        
        Parameters
        ----------
        middle_percentile : float
            Defines the middle percentile range of the image array
            that the colormap covers. Must be in the range [0.0, 100.0].
            Defaults to 100.0.

        Returns
        -------
        middle_percentile_param : Param
            `middle_percentile` as a Param object.
        '''

        if isinstance(middle_percentile, Param):
            if not middle_percentile.name == 'middle_percentile':
                raise ValueError(
                    'middle_percentile.name must be `middle_percentile`')
            if not isinstance(middle_percentile.val, float):
                raise ValueError(f'middle_percentile.val must be float '
                                f'or None: {middle_percentile.val}')
            if middle_percentile.val < 0.0 or middle_percentile.val > 100.0:
                raise ValueError('middle_percentile.val is '
                    f'{middle_percentile.val}, must be in range [0.0, 100.0]')

            return middle_percentile

        # Construct defaults for the new Param
        default = Param(\
            name='middle_percentile',
            val=100.0,
            units='unitless',
            short_descr='Middle percentile range of the image array '
                        'that the colormap covers',
            long_descr= \
                '''
                Defines the middle percentile range of the image array
                that the colormap covers. Must be in the range [0.0, 100.0].
                Defaults to 100.0.'''            )

        if middle_percentile is None:
            # return the default Param
            warnings.warn('`middle_percentile` not provided. Using default.')
            return default

        elif isinstance(middle_percentile, float):
            if middle_percentile < 0.0 or middle_percentile > 100.0:
                raise ValueError('middle_percentile is '
                    f'{middle_percentile}, must be in range [0.0, 100.0]')
            # update the default with the new value and return it
            default.val = middle_percentile
            return default

        else:
            raise ValueError(
                f'middle_percentile must be a float, Param, or None')


    def _get_gamma_param(self, gamma):
        '''Return gamma as a Param.
        
        Parameters
        ----------
        gamma : float or None, optional
            The gamma correction parameter.
            Gamma will be applied as follows:
                array_out = normalized_array ^ gamma
            where normalized_array is a copy of the image with values
            scaled to the range [0,1]. 
            The image colorbar will be defined with respect to the input
            image values prior to normalization and gamma correction.
            Defaults to None (no normalization, no gamma correction)

        Returns
        -------
        gamma_param : Param
            `gamma` as a Param object.
        '''

        if isinstance(gamma, Param):
            if not gamma.name == 'gamma':
                raise ValueError(
                    'gamma.name must be `gamma`')
            if not isinstance(gamma.val, float):
                raise ValueError(f'gamma.val must be float '
                                f'or None: {gamma.val}')
            if gamma.val < 0.0:
                raise ValueError('gamma.val is '
                    f'{gamma.val}, must be a non-negative value')

            return gamma

        # Construct defaults for the new Param
        default = Param(\
            name='gamma',
            val=None,
            units='unitless',
            short_descr='Gamma correction applied to power image',
            long_descr= \
                '''
                The gamma correction parameter.
                Gamma will be applied as follows:
                    array_out = normalized_array ^ gamma
                where normalized_array is a copy of the image with values
                scaled to the range [0,1]. 
                The image colorbar will be defined with respect to the input
                image values prior to normalization and gamma correction.
                Defaults to None (no normalization, no gamma correction)'''
            )

        if gamma is None:
            # return the default Param
            warnings.warn('`gamma` not provided. Using default.')
            return default

        elif isinstance(gamma, float):
            if gamma < 0.0:
                raise ValueError('gamma is '
                    f'{gamma}, must be a non-negative value')
            # update the default with the new value and return it
            default.val = gamma
            return default

        else:
            raise ValueError(
                f'gamma must be a float, Param, or None')


    def _get_tile_shape_param(self, tile_shape):
        '''Return `tile_shape` as a Param.
        
        TODO - this is duplicate code to other Params dataclasses. Fix.
        
        Parameters
        ----------
        tile_shape : iterable of int, Param, None
            Preferred tile shape for processing power images by batches.
     
        Returns
        -------
        tile_shape_param : Param
            `tile_shape` as a Param object.
        '''

        if isinstance(tile_shape, Param):
            if not tile_shape.name == 'tile_shape':
                raise ValueError('tile_shape.name must be "tile_shape".')
            if not (all((isinstance(e, int) and e > 0) for e in tile_shape.val) \
                    or not len(tile_shape.val) == 2) :
                raise ValueError('tile_shape.val must be a sequence '
                                 f'of two positive integers: {tile_shape.val}')

            return tile_shape

        # Construct defaults for the new Param
        default = Param(\
            name='tile_shape',
            val=[1024, 1024],
            units='unitless',
            short_descr='Preferred tile shape for processing images by batches.',
            long_descr= \
                '''
                Preferred tile shape for processing images by batches.
                Actual tile shape used during processing might
                be smaller due to shape of image.
                Format: [num_rows, num_cols]
                -1 to indicate all rows / all columns (respectively).
                Defaults to [1024, 1024] to use all columns 
                (i.e. full rows of data).'''
            )

        if tile_shape is None:
            # return the default Param
            warnings.warn('`tile_shape` not provided.')
            return default

        elif all(isinstance(e, int) for e in tile_shape):
            if not len(tile_shape) == 2 or \
                any(e <= 0 for e in tile_shape):
                raise ValueError('tile_shape must be a sequence '
                                 f'of two positive integers: {tile_shape}')
            
            # update the default with the new value and return it
            default.val = tile_shape
            return default

        else:
            raise ValueError('`tile_shape` must be a sequence of two ints, '
                             f'a Param, or None: {tile_shape}')


    def _get_pow_units_param(self):
        template = Param(
                name='pow_units',
                val='TBD',
                units=None,
                short_descr='Units for the power image',
                long_descr='''
                    Units of the power image.
                    If `linear_units` is True, this will be set to 'linear'.
                    If `linear_units` is False, this will be set to 'dB'.'''
                )
        # Phase bin edges - allow for either radians or degrees
        if self.linear_units.val:
            template.val='linear'
        else:
            template.val='dB'

        return template

    @staticmethod
    def get_path_to_group_in_runconfig():
        return ['runconfig','groups','processing','qa','qa_reports','power_image']

    @staticmethod
    def populate_runcfg(runconfig_cm):
        '''
        Parameters
        ----------
        runconfig_cm : ruamel.yaml.comments.CommentedMap
            The base commented map to add the parameters to.
        '''
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
        Populate h5_file HDF5 file this dataclass' processing parameters.

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


@dataclass
class RSLCHistogramParams(BaseParams):
    '''
    Data structure to hold the parameters to generate the
    RSLC Power and Phase Histograms.

    Note that however data values are passed into this dataclass,
    they will always be processed, stored, and accessible as Param instances.
        - If a non-Param type is provided, default values will be used
          to fill in the additional Param attributes.
        - If a Param type is provided, then the <attribute>.val must contain
          a valid non-Param input for that attribute.
        - If no value or `None` is supplied, then the attribute will be 
          set to its default Param value.

    Parameters
    ----------
    decimation_ratio : pair of int, Param, optional
        The step size to decimate the input array for computing
        the power and phase histograms.
        For example, (2,3) means every 2nd azimuth line and
        every 3rd range line will be used to compute the histograms.
        Defaults to (1,1), i.e. no decimation will occur.
        Format: (<azimuth>, <range>)
    pow_histogram_start : numeric, Param, optional
        The starting value (in dB) for the range of the power histogram edges.
        Defaults to -80. If `pow_histogram_start` is updated, then 
        `pow_bin_edges` will be updated to match.
    pow_histogram_endpoint : numeric, Param, optional
        The endpoint value (in dB) for the range of the power histogram edges.
        Defaults to 20. If `pow_histogram_endpoint` is updated, then 
        `pow_bin_edges` will be updated to match.
    phs_in_radians : bool, Param, optional
        True to compute phase in radians units, False for degrees units.
        Defaults to True. If `phs_in_radians` is updated, then 
        `phs_bin_edges` will be updated to match.
    tile_shape : iterable of int, Param, optional
        Preferred tile shape for processing images by batches.
        Actual tile shape used during processing might
        be smaller due to shape of image.
        Format: (num_rows, num_cols) 
        -1 to indicate all rows / all columns (respectively).
        Defaults to (1024, 1024) to use all columns 
        (i.e. full rows of data).

    Attributes
    ----------
    pow_bin_edges : numpy.ndarray, Param
        The bin edges (including endpoint) to use when computing
        the power histograms. Will be set to 100 uniformly-spaced bins
        in range [`pow_histogram_start`, `pow_histogram_endpoint`],
        including endpoint. (units are dB)
    phs_bin_edges : numpy.ndarray, Param
        The bin edges (including endpoint) to use when computing
        the phase histograms.
        If `phs_in_radians` is True, this will be set to 100 
        uniformly-spaced bins in range [-pi,pi], including endpoint.
        If `phs_in_radians` is False, this will be set to 100
        uniformly-spaced bins in range [-180,180], including endpoint.
    '''

    # Attributes for generating Power and Phase Histograms
    # User-Provided attributes:
    decimation_ratio: Optional[Union[Tuple[int, int], Param]] = None
    pow_histogram_start: Optional[Union[float, Param]] = None
    pow_histogram_endpoint: Optional[Union[float, Param]] = None
    phs_in_radians: Optional[Union[bool, Param]] = None
    tile_shape: Optional[Union[Iterable[int], Param]] = None

    # Auto-generated attributes
    # Power Bin Edges (generated based upon
    # `pow_histogram_start` and `pow_histogram_endpoint`)
    pow_bin_edges: Param = field(init=False)

    # Phase bin edges (generated based upon `phs_in_radians`)
    phs_bin_edges: Param = field(init=False)


    def __setattr__(self, key, value):
        if key == 'decimation_ratio':
            super().__setattr__(key, self._get_decimation_ratio_param(value))

        elif key == 'pow_histogram_start':
            super().__setattr__(key,
                                self._get_pow_histogram_start_param(value))

        elif key == 'pow_histogram_endpoint':
            super().__setattr__(key,
                                self._get_pow_histogram_endpoint_param(value))

        elif key == 'phs_in_radians':
            super().__setattr__(key, self._get_phs_in_radians_param(value))
            super().__setattr__('phs_bin_edges',
                                self._get_phs_bin_edges_param())

        elif key == 'tile_shape':
            super().__setattr__(key, self._get_tile_shape_param(value))

        elif key in ['pow_bin_edges', 'phs_bin_edges']:
            raise KeyError(f'`{key}` is only updated when setting '
                            'corresponding input parameters.')
        else:
            raise KeyError(f'{key}')

        # once both histogram start point and end point have been instantiated
        # and/or when one is modified update pow_bin_edges to account for
        # the new value.
        if (key in ['pow_histogram_start', 'pow_histogram_endpoint']) \
            and isinstance(self.pow_histogram_start, Param) \
                and isinstance(self.pow_histogram_endpoint, Param):
            # Update bin edges
            super().__setattr__('pow_bin_edges',
                                self._get_pow_bin_edges_param())


    def _get_decimation_ratio_param(self, decimation_ratio):
        '''Return `decimation_ratio` as a Param.'''

        if isinstance(decimation_ratio, Param):
            if not decimation_ratio.name == 'decimation_ratio':
                raise ValueError(
                    'decimation_ratio.name must be "decimation_ratio".')
            if not (all((isinstance(e, int) and e > 0) for e in decimation_ratio.val) \
                    or not len(decimation_ratio.val) == 2) :
                raise ValueError(
                    'decimation_ratio.val must be a sequence '
                    f'of two positive integers: {decimation_ratio.val}')

            return decimation_ratio

        # Construct defaults for the new Param
        default = Param(\
            name='decimation_ratio',
            val=[1, 1],
            units='unitless',
            short_descr='Decimation ratio for processing power and phase '
                        'histograms.',
            long_descr= \
                '''
                The step size to decimate the input array for computing
                the power and phase histograms.
                For example, [2,3] means every 2nd azimuth line and
                every 3rd range line will be used to compute the histograms.
                Defaults to [1,1], i.e. no decimation will occur.
                Format: [<azimuth>, <range>]'''
            )

        if decimation_ratio is None:
            # return the default Param
            warnings.warn('`decimation_ratio` not provided.')
            return default

        elif all(isinstance(e, int) for e in decimation_ratio):
            if not len(decimation_ratio) == 2 or \
                any(e <= 0 for e in decimation_ratio):
                raise ValueError('decimation_ratio must be a sequence '
                                f'of two positive integers: {decimation_ratio}')
            
            # update the default with the new value and return it
            default.val = decimation_ratio
            return default

        else:
            raise ValueError('`decimation_ratio` must be a sequence of two '
                             f'ints, a Param, or None: {decimation_ratio}')


    def _get_pow_histogram_start_param(self, pow_histogram_start):
        '''Return pow_histogram_start as a Param.'''

        if isinstance(pow_histogram_start, Param):
            if not pow_histogram_start.name == 'pow_histogram_start':
                raise ValueError(
                    'pow_histogram_start.name must be `pow_histogram_start`')
            if not isinstance(pow_histogram_start.val, float):
                raise ValueError(f'pow_histogram_start.val must be float: '
                                f'{pow_histogram_start.val}')

            return pow_histogram_start

        # Construct defaults for the new Param
        default = Param(\
            name='pow_histogram_start',
            val=-80.0,
            units='dB',
            short_descr='Starting value for the range of the '
                        'power histogram edges.',
            long_descr= \
                '''
                Starting value (in dB) for the range of the power
                histogram edges. Defaults to -80.0.'''
            )

        if pow_histogram_start is None:
            # return the default Param
            warnings.warn('`pow_histogram_start` not provided.')
            return default

        elif isinstance(pow_histogram_start, float):
            # update the default with the new value and return it
            default.val = pow_histogram_start
            return default

        else:
            raise ValueError(
                f'pow_histogram_start must be a float, Param, or None')

    def _get_pow_histogram_endpoint_param(self, pow_histogram_endpoint):
        '''Return pow_histogram_endpoint as a Param.'''

        if isinstance(pow_histogram_endpoint, Param):
            if not pow_histogram_endpoint.name == 'pow_histogram_endpoint':
                raise ValueError('pow_histogram_endpoint.name must be '
                                 '`pow_histogram_endpoint`')
            if not isinstance(pow_histogram_endpoint.val, float):
                raise ValueError(f'pow_histogram_endpoint.val must be float: '
                                f'{pow_histogram_endpoint.val}')

            return pow_histogram_endpoint

        # Construct defaults for the new Param
        default = Param(\
            name='pow_histogram_endpoint',
            val=20.0,
            units='dB',
            short_descr='Endpoint value for the range of the '
                        'power histogram edges.',
            long_descr= \
                '''
                Endpoint value (in dB) for the range of the power
                histogram edges. Defaults to 20.0.'''
            )

        if pow_histogram_endpoint is None:
            # return the default Param
            warnings.warn('`pow_histogram_endpoint` not provided.')
            return default

        elif isinstance(pow_histogram_endpoint, float):
            # update the default with the new value and return it
            default.val = pow_histogram_endpoint
            return default

        else:
            raise ValueError(
                f'pow_histogram_endpoint must be a float, Param, or None')


    def _get_phs_in_radians_param(self, phs_in_radians):
        '''Return phs_in_radians as a Param.'''

        if isinstance(phs_in_radians, Param):
            if not phs_in_radians.name == 'phs_in_radians':
                raise ValueError('phs_in_radians.name must be `phs_in_radians`')
            if not isinstance(phs_in_radians.val, bool):
                raise ValueError('phs_in_radians.val must be bool: '
                                 f'{phs_in_radians.val}')

            return phs_in_radians

        # Construct defaults for the new Param
        default = Param(\
            name='phs_in_radians',
            val=True,
            units=None,
            short_descr='True to compute phase in radians units, False for degrees units',
            long_descr= \
                '''True to compute phase in radians units, False for 
                degrees units. Defaults to True.'''
            )

        if phs_in_radians is None:
            # return the default Param
            warnings.warn('`phs_in_radians` not provided.')
            return default

        elif isinstance(phs_in_radians, bool):
            # update the default with the new value and return it
            default.val = phs_in_radians
            return default

        else:
            raise ValueError(
                f'phs_in_radians must be a bool or None')


    def _get_tile_shape_param(self, tile_shape):
        '''Return `tile_shape` as a Param.

        TODO - this is duplicate code to other Params dataclasses. Fix.
        
        Parameters
        ----------
        tile_shape : iterable of int, Param, None
            Preferred tile shape for processing power images by batches.
     
        Returns
        -------
        tile_shape_param : Param
            `tile_shape` as a Param object.
        '''

        if isinstance(tile_shape, Param):
            if not tile_shape.name == 'tile_shape':
                raise ValueError('tile_shape.name must be "tile_shape".')
            if not (all((isinstance(e, int) and e > 0) for e in tile_shape.val) \
                    or not len(tile_shape.val) == 2) :
                raise ValueError('tile_shape.val must be a sequence '
                                 f'of two positive integers: {tile_shape.val}')

            return tile_shape

        # Construct defaults for the new Param
        default = Param(\
            name='tile_shape',
            val=[1024, 1024],
            units='unitless',
            short_descr='Preferred tile shape for processing images by batches.',
            long_descr= \
                '''
                Preferred tile shape for processing images by batches.
                Actual tile shape used during processing might
                be smaller due to shape of image.
                Format: [num_rows, num_cols]
                -1 to indicate all rows / all columns (respectively).
                Defaults to [1024, 1024] to use all columns 
                (i.e. full rows of data).'''
            )

        if tile_shape is None:
            # return the default Param
            warnings.warn('`tile_shape` not provided.')
            return default

        elif all(isinstance(e, int) for e in tile_shape):
            if not len(tile_shape) == 2 or \
                any(e <= 0 for e in tile_shape):
                raise ValueError('tile_shape must be a sequence '
                                 f'of two positive integers: {tile_shape}')
            
            # update the default with the new value and return it
            default.val = tile_shape
            return default

        else:
            raise ValueError('`tile_shape` must be a sequence of two ints, '
                             f'a Param, or None: {tile_shape}')


    def _get_pow_bin_edges_param(self):

        # Power Bin Edges - hardcode to be in decibels
        # 101 bin edges => 100 bins
        bin_edges = np.linspace(self.pow_histogram_start.val,
                                self.pow_histogram_endpoint.val,
                                num=101,
                                endpoint=True)

        template = Param(
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

        return template


    def _get_phs_bin_edges_param(self):

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

        template = Param(
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

        return template


    @staticmethod
    def get_path_to_group_in_runconfig():
        return ['runconfig','groups','processing','qa','qa_reports','histogram']


    @staticmethod
    def populate_runcfg(runconfig_cm):
        '''
        Parameters
        ----------
        runconfig_cm : ruamel.yaml.comments.CommentedMap
            The base commented map to add the parameters to.
        '''
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
        Populate h5_file HDF5 file this dataclass' processing parameters.

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



@dataclass
class RSLCRootParams:

    workflows: WorkflowsParams
    prodpath: Optional[ProductPathGroupParams] = None
    anc_files: Optional[DynamicAncillaryFileParams] = None
    power_img: Optional[RSLCPowerImageParams] = None
    histogram: Optional[RSLCHistogramParams] = None

    # abs_cal: Optional[AbsCalParams] = None
    # nesz: Optional[NESZParams] = None
    # pta: Optional[PointTargetAnalyzerParams] = None

    def __post_init__(self):

        # Ensure that the minimum parameters were provided

        # If any of the workflows requested, then prodpath must be an
        # instance of ProductPathGroupParams.
        # prodpath is only optional in the case of doing a dumpconfig
        if any([getattr(self.workflows, field.name) \
                            for field in fields(self.workflows)]):
            if not isinstance(self.prodpath, ProductPathGroupParams):
                raise ValueError('prodpath parameter of type '
                    'ProductPathGroupParams is required to run any of the '
                    'QA workflows.')

        if self.workflows.qa_reports:
            if self.power_img is None or \
                not isinstance(self.power_img, RSLCPowerImageParams):
                raise ValueError('power_img parameter of type '
                    'RSLCPowerImageParams is required to run the '
                    'requested qa_reports workflow')

            if self.histogram is None or \
                not isinstance(self.histogram, RSLCHistogramParams):
                raise ValueError('histogram parameter of type '
                    'RSLCHistogramParams is required to run the '
                    'requested qa_reports workflow')

        if self.workflows.absolute_calibration_factor:
            if self.anc_files is None or \
                not isinstance(self.anc_files, DynamicAncillaryFileParams):
                raise ValueError('anc_files parameter of type '
                    'DynamicAncillaryFileParams is required to run the '
                    'requested absolute_calibration_factor workflow')

            # if self.abs_cal is None or \
            #     not isinstance(self.abs_cal, AbsCalParams):
            #     raise ValueError('abs_cal parameter of type '
            #         'AbsCalParams is required to run the '
            #         'requested absolute_calibration_factor workflow')

        # if self.workflows.nesz:
        #     if self.nesz is None or \
        #         not isinstance(self.nesz, NESZParams):
        #         raise ValueError('nesz parameter of type '
        #             'NESZParams is required to run the '
        #             'requested nesz workflow')

        if self.workflows.point_target_analyzer:
            if self.anc_files is None or \
                not isinstance(self.anc_files, DynamicAncillaryFileParams):
                raise ValueError('anc_files parameter of type '
                    'DynamicAncillaryFileParams is required to run the '
                    'requested point_target_analyzer workflow')

            # if self.abs_cal is None or \
            #     not isinstance(self.pta, PointTargetAnalyzerParams):
            #     raise ValueError('pta parameter of type '
            #         'PointTargetAnalyzerParams is required to run the '
            #         'requested point_target_analyzer workflow')

        # Ensure all provided attributes are a subtype of BaseParams
        for attr_name in self.__annotations__:
            attr = getattr(self, attr_name)
            if attr is not None:
                assert issubclass(type(attr), BaseParams), \
                    f'{attr_name} attribute must be a subclass of BaseParams'

    @staticmethod
    def dump_runconfig_template():
        '''Outputs the runconfig template (with default values) to stdout.
        '''

        # Build a ruamel yaml object that contains the runconfig structure
        yaml = YAML()
        yaml.indent(mapping=4, offset=4)

        runconfig_cm = CM()

        # Populate the yaml object. This order determines the order
        # the groups will appear in the runconfig.
        DynamicAncillaryFileParams.populate_runcfg(runconfig_cm)
        ProductPathGroupParams.populate_runcfg(runconfig_cm)
        WorkflowsParams.populate_runcfg(runconfig_cm)
        RSLCPowerImageParams.populate_runcfg(runconfig_cm)
        RSLCHistogramParams.populate_runcfg(runconfig_cm)

        # output to console. Let user stream that into a file.
        yaml.dump(runconfig_cm, sys.stdout)


    def save_params_to_stats_file(self, h5_file, bands=('LSAR')):
        '''Save input parameters to stats.h5 file.

        Parameters
        ----------
        h5_file : h5py.File
            The output file to save QA metrics, etc. to
        '''
        for params_obj in fields(self):
            # If a workflow was not requested, its RootParams attribute
            # will be None, so there will be no params to add to the h5 file
            po = getattr(self, params_obj.name)
            if po is not None:
                po.write_params_to_h5(h5_file, bands=bands)


def parse_rslc_runconfig(runconfig_yaml):
    # parse runconfig into a dict structure
    parser = YAML(typ='safe')
    with open(runconfig_yaml, 'r') as f:
        user_runconfig = parser.load(f)

    # Construct WorkflowsParams dataclass (required for all workflows)
    rncfg_path = WorkflowsParams.get_path_to_group_in_runconfig()
    params_dict = nisarqa.get_nested_element_in_dict(user_runconfig, rncfg_path)
    workflows_params = WorkflowsParams(**params_dict)

    # Construct DynamicAncillaryFileParams dataclass
    # Only two of the CalVal workflows use the dynamic_ancillary_file_group
    if workflows_params.absolute_calibration_factor or \
        workflows_params.point_target_analyzer:
        rncfg_path = DynamicAncillaryFileParams.get_path_to_group_in_runconfig()
        params_dict = nisarqa.get_nested_element_in_dict(user_runconfig, rncfg_path)
        dyn_anc_files = DynamicAncillaryFileParams(**params_dict)
    else:
        dyn_anc_files = None

    # Construct ProductPathGroupParams dataclass (required for all workflows)
    rncfg_path = ProductPathGroupParams.get_path_to_group_in_runconfig()
    params_dict = nisarqa.get_nested_element_in_dict(user_runconfig, rncfg_path)
    product_path_params = ProductPathGroupParams(**params_dict)

    # Construct RSLCPowerImageParams dataclass
    if workflows_params.qa_reports:
        rncfg_path = RSLCPowerImageParams.get_path_to_group_in_runconfig()
        params_dict = nisarqa.get_nested_element_in_dict(user_runconfig, rncfg_path)
        pow_img_params = RSLCPowerImageParams(**params_dict)
    else:
        pow_img_params = None

    # Construct RSLCHistogramParams dataclass
    if workflows_params.qa_reports:
        rncfg_path = RSLCHistogramParams.get_path_to_group_in_runconfig()
        params_dict = nisarqa.get_nested_element_in_dict(user_runconfig, rncfg_path)
        histogram_params = RSLCHistogramParams(**params_dict)
    else:
        histogram_params = None


    rslc_params = RSLCRootParams(workflows=workflows_params,
                                 anc_files=dyn_anc_files,
                                 prodpath=product_path_params,
                                 power_img=pow_img_params,
                                 histogram=histogram_params
                                 )

    return rslc_params


__all__ = nisarqa.get_all(__name__, objects_to_skip)
