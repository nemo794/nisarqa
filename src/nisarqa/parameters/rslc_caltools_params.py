import os
import sys
import warnings
from dataclasses import dataclass, field, fields
from typing import ClassVar, Iterable, Optional, Union

import nisarqa
import numpy as np
from nisarqa.parameters.nisar_params import *
from ruamel.yaml import YAML
from ruamel.yaml import CommentedMap as CM

objects_to_skip = nisarqa.get_all(__name__)


@dataclass
class WorkflowsParamGroup(YamlParamGroup):
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

    # default value for all workflows
    def_val: ClassVar[bool] = False

    # Generic description for all workflows
    descr: ClassVar[str] = f'Flag to run `%s` workflow. Default: `{def_val}`'

    # Dataclass fields
    name = 'validate', 
    yaml_attrs = YamlAttrs(name=name, descr=descr % name)
    validate: bool = field(init=False, default=def_val,
                             metadata={'yaml_attrs': yaml_attrs})
 
    name = 'qa_reports'
    yaml_attrs = YamlAttrs(name=name, descr=descr % name)
    qa_reports: bool = field(init=False, default=def_val,
                             metadata={'yaml_attrs': yaml_attrs})

    name = 'absolute_calibration_factor', 
    yaml_attrs = YamlAttrs(name=name, descr=descr % name)
    absolute_calibration_factor: bool = field(init=False, default=def_val,
                             metadata={'yaml_attrs': yaml_attrs})

    name = 'nesz', 
    yaml_attrs = YamlAttrs(name=name, descr=descr % name)
    nesz: bool = field(init=False, default=def_val,
                             metadata={'yaml_attrs': yaml_attrs})

    name = 'point_target_analyzer', 
    yaml_attrs = YamlAttrs(name=name, descr=descr % name)
    point_target_analyzer: bool = field(init=False,
                            default=def_val,
                            metadata={'yaml_attrs': yaml_attrs})

    # def __init__(
    #     self, 
    #     validate: bool,
    #     qa_reports: bool,
    #     absolute_calibration_factor: bool,
    #     nesz: bool,
    #     point_target_analyzer: bool, 
    #     ):

    #     # validate and initialize all attributes.
    #     self.validate = validate
    #     self.qa_reports = qa_reports
    #     self.absolute_calibration_factor = absolute_calibration_factor
    #     self.nesz = nesz
    #     self.point_target_analyzer = point_target_analyzer


    @staticmethod
    def get_path_to_group_in_runconfig():
        return ['runconfig','groups','qa','workflows']

    # def _arg_2_param(self, attr_name, arg_val):
    #     '''
    #     Wrap the argument value into a YamlParam instance.

    #     The returned YamlParam instance will use pre-set information
    #     as documentation for the input `arg_val`.

    #     Parameters
    #     ----------
    #     attr_name : str
    #         The name of the attribute of WorkflowsParamGroup that `yaml_param`
    #         will be assigned to.
    #     arg_val : Any
    #         Input argument value. Will be assigned to `val` attribute
    #         of `yaml_param`.

    #     Returns
    #     -------
    #     yaml_param : YamlParam
    #         `arg_val` wrapped in a YamlParam.
    #     '''
    #     # Wrap in a Param.
    #     def_val = self.get_default_arg_for_yaml(attr_name)
    #     attrs = YamlAttrs(name=attr_name,
    #                       descr=f'Flag to run `{attr_name}` workflow. '
    #                             f'Default: `{def_val}`'
    #                       )
    #     return YamlParam(val=arg_val, yaml_attrs=attrs)


    @staticmethod
    def _validate_arg(attr, attr_name):
        '''
        Validate that `attr` is of the correct type for the
        WorkflowsParamGroup's attribute `attr_name`.

        Parameters
        ----------
        attr : bool
            Argument value for `attr_name`.
        attr_name : str
            The name of the attribute of WorkflowsParamGroup for `attr`
        '''

        # Validate the attr.val
        if not isinstance(attr, bool):
            raise TypeError(f'`{attr_name}` must be of type bool. '
                            f'It is {type(attr)}')

    # def _validate_2_param(self, validate):
    #     # Wrap in a Param.
    #     name = 'validate'
    #     def_val = self.get_default_arg_for_yaml(name)
    #     attrs = YamlAttrs(name=name,
    #                       descr=f'Flag to run validate workflow. Default: {def_val}'
    #                       )
    #     return YamlParam(val=validate, yaml_attrs=attrs)
    
    @property
    def validate(self) -> bool:
        return self._validate
    
    @validate.setter
    def validate(self, val: bool):
        # Validate input
        self._validate_arg(val, 'validate')
        # Set attribute
        self._validate = val


    @property
    def qa_reports(self) -> bool:
        return self._qa_reports
    
    @qa_reports.setter
    def qa_reports(self, val: bool):
        # Validate input
        self._validate_arg(val, 'qa_reports')
        # Set attribute
        self._qa_reports = val


    @property
    def abs_cal(self) -> bool:
        return self._abs_cal
    
    @abs_cal.setter
    def abs_cal(self, val: bool):
        # Validate input
        self._validate_arg(val, 'abs_cal')
        # Set attribute
        self._abs_cal = val


    @property
    def nesz(self) -> bool:
        return self._nesz
    
    @nesz.setter
    def nesz(self, val: bool):
        # Validate input
        self._validate_arg(val, 'nesz')
        # Set attribute
        self._nesz = val


    @property
    def pta(self) -> YamlParam[bool]:
        return self._pta
    
    @pta.setter
    def pta(self, pta_attr: YamlParam[bool]):
        # Validate input
        self._validate_arg(pta_attr, 'pta')
        # Set attribute
        self._pta = pta_attr


@dataclass
class InputFileGroupParamGroup(YamlParamGroup):
    '''
    Parameters from the Input File Group runconfig group.

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
    qa_input_file : str
        The input NISAR product file name (with path).
    '''

    yaml_attrs: ClassVar[YamlAttrs] = YamlAttrs(
        name='qa_input_file',
        descr='''
        Filename of the input file for QA.
        REQUIRED for QA. NOT REQUIRED if only running Product SAS.
        If Product SAS and QA SAS are run back-to-back,
        this field should be identical to `sas_output_file`.
        Otherwise, this field should contain the filename of the single
        NISAR product for QA to process.'''
        )

    # `qa_input_file` is required.
    qa_input_file: str = field(metadata={'yaml_attrs': yaml_attrs})

    def __init__(self, qa_input_file: str):
        self.qa_input_file = qa_input_file


    @staticmethod
    def get_path_to_group_in_runconfig():
        return ['runconfig','groups','input_file_group']

    @property
    def qa_input_file(self) -> str:
        return self._qa_input_file
    
    @qa_input_file.setter
    def qa_input_file(self, qa_input_file: str):
        # Validate input
        nisarqa.validate_is_file(qa_input_file, 'qa_input_file', '.h5')

        # Set attribute
        self._qa_input_file = qa_input_file


@dataclass
class DynamicAncillaryFileParamGroup(YamlParamGroup):
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

    Parameters
    ----------
    corner_reflector_file : str
        The input corner reflector file's file name (with path).
        Required for the Absolute Calibration Factor and Point Target
        Analyzer workflows.
    '''

    # Set attributes to Param type for correct downstream type checking
    corner_reflector_file: YamlParam[str]

    def __init__(self, corner_reflector_file: str):

        self.corner_reflector_file = \
            self._corner_reflector_file_2_param(corner_reflector_file)


    @staticmethod
    def get_path_to_group_in_runconfig():
        return ['runconfig','groups','dynamic_ancillary_file_group']


    def _corner_reflector_file_2_param(self, corner_reflector_file):
        '''Return `corner_reflector_file` as a YamlParam'''

        # Construct defaults for the new Param
        attrs = YamlAttrs(
            name='corner_reflector_file',
            descr='''
            Locations of the corner reflectors in the input product.
            Only required if `absolute_calibration_factor` or
            `point_target_analyzer` runconfig params are set to True for QA.'''
            )

        return YamlParam(corner_reflector_file, attrs)

    @property
    def corner_reflector_file(self) -> YamlParam[str]:
        return self._corner_reflector_file
    
    @corner_reflector_file.setter
    def corner_reflector_file(self, corner_reflector_file_attr: YamlParam[str]):
        # Validate input
        if not issubclass(type(corner_reflector_file_attr), YamlParam):
            raise TypeError(f'`corner_reflector_file` must be a subclass of '
                        f'YamlParam. It is {type(corner_reflector_file_attr)}')

        val = corner_reflector_file_attr.val
        nisarqa.validate_is_file(val, 'corner_reflector_file.val`', '.csv')

        # Set attribute
        self._corner_reflector_file = corner_reflector_file_attr


@dataclass
class ProductPathGroupParamGroup(YamlParamGroup):
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

    Parameters
    ----------
    qa_output_dir : str, optional
        Filepath to the output directory to store NISAR QA output files.
        Defaults to './qa'
    '''

    qa_output_dir: YamlParam[str]

    def __init__(self, qa_output_dir: Optional[str] = './qa'):
        self.qa_output_dir = self._qa_output_dir_2_param(qa_output_dir)


    def _qa_output_dir_2_param(self, qa_output_dir):
        '''Return `qa_output_dir` as a YamlParam.'''

        default_val = self.get_default_arg_for_yaml(attr_name='qa_output_dir')

        # Construct attributes for the new YamlParam
        attrs = YamlAttrs(
            name='corner_reflector_file',
            descr=f'''
                Output directory to store all QA output files.
                Defaults to "{default_val}"'''
            )

        return YamlParam(qa_output_dir, attrs)


    @property
    def qa_output_dir(self) -> YamlParam[str]:
        return self._qa_output_dir
    
    @qa_output_dir.setter
    def qa_output_dir(self, qa_output_dir_attr: YamlParam[str]):
        # Validate input
        if not issubclass(type(qa_output_dir_attr), YamlParam):
            raise TypeError(f'`qa_output_dir_attr` must be a subclass of '
                            f'YamlParam. It is {type(qa_output_dir_attr)}')

        val = qa_output_dir_attr.val

        if not isinstance(val, str):
            raise TypeError(f'`qa_output_dir_attr.val` must be a str')

        # If this directory does not exist, make it.
        if not os.path.isdir(val):
            print(f'Creating QA output directory: {val}')
            os.makedirs(val, exist_ok=True)

        # Set attribute
        self._qa_output_dir = qa_output_dir_attr


    @staticmethod
    def get_path_to_group_in_runconfig():
        return ['runconfig','groups','product_path_group']


@dataclass
class RSLCPowerImageParamGroup(YamlHDF5ParamGroup):
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
    linear_units: YamlParam[bool]
    nlooks_freqa: YamlParam[Optional[Union[int, Iterable[int]]]]
    nlooks_freqb: YamlParam[Optional[Union[int, Iterable[int]]]]
    num_mpix: YamlParam[int]
    middle_percentile: YamlHDF5Param[float]
    gamma: YamlHDF5Param[Optional[float]]
    tile_shape: YamlParam[list[int]]

    # Auto-generated attributes.
    # `pow_units` is set by the `linear_units` attribute.
    pow_units: HDF5Param = field(init=False)

    def __init__(self,
                 linear_units: bool = True,
                 nlooks_freqa: Optional[Union[int, Iterable[int]]] = None,
                 nlooks_freqb: Optional[Union[int, Iterable[int]]] = None,
                 num_mpix: float = 4.0,
                 middle_percentile: float = 95.0,
                 gamma: Optional[float] = None,
                 tile_shape: Iterable[int] = (1024,1024)
                 ):

        self.linear_units = self._linear_units_2_param(linear_units)
        self.nlooks_freqa = self._nlooks_2_param(nlooks=nlooks_freqa, freq='A')
        self.nlooks_freqb = self._nlooks_2_param(nlooks=nlooks_freqb, freq='B')
        self.num_mpix = self._num_mpix_2_param(num_mpix=num_mpix)
        self.middle_percentile = \
            self._middle_percentile_2_param(middle_percentile)
        self.gamma = self._gamma_2_param(gamma)
        self.tile_shape = self._tile_shape_2_param(tile_shape)


    @staticmethod
    def get_path_to_group_in_runconfig():
        return ['runconfig','groups','qa','qa_reports','power_image']


    def _linear_units_2_param(self, linear_units):
        '''Return `linear_units` as a Param.'''

        # Construct attributes for the new Param
        yaml_attrs = YamlAttrs(
            name='linear_units',
            descr='''
                True to compute power in linear units when generating 
                the power image for the browse images and graphical
                summary PDF. False for decibel units.
                Defaults to True.'''
            )

        return YamlParam(val=linear_units, yaml_attrs=yaml_attrs)


    @property
    def linear_units(self) -> YamlParam[bool]:
        return self._linear_units


    @linear_units.setter
    def linear_units(self, linear_units_attr: YamlParam[str]):
        # Validate input
        val = linear_units_attr.val
        if not isinstance(val, bool):
            raise TypeError(f'`linear_units.val` must be a bool: {val}')

        # Set attributes
        self._linear_units = linear_units_attr

        self._pow_units = self._pow_units_2_param()


    def _nlooks_2_param(self, nlooks, freq):
        '''Return the number of looks for given frequency as a Param.
        
        Parameters
        ----------
        nlooks : int or iterable of int or None
            Number of looks along each axis of the input array 
            for the specified frequency.
        freq : str
            The frequnecy to assign this number of looks to.
            Options: 'A' or 'B'

        Returns
        -------
        nlooks_param : YamlParam
            `nlooks` for frequency `freq` as a Param object.
        '''

        # Construct attributes for the new Param
        default_val = self.get_default_arg_for_yaml(f'nlooks_freq{freq.lower()}')
        yaml_attrs = YamlAttrs(
            name=f'nlooks_freq{freq.lower()}',
            descr=f'''
                Number of looks along each axis of the Frequency {freq.upper()}
                image arrays for multilooking the power image.
                Format: [<num_rows>, <num_cols>]
                Example: [6,7]
                If not provided, the QA code to compute the nlooks values 
                that would produce an approx. `num_mpix` MPix browse image.
                Defaults to {default_val}.'''
            )

        return YamlParam(val=nlooks, yaml_attrs=yaml_attrs)


    @staticmethod
    def _validate_nlooks(nlooks, freq):
        '''
        Raise exception if `nlooks` is not a valid input.

        Parameters
        ----------
        nlooks : int or iterable of int or None
            Number of looks along each axis of the input array 
            for the specified frequency.
        freq : str
            The frequency to assign this number of looks to.
            Options: 'A' or 'B'
        '''
        if isinstance(nlooks, int):
            if nlooks < 1:
                raise ValueError(
                    f'`nlooks_freq{freq.lower()}` must be >= 1: {nlooks}')

        elif isinstance(nlooks, (list, tuple)):
            if all(isinstance(e, int) for e in nlooks):
                if any((e < 1) for e in nlooks) or not len(nlooks) == 2:
                    raise TypeError(
                        f'nlooks_freq{freq.lower()} must be an int or a '
                        f'sequence of two ints, which are >= 1: {nlooks}')
        elif nlooks is None:
            # the code will use num_mpix to compute `nlooks` instead.
            pass
        else:
            raise TypeError('`nlooks` must be of type int, iterable of int, '
                            f'or None: {nlooks}')

    @property
    def nlooks_freqa(self) -> YamlHDF5Param[str]:
        return self._nlooks_freqa
    
    @nlooks_freqa.setter
    def nlooks_freqa(self, nlooks_freqa_attr: YamlParam[str]):
        # Validate input
        val = nlooks_freqa_attr.val
        self._validate_nlooks(val, 'A')

        # Set attribute
        self._nlooks_freqa = nlooks_freqa_attr

    @property
    def nlooks_freqb(self) -> YamlHDF5Param[str]:
        return self._nlooks_freqb
    
    @nlooks_freqb.setter
    def nlooks_freqb(self, nlooks_freqb_attr: YamlParam[str]):
        # Validate input
        val = nlooks_freqb_attr.val
        self._validate_nlooks(val, 'B')

        # Set attribute
        self._nlooks_freqb = nlooks_freqb_attr


    def _num_mpix_2_param(self, num_mpix):
        '''Return `num_mpix` as a Param.'''

        # Construct attributes for the new Param
        default_val = self.get_default_arg_for_yaml(attr_name='num_mpix')
        yaml_attrs = YamlAttrs(
            name='num_mpix',
            descr=f'''
                The approx. size (in megapixels) for the final
                multilooked browse image(s). Defaults to {default_val} MPix.
                If `nlooks_freq*` parameter(s) is not None, nlooks
                values will take precedence.'''
            )

        return YamlParam(val=num_mpix, yaml_attrs=yaml_attrs)


    @property
    def num_mpix(self) -> YamlParam[bool]:
        return self._num_mpix
    
    @num_mpix.setter
    def num_mpix(self, num_mpix_attr: YamlParam[str]):
        # Validate input
        val = num_mpix_attr.val
        if not isinstance(val, float):
            raise TypeError(f'`num_mpix_attr.val` must be a float: {val}')

        if val <= 0.0:
            raise TypeError(f'`num_mpix_attr.val` must be >= 0.0: {val}')


        # Set attribute
        self._num_mpix = num_mpix_attr


    def _middle_percentile_2_param(self, middle_percentile):
        '''Return `middle_percentile` as a Param.'''

        # Construct attributes for the new Param
        default_val = self.get_default_arg_for_yaml(attr_name='middle_percentile')
        yaml_attrs = YamlAttrs(
            name='middle_percentile',
            descr=f'''
                Defines the middle percentile range of the image array
                that the colormap covers. Must be in the range [0.0, 100.0].
                Defaults to {default_val}.'''
                )

        hdf5_attrs = HDF5Attrs(
            name='powerImageMiddlePercentile',
            units='unitless',
            descr='Middle percentile range of the image array '
                  'that the colormap covers',
            path=self.path_to_processing_group_in_stats_h5
            )

        return YamlHDF5Param(val=middle_percentile,
                             yaml_attrs=yaml_attrs,
                             hdf5_attrs=hdf5_attrs)


    @property
    def middle_percentile(self) -> YamlHDF5Param[float]:
        return self._middle_percentile
    
    @middle_percentile.setter
    def middle_percentile(self, middle_percentile_attr: YamlHDF5Param[float]):
        # Validate input
        val = middle_percentile_attr.val
        if not isinstance(val, float):
            raise TypeError(f'`middle_percentile_attr.val` must be a float: {val}')

        if val < 0.0 or val > 100.0:
            raise TypeError('`middle_percentileattr.val` is '
                f'{val}, must be in range [0.0, 100.0]')

        # Set attribute
        self._middle_percentile = middle_percentile_attr


    def _gamma_2_param(self, gamma):
        '''Return `gamma` as a Param.'''

        # Construct attributes for the new Param
        default_val = self.get_default_arg_for_yaml(attr_name='gamma')
        yaml_attrs = YamlAttrs(
            name='gamma',
            descr=f'Gamma correction applied to power image. Default: {default_val}.'
                )

        hdf5_attrs = HDF5Attrs(
            name='powerImageGammaCorrection',
            units='unitless',
            descr='''
                The gamma correction parameter.
                Gamma will be applied as follows:
                    array_out = normalized_array ^ gamma
                where normalized_array is a copy of the image with values
                scaled to the range [0,1]. 
                The image colorbar will be defined with respect to the input
                image values prior to normalization and gamma correction.
                Defaults to None (no normalization, no gamma correction)''',
            path=self.path_to_processing_group_in_stats_h5
            )

        return YamlHDF5Param(val=gamma,
                             yaml_attrs=yaml_attrs,
                             hdf5_attrs=hdf5_attrs)

    @property
    def gamma(self) -> YamlHDF5Param[Optional[float]]:
        return self._gamma
    
    @gamma.setter
    def gamma(self, gamma_attr: YamlHDF5Param[Optional[float]]):
        # Validate input
        val = gamma_attr.val

        if isinstance(val, float):
            if (val < 0.0):
                raise ValueError('If `gamma_attr.val` is a float, it must be'
                                f' non-negative: {val}')
        elif val is not None:
            raise TypeError('`gamma_attr.val` must be a float or None. '
                            f'Value: {val}, Type: {type(val)}')

        # Set attribute
        self._gamma = gamma_attr


    def _tile_shape_2_param(self, tile_shape):
        '''Return `tile_shape` as a Param.
                
        TODO - this is duplicate code to other Params dataclasses. Fix.
        '''

        # Construct attributes for the new Param
        default_val = self.get_default_arg_for_yaml(attr_name='tile_shape')
        yaml_attrs = YamlAttrs(
            name='tile_shape',
            descr=f'''
                Preferred tile shape for processing images by batches.
                Actual tile shape used during processing might
                be smaller due to shape of image.
                Format: [<num_rows>, <num_cols>]
                -1 to indicate all rows / all columns (respectively).
                Defaults to {list(default_val)} to use all columns 
                (i.e. full rows of data).'''
            )

        return YamlParam(val=tile_shape, yaml_attrs=yaml_attrs)


    @property
    def tile_shape(self) -> YamlParam[Iterable[int]]:
        return self._tile_shape
    

    @tile_shape.setter
    def tile_shape(self, tile_shape_attr: YamlParam[Iterable[int]]):
        # Validate input
        val = tile_shape_attr.val
        if not isinstance(val, (list, tuple)):
            raise TypeError('`tile_shape_attr.val` must be a list or tuple: '
                                f'{val}')

        if not len(val) == 2:
            raise TypeError('`tile_shape_attr.val` must have a length'
                                f'of two: {val}')

        if not all(isinstance(e, int) for e in val):
            raise TypeError('`tile_shape_attr.val` must contain only '
                                f'integers: {val}')

        if any(e < -1 for e in val):
            raise TypeError('`tile_shape_attr.val` must contain only '
                                f' values >= -1: {val}')

        # Set attribute
        self._tile_shape = tile_shape_attr


    def _pow_units_2_param(self):
        '''Return `pow_units` as a Param.'''

        if self.linear_units.val:
            pow_units='linear'
        else:
            pow_units='dB'

        # Construct attributes for the new Param
        hdf5_attrs = HDF5Attrs(
            name='powerImagePowerUnits',
            units=None,
            descr='''
                Units of the power image.
                If `linear_units` is True, this will be set to 'linear'.
                If `linear_units` is False, this will be set to 'dB'.''',
            path=self.path_to_processing_group_in_stats_h5
            )

        return HDF5Param(val=pow_units, hdf5_attrs=hdf5_attrs)


    @property
    def pow_units(self) -> HDF5Param[str]:
        # There is no public setter for this property. It can only be set
        # within the `linear_units` setter.
        return self._pow_units
    

@dataclass
class RSLCHistogramParamGroup(YamlHDF5ParamGroup):
    # each param group will define the internal nesting path within its file (ABC property),
    # and how to populate its file.
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
        When `decimation_ratio` is updated, the attributes
        `pow_bin_edges` will be updated to match.
    pow_histogram_bin_edges_range : pair of float, optional
        The dB range for the power histogram's bin edges. Endpoint will
        be included. Defaults to (-80.0,20.0).
        Format: (<starting value>, <endpoint>)
        When `pow_histogram_bin_edges_range` is updated, the attribute
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
        This is set whenever `pow_histogram_bin_edges_range` is set.
        This will be stored as a numpy.ndarray in `pow_bin_edges.val`
    phs_bin_edges : Param
        The bin edges (including endpoint) to use when computing
        the phase histograms. This is set whenever `phs_in_radians` is set.
        This will be stored as a numpy.ndarray in `phs_bin_edges.val`
        If `phs_in_radians` is True, this will be set to 100 
        uniformly-spaced bins in range [-pi,pi], including endpoint.
        If `phs_in_radians` is False, this will be set to 100
        uniformly-spaced bins in range [-180,180], including endpoint.
    '''

    # Set attributes to Param type for correct downstream type checking
    # User-Provided attributes:
    decimation_ratio: YamlParam
    pow_histogram_bin_edges_range: YamlParam
    phs_in_radians: YamlParam
    tile_shape: YamlParam

    # Auto-generated attributes
    # Power Bin Edges (generated from `pow_histogram_bin_edges_range`)
    pow_bin_edges: HDF5Param = field(init=False)

    # Phase bin edges (generated from `phs_in_radians`)
    phs_bin_edges: HDF5Param = field(init=False)

    # HDF5-specific attributes, generated from `decimation_ratio`
    az_decimation: HDF5Param = field(init=False)
    rng_decimation: HDF5Param = field(init=False)

    # # scratch thoughts for populate runconfig
    # flag = False
    # for field in fields(cls):
    #     if field has yaml_attrs:
    #         if field.name in dict_of_defaults:
    #             set it in the runconfig
    #         else:
    #             this means it is a required argument, so set this to an empty string


    def __init__(self,
                decimation_ratio: Iterable[int] = (10,10),
                pow_histogram_bin_edges_range: Iterable[float] = (-80.0,20.0),
                phs_in_radians: bool = True,
                tile_shape: Iterable[int] = (1024,1024)
                ):

        self.decimation_ratio = \
            self._decimation_ratio_2_param(decimation_ratio)
        self.pow_histogram_bin_edges_range = \
            self._pow_histogram_bin_edges_range_2_param(
                pow_histogram_bin_edges_range)
        self.phs_in_radians = self._phs_in_radians_2_param(phs_in_radians)
        self.tile_shape = self._tile_shape_2_param(tile_shape)


    @staticmethod
    def get_path_to_group_in_runconfig():
        return ['runconfig','groups','qa','qa_reports','histogram']


    def _decimation_ratio_2_param(self, decimation_ratio):
        '''Return `decimation_ratio` as a Param.'''

        # Construct attributes for the new Param
        def_val = self.get_default_arg_for_yaml(attr_name='decimation_ratio')
        yaml_attrs = YamlAttrs(
            name='decimation_ratio',
            descr=f'''
                The step size to decimate the input array for computing
                the power and phase histograms.
                For example, [2,3] means every 2nd azimuth line and
                every 3rd range line will be used to compute the histograms.
                Defaults to {list(def_val)}.
                Format: [<azimuth>, <range>]'''
            )
        
        return YamlParam(val=decimation_ratio, yaml_attrs=yaml_attrs)


    @property
    def decimation_ratio(self) -> YamlParam[Iterable[int]]:
        return self._decimation_ratio


    @decimation_ratio.setter
    def decimation_ratio(self, decimation_ratio_attr: YamlParam[Iterable[int]]):
        # Validate input
        val = decimation_ratio_attr.val

        if not isinstance(val, (list, tuple)):
            raise TypeError('`decimation_ratio_attr.val` must be a list or'
                            f' tuple: {val}')

        if not len(val) == 2:
            raise ValueError('`decimation_ratio_attr.val` must have a length of '
                            f'two: {val}')

        if not all(isinstance(e, int) for e in val):
            raise TypeError('`decimation_ratio_attr.val` must contain only '
                            f'integers: {val}')

        if any(e <= 0 for e in val):
            raise ValueError('`decimation_ratio_attr.val` must contain only'
                            f' positive values: {val}')

        # Set attributes
        # Decimation Ratio must be set first, before setting the individual
        # azimuth and range attributes.
        self._decimation_ratio = decimation_ratio_attr

        self._az_decimation = self._decimation_direction_2_param(
                                                        direction='az')
        self._rng_decimation = self._decimation_direction_2_param(
                                                        direction='range')


    def _decimation_direction_2_param(self, direction):
        '''
        Take the value from `self.decimation_ratio` that corresponds to 
        `direction`, and return that value wrapped in a HDF5Param.

        Parameters
        ----------
        direction : str
            'az' for azimuth direction, 'range' for range direction.

        Returns
        -------
        hdf5_param : HDF5Param
            The value from `self.decimation_ratio` that corresponds to 
            `direction` that has been wrapped into a HDF5Param.
        '''
        # Validate input
        if not isinstance(self.decimation_ratio, YamlParam):
            raise TypeError('`self.decimation_ratio` must be a YamlParam:'
                            f' {self.decimation_ratio}')

        if direction not in ('az', 'range'):
            raise ValueError(f'`direction` must be "az" or "range": {direction}')

        if direction == 'az':
            names = ('Az', 'Azimuth')
            decimation_val = self.decimation_ratio.val[0]
        else:
            names = ('Range', 'Range')
            decimation_val = self.decimation_ratio.val[1]

        # Construct attributes for the new Param
        hdf5_attrs = HDF5Attrs(
            name=f'histogramDecimation{names[0]}',
            units='unitless',
            descr=f'{names[1]} decimation stride used to compute power and '
                   'phase histograms',
            path=self.path_to_processing_group_in_stats_h5
            )
        
        return HDF5Param(val=decimation_val, hdf5_attrs=hdf5_attrs)


    @property
    def az_decimation(self):
        '''Get az_decimation attribute value.
        
        Note that there is no setter for this property; the value
        can only be set within the @decimation_ratio.setter
        '''
        return self._az_decimation


    @property
    def rng_decimation(self):
        '''Get rng_decimation attribute value.
        
        Note that there is no setter for this property; the value
        can only be set within the @decimation_ratio.setter
        '''
        return self._rng_decimation


    def _pow_histogram_bin_edges_range_2_param(self, pow_histogram_bin_edges_range):
        '''Return `pow_histogram_bin_edges_range` as a Param.'''

        # Construct defaults for the new Param
        yaml_attrs = YamlAttrs(
            name='pow_histogram_bin_edges_range',
            descr='''
                The dB range for the power histogram's bin edges. Endpoint will
                be included. Defaults to [-80.0,20.0].
                Format: [<starting value>, <endpoint>]'''
            )

        return YamlParam(val=pow_histogram_bin_edges_range, yaml_attrs=yaml_attrs)


    @property
    def pow_histogram_bin_edges_range(self) -> YamlParam[float]:
        return self._pow_histogram_bin_edges_range

    
    @pow_histogram_bin_edges_range.setter
    def pow_histogram_bin_edges_range(self, 
        pow_histogram_bin_edges_range_attr: YamlParam[Iterable[float]]):

        # Validate input
        val = pow_histogram_bin_edges_range_attr.val

        if not isinstance(val, (list, tuple)):
            raise TypeError('`pow_histogram_bin_edges_range_attr.val` must'
                            f' be a list or tuple: {val}')

        if not len(val) == 2:
            raise ValueError('`pow_histogram_bin_edges_range_attr.val` must'
                            f' have a length of two: {val}')

        if not all(isinstance(e, float) for e in val):
            raise TypeError('`pow_histogram_bin_edges_range_attr.val` must'
                            f' contain only float: {val}')

        if val[0] >= val[1]:
            raise ValueError(
                '`pow_histogram_bin_edges_range_attr.val` has format '
                f'[<starting value>, <endpoint>]; <starting value> '
                f'must be less than <ending value>: {val}')

        # Set attributes
        self._pow_histogram_bin_edges_range = pow_histogram_bin_edges_range_attr

        self._pow_bin_edges = self._pow_bin_edges_2_param()


    @property
    def pow_bin_edges(self):
        return self._pow_bin_edges


    def _pow_bin_edges_2_param(self):
        '''Return `pow_bin_edges` as a Param.'''

        # Validate input
        if not isinstance(self.pow_histogram_bin_edges_range, YamlParam):
            raise TypeError('`self.pow_histogram_bin_edges_range` must be a '
                            f'YamlParam: {self.pow_histogram_bin_edges_range}')

        # Power Bin Edges - hardcode to be in decibels
        # 101 bin edges => 100 bins
        bin_edges = np.linspace(self.pow_histogram_bin_edges_range.val[0],
                                self.pow_histogram_bin_edges_range.val[1],
                                num=101,
                                endpoint=True)

        # Construct attributes for the new Param
        hdf5_attrs = HDF5Attrs(
            name='histogramEdgesPower',
            units='dB',
            descr='Bin edges (including endpoint) for power histogram',
            path=self.path_to_processing_group_in_stats_h5
            )
        
        return HDF5Param(val=bin_edges, hdf5_attrs=hdf5_attrs)


    def _phs_in_radians_2_param(self, phs_in_radians):
        '''Return `phs_in_radians` as a Param.'''

        # Construct defaults for the new Param
        default_val = self.get_default_arg_for_yaml(attr_name='phs_in_radians')
        yaml_attrs = YamlAttrs(
            name='phs_in_radians',
            descr=f'''
                True to compute phase in radians units, False for 
                degrees units. Defaults to {default_val}.'''
        )

        return YamlParam(val=phs_in_radians, yaml_attrs=yaml_attrs)


    @property
    def phs_in_radians(self):
        return self._phs_in_radians


    @phs_in_radians.setter
    def phs_in_radians(self, phs_in_radians_attr):

        val = phs_in_radians_attr.val
        if not isinstance(val, bool):
            raise TypeError('phs_in_radians_attr.va; must be bool: '
                                f'{val}')

        # Set Attributes
        # `self._phs_in_radians` must be set first, before setting the
        # phase bin attribute.
        self._phs_in_radians = phs_in_radians_attr
        self._phs_bin_edges = self._phs_bin_edges_2_param()


    def _phs_bin_edges_2_param(self):
        '''Return `phs_bin_edges` as a Param.'''

        # Phase bin edges - allow for either radians or degrees
        if self.phs_in_radians.val:
            phs_units = 'radians'
            start = -np.pi
            stop = np.pi
        else:
            phs_units = 'degrees'
            start = -180
            stop = 180

        # 101 bin edges => 100 bins
        bin_edges = np.linspace(start, stop, num=101, endpoint=True)

        hdf5_attrs = HDF5Attrs(
            name='histogramEdgesPhase',
            units=phs_units,
            descr='Bin edges (including endpoint) for phase histogram',
            path=self.path_to_processing_group_in_stats_h5
        )

        return HDF5Param(val=bin_edges, hdf5_attrs=hdf5_attrs)


    @property
    def phs_bin_edges(self):
        return self._phs_bin_edges


    def _tile_shape_2_param(self, tile_shape):
        '''Return `tile_shape` as a Param.
                
        TODO - this is duplicate code to other Params dataclasses. Fix.
        '''

        # Construct attributes for the new Param
        default_val = self.get_default_arg_for_yaml(attr_name='tile_shape')
        yaml_attrs = YamlAttrs(
            name='tile_shape',
            descr=f'''
                Preferred tile shape for processing images by batches.
                Actual tile shape used during processing might
                be smaller due to shape of image.
                Format: [<num_rows>, <num_cols>]
                -1 to indicate all rows / all columns (respectively).
                Defaults to {list(default_val)} to use all columns 
                (i.e. full rows of data).'''
            )

        return YamlParam(val=tile_shape, yaml_attrs=yaml_attrs)


    @property
    def tile_shape(self) -> YamlParam[Iterable[int]]:
        return self._tile_shape
    

    @tile_shape.setter
    def tile_shape(self, tile_shape_attr: YamlParam[Iterable[int]]):
        # Validate input
        val = tile_shape_attr.val
        if not isinstance(val, (list, tuple)):
            raise TypeError('`tile_shape_attr.val` must be a list or tuple: '
                                f'{val}')

        if not len(val) == 2:
            raise TypeError('`tile_shape_attr.val` must have a length'
                                f'of two: {val}')

        if not all(isinstance(e, int) for e in val):
            raise TypeError('`tile_shape_attr.val` must contain only '
                                f'integers: {val}')

        if any(e < -1 for e in val):
            raise TypeError('`tile_shape_attr.val` must contain only '
                                f' values >= -1: {val}')

        # Set attribute
        self._tile_shape = tile_shape_attr


@dataclass
class AbsCalParamGroup(YamlHDF5ParamGroup):
    '''
    Parameters from the QA-CalTools Absolute Calibration Factor
    runconfig group.

    Parameters
    ----------
    attr1 : float, optional
        Placeholder Attribute 1.
    '''

    attr1: YamlHDF5Param[float]

    # Override the parent class variable
    path_to_processing_group_in_stats_h5: ClassVar[str] = \
                        '/science/%s/absoluteCalibrationFactor/processing'


    def __init__(self, attr1: float = 2.3):

        # validate and initialize all attributes.
        self.attr1 = self._attr1_2_param(attr1)


    @staticmethod
    def get_path_to_group_in_runconfig():
        return ['runconfig','groups','qa','absolute_calibration_factor']


    def _attr1_2_param(self, attr1):
        '''Return `attr1` as a Param'''

        # Construct defaults for the new Param
        yaml_attrs = YamlAttrs(
            name='attr1',
            descr='''
            Placeholder: Attribute 1 description for runconfig. Each new line
            of text will be a separate line in the runconfig template.
            `attr1` is a non-negative float value. Default: 2.3'''
        )

        hdf5_attrs = HDF5Attrs(
            name='attribute1',
            units='smoot',
            descr='Description of `attr1` for stats.h5 file',
            path=self.path_to_processing_group_in_stats_h5
        )

        return YamlHDF5Param(val=attr1,
                             yaml_attrs=yaml_attrs,
                             hdf5_attrs=hdf5_attrs)


    @property
    def attr1(self):
        return self._attr1
    

    @attr1.setter
    def attr1(self, attr1_attr):

        # Validate
        val = attr1_attr.val
        if not isinstance(val, float):
            raise TypeError(f'`attr1_attr.val` must be a float: {val}')

        if val < 0.0:
            raise TypeError(f'`attr1_attr.val` must be non-negative: {val}')

        # Set Attribute
        self._attr1 = attr1_attr


@dataclass
class NESZParamGroup(YamlHDF5ParamGroup):
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
        Placeholder parameter of type bool. This is set by updating `attr1`.
    '''

    # Attributes for running the NESZ workflow
    attr1: YamlParam[float]

    # Auto-generated attributes
    attr2: HDF5Param[bool] = field(init=False)

    # Override the parent class' class variable
    path_to_processing_group_in_stats_h5: ClassVar[str] = \
                                                '/science/%s/NESZ/processing'


    def __init__(self, attr1: float = 11.9):

        # validate and initialize all attributes.
        self.attr1 = self._attr1_2_param(attr1)


    @staticmethod
    def get_path_to_group_in_runconfig():
        return ['runconfig','groups','qa','nesz']


    def _attr1_2_param(self, attr1):
        '''Return `attr1` as a Param'''

        # Construct defaults for the new Param
        default_value = self.get_default_arg_for_yaml(attr_name='attr1')
        yaml_attrs = YamlAttrs(
            name='attr1',
            descr=f'''
            Placeholder: Attribute 1 description for runconfig. Each new line
            of text will be a separate line in the runconfig template.
            `attr1` is a non-negative float value. Default: {default_value}'''
        )

        return YamlParam(val=attr1, yaml_attrs=yaml_attrs)


    @property
    def attr1(self):
        return self._attr1
    

    @attr1.setter
    def attr1(self, attr1_attr):

        # Validate
        val = attr1_attr.val
        if not isinstance(val, float):
            raise TypeError(f'`attr1_attr.val` must be a float: {val}')

        if val < 0.0:
            raise TypeError(f'`attr1_attr.val` must be non-negative: {val}')

        # Set Attributes
        self._attr1 = attr1_attr

        # attr2 is dependent upon attr1, but they should always stay in sync
        # with each other. So, whenever attr1 is updated, we should next 
        # update attr2 at the same time.
        # This also prevents an outside user from updating attr2 independently,
        # which would cause the attributes to fall out of sync with each other.
        self._attr2 = self._attr2_2_param()


    def _attr2_2_param(self):
        '''Return `attr2` as a Param.'''

        # Here is where the dependency upon attr1 occurs:
        if self.attr1.val < 12.0:
            val = True
        else:
            val = False

        hdf5_attrs = HDF5Attrs(
            name='attribute2',
            units='parsecs',
            descr='True if K-run was less than 12.0',
            path=self.path_to_processing_group_in_stats_h5
        )

        return HDF5Param(val=val, hdf5_attrs=hdf5_attrs)

    @property
    def attr2(self):
        # Because attr2 can only be set when the attr1 is set
        # (which is inside @attr1.setter), we should only define
        # the @property for attr2. We do not need a separate setter for
        # attr2.
        return self._attr2


@dataclass
class PointTargetAnalyzerParamGroup(YamlHDF5ParamGroup):
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
    attr1 : float, optional
        Placeholder Attribute 1.
    '''

    attr1: YamlHDF5Param[float]

    # Override the parent class variable
    path_to_processing_group_in_stats_h5: ClassVar[str] = \
                        '/science/%s/pointTargetAnalyzer/processing'


    def __init__(self, attr1: float = 2300.5):
        self.attr1 = self._attr1_2_param(attr1)


    @staticmethod
    def get_path_to_group_in_runconfig():
        return ['runconfig','groups','qa','point_target_analyzer']


    def _attr1_2_param(self, attr1):
        '''Return `attr1` as a Param'''

        # Construct defaults for the new Param
        yaml_attrs = YamlAttrs(
            name='attr1',
            descr='''
            Placeholder: Attribute 1 description for runconfig. Each new line
            of text will be a separate line in the runconfig template.
            `attr1` is a non-negative float value. Default: 2.3'''
        )

        hdf5_attrs = HDF5Attrs(
            name='attribute1',
            units='beard-second',
            descr='Description of `attr1` for stats.h5 file',
            path=self.path_to_processing_group_in_stats_h5
        )

        return YamlHDF5Param(val=attr1,
                             yaml_attrs=yaml_attrs,
                             hdf5_attrs=hdf5_attrs)


    @property
    def attr1(self):
        return self._attr1
    

    @attr1.setter
    def attr1(self, attr1_attr):

        # Validate
        val = attr1_attr.val
        if not isinstance(val, float):
            raise TypeError(f'`attr1_attr.val` must be a float: {val}')

        if val < 0.0:
            raise TypeError(f'`attr1_attr.val` must be non-negative: {val}')

        # Set attribute
        self._attr1 = attr1_attr


@dataclass
class RSLCRootParamGroup:
    '''
    Dataclass of all *Params objects to process QA for NISAR RSLC products.

    `workflows` is the only required parameter. Based on the workflows set to
    True in `workflows`, the other RSLCRootParamGroup parameters will be set
    per these rules:
        a) If a *Params object is needed by any workflow, it will be
           set to an instance of that *Params object.

                i) If a *Params object is provided by the caller, the
                corresponding attribute in RSLCRootParamGroup will be set
                to that.

                i) If a *Params object is not provided, one will be 
                instantiated using all default value, and the
                corresponding attribute in RSLCRootParamGroup will be set
                to that.

        b) If a *Params object is not needed by any workflow,
           it will be set to `None`, regardless of the input.
    
    Parameters
    ----------
    workflows : WorkflowsParamGroup
        RSLC QA Workflows parameters
    input_f : InputFileGroupParamGroup, optional
        Input File Group parameters for RSLC QA
    prodpath : ProductPathGroupParamGroup, optional
        Product Path Group parameters for RSLC QA
    power_img : RSLCPowerImageParamGroup
        Power Image Group parameters for RSLC QA
    histogram : RSLCHistogramParamGroup
        Histogram Group parameters for RSLC QA
    anc_files : DynamicAncillaryFileParamGroup, optional
        Dynamic Ancillary File Group parameters for RSLC QA-Caltools
    abs_cal : AbsCalParamGroup, optional
        Absolute Calibration Factor group parameters for RSLC QA-Caltools
    nesz : NESZParamGroup, optional
        NESZ group parameters for RSLC QA-Caltools
    pta : PointTargetAnalyzerParamGroup, optional
        Point Target Analyzer group parameters for RSLC QA-Caltools
    '''

    # Shared parameters
    workflows: WorkflowsParamGroup
    input_f: Optional[InputFileGroupParamGroup] = None
    prodpath: Optional[ProductPathGroupParamGroup] = None

    # QA parameters
    power_img: Optional[RSLCPowerImageParamGroup] = None
    histogram: Optional[RSLCHistogramParamGroup] = None

    # CalTools parameters
    anc_files: Optional[DynamicAncillaryFileParamGroup] = None
    abs_cal: Optional[AbsCalParamGroup] = None
    nesz: Optional[NESZParamGroup] = None
    pta: Optional[PointTargetAnalyzerParamGroup] = None

    def __post_init__(self):

        # Ensure that the minimum parameters were provided

        # If any of the workflows requested, then prodpath must be an
        # instance of ProductPathGroupParamGroup.
        # prodpath is only optional in the case of doing a dumpconfig
        if any([getattr(self.workflows, field.name) \
                            for field in fields(self.workflows)]):
            if not isinstance(self.input_f, InputFileGroupParamGroup):
                raise TypeError('`input_f` parameter of type '
                    'InputFileGroupParamGroup is required to run any of the '
                    'QA workflows.')

        # If any of the workflows requested, then prodpath must be an
        # instance of ProductPathGroupParamGroup.
        # prodpath is only optional in the case of doing a dumpconfig
        if any([getattr(self.workflows, field.name) \
                            for field in fields(self.workflows)]):
            if not isinstance(self.prodpath, ProductPathGroupParamGroup):
                raise TypeError('`prodpath` parameter of type '
                    'ProductPathGroupParamGroup is required to run any of the '
                    'QA workflows.')

        if self.workflows.qa_reports:
            if self.power_img is None or \
                not isinstance(self.power_img, RSLCPowerImageParamGroup):
                raise TypeError('`power_img` parameter of type '
                    'RSLCPowerImageParamGroup is required to run the '
                    'requested qa_reports workflow')

            if self.histogram is None or \
                not isinstance(self.histogram, RSLCHistogramParamGroup):
                raise TypeError('`histogram` parameter of type '
                    'RSLCHistogramParamGroup is required to run the '
                    'requested qa_reports workflow')

        if self.workflows.absolute_calibration_factor:
            if self.abs_cal is None or \
                not isinstance(self.abs_cal, AbsCalParamGroup):
                raise TypeError('`abs_cal` parameter of type '
                    'AbsCalParamGroup is required to run the '
                    'requested absolute_calibration_factor workflow')

            if self.anc_files is None or \
                not isinstance(self.anc_files, DynamicAncillaryFileParamGroup):
                raise TypeError('`anc_files` parameter of type '
                    'DynamicAncillaryFileParamGroup is required to run the '
                    'requested absolute_calibration_factor workflow')

        if self.workflows.nesz:
            if self.nesz is None or \
                not isinstance(self.nesz, NESZParamGroup):
                raise TypeError('`nesz` parameter of type '
                    'NESZParamGroup is required to run the '
                    'requested nesz workflow')

        if self.workflows.point_target_analyzer:
            if self.pta is None or \
                not isinstance(self.pta, PointTargetAnalyzerParamGroup):
                raise TypeError('`pta` parameter of type '
                    'PointTargetAnalyzerParamGroup is required to run the '
                    'requested point_target_analyzer workflow')

            if self.anc_files is None or \
                not isinstance(self.anc_files, DynamicAncillaryFileParamGroup):
                raise TypeError('`anc_files` parameter of type '
                    'DynamicAncillaryFileParamGroup is required to run the '
                    'requested point_target_analyzer workflow')


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
        # TODO - create a list of the param groups, pass this into the helper fxn,
        # then do the standard loop.
        InputFileGroupParamGroup.populate_runcfg(runconfig_cm)
        DynamicAncillaryFileParamGroup.populate_runcfg(runconfig_cm)
        ProductPathGroupParamGroup.populate_runcfg(runconfig_cm)
        WorkflowsParamGroup.populate_runcfg(runconfig_cm)
        RSLCPowerImageParamGroup.populate_runcfg(runconfig_cm)
        RSLCHistogramParamGroup.populate_runcfg(runconfig_cm)
        AbsCalParamGroup.populate_runcfg(runconfig_cm)
        NESZParamGroup.populate_runcfg(runconfig_cm)
        PointTargetAnalyzerParamGroup.populate_runcfg(runconfig_cm)

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
            po = getattr(self, params_obj.name)
            # If a workflow was not requested, its RootParams attribute
            # will be None, so there will be no params to add to the h5 file
            if po is not None:
                if issubclass(type(po), HDF5ParamGroup):
                    po.write_params_to_h5(h5_file, bands=bands)


def parse_rslc_runconfig(runconfig_yaml):
    '''
    Parse a QA RSLC Runconfig yaml file into a RSLCRootParamGroup object.
    
    Parameters
    ----------
    runconfig_yaml : str
        Filename (with path) to an RSLC QA runconfig yaml file.
    
    Returns
    -------
    rslc_params : RSLCRootParamGroup
        RSLCRootParamGroup object populated with runconfig values where provided,
        and default values for missing runconfig parameters.
    '''
    # parse runconfig into a dict structure
    parser = YAML(typ='safe')
    with open(runconfig_yaml, 'r') as f:
        user_rncfg = parser.load(f)

    # Construct WorkflowsParamGroup dataclass
    rncfg_path = WorkflowsParamGroup.get_path_to_group_in_runconfig()
    try:
        params_dict = nisarqa.get_nested_element_in_dict(user_rncfg, rncfg_path)
    except KeyError:
        # if group does not exist in runconfig, use defaults
        workflows_params = WorkflowsParamGroup()
    else:
        workflows_params = WorkflowsParamGroup(**params_dict)
    finally:
        # if all functionality is off, then exit
        # All workflows default to false. So, we only need to check if
        # any workflows were turned on via the runconfig.
        if not any(params_dict.values()):
            return

    # Construct InputFileGroupParamGroup dataclass (required if any workflows are True)
    rncfg_path = InputFileGroupParamGroup.get_path_to_group_in_runconfig()
    try:
        params_dict = nisarqa.get_nested_element_in_dict(user_rncfg,
                                                            rncfg_path)
    except KeyError as e:
        raise KeyError('`input_file_group` is a required runconfig group') from e

    try:
        input_file_params = InputFileGroupParamGroup(
                        qa_input_file=params_dict['qa_input_file'])
    except KeyError as e:
        raise KeyError('`qa_input_file` is a required parameter for QA') from e

    # Construct DynamicAncillaryFileParamGroup dataclass
    # Only two of the CalVal workflows use the dynamic_ancillary_file_group
    # YES - this file is required for these tools. Treat same as RSLC Input File.
    if workflows_params.absolute_calibration_factor or \
        workflows_params.point_target_analyzer:

        rncfg_path = DynamicAncillaryFileParamGroup.get_path_to_group_in_runconfig()
        try:
            params_dict = nisarqa.get_nested_element_in_dict(user_rncfg, 
                                                                rncfg_path)
        except KeyError as e:
            raise KeyError('`dynamic_ancillary_file_group` is a required '
                           'runconfig group to run Absolute Calibration Factor'
                           ' or Point Target Analyzer workflows.') from e
        try:
            dyn_anc_files = DynamicAncillaryFileParamGroup(
                    corner_reflector_file=params_dict['corner_reflector_file'])
        except KeyError as e:
            raise KeyError('`corner_reflector_file` is a required runconfig '
                           'parameter for Absolute Calibration Factor '
                           'or Point Target Analyzer workflows') from e
        
        # TODO - add in orbit file param for AbsCal. But, it is optional, very rare.

    else:
        dyn_anc_files = None

    # Construct ProductPathGroupParamGroup dataclass
    rncfg_path = ProductPathGroupParamGroup.get_path_to_group_in_runconfig()
    try:
        params_dict = nisarqa.get_nested_element_in_dict(
                                user_rncfg, rncfg_path)
    except KeyError:
        # group not found in runconfig. Use defaults.
        warnings.warn('`product_path_group` not found in runconfig. '
                      'Using default output directory.')
        product_path_params = ProductPathGroupParamGroup()
    else:
        try:
            product_path_params = ProductPathGroupParamGroup(
                                    qa_output_dir=params_dict['qa_output_dir'])
        except KeyError:
            # parameter not found in runconfig. Use defaults.
            warnings.warn('`qa_output_dir` not found in runconfig. '
                        'Using default output directory.')
            product_path_params = ProductPathGroupParamGroup()

    # Construct RSLCPowerImageParamGroup dataclass
    if workflows_params.qa_reports:
        rncfg_path = RSLCPowerImageParamGroup.get_path_to_group_in_runconfig()
        try:
            params_dict = nisarqa.get_nested_element_in_dict(user_rncfg, 
                                                                rncfg_path)
        except KeyError:
            pow_img_params = RSLCPowerImageParamGroup()
        else:
            pow_img_params = RSLCPowerImageParamGroup(**params_dict)
    else:
        pow_img_params = None

    # Construct RSLCHistogramParamGroup dataclass
    if workflows_params.qa_reports:
        rncfg_path = RSLCHistogramParamGroup.get_path_to_group_in_runconfig()
        try:
            params_dict = nisarqa.get_nested_element_in_dict(user_rncfg, 
                                                                rncfg_path)
        except KeyError:
            histogram_params = RSLCHistogramParamGroup()
        else:
            histogram_params = RSLCHistogramParamGroup(**params_dict)
    else:
        histogram_params = None

    # Construct AbsCalParamGroup dataclass
    if workflows_params.absolute_calibration_factor:
        # TODO: This code is commented out for R3.2. Once CalTools and its
        # runconfig parameters are integrated into QA SAS, then uncomment
        # this section.

        # rncfg_path = AbsCalParamGroup.get_path_to_group_in_runconfig()
        # try:
        #     params_dict = nisarqa.get_nested_element_in_dict(user_rncfg,
        #                                                         rncfg_path)
        # except KeyError:
        #     abscal_params = AbsCalParamGroup()
        # else:
        #     abscal_params = AbsCalParamGroup(**params_dict)

        # For R3.2 only, always use the default parameters
        abscal_params = AbsCalParamGroup()
    else:
        abscal_params = None

    # Construct NESZ dataclass
    if workflows_params.nesz:
        # TODO: This code is commented out for R3.2. Once CalTools and its
        # runconfig parameters are integrated into QA SAS, then uncomment
        # this section.

        # rncfg_path = NESZParamGroup.get_path_to_group_in_runconfig()
        # try:
        #     params_dict = nisarqa.get_nested_element_in_dict(user_rncfg, 
        #                                                         rncfg_path)
        # except KeyError:
        #     nesz_params = NESZParamGroup()
        # else:
        #     nesz_params = NESZParamGroup(**params_dict)
    
        # For R3.2 only, always use the default parameters
        nesz_params = NESZParamGroup()

    else:
        nesz_params = None

    # Construct PointTargetAnalyzerParamGroup dataclass
    if workflows_params.point_target_analyzer:
        # TODO: This code is commented out for R3.2. Once CalTools and its
        # runconfig parameters are integrated into QA SAS, then uncomment
        # this section.

        # rncfg_path = PointTargetAnalyzerParamGroup.get_path_to_group_in_runconfig()
        # try:
        #     params_dict = nisarqa.get_nested_element_in_dict(user_rncfg,
        #                                                         rncfg_path)
        # except KeyError:
        #     pta_params = PointTargetAnalyzerParamGroup()
        # else:
        #     pta_params = PointTargetAnalyzerParamGroup(**params_dict)

        # For R3.2 only, always use the default parameters
        pta_params = PointTargetAnalyzerParamGroup()

    else:
        pta_params = None

    # Construct RSLCRootParamGroup
    rslc_params = RSLCRootParamGroup(workflows=workflows_params,
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
