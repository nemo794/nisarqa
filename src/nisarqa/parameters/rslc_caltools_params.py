import os
import sys
import warnings
from dataclasses import Field, InitVar, dataclass, field, fields
from typing import ClassVar, Iterable, Optional, Union

import nisarqa
import numpy as np
from nisarqa.parameters.nisar_params import *
from numpy.typing import ArrayLike
from ruamel.yaml import YAML
from ruamel.yaml import CommentedMap as CM

objects_to_skip = nisarqa.get_all(__name__)


@dataclass(frozen=True)
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
    descr: ClassVar[str] = f'Flag to run `%s` workflow.'

    validate: bool = field(
        default=def_val,
        metadata={'yaml_attrs': YamlAttrs(name='validate',
                                          descr=descr % 'validate')})
    
    qa_reports: bool = field(
        default=def_val,
        metadata={'yaml_attrs': YamlAttrs(name='qa_reports',
                                          descr=descr % 'qa_reports')})

    absolute_calibration_factor: bool = field(
        default=def_val,
        metadata={'yaml_attrs': YamlAttrs(name='absolute_calibration_factor',
                                descr=descr % 'absolute_calibration_factor')})

    nesz: bool = field(
        default=def_val,
        metadata={'yaml_attrs': YamlAttrs(name='nesz', 
                                          descr=descr % 'nesz')})

    point_target_analyzer: bool = field(
        default=def_val,
        metadata={'yaml_attrs': YamlAttrs(name='point_target_analyzer',
                                        descr=descr % 'point_target_analyzer')})

    def __post_init__(self):

        # VALIDATE INPUTS
        self._check_workflows_arg('validate', self.validate)
        self._check_workflows_arg('qa_reports', self.qa_reports)
        self._check_workflows_arg('absolute_calibration_factor', self.absolute_calibration_factor)
        self._check_workflows_arg('nesz', self.nesz)
        self._check_workflows_arg('point_target_analyzer', self.point_target_analyzer)


    @staticmethod
    def get_path_to_group_in_runconfig():
        return ['runconfig','groups','qa','workflows']


    @staticmethod
    def _check_workflows_arg(attr_name, val):
        '''
        Validate that `val` is of the correct type for the
        WorkflowsParamGroup's attribute `attr_name`.

        Parameters
        ----------
        attr_name : str
            The name of the attribute of WorkflowsParamGroup for `attr`
        val : bool
            Argument value for `attr_name`.
        '''
        # Validate `val`
        if not isinstance(val, bool):
            raise TypeError(f'`{attr_name}` must be of type bool. '
                            f'It is {type(val)}')


@dataclass(frozen=True)
class InputFileGroupParamGroup(YamlParamGroup):
    '''
    Parameters from the Input File Group runconfig group.

    Parameters
    ----------
    qa_input_file : str
        The input NISAR product file name (with path).
    '''

    # Required parameter
    qa_input_file: str = field(
        metadata={'yaml_attrs': 
            YamlAttrs(
                name='qa_input_file',
                descr='''
                Filename of the input file for QA.
                REQUIRED for QA. NOT REQUIRED if only running Product SAS.
                If Product SAS and QA SAS are run back-to-back,
                this field should be identical to `sas_output_file`.
                Otherwise, this field should contain the filename of the single
                NISAR product for QA to process.''')})

    def __post_init__(self):
        # VALIDATE INPUTS
        nisarqa.validate_is_file(filepath=self.qa_input_file, 
                                 parameter_name='qa_input_file',
                                 extension='.h5')

    @staticmethod
    def get_path_to_group_in_runconfig():
        return ['runconfig','groups','input_file_group']


@dataclass(frozen=True)
class DynamicAncillaryFileParamGroup(YamlParamGroup):
    '''
    The parameters from the QA Dynamic Ancillary File runconfig group.

    Parameters
    ----------
    corner_reflector_file : str
        The input corner reflector file's file name (with path).
        Required for the Absolute Calibration Factor and Point Target
        Analyzer workflows.
    '''

    # Required parameter
    corner_reflector_file: str = field(
        metadata={
            'yaml_attrs' : YamlAttrs(
                name='corner_reflector_file',
                descr='''Locations of the corner reflectors in the input product.
                Only required if `absolute_calibration_factor` or
                `point_target_analyzer` runconfig params are set to True for QA.'''
            )})

    def __post_init__(self):
        # VALIDATE INPUTS

        nisarqa.validate_is_file(filepath=self.corner_reflector_file, 
                                 parameter_name='corner_reflector_file',
                                 extension='.csv')


    @staticmethod
    def get_path_to_group_in_runconfig():
        return ['runconfig','groups','dynamic_ancillary_file_group']


@dataclass(frozen=True)
class ProductPathGroupParamGroup(YamlParamGroup):
    '''
    Parameters from the Product Path Group runconfig group.

    Parameters
    ----------
    qa_output_dir : str, optional
        Filepath to the output directory to store NISAR QA output files.
        Defaults to './qa'
    '''

    qa_output_dir: str = field(
        default='./qa',
        metadata={'yaml_attrs' : YamlAttrs(
            name='corner_reflector_file',
            descr='''Output directory to store all QA output files.'''
        )}
    )

    def __post_init__(self):
        # VALIDATE INPUTS

        if not isinstance(self.qa_output_dir, str):
            raise TypeError(f'`qa_output_dir` must be a str')

        # If this directory does not exist, make it.
        if not os.path.isdir(self.qa_output_dir):
            print(f'Creating QA output directory: {self.qa_output_dir}')
            os.makedirs(self.qa_output_dir, exist_ok=True)


    @staticmethod
    def get_path_to_group_in_runconfig():
        return ['runconfig','groups','product_path_group']


@dataclass(frozen=True)
class RSLCPowerImageParamGroup(YamlHDF5ParamGroup):
    '''
    Parameters to generate RSLC Power Images and Browse Image.
    
    This corresponds to the `qa_reports: power_image` runconfig group.

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
        If None, then no normalization, no gamma correction will be applied.
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
        Units of the power image.
        If `linear_units` is True, this will be set to 'linear'.
        If `linear_units` is False, this will be set to 'dB'.
    '''

    linear_units: bool = field(
        default=True,
        metadata={
        'yaml_attrs' : YamlAttrs(
            name='linear_units',
            descr='''True to compute power in linear units when generating 
                the power image for the browse images and graphical
                summary PDF. False for decibel units.'''
        )})

    nlooks_descr_template: ClassVar[str] = \
        '''Number of looks along each axis of the Frequency %s
        image arrays for multilooking the power image.
        Format: [<num_rows>, <num_cols>]
        Example: [6,7]
        If not provided, the QA code to compute the nlooks values 
        based on `num_mpix`.'''

    nlooks_freqa: Optional[Union[int, Iterable[int]]] = field(
        default=None,
        metadata={
        'yaml_attrs' : YamlAttrs(
            name=f'nlooks_freqa',
            descr=nlooks_descr_template % 'A'
        )})

    nlooks_freqb: Optional[Union[int, Iterable[int]]] = field(
        default=None,
        metadata={
        'yaml_attrs' : YamlAttrs(
            name='nlooks_freqb',
            descr=nlooks_descr_template % 'B'
        )})

    num_mpix: int = field(
        default=4.0,
        metadata={
        'yaml_attrs' : YamlAttrs(
            name='num_mpix',
            descr='''Approx. size (in megapixels) for the final
                multilooked browse image(s). If `nlooks_freq*` parameter(s)
                is not None, nlooks values will take precedence.'''
        )})

    middle_percentile: float = field(
        default=90.0,
        metadata={
        'yaml_attrs' : YamlAttrs(
            name='middle_percentile',
            descr='''The middle percentile range of the image array
                that the colormap covers. Must be in the range [0.0, 100.0].'''
            ),
        'hdf5_attrs' : HDF5Attrs(
            name='powerImageMiddlePercentile',
            units='unitless',
            descr='Middle percentile range of the image array '
                  'that the colormap covers',
            path=HDF5ParamGroup.path_to_stats_h5_qa_processing_group
            )
        })

    gamma: Optional[float] = field(
        default=None,
        metadata={
        'yaml_attrs' : YamlAttrs(
            name='gamma',
            descr='''Gamma correction parameter applied to power and browse image(s).
                Gamma will be applied as follows:
                    array_out = normalized_array ^ gamma
                where normalized_array is a copy of the image with values
                scaled to the range [0,1]. 
                The image colorbar will be defined with respect to the input
                image values prior to normalization and gamma correction.
                Defaults to None (no normalization, no gamma correction)'''
            ),
        'hdf5_attrs' : HDF5Attrs(
            name='powerImageGammaCorrection',
            units='unitless',
            descr='Gamma correction parameter applied to power and browse image(s).',
            path=HDF5ParamGroup.path_to_stats_h5_qa_processing_group
            )
        })

    tile_shape: list[int] = field(
        default=(1024,1024),
        metadata={
        'yaml_attrs' : YamlAttrs(
            name='tile_shape',
            descr='''Preferred tile shape for processing images by batches.
                Actual tile shape may be modified by QA-SAS.
                Format: [<num_rows>, <num_cols>]
                -1 to indicate all rows / all columns (respectively).'''
            )
        })

    # Auto-generated attributes, so set init=False and have no default.
    # `pow_units` is determined by the `linear_units` attribute.
    pow_units: str = field(
        init=False,
        metadata={
            'hdf5_attrs' : HDF5Attrs(
                name='powerImagePowerUnits',
                units=None,
                descr='''Units of the power image.''',
                path=HDF5ParamGroup.path_to_stats_h5_qa_processing_group
            )
        })

    def __post_init__(self):
        # VALIDATE INPUTS

        # validate linear_units
        if not isinstance(self.linear_units, bool):
            raise TypeError(f'`linear_units` must be bool: {self.linear_units}')

        # validate nlooks_freq*
        self._validate_nlooks(self.nlooks_freqa, 'A')
        self._validate_nlooks(self.nlooks_freqa, 'B')

        # validate num_mpix
        if not isinstance(self.num_mpix, float):
            raise TypeError(f'`num_mpix` must be a float: {self.num_mpix}')
        if self.num_mpix <= 0.0:
            raise TypeError(f'`num_mpix` must be >= 0.0: {self.num_mpix}')
        
        # validate middle_percentile
        if not isinstance(self.middle_percentile, float):
            raise TypeError(
                f'`middle_percentile` must be float: {self.middle_percentile}')

        if self.middle_percentile < 0.0 or self.middle_percentile > 100.0:
            raise TypeError('`middle_percentile` is '
                f'{self.middle_percentile}, must be in range [0.0, 100.0]')

        # validate gamma
        if isinstance(self.gamma, float):
            if (self.gamma < 0.0):
                raise ValueError('If `gamma` is a float, it must be'
                                f' non-negative: {self.gamma}')
        elif self.gamma is not None:
            raise TypeError('`gamma` must be a float or None. '
                            f'Value: {self.gamma}, Type: {type(self.gamma)}')

        # validate tile_shape
        val = self.tile_shape
        if not isinstance(val, (list, tuple)):
            raise TypeError(f'`tile_shape` must be a list or tuple: {val}')
        if not len(val) == 2:
            raise TypeError(f'`tile_shape` must have a length of two: {val}')
        if not all(isinstance(e, int) for e in val):
            raise TypeError(f'`tile_shape` must contain only integers: {val}')
        if any(e < -1 for e in val):
            raise TypeError(f'Values in `tile_shape` must be >= -1: {val}')


        # SET ATTRIBUTES DEPENDENT UPON INPUT PARAMETERS
        # This dataclass is frozen to ensure that all inputs are validated, 
        # so we need to use object.__setattr__()

        # use linear_units to set pow_units
        object.__setattr__(self, 'pow_units',
                                 'linear' if self.linear_units else 'dB')


    @staticmethod
    def get_path_to_group_in_runconfig():
        return ['runconfig','groups','qa','qa_reports','power_image']

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


@dataclass(frozen=True)
class RSLCHistogramParamGroup(YamlHDF5ParamGroup):
    # each param group will define the internal nesting path within its file (ABC property),
    # and how to populate its file.
    '''
    Parameters to generate the RSLC Power and Phase Histograms;
    this corresponds to the `qa_reports: histogram` runconfig group.

    Parameters
    ----------
    decimation_ratio : pair of int, optional
        The step size to decimate the input array for computing
        the power and phase histograms.
        For example, (2,3) means every 2nd azimuth line and
        every 3rd range line will be used to compute the histograms.
        Defaults to (10,10).
        Format: (<azimuth>, <range>)
    pow_histogram_bin_edges_range : pair of float, optional
        The dB range for the power histogram's bin edges. Endpoint will
        be included. Defaults to [-80.0, 20.0].
        Format: (<starting value>, <endpoint>)
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

    Attributes
    ----------
    pow_bin_edges : numpy.ndarray
        The bin edges (including endpoint) to use when computing
        the power histograms. Will be set to 100 uniformly-spaced bins
        in range `pow_histogram_bin_edges_range`, including endpoint.
    phs_bin_edges : numpy.ndarray
        The bin edges (including endpoint) to use when computing
        the phase histograms.
        If `phs_in_radians` is True, this will be set to 100 
        uniformly-spaced bins in range [-pi,pi], including endpoint.
        If `phs_in_radians` is False, this will be set to 100
        uniformly-spaced bins in range [-180,180], including endpoint.
    az_decimation : int
        The azimuth decimation ratio value; a copy of `decimation_ratio[0]`,
        but with `hdf5_attrs` information in the dataclasses.field metadata.
    rng_decimation : int
        The range decimation ratio value; a copy of `decimation_ratio[1]`,
        but with `hdf5_attrs` information in the dataclasses.field metadata.
    '''

    decimation_ratio: Iterable[int] = field(
        default=(10,10),
        metadata={
        'yaml_attrs' : YamlAttrs(
            name='decimation_ratio',
            descr='''Step size to decimate the input array for computing
                the power and phase histograms.
                For example, [2,3] means every 2nd azimuth line and
                every 3rd range line will be used to compute the histograms.
                Format: [<azimuth>, <range>]'''
        )})

    pow_histogram_bin_edges_range: Iterable[float] = field(
        default=(-80.0,20.0),
        metadata={
        'yaml_attrs' : YamlAttrs(
            name='pow_histogram_bin_edges_range',
            descr='''The dB range for the power histogram's bin edges. Endpoint will
                be included. Format: [<starting value>, <endpoint>]'''
        )})

    phs_in_radians: bool = field(
        default=True,
        metadata={
        'yaml_attrs' : YamlAttrs(
            name='phs_in_radians',
            descr='''True to compute phase histogram in radians units,
                False for degrees units.'''
        )})

    tile_shape: list[int] = field(
        default=(1024,-1),
        metadata={
        'yaml_attrs' : YamlAttrs(
            name='tile_shape',
            descr='''Preferred tile shape for processing images by batches.
                Actual tile shape may be modified by QA-SAS.
                Format: [<num_rows>, <num_cols>]
                -1 to indicate all rows / all columns (respectively).'''
            )
        })

    # Auto-generated attributes
    # Power Bin Edges (generated from `pow_histogram_bin_edges_range`)
    pow_bin_edges: ArrayLike = field(
        init=False,
        metadata={
        'hdf5_attrs' : HDF5Attrs(
            name='histogramEdgesPower',
            units='dB',
            descr='Bin edges (including endpoint) for power histogram',
            path=HDF5ParamGroup.path_to_stats_h5_qa_processing_group
        )})

    # Phase bin edges (generated from `phs_in_radians`)
    phs_bin_edges: ArrayLike = field(
        init=False,
        metadata={
        'hdf5_func' : 
            lambda obj : HDF5Attrs(
                name='histogramEdgesPhase',
                units='radians' if obj.phs_in_radians else 'degrees',
                descr='Bin edges (including endpoint) for phase histogram',
                path=HDF5ParamGroup.path_to_stats_h5_qa_processing_group) \
            if (isinstance(obj, RSLCHistogramParamGroup)) \
            else nisarqa.raise_(TypeError(
            f'`obj` is {type(obj)}, but must be type RSLCHistogramParamGroup'))
        })

    # Attributes derived from `decimation_ratio`
    az_decimation: int = field(
        init=False,
        metadata={
        'hdf5_attrs' : HDF5Attrs(
            name='histogramDecimationAz',
            units='unitless',
            descr='Azimuth decimation stride used to compute power'
                  ' and phase histograms',
            path=HDF5ParamGroup.path_to_stats_h5_qa_processing_group
            )})

    rng_decimation: int = field(
        init=False,
        metadata={
        'hdf5_attrs' : HDF5Attrs(
            name='histogramDecimationRange',
            units='unitless',
            descr='Range decimation stride used to compute power'
                  ' and phase histograms',
            path=HDF5ParamGroup.path_to_stats_h5_qa_processing_group
            )})

    def __post_init__(self):
        # VALIDATE INPUTS

        # validate decimation_ratio
        val = self.decimation_ratio
        if not isinstance(val, (list, tuple)):
            raise TypeError(f'`decimation_ratio` must be list or tuple: {val}')
        if not len(val) == 2:
            raise ValueError(f'`decimation_ratio` must have length of 2: {val}')
        if not all(isinstance(e, int) for e in val):
            raise TypeError(f'`decimation_ratio` must contain integers: {val}')
        if any(e <= 0 for e in val):
            raise ValueError(
                f'`decimation_ratio` must contain positive values: {val}')

        # Validate pow_histogram_bin_edges_range
        val = self.pow_histogram_bin_edges_range
        if not isinstance(val, (list, tuple)):
            raise TypeError('`pow_histogram_bin_edges_range` must'
                            f' be a list or tuple: {val}')
        if not len(val) == 2:
            raise ValueError('`pow_histogram_bin_edges_range` must'
                            f' have a length of two: {val}')
        if not all(isinstance(e, float) for e in val):
            raise TypeError('`pow_histogram_bin_edges_range` must'
                            f' contain only float values: {val}')
        if val[0] >= val[1]:
            raise ValueError(
                '`pow_histogram_bin_edges_range` has format '
                f'[<starting value>, <endpoint>] where <starting value> '
                f'must be less than <ending value>: {val}')

        # validate phs_in_radians
        if not isinstance(self.phs_in_radians, bool):
            raise TypeError(f'phs_in_radians` must be bool: {val}')

        # validate tile_shape
        val = self.tile_shape
        if not isinstance(val, (list, tuple)):
            raise TypeError(f'`tile_shape` must be a list or tuple: {val}')
        if not len(val) == 2:
            raise TypeError(f'`tile_shape` must have a length of two: {val}')
        if not all(isinstance(e, int) for e in val):
            raise TypeError(f'`tile_shape` must contain only integers: {val}')
        if any(e < -1 for e in val):
            raise TypeError(f'Values in `tile_shape` must be >= -1: {val}')


        # SET ATTRIBUTES DEPENDENT UPON INPUT PARAMETERS
        # This dataclass is frozen to ensure that all inputs are validated, 
        # so we need to use object.__setattr__()

        # Set attributes dependent upon decimation_ratio
        object.__setattr__(self, 'az_decimation', self.decimation_ratio[0])
        object.__setattr__(self, 'rng_decimation', self.decimation_ratio[1])

        # Set attributes dependent upon pow_histogram_bin_edges_range
        # Power Bin Edges - hardcode to be in decibels
        # 101 bin edges => 100 bins
        object.__setattr__(self, 'pow_bin_edges',
                           np.linspace(self.pow_histogram_bin_edges_range[0],
                                       self.pow_histogram_bin_edges_range[1],
                                       num=101,
                                       endpoint=True))
  
        # Set attributes dependent upon phs_in_radians
        if self.phs_in_radians:
            object.__setattr__(self, 'phs_bin_edges', 
                np.linspace(start=-np.pi, stop=np.pi, num=101, endpoint=True))
        else:  # phase in dB
            object.__setattr__(self, 'phs_bin_edges', 
                    np.linspace(start=-180, stop=180, num=101, endpoint=True))


    @staticmethod
    def get_path_to_group_in_runconfig():
        return ['runconfig','groups','qa','qa_reports','histogram']


@dataclass(frozen=True)
class AbsCalParamGroup(YamlHDF5ParamGroup):
    '''
    Parameters from the QA-CalTools Absolute Calibration Factor
    runconfig group.

    Parameters
    ----------
    attr1 : float, optional
        Placeholder Attribute 1.
    '''
    
    path_to_stats_h5_abscal_processing_group: ClassVar[str] = \
                        '/science/%s/absoluteCalibrationFactor/processing'

    # Attributes
    attr1: float = field(
        default=2.3,
        metadata={
        'yaml_attrs' : YamlAttrs(
            name='attr1',
            descr='''
            Placeholder: Attribute 1 description for runconfig. Each new line
            of text will be a separate line in the runconfig template.
            `attr1` is a non-negative float value.'''
        ),
        'hdf5_attrs' : HDF5Attrs(
            name='attribute1',
            units='smoot',
            descr='Description of `attr1` for stats.h5 file',
            path=path_to_stats_h5_abscal_processing_group
        )})


    def __post_init__(self):
        # validate all attributes in __post_init__

        # validate attr1
        if not isinstance(self.attr1, float):
            raise TypeError(f'`attr1` must be a float: {self.attr1}')
        if self.attr1 < 0:
            raise TypeError(f'`attr1` must be positive: {self.attr1}')


    @staticmethod
    def get_path_to_group_in_runconfig():
        return ['runconfig','groups','qa','absolute_calibration_factor']


@dataclass(frozen=True)
class NESZParamGroup(YamlHDF5ParamGroup):
    '''
    Parameters from the QA-CalTools Noise Estimator (NESZ) runconfig group.

    Parameters
    ----------
    attr1 : float, optional
        Placeholder Attribute 1.

    Attributes
    ----------
    attr2 : Param
        Placeholder parameter of type bool. This is set by updating `attr1`.
    '''

    path_to_stats_h5_nesz_processing_group: ClassVar[str] = \
                                                '/science/%s/NESZ/processing'

    # Attributes for running the NESZ workflow
    attr1: float = field(
        default=11.9,
        metadata={
        'yaml_attrs' : YamlAttrs(
            name='attr1',
            descr=f'''Placeholder: Attribute 1 description for runconfig.
            Each new line of text will be a separate line in the runconfig
            template. The Default value will be auto-appended to this
            description by the QA code during generation of the template.
            `attr1` is a positive float value.'''
        )})

    # Auto-generated attributes. Set init=False for auto-generated attributes.
    # attr2 is dependent upon attr1
    attr2: bool = field(
        init=False,
        metadata={
        'hdf5_attrs' : HDF5Attrs(
            name='attribute2',
            units='parsecs',
            descr='True if K-run was less than 12.0',
            path=path_to_stats_h5_nesz_processing_group
        )})


    def __post_init__(self):
        # VALIDATE INPUTS

        # Validate attr1
        if not isinstance(self.attr1, float):
            raise TypeError(f'`attr1` must be a float: {self.attr1}')
        if self.attr1 < 0.0:
            raise TypeError(f'`attr1` must be postive: {self.attr1}')


        # SET ATTRIBUTES DEPENDENT UPON INPUT PARAMETERS
        # This dataclass is frozen to ensure that all inputs are validated, 
        # so we need to use object.__setattr__()

        # set attr2 based on attr1
        object.__setattr__(self, 'attr2', (self.attr1 < 12.0))


    @staticmethod
    def get_path_to_group_in_runconfig():
        return ['runconfig','groups','qa','nesz']


@dataclass(frozen=True)
class PointTargetAnalyzerParamGroup(YamlHDF5ParamGroup):
    '''
    Parameters from the QA-CalTools Point Target Analyzer runconfig group.

    Parameters
    ----------
    attr1 : float, optional
        Placeholder Attribute 1.
    '''

    # Override the parent class variable
    path_to_stats_h5_pta_processing_group: ClassVar[str] = \
                        '/science/%s/pointTargetAnalyzer/processing'

    attr1: float = field(
        default=2300.5,
        metadata={
        'yaml_attrs' : YamlAttrs(
            name='attr1',
            descr='''Placeholder: Attribute 1 description for runconfig.
            Each new line of text will be a separate line in the runconfig
            template.
            `attr1` is a non-negative float value.'''
        ),
        'hdf5_attrs' : HDF5Attrs(
            name='attribute1',
            units='beard-second',
            descr='Description of `attr1` for stats.h5 file',
            path=path_to_stats_h5_pta_processing_group
        )})


    def __post_init__(self):
        # validate attr1
        if not isinstance(self.attr1, float):
            raise TypeError(f'`attr1` must be a float: {self.attr1}')
        if self.attr1 < 0.0:
            raise TypeError(f'`attr1` must be non-negative: {self.attr1}')

    @staticmethod
    def get_path_to_group_in_runconfig():
        return ['runconfig','groups','qa','point_target_analyzer']


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
        WorkflowsParamGroup.populate_runcfg(runconfig_cm)
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
