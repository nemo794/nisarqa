import os
import sys
from dataclasses import dataclass, field, fields
from typing import ClassVar, Iterable, List, Optional, Type, Union

import nisarqa
import numpy as np
from nisarqa import (HDF5Attrs, HDF5ParamGroup, RootParamGroup,
                     WorkflowsParamGroup, YamlAttrs, YamlParamGroup)
from numpy.typing import ArrayLike
from ruamel.yaml import YAML
from ruamel.yaml import CommentedMap as CM

objects_to_skip = nisarqa.get_all(__name__)


@dataclass(frozen=True)
class RSLCWorkflowsParamGroup(WorkflowsParamGroup):
    '''
    The parameters specifying which RSLC-Caltools QA workflows should be run.

    This corresponds to the `qa: workflows` runconfig group.

    Parameters
    ----------
    validate : bool, optional
        True to run the validate workflow. Default: False
        (inherited from WorkflowsParamGroup class)
    qa_reports : bool, optional
        True to run the QA Reports workflow. Default: False
        (inherited from WorkflowsParamGroup class)
    abs_cal : bool, optional
        True to run the Absolute Radiometric Calibration Factor CalTool workflow
        Default: False
    nesz : bool, optional
        True to run the Noise Estimator (NESZ) workflow. Default: False
    point_target : bool, optional
        True to run the Point Target Analyzer (PTA) workflow. Default: False
    '''

    abs_cal: bool = field(
        default=WorkflowsParamGroup._default_val,
        metadata={
        'yaml_attrs': YamlAttrs(
            name='absolute_radiometric_calibration',
            descr=WorkflowsParamGroup._descr % 'Absolute Radiometric Calibration calibration tool')})

    nesz: bool = field(
        default=WorkflowsParamGroup._default_val,
        metadata={
        'yaml_attrs': YamlAttrs(
            name='nesz', 
            descr=WorkflowsParamGroup._descr % 'NESZ calibration tool')})

    point_target: bool = field(
        default=WorkflowsParamGroup._default_val,
        metadata={
        'yaml_attrs': YamlAttrs(
            name='point_target_analyzer',
            descr=WorkflowsParamGroup._descr % 'Point Target Analyzer calibration tool')})

    def __post_init__(self):

        # VALIDATE INPUTS
        super().__post_init__()
        self._check_workflows_arg('abs_cal', self.abs_cal)
        self._check_workflows_arg('nesz', self.nesz)
        self._check_workflows_arg('point_target',
                                    self.point_target)


@dataclass(frozen=True)
class InputFileGroupParamGroup(YamlParamGroup):
    '''
    Parameters from the Input File Group runconfig group.

    This corresponds to the `groups: input_file_group` runconfig group.

    Parameters
    ----------
    qa_input_file : str
        The input NISAR product file name (with path).
    '''

    # Required parameter - do not set a default
    qa_input_file: str = field(
        metadata={'yaml_attrs': 
            YamlAttrs(
                name='qa_input_file',
                descr='''Filename of the input file for QA.
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

    This corresponds to the `groups: dynamic_ancillary_file_group`
    runconfig group.

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

    This corresponds to the `groups: product_path_group` runconfig group.

    Parameters
    ----------
    qa_output_dir : str, optional
        Filepath to the output directory to store NISAR QA output files.
        Defaults to './qa'
    '''

    qa_output_dir: str = field(
        default='./qa',
        metadata={'yaml_attrs' : YamlAttrs(
            name='qa_output_dir',
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
class RSLCPowerImageParamGroup(YamlParamGroup, HDF5ParamGroup):
    '''
    Parameters to generate RSLC Power Images and Browse Image.
    
    This corresponds to the `qa_reports: power_image` runconfig group.

    Parameters
    ----------
    linear_units : bool, optional
        True to compute power image in linear units, False for decibel units.
        Defaults to True.
    nlooks_freqa, nlooks_freqb : iterable of int, None, optional
        Number of looks along each axis of the input array 
        for the specified frequency. If None, then nlooks will be computed
        on-the-fly based on `num_mpix`.
    num_mpix : float, optional
        The approx. size (in megapixels) for the final multilooked image.
        Superseded by nlooks_freq* parameters. Defaults to 4.0 MPix.
    middle_percentile : float, optional
        Defines the middle percentile range of the image array
        that the colormap covers. Must be in the range [0.0, 100.0].
        Defaults to 90.0.
    gamma : float, None, optional
        The gamma correction parameter.
        Gamma will be applied as follows:
            array_out = normalized_array ^ gamma
        where normalized_array is a copy of the image with values
        scaled to the range [0,1]. 
        The image colorbar will be defined with respect to the input
        image values prior to normalization and gamma correction.
        If None, then no normalization, no gamma correction will be applied.
        Default: 0.5
    tile_shape : iterable of int, optional
        User-preferred tile shape for processing images by batches.
        Actual tile shape may be modified by QA to be an integer
        multiple of the number of looks for multilooking, of the
        decimation ratio, etc.
        Format: (num_rows, num_cols) 
        -1 to indicate all rows / all columns (respectively).
        Defaults to (1024, 1024).

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

    _nlooks_descr_template: ClassVar[str] = \
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
            descr=_nlooks_descr_template % 'A'
        )})

    nlooks_freqb: Optional[Union[int, Iterable[int]]] = field(
        default=None,
        metadata={
        'yaml_attrs' : YamlAttrs(
            name='nlooks_freqb',
            descr=_nlooks_descr_template % 'B'
        )})

    num_mpix: int = field(
        default=4.0,
        metadata={
        'yaml_attrs' : YamlAttrs(
            name='num_mpix',
            descr='''Approx. size (in megapixels) for the final
                multilooked browse image(s). When `nlooks_freq*` parameter(s)
                is not None, those nlooks values will take precedence.'''
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
            path=nisarqa.STATS_H5_QA_PROCESSING_GROUP
            )
        })

    gamma: Optional[float] = field(
        default=0.5,
        metadata={
        'yaml_attrs' : YamlAttrs(
            name='gamma',
            descr=\
            '''Gamma correction parameter applied to power and browse image(s).
            Gamma will be applied as follows:
                array_out = normalized_array ^ gamma
            where normalized_array is a copy of the image with values
            scaled to the range [0,1]. 
            The image colorbar will be defined with respect to the input
            image values prior to normalization and gamma correction.
            If None, then no normalization and no gamma correction will be applied.'''
            ),
        'hdf5_attrs' : HDF5Attrs(
            name='powerImageGammaCorrection',
            units='unitless',
            descr='Gamma correction parameter applied to power and browse image(s). Dataset will be type float if gamma was applied, otherwise it is the string \'None\'',
            path=nisarqa.STATS_H5_QA_PROCESSING_GROUP
            )
        })

    tile_shape: List[int] = field(
        default=(1024,1024),
        metadata={
        'yaml_attrs' : YamlAttrs(
            name='tile_shape',
            descr='''User-preferred tile shape for processing images by batches.
                Actual tile shape may be modified by QA to be an integer
                multiple of the number of looks for multilooking, of the
                decimation ratio, etc.
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
                path=nisarqa.STATS_H5_QA_PROCESSING_GROUP
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
        nlooks : iterable of int or None
            Number of looks along each axis of the input array 
            for the specified frequency.
        freq : str
            The frequency to assign this number of looks to.
            Options: 'A' or 'B'
        '''
        if isinstance(nlooks, (list, tuple)):
            if all(isinstance(e, int) for e in nlooks):
                if any((e < 1) for e in nlooks) or not len(nlooks) == 2:
                    raise TypeError(
                        f'nlooks_freq{freq.lower()} must be an int or a '
                        f'sequence of two ints, which are >= 1: {nlooks}')
        elif nlooks is None:
            # the code will use num_mpix to compute `nlooks` instead.
            pass
        else:
            raise TypeError('`nlooks` must be of type iterable of int, '
                            f'or None: {nlooks}')


@dataclass(frozen=True)
class RSLCHistogramParamGroup(YamlParamGroup, HDF5ParamGroup):
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
        Defaults to True.
    tile_shape : iterable of int, optional
        User-preferred tile shape for processing images by batches.
        Actual tile shape may be modified by QA to be an integer
        multiple of the number of looks for multilooking, of the
        decimation ratio, etc.
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
                Format: [<azimuth>, <range>]'''),
        'hdf5_attrs' : HDF5Attrs(
            name='histogramDecimationRatio',
            units='unitless',
            descr='Image decimation strides used to compute power'
                  ' and phase histograms. Format: [<azimuth>, <range>]',
            path=nisarqa.STATS_H5_QA_PROCESSING_GROUP
        )})

    pow_histogram_bin_edges_range: Iterable[float] = field(
        default=(-80.0,20.0),
        metadata={
        'yaml_attrs' : YamlAttrs(
            name='pow_histogram_bin_edges_range',
            descr='''Range in dB for the power histogram's bin edges. Endpoint will
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

    tile_shape: List[int] = field(
        default=(1024,-1),
        metadata={
        'yaml_attrs' : YamlAttrs(
            name='tile_shape',
            descr='''User-preferred tile shape for processing images by batches.
                Actual tile shape may be modified by QA to be an integer
                multiple of the number of looks for multilooking, of the
                decimation ratio, etc.
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
            path=nisarqa.STATS_H5_QA_PROCESSING_GROUP
        )})

    # Phase bin edges (generated from `phs_in_radians`)
    # Note: `phs_bin_edges` is dependent upon `phs_in_radians` being set
    # first. The value of `phs_bin_edges` can be set in __post_init__,
    # but the contents of the field metadata cannot be modified 
    # after initialization. It raises this error:
    #     TypeError: 'mappingproxy' object does not support item assignment
    # So, use a lambda function; this can be called to generate the correct
    # HDF5Attrs when needed, and it does not clutter the dataclass much.
    # Usage: `obj` is an instance of RSLCHistogramParamGroup()
    phs_bin_edges: ArrayLike = field(
        init=False,
        metadata={
        'hdf5_attrs_func' : 
            lambda obj : HDF5Attrs(
                name='histogramEdgesPhase',
                units='radians' if obj.phs_in_radians else 'degrees',
                descr='Bin edges (including endpoint) for phase histogram',
                path=nisarqa.STATS_H5_QA_PROCESSING_GROUP) \
            if (isinstance(obj, RSLCHistogramParamGroup)) \
            else nisarqa.raise_(TypeError(
            f'`obj` is {type(obj)}, but must be type RSLCHistogramParamGroup'))
        })


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

        # Set attributes dependent upon pow_histogram_bin_edges_range
        # Power Bin Edges - hardcode to be in decibels
        # 101 bin edges => 100 bins
        object.__setattr__(self, 'pow_bin_edges',
                           np.linspace(self.pow_histogram_bin_edges_range[0],
                                       self.pow_histogram_bin_edges_range[1],
                                       num=101,
                                       endpoint=True))
  
        # Set attributes dependent upon phs_in_radians
        start = -np.pi if self.phs_in_radians else -180
        stop  =  np.pi if self.phs_in_radians else  180
        object.__setattr__(self, 'phs_bin_edges', 
            np.linspace(start=start, stop=stop, num=101, endpoint=True))


    @staticmethod
    def get_path_to_group_in_runconfig():
        return ['runconfig','groups','qa','qa_reports','histogram']


@dataclass(frozen=True)
class AbsCalParamGroup(YamlParamGroup, HDF5ParamGroup):
    '''
    Parameters from the QA-CalTools Absolute Calibration Factor
    runconfig group.

    Parameters
    ----------
    attr1 : float, optional
        Placeholder Attribute 1.
    '''
    
    # Attributes
    attr1: float = field(
        default=2.3,
        metadata={
        'yaml_attrs' : YamlAttrs(
            name='attr1',
            descr='''Placeholder: Attribute 1 description for runconfig.
            Each new line of text will be a separate line in the runconfig
            template. `attr1` is a non-negative float value.'''
        ),
        'hdf5_attrs' : HDF5Attrs(
            name='attribute1',
            units='smoot',
            descr='Description of `attr1` for stats.h5 file',
            path=nisarqa.STATS_H5_ABSCAL_PROCESSING_GROUP
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
class NESZParamGroup(YamlParamGroup, HDF5ParamGroup):
    '''
    Parameters from the QA-CalTools Noise Estimator (NESZ) runconfig group.

    Parameters
    ----------
    attr1 : float, optional
        Placeholder Attribute 1.

    Attributes
    ----------
    attr2 : Param
        Placeholder parameter of type bool. This is set based on `attr1`.
    '''

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
            path=nisarqa.STATS_H5_NESZ_PROCESSING_GROUP
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
class PointTargetAnalyzerParamGroup(YamlParamGroup, HDF5ParamGroup):
    '''
    Parameters from the QA-CalTools Point Target Analyzer runconfig group.

    Parameters
    ----------
    attr1 : float, optional
        Placeholder Attribute 1.
    '''

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
            path=nisarqa.STATS_H5_PTA_PROCESSING_GROUP
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
class RSLCRootParamGroup(RootParamGroup):
    '''
    Dataclass of all *ParamGroup objects to process QA for NISAR RSLC products.

    `workflows` is the only required parameter; this *ParamGroup contains
    boolean attributes that indicate which QA workflows to run.
    
    All other parameters are optional, but they each correspond to (at least)
    one of the QA workflows. Based on the workflows set to True in 
    `workflows`, certain others of these parameters will become required.

    Parameters
    ----------
    workflows : RSLCWorkflowsParamGroup
        RSLC QA Workflows parameters
    input_f : InputFileGroupParamGroup or None, optional
        Input File Group parameters for RSLC QA
    prodpath : ProductPathGroupParamGroup or None, optional
        Product Path Group parameters for RSLC QA
    power_img : RSLCPowerImageParamGroup or None, optional
        Power Image Group parameters for RSLC QA
    histogram : RSLCHistogramParamGroup or None, optional
        Histogram Group parameters for RSLC QA
    anc_files : DynamicAncillaryFileParamGroup or None, optional
        Dynamic Ancillary File Group parameters for RSLC QA-Caltools
    abs_cal : AbsCalParamGroup or None, optional
        Absolute Radiometric Calibration group parameters for RSLC QA-Caltools
    nesz : NESZParamGroup or None, optional
        NESZ group parameters for RSLC QA-Caltools
    pta : PointTargetAnalyzerParamGroup or None, optional
        Point Target Analyzer group parameters for RSLC QA-Caltools
    '''

    # Shared parameters
    workflows: RSLCWorkflowsParamGroup  # overwrite parent's `workflows` b/c new type
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
        msg = '`%s` parameter of type `%s` is required for the requested ' \
              'QA workflow(s).'

        mapping_of_req_wkflws2params = \
            self.get_mapping_of_workflows2param_grps_from_self()

        for param_grp in mapping_of_req_wkflws2params:
            if param_grp.flag_param_grp_req:
                attr = getattr(self, param_grp.root_param_grp_attr_name)
                if not isinstance(attr, param_grp.param_grp_cls_obj):
                    raise TypeError(msg % (param_grp.root_param_grp_attr_name,
                                           str(param_grp.param_grp_cls_obj)))


    @staticmethod
    def get_mapping_of_workflows2param_grps(workflows):
        Grp = RootParamGroup.ReqParamGrp  # class object for our named tuple

        flag_any_workflows_true = any([getattr(workflows, field.name) \
                            for field in fields(workflows)])

        grps_to_parse = (
            Grp(flag_param_grp_req=flag_any_workflows_true, 
                root_param_grp_attr_name='input_f',
                param_grp_cls_obj=InputFileGroupParamGroup),

            Grp(flag_param_grp_req=flag_any_workflows_true, 
                root_param_grp_attr_name='prodpath',
                param_grp_cls_obj=ProductPathGroupParamGroup),

            Grp(flag_param_grp_req=workflows.qa_reports, 
                root_param_grp_attr_name='power_img',
                param_grp_cls_obj=RSLCPowerImageParamGroup),

            Grp(flag_param_grp_req=workflows.qa_reports, 
                root_param_grp_attr_name='histogram',
                param_grp_cls_obj=RSLCHistogramParamGroup),

            Grp(flag_param_grp_req=workflows.abs_cal, 
                root_param_grp_attr_name='abs_cal',
                param_grp_cls_obj=AbsCalParamGroup),

            Grp(flag_param_grp_req= \
                    workflows.abs_cal or workflows.point_target, 
                root_param_grp_attr_name='anc_files',
                param_grp_cls_obj=DynamicAncillaryFileParamGroup),

            Grp(flag_param_grp_req=workflows.nesz, 
                root_param_grp_attr_name='nesz',
                param_grp_cls_obj=NESZParamGroup),

            Grp(flag_param_grp_req=workflows.point_target, 
                root_param_grp_attr_name='pta',
                param_grp_cls_obj=PointTargetAnalyzerParamGroup)
            )

        return grps_to_parse


    @staticmethod
    def dump_runconfig_template(indent=4):
        '''Output the runconfig template (with default values) to stdout.

        Parameters
        ----------
        indent : int, optional
            Number of spaces of an indent. Defaults to 4.
        '''

        # Build a ruamel yaml object that contains the runconfig structure
        yaml = YAML()

        # Here, the `mapping` parameter sets the size of an indent for the
        # mapping keys (aka the variable names) in the output yaml file. But,
        # it does not set the indent for the in-line comments in the output
        # yaml file; the indent spacing for inline comments will need to be
        # set later while creating the commented maps.
        # Re: `sequence` and `offset` parameters -- At the time of writing,
        # the current QA implementation of `add_param_to_cm` specifies that
        # lists should always be dumped inline, which means that these
        # `sequence` and `offset` parameters are a moot point. However,
        # should that underlying implementation change, settings sequence=4, 
        # offset=2 results in nicely-indented yaml files.
        # Ref: https://yaml.readthedocs.io/en/latest/detail.html#indentation-of-block-sequences
        yaml.indent(mapping=indent, sequence=indent, offset=max(indent-2, 0))

        runconfig_cm = CM()

        # Populate the yaml object. This order determines the order
        # the groups will appear in the runconfig.
        param_group_class_objects = (
            InputFileGroupParamGroup,
            DynamicAncillaryFileParamGroup,
            ProductPathGroupParamGroup,
            RSLCWorkflowsParamGroup,
            RSLCPowerImageParamGroup,
            RSLCHistogramParamGroup,
            AbsCalParamGroup,
            NESZParamGroup,
            PointTargetAnalyzerParamGroup
            )
        
        for param_grp in param_group_class_objects:
            param_grp.populate_runcfg(runconfig_cm, indent=indent)

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


def build_root_params(product_type, user_rncfg):
    '''
    Build the *RootParamGroup object for the specified product type.
    
    Parameters
    ----------
    product_type : str
        One of: 'rslc', 'gslc', 'gcov', 'rifg', 'runw', 'gunw', 'roff', 'goff'
    user_rncfg : dict
        A dictionary of parameters; the structure if this dict must match
        the QA runconfig file for the specified `product_type`.
    
    Returns
    -------
    root_params : RSLCRootParamGroup
        *RootParamGroup object for the specified product type. This will be 
        populated with runconfig values where provided,
        and default values for missing runconfig parameters.
    '''
    if product_type not in nisarqa.LIST_OF_NISAR_PRODUCTS:
        raise ValueError(f'`product_type` is {product_type}; must one of:'
                         f' {nisarqa.LIST_OF_NISAR_PRODUCTS}')

    if product_type == 'rslc':
        workflows_param_cls_obj = RSLCWorkflowsParamGroup
        root_param_class_obj = RSLCRootParamGroup
    else:
        raise NotImplementedError(
            f'{product_type} code not implemented yet.')
        
    # Dictionary to hold the *ParamGroup objects. Will be used as
    # kwargs for the *RootParamGroup instance.
    root_inputs = {}

    # Construct *WorkflowsParamGroup dataclass (necessary for all workflows)
    try:
        root_inputs['workflows'] = \
            _get_param_group_instance_from_runcfg(
                param_grp_cls_obj=workflows_param_cls_obj,
                user_rncfg=user_rncfg)

    except KeyError as e:
        raise KeyError('`workflows` group is a required runconfig group') from e

    # If all functionality is off (i.e. all workflows are set to false),
    # then exit early. We will not need any of the other runconfig groups.
    if not root_inputs['workflows'].at_least_one_wkflw_requested():
        # Construct *RootParamGroup with only the workflows group
        return root_param_class_obj(**root_inputs)

    workflows = root_inputs['workflows']

    wkflws2params_mapping = \
        root_param_class_obj.get_mapping_of_workflows2param_grps(
                                                        workflows=workflows)

    for param_grp in wkflws2params_mapping:
        if param_grp.flag_param_grp_req:
            try:
                root_inputs[param_grp.root_param_grp_attr_name] = \
                    _get_param_group_instance_from_runcfg(
                        param_grp_cls_obj=param_grp.param_grp_cls_obj,
                        user_rncfg=user_rncfg)

            # Some custom exception handling, such as to help make errors
            # from missing required input files less cryptic.
            except KeyError as e:
                if (product_type == 'rslc') \
                    and (param_grp.root_param_grp_attr_name == 'input_f'):

                    raise KeyError(
                    '`*qa_input_file` is a required runconfig parameter') from e
                else:
                    raise e
            except TypeError as e:
                if (product_type == 'rslc') \
                    and (param_grp.root_param_grp_attr_name == 'anc_files'):

                    raise KeyError('`corner_reflector_file` is a required '
                            'runconfig parameter for Absolute Calibration '
                            'Factor or Point Target Analyzer workflows') from e
                else:
                    raise e

    # Construct *RootParamGroup
    root_params = root_param_class_obj(**root_inputs)

    return root_params


def _get_param_group_instance_from_runcfg(
        param_grp_cls_obj: Type[YamlParamGroup],
        user_rncfg: Optional[dict] = None):
    '''
    Generate an instance of a YamlParamGroup subclass) object
    where the values from a user runconfig take precedence.
    
    Parameters
    ----------
    param_grp_cls_obj : Type[YamlParamGroup]
        A class instance of a subclass of YamlParamGroup.
        For example, `RSLCHistogramParamGroup`.
    user_rncfg : nested dict, optional
        A dict containing the user's runconfig values that (at minimum)
        correspond to the `param_grp_cls_obj` parameters. (Other values
        will be ignored.) For example, a QA runconfig yaml loaded directly
        into a dict would be a perfect input for `user_rncfg`.
        The nested structure of `user_rncfg` must match the structure
        of the QA runconfig yaml file for this parameter group.
        To see the expected yaml structure for e.g. RSLC, run  
        `nisarqa dumpconfig rslc` from the command line.
        If `user_rncfg` contains entries that do not correspond to attributes
        in `param_grp_cls_obj`, they will be ignored.
        If `user_rncfg` is either None, an empty dict, or does not contain
        values for `param_grp_cls_obj` in a nested structure that matches
        the QA runconfig group that corresponds to `param_grp_cls_obj`,
        then an instance with all default values will be returned.

    Returns
    -------
    param_grp_instance : `param_grp_cls_obj` instance
        An instance of `param_grp_cls_obj` that is fully instantiated
        using default values and the arguments provided in `user_rncfg`.
        The values in `user_rncfg` have precedence over the defaults.
    '''

    if not user_rncfg:
        # If user_rncfg is None or is an empty dict, then return the default
        return param_grp_cls_obj()

    # Get the runconfig path for this *ParamGroup
    rncfg_path = param_grp_cls_obj.get_path_to_group_in_runconfig()

    try:
        runcfg_grp_dict = nisarqa.get_nested_element_in_dict(user_rncfg, 
                                                            rncfg_path)
    except KeyError:
        # Group was not found, so construct an instance using all defaults.
        # If a dataclass has a required parameter, this will (correctly)
        # throw another error.
        return param_grp_cls_obj()
    else:
        # Get the relevant yaml runconfig parameters for this ParamGroup
        yaml_names = param_grp_cls_obj.get_dict_of_yaml_names()

        # prune extraneous fields from the runconfig group
        # (aka keep only the runconfig fields that are relevant to QA)
        # The "if..." logic will allow us to skip missing runconfig fields.
        user_input_args = \
            {cls_attr_name : runcfg_grp_dict[yaml_name] \
                for cls_attr_name, yaml_name in yaml_names.items() 
                    if yaml_name in runcfg_grp_dict}

        return param_grp_cls_obj(**user_input_args)


__all__ = nisarqa.get_all(__name__, objects_to_skip)
