from dataclasses import dataclass, field, fields
from typing import Iterable
import os
import h5py

import nisarqa

# List of objects from the import statements that
# should not be included when importing this module
objects_to_skip = nisarqa.get_all(name=__name__)

@dataclass
class CoreCalToolsParams:
    '''
    Data structure to hold the core parameters for the
    CalTools code's output files that are
    common to all NISAR CalTools workflows.

    Parameters
    ----------
    stats_h5 : h5py.File
        The output file to save CalTools metrics, etc. to
    bands : sequence of str
        A sequence of the bands in the input file,
        Examples: ['LSAR','SSAR'] or ('LSAR')
    '''

    # Attributes that are common to all NISAR CalTools workflows
    stats_h5: h5py.File
    bands: Iterable[str]

    @classmethod
    def from_parent(cls, core, **kwargs):
        '''
        Construct a child of CoreCalToolsParams
        from an existing instance of CoreCalToolsParams.

        Parameters
        ----------
        core : CoreCalToolsParams
            Instance of CoreCalToolsParams whose attributes will
            populate the new child class instance.
            Note that a only shallow copy is performed when populating
            the new instance; for parent attributes that contain
            references, the child object will reference the same
            same 
        **kwargs : optional
            Attributes specific to the child class of CoreCalToolsParams.

        Example
        -------
        >>> parent = CoreCalToolsParams()
        >>> @dataclass
        ... class ChildParams(CoreCalToolsParams):
        ...     x: int
        ... 
        >>> y = ChildParams.from_parent(core=parent, x=2)
        >>> print(y)
        ChildParams(stats_h5=<contextlib._GeneratorContextManager object at 0x7fd04cab6690>, x=2)        
        '''
        if not isinstance(core, CoreCalToolsParams):
            raise ValueError('`core` input must be of type CoreCalToolsParams.')

        # Create shallow copy of the dataclass into a dict.
        # (Using the asdict() method to create a deep copy throws a
        # "TypeError: cannot serialize '_io.BufferedWriter' object" 
        # exception when copying the field with the PdfPages object.)
        core_dict = dict((field.name, getattr(core, field.name)) 
                                            for field in fields(core))

        return cls(**core_dict, **kwargs)


@dataclass
class AbsCalParams(CoreCalToolsParams):
    '''
    Data structure to hold the parameters for the NISAR
    CalTools Absolute Calibration Factor workflow.

    Use the class method .from_parent() to construct
    an instance from an existing CoreCalToolsParams object.

    Parameters
    ----------
    **core : CoreCalToolsParams
        All fields from the parent class CoreCalToolsParams.
    corner_reflector_filename : str
        Filename (with path) to the corner reflector
        locations.
    attr1 : int
        Placeholder parameter; should be removed once the
        actual parameters are provided.

    Attributes
    ----------
    attr2 : str
        Placeholder parameter; should be removed once the
        actual parameters are provided.
    '''

    # Attributes for running the Abs-Cal workflow
    corner_reflector_filename: str  # required input
    attr1: float = 1.0

    # Auto-generated attributes
    attr2: str = field(init=False)

    def __post_init__(self):
        # Use this section to auto-generate any new 
        # attributes for this dataclass.
        if self.attr1 > 2:
            self.attr2 = 'Big Value!'
        else:
            self.attr2 = 'Smaller Value'

    def save_processing_params_to_h5(self, path_to_group):
        '''
        Populate this instance's `stats_h5` HDF5 file 
        with its processing parameters.

        This function will populate the following fields
        in `stats_h5` for all bands in self.bands:
            /science/<band>/<path_to_group>/cornerReflectorFile
            /science/<band>/<path_to_group>/attr1
            /science/<band>/<path_to_group>/attr2
        
        Parameters
        ----------
        path_to_group : str
            Internal path in `stats_h5` to the HDF5 group where
            the processing parameters will be stored.
            Full path will be constructed in this format:
                /science/<band>/<path_to_group>/<dataset>

        '''
        for band in self.bands:
            # Open the group in the file, creating it if it doesn’t exist.
            grp_path = os.path.join('/science', band, path_to_group.lstrip('/'))
            proc_grp = self.stats_h5.require_group(grp_path)

            nisarqa.create_dataset_in_h5group(grp=proc_grp,
                                            ds_name='cornerReflectorFile',
                                            ds_data=self.corner_reflector_filename,
                                            ds_units='unitless',
                                            ds_description='TODO Source file for corner reflector locations')

            nisarqa.create_dataset_in_h5group(grp=proc_grp,
                                            ds_name='attr1',
                                            ds_data=self.attr1,
                                            ds_units='seconds',
                                            ds_description='TODO Seconds since start of epoch')

            nisarqa.create_dataset_in_h5group(grp=proc_grp,
                                            ds_name='attr2',
                                            ds_data=self.attr2,
                                            ds_units='unitless',
                                            ds_description='TODO Description of Attribute 2')


@dataclass
class NESZParams(CoreCalToolsParams):
    '''
    Data structure to hold the parameters for the NISAR
    CalTools NESZ workflow.

    Use the class method .from_parent() to construct
    an instance from an existing CoreCalToolsParams object.

    Parameters
    ----------
    **core : CoreCalToolsParams
        All fields from the parent class CoreCalToolsParams.
    attr1 : int
        Placeholder parameter; should be removed once the
        actual parameters are provided.

    Attributes
    ----------
    attr2 : str
        Placeholder parameter; should be removed once the
        actual parameters are provided.
    '''

    # Attributes for running the NESZ workflow
    attr1: float = 1.0

    # Auto-generated attributes
    attr2: str = field(init=False)

    def __post_init__(self):
        # Use this section to auto-generate any new 
        # attributes for this dataclass.
        if self.attr1 > 2:
            self.attr2 = 'Big Value!'
        else:
            self.attr2 = 'Smaller Value'

    def save_processing_params_to_h5(self, path_to_group):
        '''
        Populate this instance's `stats_h5` HDF5 file 
        with its processing parameters.

        This function will populate the following fields
        in `stats_h5` for all bands in self.bands:
            /science/<band>/<path_to_group>/attr1
            /science/<band>/<path_to_group>/attr2

        Parameters
        ----------
        path_to_group : str
            Internal path in `stats_h5` to the HDF5 group where
            the processing parameters will be stored.
            Full path will be constructed in this format:
                /science/<band>/<path_to_group>/<dataset>
        '''

        # Open the group in the file, creating it if it doesn’t exist.
        for band in self.bands:
            grp_path = os.path.join('/science', band, path_to_group.lstrip('/'))
            proc_grp = self.stats_h5.require_group(grp_path)

            nisarqa.create_dataset_in_h5group(grp=proc_grp,
                                            ds_name='attr1',
                                            ds_data=self.attr1,
                                            ds_units='seconds',
                                            ds_description='TODO Seconds since start of epoch')

            nisarqa.create_dataset_in_h5group(grp=proc_grp,
                                            ds_name='attr2',
                                            ds_data=self.attr2,
                                            ds_units='unitless',
                                            ds_description='TODO Description of Attribute 2')


@dataclass
class PointTargetAnalyzerParams(CoreCalToolsParams):
    '''
    Data structure to hold the parameters for the NISAR
    CalTools Point Target AnalyzerParams workflow.

    Use the class method .from_parent() to construct
    an instance from an existing CoreCalToolsParams object.

    Parameters
    ----------
    **core : CoreCalToolsParams
        All fields from the parent class CoreCalToolsParams.
    corner_reflector_filename : str
        Filename (with path) to the corner reflector
        locations.
    attr1 : int
        Placeholder parameter; should be removed once the
        actual parameters are provided.

    Attributes
    ----------
    attr2 : str
        Placeholder parameter; should be removed once the
        actual parameters are provided.
    '''

    # Attributes for running the Point Target Analyzer workflow
    corner_reflector_filename: str  # required input
    attr1: float = 1.0

    # Auto-generated attributes
    attr2: str = field(init=False)

    def __post_init__(self):
        # Use this section to auto-generate any new 
        # attributes for this dataclass.
        if self.attr1 > 2:
            self.attr2 = 'Big Value!'
        else:
            self.attr2 = 'Smaller Value'

    def save_processing_params_to_h5(self, path_to_group):
        '''
        Populate this instance's `stats_h5` HDF5 file 
        with its processing parameters.

        This function will populate the following fields
        in `stats_h5` for all bands in self.bands:
            /science/<band>/<path_to_group>/cornerReflectorFile
            /science/<band>/<path_to_group>/Attribute1
            /science/<band>/<path_to_group>/PTAAttribute2

        Parameters
        ----------
        path_to_group : str
            Internal path in `stats_h5` to the HDF5 group where
            the processing parameters will be stored.
            Full path will be constructed in this format:
                /science/<band>/<path_to_group>/<dataset>
        '''
        for band in self.bands:
            # Open the group in the file, creating it if it doesn’t exist.
            grp_path = os.path.join('/science', band, path_to_group.lstrip('/'))
            proc_grp = self.stats_h5.require_group(grp_path)

            nisarqa.create_dataset_in_h5group(grp=proc_grp,
                                            ds_name='cornerReflectorFile',
                                            ds_data=self.corner_reflector_filename,
                                            ds_units='unitless',
                                            ds_description='TODO Source file for corner reflector locations')

            nisarqa.create_dataset_in_h5group(grp=proc_grp,
                                            ds_name='attr1',
                                            ds_data=self.attr1,
                                            ds_units='seconds',
                                            ds_description='TODO Seconds since start of epoch')

            nisarqa.create_dataset_in_h5group(grp=proc_grp,
                                            ds_name='attr2',
                                            ds_data=self.attr2,
                                            ds_units='unitless',
                                            ds_description='TODO Description of Attribute 2')


def run_absolute_cal_factor(params):
    '''
    Run the Absolute Calibration Factor workflow.

    Parameters
    ----------
    params : AbsCalParams
        A dataclass containing the parameters for processing
        and outputting the Absolute Calibration Factor workflow.
    '''
    print('TODO: Integrate the Absolute Calibration Factor workflow')

    # TODO: implement this CalTool workflow

    # Save results to stats.h5
    for band in params.bands:
        grp_path = os.path.join('/science', band, 'absoluteCalibrationFactor/data')
        grp = params.stats_h5.require_group(grp_path)
        nisarqa.create_dataset_in_h5group(grp=grp,
                                        ds_name='abscalResult',
                                        ds_data=5.3,
                                        ds_units='meters',
                                        ds_description='TODO Description for abscalResult')


def run_nesz(params):
    '''
    Run the NESZ workflow.

    Parameters
    ----------
    params : NESZParams
        A dataclass containing the parameters for processing
        and outputting the NESZ workflow.
    '''
    print('TODO: Integrate the NESZ workflow')

    # TODO: implement this CalTool workflow

    for band in params.bands:
        grp_path = os.path.join('/science', band, 'NESZ/data')

        # Save results to stats.h5
        grp = params.stats_h5.require_group(grp_path)
        nisarqa.create_dataset_in_h5group(grp=grp,
                                        ds_name='NESZResult',
                                        ds_data=5.3,
                                        ds_units='meters',
                                        ds_description='TODO Description for NESZResult')


def run_point_target_analyzer(params):
    '''
    Run the Point Target Analyzer workflow.

    Parameters
    ----------
    params : PointTargetAnalyzerParams
        A dataclass containing the parameters for processing
        and outputting the Point Target Analyzer workflow.
    '''
    print('TODO: Integrate the Point Target Analyzer workflow')
    
    # TODO: implement this CalTool workflow

    for band in params.bands:
        grp_path = os.path.join('/science', band, 'pointTargetAnalyzer/data')

        # Save results to stats.h5
        grp = params.stats_h5.require_group(grp_path)
        nisarqa.create_dataset_in_h5group(grp=grp,
                                        ds_name='pointTargetAnalyzerResult',
                                        ds_data=5.3,
                                        ds_units='meters',
                                        ds_description='TODO Description for pointTargetAnalyzerResult')


__all__ = nisarqa.get_all(__name__, objects_to_skip)
