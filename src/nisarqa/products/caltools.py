import os
from dataclasses import dataclass, field, fields
from typing import Iterable

import h5py
import nisarqa

# List of objects from the import statements that
# should not be included when importing this module
objects_to_skip = nisarqa.get_all(name=__name__)

def run_abscal_tool(params, input_filename, stats_filename):
    '''
    Run the Absolute Calibration Factor workflow.

    Parameters
    ----------
    params : AbsCalParams
        A dataclass containing the parameters for processing
        and outputting the Absolute Calibration Factor workflow.
    input_filename : str
        Filename (with path) for input NISAR Product
    stats_filename : str
        Filename (with path) for output STATS.h5 file. This is where
        outputs from the CalTool should be stored.
    '''
    # TODO: implement this CalTool workflow

    # Get list of bands from the input file.
    # QA must be able to handle both LSAR and SSAR.
    bands = []
    with nisarqa.open_h5_file(input_filename, mode='r') as in_file:
        for band in nisarqa.BANDS:
            grp_path = f'/science/{band}'
            if grp_path in in_file:
                bands.append(band)

    # Save placeholder data to the STATS.h5 file
    # QA code workflows have probably already written to this HDF5 file,
    # so it could be bad to open in 'w' mode. Open in 'a' mode instead.
    with nisarqa.open_h5_file(stats_filename, mode='a') as stats_h5:
        for band in bands:
            # Step 1: Run the tool; get some results
            result = -63.12

            # Step 2: store the data
            grp_path = f'/science/{band}/absoluteCalibrationFactor/data'
            nisarqa.create_dataset_in_h5group(
                    h5_file = stats_h5,
                    grp_path=grp_path,
                    ds_name='abscalResult',
                    ds_data=result,
                    ds_description='TODO Description for abscalResult',
                    ds_units=params.attr1.units  # use the same units as attr1
            )


def run_nesz_tool(params, input_filename, stats_filename):
    '''
    Run the NESZ workflow.

    Parameters
    ----------
    params : NESZParams
        A dataclass containing the parameters for processing
        and outputting the NESZ workflow.
    input_filename : str
        Filename (with path) for input NISAR Product
    stats_filename : str
        Filename (with path) for output STATS.h5 file. This is where
        outputs from the CalTool should be stored.
    '''
    # TODO: implement this CalTool workflow

    # Get list of bands from the input file.
    # QA must be able to handle both LSAR and SSAR.
    bands = []
    with nisarqa.open_h5_file(input_filename, mode='r') as in_file:
        for band in nisarqa.BANDS:
            grp_path = f'/science/{band}'
            if grp_path in in_file:
                bands.append(band)

    # Save placeholder data to the STATS.h5 file
    # QA code workflows have probably already written to this HDF5 file,
    # so it could be very bad to open in 'w' mode. Open in 'a' mode instead.
    with nisarqa.open_h5_file(stats_filename, mode='a') as stats_h5:
        for band in bands:
            # Step 1: Run the tool; get some results
            result = ((12.0 - params.attr1.val) / params.attr1.val) * 100.

            # Step 2: store the data
            grp_path = f'/science/{band}/NESZ/data'
            nisarqa.create_dataset_in_h5group(
                    h5_file=stats_h5,
                    grp_path=grp_path,
                    ds_name='NESZResult',
                    ds_data=result,
                    ds_description='Percent better than 12.0 parsecs',
                    ds_units='unitless'
            )


def run_pta_tool(params, input_filename, stats_filename):
    '''
    Run the Point Targer Analyzer workflow.

    Parameters
    ----------
    params : PointTargetAnalyzerParams
        A dataclass containing the parameters for processing
        and outputting the Point Target Analyzer workflow.
    input_filename : str
        Filename (with path) for input NISAR Product
    stats_filename : str
        Filename (with path) for output STATS.h5 file. This is where
        outputs from the CalTool should be stored.
    '''
    # TODO: implement this CalTool workflow

    # Get list of bands from the input file.
    # QA must be able to handle both LSAR and SSAR.
    bands = []
    with nisarqa.open_h5_file(input_filename, mode='r') as in_file:
        for band in nisarqa.BANDS:
            grp_path = f'/science/{band}'
            if grp_path in in_file:
                bands.append(band)

    # Save placeholder data to the STATS.h5 file
    # QA code workflows have probably already written to this HDF5 file,
    # so it could be very bad to open in 'w' mode. Open in 'a' mode instead.
    with nisarqa.open_h5_file(stats_filename, mode='a') as stats_h5:
        for band in bands:
            # Step 1: Run the tool; get some results
            result = 'PLACEHOLDER'

            # Step 2: store the data
            grp_path = f'/science/{band}/pointTargetAnalyzer/data'
            nisarqa.create_dataset_in_h5group(
                    h5_file = stats_h5,
                    grp_path=grp_path,
                    ds_name='pointTargetAnalyzerResult',
                    ds_data=result,
                    ds_description='PLACEHOLDER short description',
                    ds_units=None
            )


__all__ = nisarqa.get_all(__name__, objects_to_skip)
