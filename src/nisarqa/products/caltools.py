import nisarqa

# List of objects from the import statements that
# should not be included when importing this module
objects_to_skip = nisarqa.get_all(name=__name__)

def run_abscal_tool(abscal_params, dyn_anc_params, 
                    input_filename, stats_filename):
    '''
    Run the Absolute Calibration Factor workflow.

    Parameters
    ----------
    abscal_params : AbsCalParams
        A dataclass containing the parameters for processing
        and outputting the Absolute Calibration Factor workflow.
    dyn_anc_params : DynamicAncillaryFileParams
        A dataclass containing the parameters for the dynamic
        ancillary files.
    input_filename : str
        Filename (with path) for input NISAR Product
    stats_filename : str
        Filename (with path) for output STATS.h5 file. This is where
        outputs from the CalTool should be stored.
    '''
    # TODO: implement this CalTool workflow

    # Check that the *Params were properly received
    assert isinstance(abscal_params, nisarqa.AbsCalParamGroup)
    assert isinstance(dyn_anc_params, nisarqa.DynamicAncillaryFileParamGroup)

    # Get list of bands from the input file.
    # QA must be able to handle both LSAR and SSAR.
    bands = []
    with nisarqa.open_h5_file(input_filename, mode='r') as in_file:
        for band in nisarqa.NISAR_BANDS:
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
            grp_path = nisarqa.STATS_H5_ABSCAL_DATA_GROUP % band
            ds_units = abscal_params.get_units_from_hdf5_metadata('attr1')

            nisarqa.create_dataset_in_h5group(
                    h5_file = stats_h5,
                    grp_path=grp_path,
                    ds_name='abscalResult',
                    ds_data=result,
                    ds_description='TODO Description for abscalResult',
                    ds_units=ds_units
            )


def run_noise_estimation_tool(params, input_filename, stats_filename):
    '''
    Run the Noise Estimation Tool workflow.

    Parameters
    ----------
    params : NoiseEstimationParamGroup
        A dataclass containing the parameters for processing
        and outputting the Noise Estimation Tool workflow.
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
        for band in nisarqa.NISAR_BANDS:
            grp_path = f'/science/{band}'
            if grp_path in in_file:
                bands.append(band)

    # Save placeholder data to the STATS.h5 file
    # QA code workflows have probably already written to this HDF5 file,
    # so it could be very bad to open in 'w' mode. Open in 'a' mode instead.
    with nisarqa.open_h5_file(stats_filename, mode='a') as stats_h5:
        for band in bands:
            # Step 1: Run the tool; get some results
            result = ((12.0 - params.attr1) / params.attr1) * 100.

            # Step 2: store the data
            grp_path = nisarqa.STATS_H5_NOISE_EST_DATA_GROUP % band
            nisarqa.create_dataset_in_h5group(
                    h5_file=stats_h5,
                    grp_path=grp_path,
                    ds_name='NoiseEstimationToolResult',
                    ds_data=result,
                    ds_description='Percent better than 12.0 parsecs',
                    ds_units='unitless'
            )


def run_pta_tool(pta_params, dyn_anc_params,
                    input_filename, stats_filename):
    '''
    Run the Point Targer Analyzer workflow.

    Parameters
    ----------
    pta_params : PointTargetAnalyzerParams
        A dataclass containing the parameters for processing
        and outputting the Point Target Analyzer workflow.
    dyn_anc_params : DynamicAncillaryFileParams
        A dataclass containing the parameters for the dynamic
        ancillary files.
    input_filename : str
        Filename (with path) for input NISAR Product
    stats_filename : str
        Filename (with path) for output STATS.h5 file. This is where
        outputs from the CalTool should be stored.
    '''
    # TODO: implement this CalTool workflow

    # Check that the *Params were properly received
    assert isinstance(pta_params, nisarqa.PointTargetAnalyzerParamGroup)
    assert isinstance(dyn_anc_params, nisarqa.DynamicAncillaryFileParamGroup)

    # Get list of bands from the input file.
    # QA must be able to handle both LSAR and SSAR.
    bands = []
    with nisarqa.open_h5_file(input_filename, mode='r') as in_file:
        for band in nisarqa.NISAR_BANDS:
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
            grp_path = nisarqa.STATS_H5_PTA_DATA_GROUP % band
            nisarqa.create_dataset_in_h5group(
                    h5_file=stats_h5,
                    grp_path=grp_path,
                    ds_name='pointTargetAnalyzerResult',
                    ds_data=result,
                    ds_description='PLACEHOLDER short description',
                    ds_units=None
            )


__all__ = nisarqa.get_all(__name__, objects_to_skip)
