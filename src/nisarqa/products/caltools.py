from __future__ import annotations

import json
import os
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from tempfile import NamedTemporaryFile
from typing import Any

import h5py
import numpy as np
from nisar.workflows import estimate_abscal_factor, point_target_analysis
from numpy.typing import DTypeLike

import nisarqa
from nisarqa import (
    AbsCalParamGroup,
    DynamicAncillaryFileParamGroup,
    PointTargetAnalyzerParamGroup,
)

# List of objects from the import statements that
# should not be included when importing this module
objects_to_skip = nisarqa.get_all(name=__name__)


def get_copols(rslc: nisarqa.RSLC, freq: str) -> list[str]:
    """
    Get a list of co-pol channels in a sub-band of an RSLC product.

    Returns a list of strings identifying the transmit (Tx) & receive (Rx)
    polarizations of each co-pol channel (e.g. 'HH' or 'VV') found in the
    specified sub-band of the product.

    Parameters
    ----------
    rslc : RSLC
        The RSLC product.
    freq : {'A', 'B'}
        The frequency sub-band to check for co-pols. Must be a valid sub-band in the
        input product.

    Returns
    -------
    copols : list of str
        The list of co-pols.
    """
    return [pol for pol in rslc.get_pols(freq) if pol[0] == pol[1]]


def file_is_empty(filepath: str | os.PathLike) -> bool:
    """
    Check if a file is empty.

    Parameters
    ----------
    filepath : path-like
        The path of an existing file.

    Returns
    -------
    is_empty : bool
        True if the file was empty; false otherwise.
    """
    # Check if the file size is zero.
    return os.stat(filepath).st_size == 0


def run_abscal_single_freq_pol(
    corner_reflector_csv: str | os.PathLike,
    rslc_hdf5: str | os.PathLike,
    freq: str,
    pol: str,
    abscal_params: AbsCalParamGroup,
) -> list[dict[str, Any]]:
    """
    Run the absolute radiometric calibration (AbsCal) tool.

    Run the `nisar.workflows.estimate_abscal_factor` workflow on a single
    freq/pol in the input RSLC file and parse the JSON output to a list of dicts
    containing one dict per valid corner reflector found in the scene.

    See the corresponding script in the `nisar` package for a detailed
    description of the output.

    Parameters
    ----------
    corner_reflector_csv : path-like
        A CSV file containing corner reflector data in the format defined by the
        NISAR Corner Reflector Software Interface Specification (SIS) document\
        [1]_.
    rslc_hdf5 : path-like
        A NISAR RSLC product file path.
    freq : {'A', 'B'}
        The frequency sub-band of the data.
    pol : str
        The transmit and receive polarization of the data.
    abscal_params : AbsCalParamGroup
        A dataclass containing the parameters for processing
        and outputting the Absolute Calibration Factor workflow.

    Returns
    -------
    results : list of dict
        A list of dicts containing one entry per valid corner reflector found
        within the area imaged by the RSLC product.

    Notes
    -----
    Since some corner reflectors may be outside the image bounds of a given
    radar observation, if the tool encounters an error while processing any
    particular target, it will emit a warning with the exception & traceback
    info and continue on to processing the next target.

    References
    ----------
    .. [1] B. Hawkins, "Corner Reflector Software Interface Specification," JPL
       D-107698 (2023).
    """
    # The parameter names in AbsCalParamGroup were designed to be identical to
    # the keyword arguments of `estimate_abscal_factor.main()` (with the
    # exception of `pthresh`, handled below). If/when this assumption no longer
    # holds, then a KeyError will be thrown below, and QA code will need to be
    # updated accordingly.
    kwds = asdict(abscal_params)

    # Rename 'power_threshold' to 'pthresh'.
    assert "pthresh" not in kwds
    kwds["pthresh"] = kwds.pop("power_threshold")

    # Create a temporary file to store the JSON output of the tool.
    with NamedTemporaryFile(suffix=".json") as tmpfile:
        # Run AbsCal tool.
        estimate_abscal_factor.main(
            corner_reflector_csv=corner_reflector_csv,
            csv_format="nisar",
            rslc_hdf5=rslc_hdf5,
            output_json=tmpfile.name,
            freq=freq,
            pol=pol,
            external_orbit_xml=None,  # Use the orbit in the RSLC product.
            **kwds,
        )

        # Parse the JSON output.
        # `json.load()` fails if the file is empty.
        if file_is_empty(tmpfile.name):
            return []
        else:
            return json.load(tmpfile)


def populate_abscal_hdf5_output(
    stats_h5: h5py.File,
    grp_path: str,
    abscal_results: Sequence[Mapping[str, Any]],
) -> None:
    """
    Store the output of the AbsCal tool in the stats HDF5 file.

    Parameters
    ----------
    stats_h5 : h5py.File
        Handle to an HDF5 file where the absolute radiometric calibration
        (AbsCal) results should be saved.
    grp_path : str
        The path of the group within `stats_h5` to store the results in. The
        group will be created if it did not already exist.
    abscal_results : sequence of dict
        The output of the AbsCal tool. A list of dicts containing one entry per
        valid corner reflector found within the area imaged by the RSLC product.
    """
    # Helper function to create a new dataset within `grp_path` that stores info
    # about each corner reflector in the AbsCal output.
    #
    # The contents of the dataset are obtained by extracting the value of `key`
    # for each corner reflector in the input `abscal_results` and casting the
    # resulting array to `ds_dtype`.
    #
    # The dataset name, description, and (optional) units are supplied by the
    # `ds_name`, `ds_descr`, and `ds_units` arguments.
    def create_dataset_from_abscal_results(
        key: str,
        ds_name: str,
        ds_descr: str,
        ds_dtype: DTypeLike,
        ds_units: str | None = None,
    ):
        ds_data = np.asarray([d[key] for d in abscal_results], dtype=ds_dtype)
        nisarqa.create_dataset_in_h5group(
            h5_file=stats_h5,
            grp_path=grp_path,
            ds_name=ds_name,
            ds_data=ds_data,
            ds_description=ds_descr,
            ds_units=ds_units,
        )

    create_dataset_from_abscal_results(
        key="id",
        ds_name="cornerReflectorId",
        ds_descr="The unique identifier of the corner reflector.",
        ds_dtype=np.bytes_,
    )

    create_dataset_from_abscal_results(
        key="survey_date",
        ds_name="cornerReflectorSurveyDate",
        ds_descr=(
            "The date (and time) when the corner reflector was surveyed most"
            " recently prior to the radar observation, as a string in ISO 8601"
            " format."
        ),
        ds_dtype=np.bytes_,
    )

    create_dataset_from_abscal_results(
        key="velocity",
        ds_name="cornerReflectorVelocity",
        ds_descr=(
            "The corner reflector velocity due to tectonic plate motion, as an"
            " East-North-Up (ENU) vector in meters per second (m/s). The"
            " velocity components are provided in local ENU coordinates with"
            " respect to the WGS 84 reference ellipsoid."
        ),
        ds_dtype=np.float_,
        ds_units="meters per second",
    )

    create_dataset_from_abscal_results(
        key="timestamp",
        ds_name="radarObservationDate",
        ds_descr=(
            "The radar observation date and time of the corner reflector, as a"
            " string in ISO 8601 format."
        ),
        ds_dtype=np.bytes_,
    )

    # XXX Elevation angle is always NaN currently. Should be fixed in R4.
    create_dataset_from_abscal_results(
        key="elevation_angle",
        ds_name="elevationAngle",
        ds_descr=(
            "Antenna elevation angle, in radians, measured w.r.t. antenna"
            " boresight, increasing toward the far-range direction and"
            " decreasing (becoming negative) toward the near-range direction."
        ),
        ds_dtype=np.float_,
        ds_units="radians",
    )

    create_dataset_from_abscal_results(
        key="absolute_calibration_factor",
        ds_name="absoluteCalibrationFactor",
        ds_descr=(
            "The absolute radiometric calibration error for the corner"
            " reflector (the ratio of the measured RCS to the predicted RCS),"
            " in linear units."
        ),
        ds_dtype=np.float_,
        ds_units="unitless",
    )


def run_abscal_tool(
    abscal_params: AbsCalParamGroup,
    dyn_anc_params: DynamicAncillaryFileParamGroup,
    input_filename: str | os.PathLike,
    stats_filename: str | os.PathLike,
) -> None:
    """
    Run the Absolute Calibration Factor workflow.

    Parameters
    ----------
    abscal_params : AbsCalParamGroup
        A dataclass containing the parameters for processing
        and outputting the Absolute Calibration Factor workflow.
    dyn_anc_params : DynamicAncillaryFileParamGroup
        A dataclass containing the parameters for the dynamic
        ancillary files.
    input_filename : path-like
        Filename (with path) for input NISAR Product
    stats_filename : path-like
        Filename (with path) for output STATS.h5 file. This is where
        outputs from the CalTool should be stored.
    """
    # Init RSLC product reader.
    # Note that this does not store an open file handle.
    rslc = nisarqa.RSLC(input_filename)

    for freq in rslc.freqs:
        # The scattering matrix of a canonical triangular trihedral corner
        # reflector is diagonal. We're only interested in measuring the co-pol
        # response since the cross-pol response should be negligible.
        pols = get_copols(rslc, freq)

        for pol in pols:
            results = run_abscal_single_freq_pol(
                corner_reflector_csv=dyn_anc_params.corner_reflector_file,
                rslc_hdf5=input_filename,
                freq=freq,
                pol=pol,
                abscal_params=abscal_params,
            )

            # Check if the results were empty (i.e. if there were no valid
            # corner reflectors in the scene). If so, don't create any HDF5
            # output for this freq/pol.
            if results:
                group_path = (
                    nisarqa.STATS_H5_ABSCAL_DATA_GROUP % rslc.band
                    + f"/frequency{freq}/{pol}"
                )

                with nisarqa.open_h5_file(stats_filename, mode="a") as stats_h5:
                    populate_abscal_hdf5_output(
                        stats_h5=stats_h5,
                        grp_path=group_path,
                        abscal_results=results,
                    )


def run_noise_estimation_tool(params, input_filename, stats_filename):
    """
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
    """
    # TODO: implement this CalTool workflow

    # Get list of bands from the input file.
    # QA must be able to handle both LSAR and SSAR.
    bands = []
    with nisarqa.open_h5_file(input_filename, mode="r") as in_file:
        for band in nisarqa.NISAR_BANDS:
            grp_path = f"/science/{band}SAR"
            if grp_path in in_file:
                bands.append(band)

    # Save placeholder data to the STATS.h5 file
    # QA code workflows have probably already written to this HDF5 file,
    # so it could be very bad to open in 'w' mode. Open in 'a' mode instead.
    with nisarqa.open_h5_file(stats_filename, mode="a") as stats_h5:
        for band in bands:
            # Step 1: Run the tool; get some results
            result = ((12.0 - params.attr1) / params.attr1) * 100.0

            # Step 2: store the data
            grp_path = nisarqa.STATS_H5_NOISE_EST_DATA_GROUP % band
            nisarqa.create_dataset_in_h5group(
                h5_file=stats_h5,
                grp_path=grp_path,
                ds_name="NoiseEstimationToolResult",
                ds_data=result,
                ds_description="Percent better than 12.0 parsecs",
                ds_units="unitless",
            )


def run_pta_single_freq_pol(
    corner_reflector_csv: str | os.PathLike,
    rslc_hdf5: str | os.PathLike,
    freq: str,
    pol: str,
    pta_params: PointTargetAnalyzerParamGroup,
) -> list[dict[str, Any]]:
    """
    Run the point target analysis (PTA) tool.

    Run the `nisar.workflows.point_target_analysis` workflow on a single
    freq/pol in the input RSLC file and parse the JSON output to a list of dicts
    containing one dict per valid corner reflector found in the scene.

    See the corresponding script in the `nisar` package for a detailed
    description of the output.

    Parameters
    ----------
    corner_reflector_csv : path-like
        A CSV file containing corner reflector data in the format defined by the
        NISAR Corner Reflector Software Interface Specification (SIS) document\
        [1]_.
    rslc_hdf5 : path-like
        A NISAR RSLC product file path.
    freq : {'A', 'B'}
        The frequency sub-band of the data.
    pol : str
        The transmit and receive polarization of the data.
    pta_params : PointTargetAnalyzerParamGroup
        A dataclass containing the parameters for processing
        and outputting the Point Target Analyzer workflow.

    Returns
    -------
    results : list of dict
        A list of dicts containing one entry per valid corner reflector found
        within the area imaged by the RSLC product.

    Notes
    -----
    Since some corner reflectors may be outside the image bounds of a given
    radar observation, if the tool encounters an error while processing any
    particular target, it will emit a warning with the exception & traceback
    info and continue on to processing the next target.

    References
    ----------
    .. [1] B. Hawkins, "Corner Reflector Software Interface Specification," JPL
       D-107698 (2023).
    """
    # The parameter names in PointTargetAnalyzerParamGroup were designed to be
    # identical to the keyword arguments of
    # `point_target_analysis.process_corner_reflector_csv()`. If/when this
    # assumption no longer holds, then a KeyError will be thrown below, and QA
    # code will need to be updated accordingly.
    kwds = asdict(pta_params)

    # Create a temporary file to store the JSON output of the tool.
    with NamedTemporaryFile(suffix=".json") as tmpfile:
        # Run PTA tool.
        point_target_analysis.process_corner_reflector_csv(
            corner_reflector_csv=corner_reflector_csv,
            csv_format="nisar",
            rslc_hdf5=rslc_hdf5,
            output_json=tmpfile.name,
            freq=freq,
            pol=pol,
            cuts=True,
            **kwds,
        )

        # Parse the JSON output.
        # `json.load()` fails if the file is empty.
        if file_is_empty(tmpfile.name):
            return []
        else:
            return json.load(tmpfile)


def populate_pta_hdf5_output(
    stats_h5: h5py.File,
    grp_path: str,
    pta_results: Sequence[Mapping[str, Any]],
) -> None:
    """
    Store the output of the PTA tool in the stats HDF5 file.

    Parameters
    ----------
    stats_h5 : h5py.File
        Handle to an HDF5 file where the point target analysis (PTA) results
        should be saved.
    grp_path : str
        The path of the group within `stats_h5` to store the results in. The
        group will be created if it did not already exist.
    pta_results : sequence of dict
        The output of the PTA tool. A list of dicts containing one entry per
        valid corner reflector found within the area imaged by the RSLC product.
    """
    # Helper function to create a new dataset within `grp_path` that stores info
    # about each corner reflector in the PTA output. This function is a bit more
    # complicated than in the AbsCal case due to the presence of nested dicts in
    # the `pta_results` that are used to populate nested groups in the HDF5 file.
    #
    # The `key` and (optionally) `subkey` arguments are used to extract data for
    # each corner reflector from the input list of (possibly nested) dicts. The
    # resulting array is then cast to `ds_dtype` to form the contents of the
    # HDF5 dataset.
    #
    # The dataset name, description, and (optional) units are supplied by the
    # `ds_name`, `ds_descr`, and `ds_units` arguments.
    def create_dataset_from_pta_results(
        key: str,
        grp_path: str,
        ds_name: str,
        ds_descr: str,
        ds_dtype: DTypeLike,
        ds_units: str | None = None,
        subkey: str | None = None,
    ):
        if subkey is None:
            data = [d[key] for d in pta_results]
        else:
            data = [d[key][subkey] for d in pta_results]

        ds_data = np.asarray(data, dtype=ds_dtype)

        nisarqa.create_dataset_in_h5group(
            h5_file=stats_h5,
            grp_path=grp_path,
            ds_name=ds_name,
            ds_data=ds_data,
            ds_description=ds_descr,
            ds_units=ds_units,
        )

    create_dataset_from_pta_results(
        key="id",
        grp_path=grp_path,
        ds_name="cornerReflectorId",
        ds_descr="The unique identifier of the corner reflector.",
        ds_dtype=np.bytes_,
    )

    create_dataset_from_pta_results(
        key="survey_date",
        grp_path=grp_path,
        ds_name="cornerReflectorSurveyDate",
        ds_descr=(
            "The date (and time) when the corner reflector was surveyed most"
            " recently prior to the radar observation, as a string in ISO 8601"
            " format."
        ),
        ds_dtype=np.bytes_,
    )

    create_dataset_from_pta_results(
        key="validity",
        grp_path=grp_path,
        ds_name="cornerReflectorValidity",
        ds_descr=(
            "The integer validity code of the corner reflector. Refer to the"
            " NISAR Corner Reflector Software Interface Specification (SIS)"
            " document for details."
        ),
        ds_dtype=np.int_,
    )

    create_dataset_from_pta_results(
        key="velocity",
        grp_path=grp_path,
        ds_name="cornerReflectorVelocity",
        ds_descr=(
            "The corner reflector velocity due to tectonic plate motion, as an"
            " East-North-Up (ENU) vector in meters per second (m/s). The"
            " velocity components are provided in local ENU coordinates with"
            " respect to the WGS 84 reference ellipsoid."
        ),
        ds_dtype=np.float_,
        ds_units="meters per second",
    )

    create_dataset_from_pta_results(
        key="magnitude",
        grp_path=grp_path,
        ds_name="peakMagnitude",
        ds_descr="The peak magnitude of the impulse response.",
        ds_dtype=np.float_,
        ds_units="unitless",
    )

    create_dataset_from_pta_results(
        key="phase",
        grp_path=grp_path,
        ds_name="peakPhase",
        ds_descr="The phase at the peak location, in radians.",
        ds_dtype=np.float_,
        ds_units="radians",
    )

    # XXX Elevation angle is always NaN currently. Should be fixed in R4.
    create_dataset_from_pta_results(
        key="elevation_angle",
        grp_path=grp_path,
        ds_name="elevationAngle",
        ds_descr=(
            "Antenna elevation angle, in radians, measured w.r.t. antenna"
            " boresight, increasing toward the far-range direction and"
            " decreasing (becoming negative) toward the near-range direction."
        ),
        ds_dtype=np.float_,
        ds_units="radians",
    )

    for direction in ["azimuth", "range"]:
        create_dataset_from_pta_results(
            key=direction,
            subkey="ISLR",
            grp_path=grp_path + f"/{direction}IRF",
            ds_name="ISLR",
            ds_descr=(
                f"The integrated sidelobe ratio of the {direction} IRF, in"
                " decibels (dB). A measure of the ratio of energy in the"
                " sidelobes to the energy in the main lobe. If `predict_null`"
                " was true, the first sidelobe will be considered part of the"
                " main lobe and the ISLR will instead measure the ratio of"
                " energy in the remaining sidelobes to the energy in the main"
                " lobe + first sidelobe."
            ),
            ds_dtype=np.float_,
            ds_units="unitless",
        )

        create_dataset_from_pta_results(
            key=direction,
            subkey="PSLR",
            grp_path=grp_path + f"/{direction}IRF",
            ds_name="PSLR",
            ds_descr=(
                f"The peak-to-sidelobe ratio of the {direction} IRF, in"
                " decibels (dB). A measure of the ratio of peak sidelobe power"
                " to the peak main lobe power."
            ),
            ds_dtype=np.float_,
            ds_units="unitless",
        )

        create_dataset_from_pta_results(
            key=direction,
            subkey="index",
            grp_path=grp_path + f"/{direction}IRF",
            ds_name="peakIndex",
            ds_descr=(
                f"The real-valued {direction} index, in samples, of the"
                " estimated peak location of the IRF within the RSLC image"
                " grid."
            ),
            ds_dtype=np.float_,
            ds_units="samples",
        )

        create_dataset_from_pta_results(
            key=direction,
            subkey="offset",
            grp_path=grp_path + f"/{direction}IRF",
            ds_name="peakOffset",
            ds_descr=(
                f"The error in the predicted target location in the {direction}"
                " direction, in samples. Equal to the signed difference between"
                " the measured location of the IRF peak in the RSLC data and"
                " the predicted location of the peak based on the surveyed"
                " corner reflector location."
            ),
            ds_dtype=np.float_,
            ds_units="samples",
        )

        create_dataset_from_pta_results(
            key=direction,
            subkey="phase ramp",
            grp_path=grp_path + f"/{direction}IRF",
            ds_name="phaseSlope",
            ds_descr=(
                f"The estimated {direction} phase slope at the target location,"
                " in radians per sample."
            ),
            ds_dtype=np.float_,
            ds_units="radians per sample",
        )

        create_dataset_from_pta_results(
            key=direction,
            subkey="resolution",
            grp_path=grp_path + f"/{direction}IRF",
            ds_name="resolution",
            ds_descr=f"The measured 3dB width of the {direction} IRF, in samples.",
            ds_dtype=np.float_,
            ds_units="samples",
        )

        create_dataset_from_pta_results(
            key=direction,
            subkey="cut",
            grp_path=grp_path + f"/{direction}IRF/cut",
            ds_name="index",
            ds_descr=(
                f"The {direction} sample indices of the magnitude and phase cut"
                " values."
            ),
            ds_dtype=np.float_,
            ds_units="samples",
        )

        create_dataset_from_pta_results(
            key=direction,
            subkey="magnitude cut",
            grp_path=grp_path + f"/{direction}IRF/cut",
            ds_name="magnitude",
            ds_descr=(
                "The magnitude of the (upsampled) impulse response function in"
                f" {direction}."
            ),
            ds_dtype=np.float_,
            ds_units="unitless",
        )

        create_dataset_from_pta_results(
            key=direction,
            subkey="phase cut",
            grp_path=grp_path + f"/{direction}IRF/cut",
            ds_name="phase",
            ds_descr=(
                "The phase of the (upsampled) impulse response function in"
                f" {direction}."
            ),
            ds_dtype=np.float_,
            ds_units="radians",
        )


def run_pta_tool(
    pta_params: PointTargetAnalyzerParamGroup,
    dyn_anc_params: DynamicAncillaryFileParamGroup,
    input_filename: str | os.PathLike,
    stats_filename: str | os.PathLike,
) -> None:
    """
    Run the Point Target Analyzer workflow.

    Parameters
    ----------
    pta_params : PointTargetAnalyzerParamGroup
        A dataclass containing the parameters for processing
        and outputting the Point Target Analyzer workflow.
    dyn_anc_params : DynamicAncillaryFileParamGroup
        A dataclass containing the parameters for the dynamic
        ancillary files.
    input_filename : path-like
        Filename (with path) for input NISAR Product
    stats_filename : path-like
        Filename (with path) for output STATS.h5 file. This is where
        outputs from the CalTool should be stored.
    """
    # Init RSLC product reader.
    # Note that this does not store an open file handle.
    rslc = nisarqa.RSLC(input_filename)

    for freq in rslc.freqs:
        # The scattering matrix of a canonical triangular trihedral corner
        # reflector is diagonal. We're only interested in measuring the co-pol
        # response since the cross-pol response should be negligible.
        pols = get_copols(rslc, freq)

        for pol in pols:
            results = run_pta_single_freq_pol(
                corner_reflector_csv=dyn_anc_params.corner_reflector_file,
                rslc_hdf5=input_filename,
                freq=freq,
                pol=pol,
                pta_params=pta_params,
            )

            # Check if the results were empty (i.e. if there were no valid
            # corner reflectors in the scene). If so, don't create any HDF5
            # output for this freq/pol.
            if results:
                freq_group_path = (
                    nisarqa.STATS_H5_PTA_DATA_GROUP % rslc.band
                    + f"/frequency{freq}"
                )
                pol_group_path = freq_group_path + f"/{pol}"

                with nisarqa.open_h5_file(stats_filename, mode="a") as stats_h5:
                    populate_pta_hdf5_output(
                        stats_h5=stats_h5,
                        grp_path=pol_group_path,
                        pta_results=results,
                    )

                    # Several of the PTA outputs are expressed in pixel
                    # coordinates rather than physical units. In order to assist
                    # with interpretability of these values, we also provide the
                    # pixel spacing of the radar grid (if not already provided
                    # for this frequency sub-band).
                    freq_group = stats_h5[freq_group_path]
                    if not "slantRangeSpacing" in freq_group:
                        nisarqa.create_dataset_in_h5group(
                            h5_file=stats_h5,
                            grp_path=freq_group_path,
                            ds_name="slantRangeSpacing",
                            ds_data=rslc.get_slant_range_spacing(freq),
                            ds_description="Slant range spacing of grid.",
                            ds_units="meters",
                        )

                        assert "sceneCenterAlongTrackSpacing" not in freq_group
                        nisarqa.create_dataset_in_h5group(
                            h5_file=stats_h5,
                            grp_path=freq_group_path,
                            ds_name="sceneCenterAlongTrackSpacing",
                            ds_data=(
                                rslc.get_scene_center_along_track_spacing(freq)
                            ),
                            ds_description=(
                                "Nominal along track spacing in meters between"
                                " consecutive lines near mid swath of the RSLC"
                                " image"
                            ),
                            ds_units="meters",
                        )


__all__ = nisarqa.get_all(__name__, objects_to_skip)
