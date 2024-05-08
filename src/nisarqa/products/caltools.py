from __future__ import annotations

import itertools
import json
import os
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import asdict, dataclass
from tempfile import NamedTemporaryFile
from typing import Any

import h5py
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import to_rgba
from matplotlib.figure import Figure
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
        ds_units="1",
    )


def run_abscal_tool(
    abscal_params: AbsCalParamGroup,
    dyn_anc_params: DynamicAncillaryFileParamGroup,
    rslc: nisarqa.RSLC,
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
    rslc : nisarqa.RSLC
        The RSLC product.
    stats_filename : path-like
        Filename (with path) for output STATS.h5 file. This is where
        outputs from the CalTool should be stored.
    """
    for freq in rslc.freqs:
        # The scattering matrix of a canonical triangular trihedral corner
        # reflector is diagonal. We're only interested in measuring the co-pol
        # response since the cross-pol response should be negligible.
        pols = get_copols(rslc, freq)

        for pol in pols:
            results = run_abscal_single_freq_pol(
                corner_reflector_csv=dyn_anc_params.corner_reflector_file,
                rslc_hdf5=rslc.filepath,
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

                with h5py.File(stats_filename, mode="a") as stats_h5:
                    populate_abscal_hdf5_output(
                        stats_h5=stats_h5,
                        grp_path=group_path,
                        abscal_results=results,
                    )


def run_nes0_tool(
    rslc: nisarqa.RSLC,
    stats_filename: str | os.PathLike,
) -> None:
    """
    Run the Noise Equivalent Sigma 0 (nes0) Tool workflow.

    Parameters
    ----------
    rslc : nisarqa.RSLC
        The RSLC product.
    stats_filename : path-like
        Filename (with path) for output STATS.h5 file. This is where
        outputs from the CalTool should be stored.
    """
    # Step 1: Copy nes0 data from input RSLC to outputs STATS.h5
    with (
        h5py.File(rslc.filepath, "r") as in_file,
        h5py.File(stats_filename, "a") as stats_file,
    ):
        for freq in rslc.freqs:
            src_grp_path = rslc.get_nes0_group_path(freq=freq)
            dest_grp_path = (
                f"{nisarqa.STATS_H5_NES0_DATA_GROUP % rslc.band}"
                f"/frequency{freq}"
            )

            try:
                # Copy entire nes0 metadata group from input file to stats.h5
                in_file.copy(src_grp_path, stats_file, dest_grp_path)
            except RuntimeError:
                # h5py.File.copy() raises this error if `src_grp_path`
                # does not exist:
                #       RuntimeError: Unable to synchronously copy object
                #       (component not found)
                nisarqa.get_logger().error(
                    "Cannot copy `nes0` Group. Input RSLC product is"
                    f" missing `nes0` for frequency {freq} at {src_grp_path}"
                )
                nisarqa.get_logger().error(
                    "Cannot plot `nes0` metadata. Input RSLC product is"
                    f" missing `nes0` for frequency {freq} at {src_grp_path}"
                )
                # Return early, because we cannot create plots
                return

    # TODO: Step 2: create plots


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
        ds_units="1",
    )

    create_dataset_from_pta_results(
        key="phase",
        grp_path=grp_path,
        ds_name="peakPhase",
        ds_descr="The phase at the peak location, in radians.",
        ds_dtype=np.float_,
        ds_units="radians",
    )

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
            ds_units="1",
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
            ds_units="1",
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
            ds_descr=(
                f"The measured 3dB width of the {direction} IRF, in samples."
            ),
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
            ds_units="1",
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
    rslc: nisarqa.RSLC,
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
    rslc : nisarqa.RSLC
        The RSLC product.
    stats_filename : path-like
        Filename (with path) for output STATS.h5 file. This is where
        outputs from the CalTool should be stored.
    """

    for freq in rslc.freqs:
        # The scattering matrix of a canonical triangular trihedral corner
        # reflector is diagonal. We're only interested in measuring the co-pol
        # response since the cross-pol response should be negligible.
        pols = get_copols(rslc, freq)

        for pol in pols:
            results = run_pta_single_freq_pol(
                corner_reflector_csv=dyn_anc_params.corner_reflector_file,
                rslc_hdf5=rslc.filepath,
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

                with h5py.File(stats_filename, mode="a") as stats_h5:
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


@dataclass(frozen=True)
class IPRCut:
    """
    1-D cut of a point-like target's impulse response (IPR).

    Each cut is a 1-D cross-section of the target's impulse response through the
    center of the target along either azimuth or range. Both magnitude and phase
    information are stored.

    Parameters
    ----------
    index : (N,) numpy.ndarray
        A 1-D array of sample indices on which the impulse response function is
        sampled, relative to the approximate location of the peak.
    magnitude : (N,) numpy.ndarray
        The magnitude (linear) of the impulse response. Must have the same shape
        as `index`.
    phase : (N,) numpy.ndarray
        The phase of the impulse response, in radians. Must have the same shape
        as `index`.
    pslr : float
        The peak-to-sidelobe ratio (PSLR) of the impulse response, in dB.
    islr : float
        The integrated sidelobe ratio (ISLR) of the impulse response, in dB.
    """

    index: np.ndarray
    magnitude: np.ndarray
    phase: np.ndarray
    pslr: float
    islr: float

    def __post_init__(self) -> None:
        # Check that the `index` array is 1-D.
        if self.index.ndim != 1:
            raise ValueError(
                f"index must be a 1-D array, instead got ndim={self.index.ndim}"
            )

        # A helper function used to check that `magnitude` and `phase` each have the
        # same shape as `index`.
        def check_shape(name: str, shape: tuple[int, ...]) -> None:
            if shape != self.index.shape:
                raise ValueError(
                    f"Shape mismatch: index and {name} must have the same"
                    f" shape, instead got {shape} != {self.index.shape}"
                )

        check_shape("magnitude", self.magnitude.shape)
        check_shape("phase", self.phase.shape)


@dataclass
class CornerReflectorIPRCuts:
    """
    Azimuth & range impulse response (IPR) cuts for a single corner reflector.

    Parameters
    ----------
    id : str
        Unique corner reflector ID.
    az_cut : IPRCut
        The azimuth impulse response cut.
    rg_cut : IPRCut
        The range impulse response cut.
    """

    id: str
    az_cut: IPRCut
    rg_cut: IPRCut


def get_pta_data_group(file_: h5py.File) -> h5py.Group:
    """
    Get the 'pointTargetAnalyzer' data group in an RSLC QA STATS.h5 file.

    Returns the group in the input HDF5 file containing the outputs of the Point
    Target Analyzer (PTA) CalTool. The group name is expected to match the
    pattern '/science/{L|S}SAR/pointTargetAnalyzer/data/'.

    Parameters
    ----------
    file_ : h5py.File
        The input HDF5 file. Must be a valid STATS.h5 file created by the RSLC
        QA workflow with the PTA tool enabled.

    Returns
    -------
    pta_data_group : h5py.Group
        The HDF5 Group containing the PTA tool output.

    Raises
    ------
    nisarqa.DatasetNotFoundError
        If not such group was found in the input HDF5 file.

    Notes
    -----
    Even if the PTA tool is enabled during RSLC QA, it still may not produce an
    output 'data' group in the STATS.h5 file if the RSLC product did not contain
    any valid corner reflectors. This is a requirement imposed on the PTA tool
    in order to simplify processing rules for the NISAR RSLC PGE.
    """
    for band in nisarqa.NISAR_BANDS:
        path = nisarqa.STATS_H5_PTA_DATA_GROUP % band
        if path in file_:
            return file_[path]

    raise nisarqa.DatasetNotFoundError(
        "Input STATS.h5 file did not contain a group matching"
        f" '{nisarqa.STATS_H5_PTA_DATA_GROUP % '(L|S)'}' "
    )


def get_valid_freqs(group: h5py.Group) -> list[str]:
    """
    Get a list of frequency sub-bands contained within the input HDF5 Group.

    The group is assumed to contain child groups corresponding to each sub-band,
    with names 'frequencyA' and/or 'frequencyB'.

    Parameters
    ----------
    group : h5py.Group
        The input HDF5 Group.

    Returns
    -------
    freqs : list of str
        A list of frequency sub-bands found in the group. Each sub-band in the
        list is denoted by its single-character identifier (e.g. 'A', 'B').
    """
    return [freq for freq in nisarqa.NISAR_FREQS if f"frequency{freq}" in group]


def get_valid_pols(group: h5py.File) -> list[str]:
    """
    Get a list of polarization channels contained within the input HDF5 Group.

    The group is assumed to contain child groups corresponding to each
    polarization, with names like 'HH', 'HV', etc.

    Parameters
    ----------
    group : h5py.Group
        The input HDF5 Group.

    Returns
    -------
    pols : list of str
        A list of polarizations found in the group.
    """
    pols = ("HH", "HV", "VH", "VV", "LH", "LV", "RH", "RV")
    return [pol for pol in pols if pol in group]


def get_ipr_cut_data(group: h5py.Group) -> Iterator[CornerReflectorIPRCuts]:
    """
    Extract corner reflector IPR cuts from a group in an RSLC QA STATS.h5 file.

    Get azimuth & range impulse response (IPR) cut data for each corner
    reflector in a single RSLC image raster (i.e. a single freq/pol pair).

    Parameters
    ----------
    group : h5py.Group
        The group in the RSLC QA STATS.h5 file containing the Point Target
        Analysis (PTA) results for a single frequency & polarization, e.g.
        '/science/LSAR/pointTargetAnalyzer/data/frequencyA/HH/'.

    Yields
    ------
    cuts : CornerReflectorIPRCuts
        Azimuth and range cuts for a single corner reflector.
    """
    # Get 1-D array of corner reflector IDs and decode from
    # np.bytes_ -> np.unicode_.
    ids = group["cornerReflectorId"][()].astype(np.unicode_)

    # Total number of corner reflectors.
    num_corners = len(ids)

    # A helper function to check that each corner reflector IPR cut dataset is
    # valid and return its contents. Each dataset should contain an MxN array of
    # impulse response azimuth/range cuts, where M is the total number of corner
    # reflectors and N is the number of samples per cut. In general, N could be
    # different for azimuth vs. range cuts but M should be the same for all
    # datasets.
    def ensure_valid_2d_dataset(dataset: h5py.Dataset) -> np.ndarray:
        # The dataset must be 2-D and its number of rows must be equal to the
        # number of corner reflectors.
        if dataset.ndim != 2:
            raise ValueError(
                f"Expected dataset {dataset.name} to contain a 2-D array,"
                f" instead got ndim={dataset.ndim}"
            )
        if dataset.shape[0] != num_corners:
            raise ValueError(
                "Expected the length (the number of rows) of dataset"
                f" {dataset.name} to be equal to the number of corner"
                f" reflectors ({num_corners}), but instead got"
                f" len={dataset.shape[0]}"
            )

        # Return the contents.
        return dataset[()]

    az_index = ensure_valid_2d_dataset(group["azimuthIRF/cut/index"])
    az_magnitude = ensure_valid_2d_dataset(group["azimuthIRF/cut/magnitude"])
    az_phase = ensure_valid_2d_dataset(group["azimuthIRF/cut/phase"])

    rg_index = ensure_valid_2d_dataset(group["rangeIRF/cut/index"])
    rg_magnitude = ensure_valid_2d_dataset(group["rangeIRF/cut/magnitude"])
    rg_phase = ensure_valid_2d_dataset(group["rangeIRF/cut/phase"])

    # A helper function to check that PSLR/ISLR datasets are valid and return
    # their contents. Each dataset should be 1-D array with length equal to the
    # number of corner reflectors.
    def ensure_valid_1d_dataset(dataset: h5py.Dataset) -> np.ndarray:
        # Check dataset shape.
        if dataset.shape != (num_corners,):
            raise ValueError(
                "Expected dataset to be a 1-D array with length equal to the"
                f" number of corner reflectors ({num_corners}), but instead got"
                f" len={dataset.shape[0]} for dataset {dataset.name}"
            )

        # Return the contents.
        return dataset[()]

    az_pslr = ensure_valid_1d_dataset(group["azimuthIRF/PSLR"])
    az_islr = ensure_valid_1d_dataset(group["azimuthIRF/ISLR"])

    rg_pslr = ensure_valid_1d_dataset(group["rangeIRF/PSLR"])
    rg_islr = ensure_valid_1d_dataset(group["rangeIRF/ISLR"])

    # Iterate over corner reflectors.
    for i in range(num_corners):
        id_ = ids[i]
        az_cut = IPRCut(
            index=az_index[i],
            magnitude=az_magnitude[i],
            phase=az_phase[i],
            pslr=az_pslr[i],
            islr=az_islr[i],
        )
        rg_cut = IPRCut(
            index=rg_index[i],
            magnitude=rg_magnitude[i],
            phase=rg_phase[i],
            pslr=rg_pslr[i],
            islr=rg_islr[i],
        )
        yield CornerReflectorIPRCuts(id=id_, az_cut=az_cut, rg_cut=rg_cut)


def plot_ipr_cuts(
    cuts: CornerReflectorIPRCuts,
    freq: str,
    pol: str,
    *,
    xlim: tuple[float, float] | None = (-15.0, 15.0),
) -> Figure:
    """
    Plot corner reflector impulse response cuts.

    Parameters
    ----------
    cuts : CornerReflectorIPRCuts
        The corner reflector azimuth & range impulse response cuts.
    freq : str
        The frequency sub-band of the RSLC image data (e.g. 'A', 'B').
    pol : str
        The polarization of the RSLC image data (e.g. 'HH', 'HV').
    xlim : (float, float) or None, optional
        X-axis limits of the plots. Defines the range of sample indices,
        relative to the approximate location of the impulse response peak, to
        display in each plot. Use a smaller range to include fewer sidelobes or
        a wider range to include more sidelobes. If None, uses Matplotlib's
        default limits. Defaults to (-15, 15).

    Returns
    -------
    fig : matplotlib.figure.Figure
        The created figure.
    """
    # Create two side-by-side sub-plots to plot azimuth & range impulse response
    # cuts.
    figsize = nisarqa.FIG_SIZE_TWO_PLOTS_PER_PAGE
    fig, axes = plt.subplots(figsize=figsize, ncols=2)

    # Set figure title.
    fig.suptitle(f"Corner Reflector {cuts.id!r} (freq={freq!r}, pol={pol!r})")

    # Set sub-plot titles.
    axes[0].set_title("Azimuth Impulse Response")
    axes[1].set_title("Range Impulse Response")

    # Each sub-plot should have both a left-edge y-axis and a right-edge y-axis.
    # The two left-edge y-axes represent power, and are independent of the
    # right-edge y-axes which represent phase.
    raxes = [ax.twinx() for ax in axes]

    # Share both (left & right) y-axes between both sub-plots.
    axes[0].sharey(axes[1])
    raxes[0].sharey(raxes[1])

    # Set y-axis labels. Only label the left axis on the left sub-plot and only
    # label the right axis on the right sub-plot.
    axes[0].set_ylabel("Power (dB)")
    raxes[1].set_ylabel("Phase (rad)")

    # Get styling properties for power & phase curves. Power info should be most
    # salient -- use a thinner line for the phase plot.
    palette = nisarqa.SEABORN_COLORBLIND
    power_props = dict(color=palette[0], linewidth=2.0)
    phase_props = dict(color=palette[1], linewidth=1.0)

    for ax, rax, cut in zip(axes, raxes, [cuts.az_cut, cuts.rg_cut]):
        # Stack the power/phase plots so that power has higher precedence
        # (https://stackoverflow.com/a/30506077).
        ax.set_zorder(rax.get_zorder() + 1)
        ax.set_frame_on(False)

        # Set x-axis limits/label.
        ax.set_xlim(xlim)
        ax.set_xlabel("Rel. Sample Index")

        # Plot impulse response power, in dB.
        power_db = nisarqa.amp2db(cut.magnitude)
        lines = ax.plot(cut.index, power_db, label="power", **power_props)

        # Plot impulse response phase, in radians.
        lines += rax.plot(cut.index, cut.phase, label="phase", **phase_props)

        # Constrain the left y-axis (power) lower limit to 50dB below the peak.
        # Otherwise, the nulls tend to stretch the y-axis range too much. (Note
        # that the azimuth & range cuts both have the same peak value.)
        peak_power = np.max(power_db)
        ax.set_ylim([peak_power - 50.0, None])

        # Set right y-axis (phase) limits. Note: don't use fixed limits for the
        # left y-axis (power) -- let Matplotlib choose limits appropriate for
        # the data.
        rax.set_ylim([-np.pi, np.pi])

        # Add a text box in the upper-right corner of each plot with PSLR & ISLR
        # info.
        ax.text(
            x=0.97,
            y=0.97,
            s=f"PSLR = {cut.pslr:.3f} dB\nISLR = {cut.islr:.3f} dB",
            transform=ax.transAxes,
            horizontalalignment="right",
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.7),
        )

    # Add a legend to the left sub-plot.
    labels = [line.get_label() for line in lines]
    axes[0].legend(lines, labels, loc="upper left")

    return fig


def add_pta_plots_to_report(stats_h5: h5py.File, report_pdf: PdfPages) -> None:
    """
    Add plots of PTA results to the RSLC QA PDF report.

    Extract the Point Target Analysis (PTA) results from `stats_h5`, use them to
    generate plots of azimuth & range impulse response for each corner reflector
    in the scene and add them to QA PDF report.

    This function has no effect if the input STATS.h5 file did not contain a PTA
    data group (for example, in the case where the RSLC product did not contain
    any corner reflectors).

    Parameters
    ----------
    stats_h5 : h5py.File
        The input RSLC QA STATS.h5 file.
    report_pdf : matplotlib.backends.backend_pdf.PdfPages
        The output PDF report.
    """
    # Get the group in the HDF5 file containing the output from the PTA tool. If
    # the group does not exist, we assume that the RSLC product did not contain
    # any valid corner reflectors, so there is nothing to do here.
    try:
        pta_data_group = get_pta_data_group(stats_h5)
    except nisarqa.DatasetNotFoundError:
        return

    # Get valid frequency groups within `.../data/`. If the data group exists,
    # it must contain at least one frequency sub-group.
    freqs = get_valid_freqs(pta_data_group)
    if not freqs:
        raise RuntimeError(
            f"No frequency groups found in {pta_data_group.name}. The STATS.h5"
            " file is ill-formed."
        )

    for freq in freqs:
        freq_group = pta_data_group[f"frequency{freq}"]

        # Get polarization sub-groups. Each frequency group must contain at
        # least one polarization sub-group.
        pols = get_valid_pols(freq_group)
        if not pols:
            raise RuntimeError(
                f"No polarization groups found in {freq_group.name}. The"
                " STATS.h5 file is ill-formed."
            )

        for pol in pols:
            pol_group = freq_group[pol]

            # Loop over corner reflectors. Generate azimuth & range cut plots
            # for each and add them to the report.
            for cuts in get_ipr_cut_data(pol_group):
                fig = plot_ipr_cuts(cuts, freq, pol)
                report_pdf.savefig(fig)

                # Close the plot.
                plt.close(fig)


def make_cr_offsets_plot(
    xy_offsets: Iterable[tuple[float, float]],
    *,
    title: str,
    xlabel: str,
    ylabel: str,
) -> Figure:
    """
    Make a plot of corner reflector position errors.

    Parameters
    ----------
    xy_offsets : iterable of (float, float)
        Iterable of x & y (or range & azimuth) position offsets for each corner
        reflector w.r.t. the expected corner locations.
    title : str
        The figure title.
    xlabel, ylabel : str
        The x-axis and y-axis labels.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The created figure.
    """
    # Make a figure with a single sub-plot.
    fig, ax = plt.subplots(
        ncols=1,
        nrows=1,
        constrained_layout="tight",
        figsize=nisarqa.FIG_SIZE_ONE_PLOT_PER_PAGE,
    )

    # Plot x & y (range & azimuth) offsets.
    colors = itertools.cycle(nisarqa.SEABORN_COLORBLIND)
    for x_offset, y_offset in xy_offsets:
        ax.scatter(
            x=x_offset,
            y=y_offset,
            marker="x",
            s=100.0,
            color=next(colors),
        )

    # We want to center the subplot axes (like crosshairs). Move the left and bottom
    # spines to x=0 and y=0, respectively. Hide the top and right spines.
    ax.spines["left"].set_position(("data", 0.0))
    ax.spines["bottom"].set_position(("data", 0.0))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Draw arrows as black triangles at the end of each axis spine. Disable
    # clipping (clip_on=False) as the marker actually spills out of the axes.
    ax.plot(0.0, 0.0, "<k", transform=ax.get_yaxis_transform(), clip_on=False)
    ax.plot(1.0, 0.0, ">k", transform=ax.get_yaxis_transform(), clip_on=False)
    ax.plot(0.0, 0.0, "vk", transform=ax.get_xaxis_transform(), clip_on=False)
    ax.plot(0.0, 1.0, "^k", transform=ax.get_xaxis_transform(), clip_on=False)

    # Update axis limits as follows:
    #  - Center the axis limits at the origin (0, 0)
    #  - Use the same axis limits for both x & y
    #  - Pad the axis limits by 25% (to make more room for axis labels)
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    xymax = 1.25 * np.max(np.abs([xmin, xmax, ymin, ymax]))
    ax.set_xlim([-xymax, xymax])
    ax.set_ylim([-xymax, xymax])

    # Force the aspect ratio to be 1:1.
    ax.set_aspect("equal")

    # Plot concentric circles with radii equal to each positive tick position
    # (except the outermost ticks, which are at or beyond the axis limits).
    inner_ticks = ax.get_xticks()[1:-1]
    for tick in filter(lambda x: x > 0.0, inner_ticks):
        circle = plt.Circle(
            xy=(0.0, 0.0),
            radius=tick,
            edgecolor=to_rgba("black", alpha=0.2),
            fill=False,
        )
        ax.add_patch(circle)

    # Add title and axis labels.
    fig.suptitle(title)
    ax.set_xlabel(xlabel, loc="right")
    ax.set_ylabel(ylabel, loc="top", rotation=0.0, verticalalignment="top")

    return fig


def plot_rslc_cr_offsets_to_pdf(
    stats_h5: h5py.File, report_pdf: PdfPages
) -> None:
    """
    Plot corner reflector azimuth/range position errors to PDF.

    Extract the Point Target Analysis (PTA) results from `stats_h5`, use them to
    generate plots of azimuth & range peak offsets for each corner reflector
    in the scene and add them to QA PDF report. A single figure is generate for each
    available frequency/polarization pair.

    This function has no effect if the input STATS.h5 file did not contain a PTA
    data group (for example, in the case where the RSLC product did not contain
    any corner reflectors).

    Parameters
    ----------
    stats_h5 : h5py.File
        The input RSLC QA STATS.h5 file.
    report_pdf : matplotlib.backends.backend_pdf.PdfPages
        The output PDF report.
    """
    # Get the group in the HDF5 file containing the output from the PTA tool. If
    # the group does not exist, we assume that the RSLC product did not contain
    # any valid corner reflectors, so there is nothing to do here.
    try:
        pta_data_group = get_pta_data_group(stats_h5)
    except nisarqa.DatasetNotFoundError:
        return

    # Get valid frequency groups within `.../data/`. If the data group exists,
    # it must contain at least one frequency sub-group.
    freqs = get_valid_freqs(pta_data_group)
    if not freqs:
        raise RuntimeError(
            f"No frequency groups found in {pta_data_group.name}. The STATS.h5"
            " file is ill-formed."
        )

    for freq in freqs:
        freq_group = pta_data_group[f"frequency{freq}"]

        # Get polarization sub-groups. Each frequency group must contain at
        # least one polarization sub-group.
        pols = get_valid_pols(freq_group)
        if not pols:
            raise RuntimeError(
                f"No polarization groups found in {freq_group.name}. The"
                " STATS.h5 file is ill-formed."
            )

        for pol in pols:
            pol_group = freq_group[pol]

            # Extract azimuth & range peak offsets data from STATS.h5.
            az_offsets = pol_group["azimuthIRF/peakOffset"]
            rg_offsets = pol_group["rangeIRF/peakOffset"]

            # Check that both datasets contain 1-D arrays with the same shape.
            for dataset in [az_offsets, rg_offsets]:
                if dataset.ndim != 1:
                    raise ValueError(
                        f"Expected dataset {dataset.name} to contain a 1-D"
                        f" array, instead got ndim={dataset.ndim}"
                    )
            if az_offsets.shape != rg_offsets.shape:
                raise ValueError(
                    "Azimuth & range peak offsets must have the same shape,"
                    f" instead got {az_offsets.shape=} and {rg_offsets.shape=}"
                )

            # Make a plot of corner reflector position offsets for the current
            # freq/pol and append it to the PDF report.
            fig = make_cr_offsets_plot(
                xy_offsets=zip(rg_offsets, az_offsets),
                title=(
                    "Corner Reflector Azimuth/Range Position Error (pixels)"
                    f"\nfreq={freq!r}, pol={pol!r}"
                ),
                xlabel="Range Offset\n(pixels)",
                ylabel="Azimuth Offset\n(pixels)",
            )
            report_pdf.savefig(fig)

            # Close the figure.
            plt.close(fig)


__all__ = nisarqa.get_all(__name__, objects_to_skip)
