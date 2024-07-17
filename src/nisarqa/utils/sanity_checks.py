from __future__ import annotations

import h5py
import numpy as np
from numpy.typing import ArrayLike

import nisarqa

objects_to_skip = nisarqa.get_all(name=__name__)


def validate_spacing(
    data: ArrayLike, spacing: float, dname: str, epsilon=1e-5
) -> None:
    """
    Validate that the values in `data` are in ascending order
    and equispaced, within a tolerance of +/- epsilon (EPS = 1.0E-05).

    Example usage: validate_spacing() can be used to validate that the
    timestamps in a zero doppler time array are always increasing,
    or that the distances in a slant range array are always increasing.

    Parameters
    ----------
    data : array_like
        The 1D data array to be validated.
    spacing : numeric
        The theoretical interval between each value in `data`.
    dataset_name : string
        Name of `data`, which will be used for log messages.
    """

    log = nisarqa.get_logger()

    # Validate that data's values are strictly increasing
    delta = data[1:] - data[:-1]

    if not np.all(delta > 0.0):
        idx = np.where(delta <= 0.0)
        log.info(
            f"{dname}: Found {len(idx[0])} elements with negative spacing: "
            f"{data[idx]} at locations {idx}"
        )

    # Validate that successive values in data are separated by `spacing`,
    # within a tolerance of +/- epsilon.
    EPS = 1.0e-05
    diff = np.abs(delta - spacing)
    try:
        assert np.all(diff <= EPS)
    except AssertionError as e:
        idx = np.where(diff > EPS)
        log.error(
            f"{dname}: Found {len(idx[0])} elements with unexpected steps: "
            f"{diff[idx]} at locations {idx}"
        )


def verify_shapes_are_consistent(product: nisarqa.NisarProduct) -> None:
    """
    Verify that the shape dimensions are consistent between datasets.

    For example, if the Dataset `zeroDopplerTime` has a shape of
    `frequencyALength` specified in the XML and has dimensions (132,),
    then all other Datasets with a shape of `frequencyALength` in the XML
    must also have a shape of (132,) in the HDF5.
    """

    # Create a dict of all groups and datasets inside the input file,
    # where the path is the key and the value is an hdf5 object

    # Parse all of the xml Shapes into a dict, initializing each to None.
    # key will be the Shape name,
    # value will be set the first time it is encountered.
    # Each subsequent time it is encountered, check that the shape is consistent

    # for every item in the xml_tree:
    #     if that path exists in the input file:
    #         check that the description matches the xml_file
    #         check that the units matches the xml_file

    #         Look at the shape:
    #             if Shape is in the NISAR Shapes:
    #                 if shape.value is None:
    #                     set the Value
    #                 else:
    #                     assert shape.value == the value in the dict

    #             if the shape is a known constant, confirm the actual data
    #                 has that shape

    #             else:
    #                 raise error, because each dataset should have a shape

    #         remove that path from the input file's dict

    # if the input file's dict is not empty:
    #     raise error - input file has extraneous datasets
    pass


def dataset_sanity_checks(
    product: nisarqa.NisarProduct, number_of_tracks: int = 173
) -> None:
    """
    Perform a series of verification checks on the input product's datasets.

    -> If logic error, fail the entire program.
    -> If validation error, then log those to terminal/log, and continue and do the quality outputs.

    Parameters
    ----------
    product : nisarqa.NisarProduct
        Instance of the input product.
    number_of_tracks : int
        # TODO - make this a GLOBAL
        The total number of tracks for the NISAR missions during operations.
        As of 2024, this is 173 tracks.
        TODO: confirm that there are still 173 tracks.

    """
    log = nisarqa.get_logger()
    passes = True

    # Log which bands / freqs / pols are available
    # Sanity Check - is there at least one each of band, freq, pol?

    # Check that the bands are of the available options in the constants

    # Check that identification/listOfFrequencies set equals the frequencies available
    # Check that the frequencies are of the available options in the constants

    # Check that frequency*/listOfPolarizations set equals the pols available
    # Check that the polarizations are of the available options in the constants

    # Check that frequency*/listOfLayers set equals the pols available
    # Check that the layer groups are of the available options in the constants

    with h5py.File(product.filepath, "r") as f:
        id_path = product.identification_path
        id_grp = f[id_path]

        # TODO - break into 2 functions: 1) verify integer, 2) verify `fail_if`
        def _integer_checker(ds_name: str, fail_if) -> None:
            ds_path = f"{id_path}/{ds_name}"

            try:
                data = id_grp[ds_name][()]
            except KeyError:
                log.error(f"Missing dataset: {ds_path}")
                # look at nonlocal keyword
                passes = False
            else:
                if not isinstance(data, int):
                    log.error(
                        f"Dataset has type `{type(data)}`, must be an integer."
                        f" Dataset: {ds_path}"
                    )
                    passes = False
                elif fail_if(data):
                    log.error(
                        f"Invalid `{ds_name}`: {data}." f" Dataset: {ds_path}"
                    )
                    passes = False

        _integer_checker(
            ds_name="absoluteOrbitNumber", fail_if=lambda x: x <= 0
        )

        _integer_checker(
            ds_name="trackNumber",
            fail_if=lambda x: (x <= 0) or (x > number_of_tracks),
        )

        _integer_checker(ds_name="frameNumber", fail_if=lambda x: x <= 0)

        _integer_checker(ds_name="cycleNumber", fail_if=lambda x: x <= 0)

        def _string_checker(ds_name: str, fail_if) -> None:

            ds_path = {id_path} / {ds_name}

            try:
                data = nisarqa.byte_string_to_python_str(id_grp[ds_path][()])
            except KeyError:
                log.error(f"Missing dataset: {ds_path}")
                passes = False
            else:
                if not isinstance(data, np.bytes_):
                    log.error(
                        f"Dataset has type `{type(data)}`, must be a NumPy."
                        f" byte string. Dataset: {ds_path}"
                    )
                    passes = False
                elif fail_if(data):
                    log.error(
                        # TODO - improve error message
                        f"Invalid `{ds_name}`: {data}."
                        f" Dataset: {ds_path}"
                    )
                    passes = False

        _string_checker(
            ds_name="productType",
            fail_if=lambda x: x != product.product_type.upper(),
        )
        _string_checker(
            ds_name="lookDirection",
            fail_if=lambda x: x not in ("Left", "Right"),
        )

        # # Check Frequencies
        # Check that FreqA's centerFrequency (processedCenterFrequency for RSLC) is
        # less than FreqB's centerFrequency (processedCenterFrequency for RSLC)

    path_to_swaths_grids = "/science/LSAR/SLC/swaths"

    start_time = str(id_group["zeroDopplerStartTime"][0])
    end_time = str(id_group["zeroDopplerEndTime"][0])
    if end_time <= start_time:
        print(f"Start Time {start_time} not less than End Time {end_time}")

    # Check time
    if RSLC:
        for b in ("LSAR", "SSAR"):
            time = in_file[f"{path_to_swaths_grids}/zeroDopplerTime"][...]
            spacing = in_file[f"{path_to_swaths_grids}/zeroDopplerTimeSpacing"][
                ...
            ]

            validate_spacing(time, spacing, "%s zeroDopplerTime" % b)

    # Check time for all products

    path_to_identification = "/science/LSAR/RSLC/identification"

    for band in ("LSAR", "SSAR"):
        start_name = "zeroDopplerStartTime"
        end_name = "zeroDopplerEndTime"

        start_time = in_file[f"{path_to_identification}/{start_name}"][...]
        end_time = in_file[f"{path_to_identification}/{end_name}"][...][...]
        print(f"{band} {start_name}: {start_time}, {end_name} {end_time}")

        # try:
        #     start_time = bytes(start_time).split(b".")[0].decode("utf-8").replace("\x00", "")
        #     end_time = bytes(end_time).split(b".")[0].decode("utf-8").replace("\x00", "")
        # except UnicodeDecodeError as e:
        #     self.logger.log_message(logging_base.LogFilterError, \
        #                             "%s Start/End Times could not be read." % b)

        # TODO - check that these are the current datetime formats
        for time_format in ("%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M:%S:%f"):
            try:
                time1 = datetime.datetime.strptime(start_time, time_format)
                time2 = datetime.datetime.strptime(end_time, time_format)
            except ValueError as e:
                continue

            # time format was successfully parsed, and thus is valid. Break.
            break
        else:
            # Neither time format matched the product
            print(
                f"{band} Invalid datetime format for zeroDopplerStartTime and/or zeroDopplerEndTime"
            )

        if (time1.year < 2000) or (time1.year > 2100):
            print(
                f"{band} {start_name} must have a year between 2000 and 2100 (inclusive)"
            )
        if (time2.year < 2000) or (time2.year > 2100):
            print(
                f"{band} {end_name} must have a year between 2000 and 2100 (inclusive)"
            )

        if start_time > end_time:
            print(f"{band} {start_name} > {end_name}")

    path_to_metadata = f"{path_to_swaths_grids}/metadata"
    for band in BANDS:
        time = in_file[f"{path_to_metadata}/orbit/time"][...]
        validate_spacing(time, time[1] - time[0], f"{band} orbitTime")

    # Check slant-path spacing

    # While exploring the xml, every time a path ends with "slantRangeSpacing",
    # append that path to this list. There should be one for each Rxxx product.
    paths_to_slantRangeSpacing = []

    for path in paths_to_slantRangeSpacing:
        slant_path = in_file[path][...]

        # slantRangeSpacing and slantRangeSpacing datasets should be located
        # in the same group
        slant_spacing = in_file[
            path.replace("slantRangeSpacing", "slantRange")
        ][...]

        validate_spacing(slant_path, spacing, f"{band} {freq} SlantPath")

    # fin.check_subswaths_bounds()
    for band in BANDS:
        for freq in FREQUENCIES:

            num_subswath = in_file[f"{path_to_freq}/numberOfSubSwaths"][...]
            if (num_subswath < 1) or (num_subswath > 5):
                print(
                    f"{band} Frequency{freq} had invalid number of subswaths: {num_subswath}"
                )

        for subswath in range(1, num_subswath + 1):
            try:
                sub_bounds = self.FREQUENCIES[b][f][
                    f"validSamplesSubSwath{subswath}"
                ][...]
            except KeyError as e:
                self.logger.log_message(
                    logging_base.LogFilterWarning,
                    "%s Frequency%s had missing SubSwath%i bounds"
                    % (b, f, isub + 1),
                )
                continue

            try:
                nslantrange = self.FREQUENCIES[b][f]["slantRange"].size
                assert np.all(sub_bounds[:, 0] < sub_bounds[:, 1])
                assert np.all(sub_bounds[:, 0] >= 0)
                assert np.all(sub_bounds[:, 1] <= nslantrange)
            except KeyError:
                continue
            except AssertionError as e:
                message = (
                    "%s Frequency%s with nSlantRange %i had invalid SubSwath bounds"
                    % (b, f, nslantrange)
                )
                self.logger.log_message(logging_base.LogFilterWarning, message)

    # TODO - should `indentification/boundingPolygon` always have lat/lon/height points,
    # or are only lat/lon points ok?


__all__ = nisarqa.get_all(__name__, objects_to_skip)
