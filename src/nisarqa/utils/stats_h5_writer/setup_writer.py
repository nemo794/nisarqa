from __future__ import annotations

import h5py

import nisarqa
from nisarqa.utils.typing import RootParamGroupT

objects_to_skip = nisarqa.get_all(name=__name__)


def setup_stats_h5_all_products(
    product: nisarqa.NisarProduct,
    stats_h5: h5py.File,
    root_params: RootParamGroupT,
) -> None:
    """
    Setup the STATS.h5 file for all NISAR products.

    Parameters
    ----------
    product : nisarqa.NisarProduct
        The input product.
    stats_h5 : h5py.File
        Handle to the output HDF5 file.
    root_params : nisarqa.typing.RootParamGroupT
        *RootParamGroup object corresponding to the product type of `product`.
    """
    log = nisarqa.get_logger()
    stats_file = stats_h5.filename

    # Save the processing parameters to the stats.h5 file
    # Note: If only the validate workflow is requested,
    # this will do nothing.
    root_params.save_processing_params_to_stats_h5(
        h5_file=stats_h5, band=product.band
    )
    log.info(f"QA Processing Parameters saved to {stats_file}")

    copy_identification_group_to_stats_h5(product=product, stats_h5=stats_h5)
    log.info(f"Input file Identification group copied to {stats_file}")

    copy_src_runconfig_to_stats_h5(product=product, stats_h5=stats_h5)
    log.info(f"Input file's runconfig copied to {stats_file}")


def setup_stats_h5_non_insar_products(
    product: nisarqa.NonInsarProduct,
    stats_h5: h5py.File,
    root_params: nisarqa.typing.RootParamGroupT,
) -> None:
    """
    Setup the STATS.h5 file for Non-InSAR products (RSLC, GSLC, GCOV).

    Parameters
    ----------
    product : nisarqa.NonInsarProduct
        Instance of a NonInsarProduct (RSLC, GSLC, GCOV).
    stats_h5 : h5py.File
        Handle to an h5 file where the identification metadata
        should be saved.
    root_params : nisarqa.typing.RootParamGroupT
        *RootParamGroup object corresponding to the product type of `product`.
    """
    log = nisarqa.get_logger()
    stats_file = stats_h5.filename

    setup_stats_h5_all_products(
        product=product, stats_h5=stats_h5, root_params=root_params
    )

    copy_rfi_metadata_to_stats_h5(product=product, stats_h5=stats_h5)
    log.info(f"Input file RFI metadata copied to {stats_file}")

    if root_params.workflows.qa_reports:
        # Save frequency/polarization info to stats file
        nisarqa.save_nisar_freq_metadata_to_h5(
            stats_h5=stats_h5, product=product
        )

        # Copy imagery metrics into stats.h5
        nisarqa.copy_non_insar_imagery_metrics(
            product=product, stats_h5=stats_h5
        )
        log.info(f"Input file imagery metrics copied to {stats_file}")


def copy_identification_group_to_stats_h5(
    product: nisarqa.NisarProduct, stats_h5: h5py.File
) -> None:
    """
    Copy the identification group from the input NISAR file
    to the STATS.h5 file.

    Parameters
    ----------
    product : nisarqa.NisarProduct
        Instance of a NisarProduct
    stats_h5 : h5py.File
        Handle to an HDF5 file where the identification metadata
        should be saved.
    """

    src_grp_path = product.identification_path
    dest_grp_path = nisarqa.STATS_H5_IDENTIFICATION_GROUP % product.band

    with h5py.File(product.filepath, "r") as in_file:
        if dest_grp_path in stats_h5:
            # The identification group already exists, so copy each
            # dataset, etc. individually
            for item in in_file[src_grp_path]:
                item_path = f"{dest_grp_path}/{item}"
                in_file.copy(in_file[item_path], stats_h5, item_path)
        else:
            # Copy entire identification metadata from input file to stats.h5
            in_file.copy(in_file[src_grp_path], stats_h5, dest_grp_path)


def copy_src_runconfig_to_stats_h5(
    product: nisarqa.NisarProduct, stats_h5: h5py.File
) -> None:
    """
    Copy input granule's `runConfigurationContents` Dataset to STATS.h5.

    This copies the Dataset's contents as-is, with no further processing.
    It is a known issue that the L1/L2 product types use different formats
    for `runConfigurationContents` (e.g. JSON, YAML). If that format is
    updated within ISCE3, then this function will simply continue to
    copy the Dataset's contents as-is.

    Parameters
    ----------
    product : nisarqa.NisarProduct
        The input product.
    stats_h5 : h5py.File
        Handle to an HDF5 file where the source runconfig should be saved.
    """

    grp_path = nisarqa.STATS_H5_SOURCE_DATA % product.band
    contents = product.runconfig_contents

    nisarqa.create_dataset_in_h5group(
        h5_file=stats_h5,
        grp_path=grp_path,
        ds_name="runConfigurationContents",
        ds_data=contents,
        ds_description=(
            "Contents of the run configuration file associated with the"
            "processing of the source data"
        ),
    )


def copy_rfi_metadata_to_stats_h5(
    product: nisarqa.NonInsarProduct,
    stats_h5: h5py.File,
) -> None:
    """
    Copy the RFI metadata from input product into the STATS HDF5 file.

    Parameters
    ----------
    product : nisarqa.NonInsarProduct
        Instance of a NonInsarProduct (RSLC, GSLC, GCOV).
    stats_h5 : h5py.File
        Handle to an HDF5 file where the RFI metadata should be saved.
    """
    with h5py.File(product.filepath, "r") as in_file:
        for freq in product.freqs:
            # Use `product.get_list_of_polarizations()` instead of the
            # typical `product.get_pols()`. There should make no difference
            # for RSLC and GSLC, but there is a difference for GCOV.
            # For GCOV, `product.get_pols()` actually returns the GCOV terms
            # (e.g. ["HHHH", "HVHVH"]), instead of the polarization pairs
            # (e.g. ["HH", "HV"]). The RFI likelihood values are accessed via
            # polarization pairs for GCOV (not the GCOV terms).
            # In contrast, `get_list_of_polarizations` returns the input RSLC,
            # GSLC, or GCOV granule's `listOfPolarizations` Dataset contents,
            # which are always the polarization pairs.
            for pol in product.get_list_of_polarizations(freq=freq):
                src_path = product.get_rfi_likelihood_path(freq=freq, pol=pol)

                basename = src_path.split("/")[-1]
                dest_path = (
                    f"{nisarqa.STATS_H5_RFI_DATA_GROUP % product.band}/"
                    + f"frequency{freq}/{pol}/{basename}"
                )
                try:
                    in_file.copy(src_path, stats_h5, dest_path)
                except RuntimeError:
                    # h5py.File.copy() raises this error if `src_path`
                    # does not exist:
                    #       RuntimeError: Unable to synchronously copy object
                    #       (component not found)
                    nisarqa.get_logger().error(
                        "Cannot copy `rfiLikelihood`. Input granule is"
                        " missing `rfiLikelihood` for"
                        f" frequency {freq}, polarization {pol} at {src_path}"
                    )


def save_nisar_freq_metadata_to_h5(
    product: nisarqa.NonInsarProduct, stats_h5: h5py.File
) -> None:
    """
    Populate the `stats_h5` HDF5 file with a list of each available
    frequency's polarizations.

    If `pols` contains values for Frequency A, then this dataset will
    be created in `stats_h5`:
        /science/<band>/QA/data/frequencyA/listOfPolarizations

    If `pols` contains values for Frequency B, then this dataset will
    be created in `stats_h5`:
        /science/<band>/QA/data/frequencyB/listOfPolarizations

    * Note: The paths are pulled from the global nisarqa.STATS_H5_QA_FREQ_GROUP.
    If the value of that global changes, then the path for the
    `listOfPolarizations` dataset(s) will change accordingly.

    Parameters
    ----------
    product : nisarqa.NisarProduct
        Input NISAR product
    stats_h5 : h5py.File
        Handle to an HDF5 file where the list(s) of polarizations
        should be saved.
    """
    # Populate data group's metadata
    for freq in product.freqs:
        list_of_pols = product.get_pols(freq=freq)
        grp_path = nisarqa.STATS_H5_QA_FREQ_GROUP % (product.band, freq)
        nisarqa.create_dataset_in_h5group(
            h5_file=stats_h5,
            grp_path=grp_path,
            ds_name="listOfPolarizations",
            ds_data=list_of_pols,
            ds_description=(
                f"Polarizations for Frequency {freq} "
                "discovered in input NISAR product by QA code"
            ),
        )


__all__ = nisarqa.get_all(__name__, objects_to_skip)
