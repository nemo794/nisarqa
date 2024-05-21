from __future__ import annotations

from collections.abc import Iterable, Mapping
from datetime import datetime
from typing import Tuple

import numpy as np
from numpy.typing import DTypeLike

# List of objects from the import statements that
# should not be included when importing this module
import nisarqa
from nisarqa import (
    DataAnnotation,
    DataShape,
    HDF5Dataset,
    XMLAnnotation,
    XMLDataset,
)

objects_to_skip = nisarqa.get_all(name=__name__)


def check_xml_for_unused_shape_elements(
    shape_names: Iterable[str],
    shapes: Iterable[DataShape],
    xml_datasets: Iterable[XMLDataset],
):
    """
    Check the given XML shapes against other shapes and Datasets, and log
    any shapes that are not in use by any Dataset or other shape.

    Parameters
    ----------
    shape_names: Iterable[str]
        A set of shape names to check for.
    shapes : Iterable[DataShape]
        A set of Shape objects to check against.
    xml_datasets : Iterable[XMLDataset]
        A set of XML datasets to check against.
    """
    log = nisarqa.get_logger()

    shape_set = set(shape_names)

    used_shapes: set[str] = set()

    # DataShapes sometimes hold other shape names in their dimensions. Check all
    # dimensions for shape names and add them to the used shapes. It doesn't matter if
    # the dimensions have non-shape names, as these will be discarded when a difference
    # is taken later.
    for shape in shapes:
        for dimension in shape.dimensions:
            if dimension.name in shape_set:
                used_shapes.add(dimension.name)

    # All datasets either have an DataShape or None in their shape field. If there is an
    # DataShape, add its' name to the used shapes set.
    for dataset in xml_datasets:
        if dataset.shape is not None:
            used_shapes.add(dataset.shape.name)

    # The set of unused shapes is the difference between the set of all shapes and
    # the set of used shapes.
    unused_shapes = shape_set - used_shapes

    # Log the number of unused shapes, and then log their names, if any.
    log.info(f"\tUNUSED XML SHAPES: {len(unused_shapes)}")
    if len(unused_shapes) > 0:
        for shape_name in sorted(unused_shapes):
            log.warning(f"\t\tXML UNUSED SHAPE: {shape_name}")


def compare_dataset_lists(
    product_type: str,
    *,
    xml_datasets: Iterable[str],
    hdf5_datasets: Iterable[str],
    valid_freq_pols: Mapping[str, Iterable[str]],
    valid_layers: Iterable[str] | None = None,
    valid_subswaths: Iterable[str] | None = None,
) -> Tuple[set[str], set[str], set[str]]:
    """
    Calculate the sets of datasets in common between XML and HDF5, as well as
    those unique to each. Log the results.

    Parameters
    ----------
    product_type : str
        One of: 'rslc', 'gslc', 'gcov', 'rifg', 'runw', 'gunw', 'roff', 'goff'.
    xml_datasets : Iterable[str]
        A set of dataset names in the XML file.
    hdf5_datasets : Iterable[str]
        A set of dataset names in the HDF5 file.
    valid_freq_pols : Mapping[str, Iterable[str]]
        Dict of the expected frequency + polarization combinations that the
        input NISAR product says it contains.
            Example: { "A" : ["HH", "HV], "B" : ["VV", "VH"] }
    valid_layers : Iterable[str] or None, optional
        ROFF and GOFF products contain HDF5 datasets referred to as "layer number
        datasets" (e.g. `../layer1/..`, `../layer3/..`).
        This parameter should be a set of all valid layer groups for this product,
        e.g., {'layer1', 'layer2', 'layer3'}, or None.
        If None, layer datasets will not be checked. Defaults to None.
    valid_subswaths : Iterable[str] | None, optional
        Some products contain a number of subswaths - if the product has
        subswath information, this should be a set of subswaths expected
        in the product. If not, it should be None. Defaults to None.

    Returns
    -------
    common_dataset_names: set[str]:
        The names of all datasets in common between XML and HDF5.
    unique_xml_dataset_names: set[str]:
        The names of all datasets in the XML that are not in HDF5.
    unique_hdf5_dataset_names: set[str]:
        The names of all datasets in the HDF5 that are not in the XML.
    """
    log = nisarqa.get_logger()

    # Make these into sets so their results will also be sets.
    xml_datasets = set(xml_datasets)
    hdf5_datasets = set(hdf5_datasets)

    all_pols = list(nisarqa.get_possible_pols(product_type=product_type))

    if product_type == "gcov":
        # For GCOV, the Dataset paths can use either the four-character
        # covariance terms or the two-character polarizations (or neither).
        # Append the two-character pols from RSLC to create the full GCOV list.
        all_pols += list(nisarqa.get_possible_pols(product_type="rslc"))

    excepted_paths = nisarqa.rule_excepted_paths(product_type=product_type)

    # The shared datasets are all those in the intersect between the two
    # input sets.
    shared_dataset_names = xml_datasets.intersection(hdf5_datasets)
    exp_shared, unexp_shared = nisarqa.check_paths(
        paths=shared_dataset_names,
        valid_freq_pols=valid_freq_pols,
        valid_layers=valid_layers,
        valid_subswaths=valid_subswaths,
        rule_exceptions=excepted_paths,
        all_pols=all_pols,
    )
    log.info(f"\tNODES IN BOTH XML AND HDF5: {len(shared_dataset_names)}")
    log.info(f"\t\tCORRECTLY INCLUDED IN HDF5: {len(exp_shared)}")
    for name in sorted(exp_shared):
        log.info(f"\t\t\tCORRECTLY INCLUDED DATASET: {name}")
    log.info(f"\t\tUNEXPECTED INCLUSIONS: {len(unexp_shared)}")
    for name in sorted(unexp_shared):
        log.error(f"\t\t\tUNEXPECTED XML DATASET PRESENT IN HDF5: {name}")

    # The XML unique datasets are all those in the difference between the XML
    # datasets and the common set.
    xml_only_dataset_names = xml_datasets.difference(shared_dataset_names)
    exp_xml, unexp_xml = nisarqa.check_paths(
        paths=xml_only_dataset_names,
        valid_freq_pols=valid_freq_pols,
        valid_layers=valid_layers,
        valid_subswaths=valid_subswaths,
        rule_exceptions=excepted_paths,
        all_pols=all_pols,
    )
    log.info(f"\tXML UNIQUE NODES: {len(xml_only_dataset_names)}")
    log.info(f"\t\tMISSING FROM HDF5: {len(exp_xml)}")
    for name in sorted(exp_xml):
        log.error(f"\t\t\tDATASET MISSING FROM HDF5: {name}")
    log.info(f"\t\tCORRECTLY OMITTED FROM HDF5: {len(unexp_xml)}")

    # The HDF5 unique datasets are all those in the difference between the HDF5
    # datasets and the common set.
    hdf5_only_dataset_names = hdf5_datasets.difference(shared_dataset_names)
    log.info(f"\tHDF5 UNIQUE NODES: {len(hdf5_only_dataset_names)}")
    for name in sorted(hdf5_only_dataset_names):
        log.error(f"\t\tHDF5 DATASET NOT IN XML: {name}")

    return (
        shared_dataset_names,
        xml_only_dataset_names,
        hdf5_only_dataset_names,
    )


def compare_xml_dataset_to_hdf5(
    xml_dataset: XMLDataset,
    hdf5_dataset: HDF5Dataset,
):
    """
    Perform checks that compare an XML dataset against an HDF5 dataset.

    Parameters
    ----------
    xml_dataset : nisarqa.XMLDataset
        The XML dataset.
    hdf5_dataset : nisarqa.HDF5Dataset
        The HDF5 dataset.
    """
    # This section should not be reached. Error if the given datasets have
    # different names.
    if not xml_dataset.name == hdf5_dataset.name:
        raise ValueError(
            f"Dataset names differ: XML:{xml_dataset.name},"
            f" HDF5:{hdf5_dataset.name}"
        )

    compare_dtypes_xml_hdf5(xml_dataset, hdf5_dataset)

    for annotation in xml_dataset.annotations:
        if annotation.attributes["app"] == "io":
            continue
        common_attribute_check(
            xml_annotation=annotation,
            hdf5_annotation=hdf5_dataset.annotation,
            dataset_name=hdf5_dataset.name,
        )

        attribute_description_check(
            xml_annotation=annotation,
            hdf5_annotation=hdf5_dataset.annotation,
            dataset_name=hdf5_dataset.name,
        )

        # Strings types don't have units, so check only for others.
        if xml_dataset.dtype is not str:
            attribute_unit_check(
                xml_annotation=annotation,
                hdf5_annotation=hdf5_dataset.annotation,
                dataset_name=hdf5_dataset.name,
            )


def compare_dtypes_xml_hdf5(xml_dataset: XMLDataset, hdf5_dataset: HDF5Dataset):
    """
    Compare the dtypes of an XML dataset and HDF5 dataset, and log any
    discrepancies.

    Parameters
    ----------
    xml_dataset : nisarqa.XMLDataset
        The XML dataset to compare.
    hdf5_dataset : nisarqa.HDF5Dataset
        The HDF5 dataset to compare.
    """
    log = nisarqa.get_logger()
    xml_dtype = xml_dataset.dtype
    hdf5_dtype = hdf5_dataset.dtype

    # If either type is None (far more likely to be an XML dtype, but check for both,)
    # this constitutes a problem because the dtype could not be determined.
    # Print this error and return.
    dtype_check = True
    if xml_dtype is None:
        log.error(
            f"XML dataset dtype could not be determined: {xml_dataset.name}"
        )
        dtype_check = False
    if hdf5_dtype is None:
        log.error(
            f"HDF5 dataset dtype could not be determined: {hdf5_dataset.name}"
        )
        dtype_check = False

    if not dtype_check:
        log.warning(
            f"Skipping XML-HDF5 type checking for dataset: {xml_dataset.name}"
        )
        return

    # Strings are a special case. When the XML requests a string, the HDF5 dataset
    # should be a byte string with a length given by the XML file, or an arbitrary
    # length if the XML length is "0".
    if xml_dtype == str:
        if not np.issubdtype(hdf5_dtype, np.bytes_):
            log.error(
                f"XML expects string. HDF5 type is not numpy byte string,"
                f" but instead {hdf5_dtype}: {hdf5_dataset.name}"
            )
            return

        xml_stated_length = xml_dataset.length

        # All strings should have a length in their properties.
        if xml_stated_length is None:
            log.error(
                f"String given without length: XML dataset {xml_dataset.name}"
            )
            return

        # If the length is 0, there is no expected length. Return successfully.
        if xml_stated_length == 0:
            return

        # The length of the string in the HDF5 dataset is its' itemsize.
        # If this is different from the length value, then the actual string
        # length differs from the expected one.
        hdf5_dataset_length = hdf5_dtype.itemsize
        if xml_stated_length != hdf5_dataset_length:
            log.error(
                "Unequal string lengths. XML expects length"
                f" {xml_stated_length}, HDF5 dataset byte string length is"
                f" {hdf5_dataset_length}: Dataset {xml_dataset.name}"
            )
        return

    if nisarqa.is_complex32(hdf5_dataset.dataset):
        hdf5_dtype_name = "complex32"
    else:
        hdf5_dtype_name = str(hdf5_dtype)
    
    if xml_dtype == np.dtype([("r", np.float16), ("i", np.float16)]):
        xml_dtype_name = "complex32"
    else:
        xml_dtype_name = str(xml_dtype().dtype)

    # For non-string types, perform a simple type check.
    # The reason for using `xml_dtype().dtype` below instead of xml_dtype on
    # its own is to ensure that it prints something sensible and brief. The
    # dtype itself prints <class '[dtype name]'> which isn't really accurate.
    # This just prints the dtype name itself.
    if xml_dtype != hdf5_dtype:
        log.error(
            f"dtypes differ: XML: {xml_dtype_name},"
            f" HDF5: {hdf5_dtype_name} - Dataset {xml_dataset.name}"
        )


def common_attribute_check(
    xml_annotation: XMLAnnotation,
    hdf5_annotation: DataAnnotation,
    dataset_name: str,
) -> None:
    """
    Check for common attributes between annotations, report the differences.

    Parameters
    ----------
    xml_annotation : nisarqa.XMLAnnotation
        An XML annotation to compare.
    hdf5_annotation : nisarqa.DataAnnotation
        An HDF5 annotation to compare.
    dataset_name : str
        The name of the dataset on which both annotations exist.
    """
    log = nisarqa.get_logger()
    ignored_xml_attributes = nisarqa.ignored_xml_annotation_attributes()
    xml_attributes = xml_annotation.attribute_names - ignored_xml_attributes
    common_attribs = xml_attributes & hdf5_annotation.attribute_names

    # Keys that exist only on this annotation or the other.
    xml_unique_attribs = xml_attributes - common_attribs
    if len(xml_unique_attribs) > 0:
        log.error(
            f"Keys found only on XML annotation: {xml_unique_attribs} -"
            f" Dataset {dataset_name}"
        )
    hdf5_unique_attribs = hdf5_annotation.attribute_names - common_attribs
    if len(hdf5_unique_attribs) > 0:
        log.error(
            f"Keys found only on HDF5 annotation: {hdf5_unique_attribs} -"
            f" Dataset {dataset_name}"
        )


def attribute_unit_check(
    xml_annotation: XMLAnnotation,
    hdf5_annotation: DataAnnotation,
    dataset_name: str,
):
    """
    Check the units listed on two attributes against each other.

    Parameters
    ----------
    xml_annotation : nisarqa.XMLAnnotation
        An XML annotation to check.
    hdf5_annotation : nisarqa.DataAnnotation
        An HDF5 annotation to check.
    dataset_name : str
        The dataset being checked, for logging.
    """
    log = nisarqa.get_logger()
    proceed: bool = True
    if "units" not in xml_annotation.attributes:
        log.error(f"No units detected on XML: Dataset {dataset_name}")
        proceed = False
    if "units" not in hdf5_annotation.attributes:
        log.error(f"No units detected on HDF5: Dataset {dataset_name}")
        proceed = False

    if not proceed:
        return

    xml_units = str(xml_annotation.attributes["units"])
    hdf5_units = str(hdf5_annotation.attributes["units"])

    if xml_units == "":
        log.error(f'Empty "units" attribute on XML: Dataset {dataset_name}')
        proceed = False
    if hdf5_units == "":
        log.error(
            f'Empty "units" attribute field on HDF5: Dataset {dataset_name}'
        )
        proceed = False

    if not proceed:
        return

    # Datetime and ISO format objects in annotation units fields tend to start with
    # the string "seconds since " - do some special datetime/ISO checks for these.
    if xml_units.startswith("seconds since ") and hdf5_units.startswith(
        "seconds since "
    ):
        xml_iso_str = xml_units.removeprefix("seconds since ")
        hdf5_datetime_str = hdf5_units.removeprefix("seconds since ")

        compare_datetime_strings(
            dataset_name=dataset_name,
            xml_iso_string=xml_iso_str,
            hdf5_datetime=hdf5_datetime_str,
        )
        return

    if xml_units != hdf5_units:
        log.error(
            f'Differing units detected on annotations. XML: "{xml_units}",'
            f' HDF5: "{hdf5_units}": Dataset {dataset_name}'
        )


def compare_datetime_strings(
    dataset_name: str, xml_iso_string: str, hdf5_datetime: str
):
    """
    Compare two datetime- or ISO-formatted strings to each other.

    Parameters
    ----------
    dataset_name : str
        The name of the dataset being compared.
    xml_datetime : str
        An ISO-formatted string from XML to compare.
    hdf5_datetime : str
        A datetime-formatted string from HDF5 to compare.
    """
    log = nisarqa.get_logger()

    standard_format = nisarqa.NISAR_DATETIME_FORMAT_HUMAN
    if xml_iso_string != standard_format:
        log.error(
            f"XML datetime format string {xml_iso_string} does not match "
            f"ISCE3 standard datetime format {standard_format} - "
            f"Dataset {dataset_name}"
        )

    if not check_product_datetime(hdf5_datetime):
        log.error(
            f"HDF5 datetime string {hdf5_datetime} does not conform to "
            f"ISO-8601 standard datetime format - "
            f"Dataset {dataset_name}"
        )


def check_product_datetime(datetime_str) -> bool:
    """
    Compare a datetime string against the standard datetime format for ISCE3.

    Parameters
    ----------
    datetime_str : str
        The datetime string.

    Returns
    -------
    bool
        True if valid, False if not.
    """
    strptime_format = nisarqa.NISAR_DATETIME_FORMAT_PYTHON

    # Try to read the datetime string with the strptime format.
    # If this fails, then the format and the string don't match.
    try:
        datetime.strptime(datetime_str, strptime_format)
    except Exception:
        return False
    return True


def attribute_description_check(
    xml_annotation: XMLAnnotation,
    hdf5_annotation: DataAnnotation,
    dataset_name: str,
):
    """
    Check the descriptions listed on XML and HDF5 attributes against each other.

    Parameters
    ----------
    xml_annotation : nisarqa.XMLAnnotation
        An XML annotation to check.
    hdf5_annotation : nisarqa.DataAnnotation
        An HDF5 annotation to check.
    dataset_name : str
        The dataset being checked, for logging.
    """
    log = nisarqa.get_logger()

    proceed: bool = True
    xml_desc = str(xml_annotation.description)
    hdf5_desc = str(hdf5_annotation.description)

    if xml_desc == "":
        log.error(f"Empty description on XML: Dataset {dataset_name}")
        proceed = False
    if hdf5_desc == "":
        log.error(
            f"Empty description attribute on HDF5: Dataset {dataset_name}"
        )
        proceed = False

    if not proceed:
        return

    if xml_desc != hdf5_desc:
        log.error(
            f'Differing descriptions detected: XML: "{xml_desc}",'
            f' HDF5: "{hdf5_desc}": Dataset {dataset_name}'
        )


__all__ = nisarqa.get_all(__name__, objects_to_skip)
