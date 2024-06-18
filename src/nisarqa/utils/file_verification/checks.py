from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import datetime
from typing import Tuple

import numpy as np

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
    log.info(f"\tDATASETS IN BOTH XML AND HDF5: {len(shared_dataset_names)}")
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


@dataclass
class SingleDatasetAspectFlags:
    improper_in_xml: bool = False
    improper_in_hdf5: bool = False
    xml_and_hdf5_differ: bool = False

    @property
    def both_pass_checks(self):
        """True if Dataset aspect is correct (matches) in XML and HDF5."""
        return not (
            self.improper_in_xml
            | self.improper_in_hdf5
            | self.xml_and_hdf5_differ
        )


@dataclass(frozen=True)
class AttributeAspectStats:
    name_of_attr_aspect: str
    total_num_attr_aspects_checked: int = 0
    num_improper_in_xml: int = 0
    num_improper_in_hdf5: int = 0
    num_differ: int = 0
    num_both_pass_checks: int = 0

    def __add__(
        self, other: AttributeAspectStats | SingleDatasetAspectFlags, /
    ) -> AttributeAspectStats:

        if isinstance(other, AttributeAspectStats):
            if self.name_of_attr_aspect != other.name_of_attr_aspect:
                raise ValueError(
                    f"`{self.name_of_attr_aspect=}` but"
                    f" `{other.name_of_attr_aspect=}`; they must be identical."
                )
            other_total_num_attrs = other.total_num_attr_aspects_checked
            other_improper_in_xml = other.num_improper_in_xml
            other_improper_in_hdf5 = other.num_improper_in_hdf5
            other_differ = other.num_differ
            other_both_pass_checks = other.num_both_pass_checks
        elif isinstance(other, SingleDatasetAspectFlags):
            other_total_num_attrs = 1

            # `True` and `False` become `1` and `0` when used for addition
            other_improper_in_xml = other.improper_in_xml
            other_improper_in_hdf5 = other.improper_in_hdf5
            other_differ = other.xml_and_hdf5_differ
            other_both_pass_checks = other.both_pass_checks
        else:
            raise TypeError(
                f"The second addend has type `{type(other)}`,"
                " but only AttributeAspectStats or SingleDatasetAspectFlags"
                " are supported."
            )

        return AttributeAspectStats(
            name_of_attr_aspect=self.name_of_attr_aspect,
            total_num_attr_aspects_checked=self.total_num_attr_aspects_checked
            + other_total_num_attrs,
            num_improper_in_xml=self.num_improper_in_xml
            + other_improper_in_xml,
            num_improper_in_hdf5=self.num_improper_in_hdf5
            + other_improper_in_hdf5,
            num_differ=self.num_differ + other_differ,
            num_both_pass_checks=self.num_both_pass_checks
            + other_both_pass_checks,
        )

    def print_to_log(self) -> None:
        """Record the current status of this instance in the log file."""
        log = nisarqa.get_logger()
        name = f"`{self.name_of_attr_aspect}`"
        log.info(f"Comparing HDF5 Dataset {name} vs. XML spec {name}:")

        total = self.total_num_attr_aspects_checked
        log.info(
            f"\t{name}: TOTAL NUMBER CHECKED IN BOTH XML AND HDF5: {total}"
        )
        impr_xml = self.num_improper_in_xml
        log.info(
            f"\t{name}: NUMBER MISSING OR IMPROPERLY FORMED IN XML:"
            f" {impr_xml} ({100*impr_xml / total:.1f} %)"
        )
        impr_hdf5 = self.num_improper_in_hdf5
        log.info(
            f"\t{name}: NUMBER MISSING OR IMPROPERLY FORMED IN HDF5:"
            f" {impr_hdf5} ({100*impr_hdf5 / total:.1f} %)"
        )
        num_diff = self.num_differ
        log.info(
            f"\t{name}: NUMBER DIFFER IN XML VS. HDF5:"
            f" {num_diff} ({100*num_diff/total:.1f} %)"
        )

        correct = self.num_both_pass_checks
        log.info(
            f"\t{name}: NUMBER PASS ALL CHECKS AND ARE CONSISTENT IN XML AND HDF5:"
            f" {correct} ({100*correct/total:.1f} %)"
        )


@dataclass(frozen=True)
class AttributeStats:
    """Class to hold all overview stats on the attributes."""

    attr_names_stats: AttributeAspectStats = AttributeAspectStats(
        name_of_attr_aspect="Attribute names"
    )

    dtype_stats: AttributeAspectStats = AttributeAspectStats(
        name_of_attr_aspect="dtype"
    )
    description_stats: AttributeAspectStats = AttributeAspectStats(
        name_of_attr_aspect="description"
    )
    units_stats: AttributeAspectStats = AttributeAspectStats(
        name_of_attr_aspect="units (only non-string Datasets)"
    )

    def __add__(self, other: AttributeStats, /) -> AttributeStats:
        return AttributeStats(
            attr_names_stats=self.attr_names_stats + other.attr_names_stats,
            dtype_stats=self.dtype_stats + other.dtype_stats,
            description_stats=self.description_stats + other.description_stats,
            units_stats=self.units_stats + other.units_stats,
        )

    def print_to_log(self) -> None:
        """Record the current status of this instance in the log file."""
        self.attr_names_stats.print_to_log()
        self.dtype_stats.print_to_log()
        self.description_stats.print_to_log()
        self.units_stats.print_to_log()


def compare_xml_dataset_to_hdf5(
    xml_dataset: XMLDataset,
    hdf5_dataset: HDF5Dataset,
) -> AttributeStats:
    """
    Perform checks that compare an XML dataset against an HDF5 dataset.

    Parameters
    ----------
    xml_dataset : nisarqa.XMLDataset
        The XML dataset.
    hdf5_dataset : nisarqa.HDF5Dataset
        The HDF5 dataset.

    Returns
    -------
    stats : nisarqa.AttributeStats
        Metrics for attributes and aspects of `xml_dataset` and `hdf5_dataset`.
    """
    # This section should not be reached. Error if the given datasets have
    # different names.
    if not xml_dataset.name == hdf5_dataset.name:
        raise ValueError(
            f"Dataset names differ: XML:{xml_dataset.name},"
            f" HDF5:{hdf5_dataset.name}"
        )

    dtype_stats = AttributeAspectStats(name_of_attr_aspect="dtype")
    dtype_stats += compare_dtypes_xml_hdf5(xml_dataset, hdf5_dataset)

    name_of_attr_name_aspect = "Attribute names"
    attr_names_stats = AttributeAspectStats(
        name_of_attr_aspect=name_of_attr_name_aspect
    )
    descr_stats = AttributeAspectStats(name_of_attr_aspect="description")
    units_stats = AttributeAspectStats(
        name_of_attr_aspect="units (only non-string Datasets)"
    )

    for annotation in xml_dataset.annotations:
        if annotation.attributes["app"] == "io":
            continue

        attr_names_stats += attribute_names_check(
            xml_annotation=annotation,
            hdf5_annotation=hdf5_dataset.annotation,
            dataset_name=hdf5_dataset.name,
            name_of_attr_aspect=name_of_attr_name_aspect,
        )

        descr_stats += attribute_description_check(
            xml_annotation=annotation,
            hdf5_annotation=hdf5_dataset.annotation,
            dataset_name=hdf5_dataset.name,
        )

        # Strings types should not have units Attribute; but all others should.
        if (xml_dataset.dtype is not str) and (
            not xml_dataset.name.endswith(("epsg", "projection"))
        ):
            units_stats += attribute_units_check(
                xml_annotation=annotation,
                hdf5_annotation=hdf5_dataset.annotation,
                dataset_name=hdf5_dataset.name,
            )

    return AttributeStats(
        attr_names_stats=attr_names_stats,
        dtype_stats=dtype_stats,
        description_stats=descr_stats,
        units_stats=units_stats,
    )


def compare_dtypes_xml_hdf5(
    xml_dataset: XMLDataset, hdf5_dataset: HDF5Dataset
) -> SingleDatasetAspectFlags:
    """
    Compare dtypes of an XML dataset and an HDF5 dataset; log any discrepancies.

    Parameters
    ----------
    xml_dataset : nisarqa.XMLDataset
        The XML dataset to compare.
    hdf5_dataset : nisarqa.HDF5Dataset
        The HDF5 dataset to compare.

    Returns
    -------
    flags : nisarqa.SingleDatasetAspectFlags
        Metrics for dtypes in `xml_annotation` and `hdf5_annotation`.
    """
    log = nisarqa.get_logger()
    xml_dtype = xml_dataset.dtype
    hdf5_dtype = hdf5_dataset.dtype

    flags = SingleDatasetAspectFlags()

    # If either dtype is None (more likely to be for an XML, but check for both)
    # means the dtype could not be determined. Print error and return.
    if xml_dtype is None:
        log.error(
            f"XML dataset dtype could not be determined: {xml_dataset.name}"
        )
        flags.improper_in_xml = True
    if hdf5_dtype is None:
        log.error(
            f"HDF5 dataset dtype could not be determined: {hdf5_dataset.name}"
        )
        flags.improper_in_hdf5 = True

    # Strings are a special case. When the XML requests a string, the HDF5
    # Dataset should be a byte string with a length specified by the XML file,
    # or an arbitrary length if the XML length is "0".
    if xml_dtype == str:
        if not np.issubdtype(hdf5_dtype, np.bytes_):
            # Python bytes object, such as a variable length string. Boo.
            log.error(
                f"`XML expects string. HDF5 has type `{hdf5_dtype}`, but"
                f" should be a NumPy byte string. Dataset {hdf5_dataset.name}"
            )
            flags.improper_in_hdf5 = True
        else:
            # NumPy byte string, or a list of numpy bytes strings. Perfect!
            xml_stated_length = xml_dataset.length

            # All strings should have a length in their properties.
            if xml_stated_length is None:
                log.error(
                    f"String given without length: XML dataset {xml_dataset.name}"
                )
                flags.improper_in_xml = True

            # If the length is 0, there is no expected length. Return successfully.
            elif xml_stated_length == 0:
                pass

            else:
                # The length of the string in the HDF5 dataset is its itemsize.
                # If this is different from the length value, then the actual string
                # length differs from the expected one.
                hdf5_dataset_length = hdf5_dtype.itemsize
                if xml_stated_length != hdf5_dataset_length:
                    log.error(
                        "Unequal string lengths. XML expects length"
                        f" {xml_stated_length}, HDF5 dataset byte string length is"
                        f" {hdf5_dataset_length}: Dataset {xml_dataset.name}"
                    )
                    flags.xml_and_hdf5_differ = True

    else:
        # For non-string types, perform a simple type check.
        # The reason for using `xml_dtype().dtype` below instead of xml_dtype on
        # its own is to ensure that it prints something sensible and brief. The
        # dtype itself prints <class '[dtype name]'> which isn't really accurate.
        # This just prints the dtype name itself.
        if nisarqa.is_complex32(hdf5_dataset.dataset):
            hdf5_dtype_name = "complex32"
        else:
            hdf5_dtype_name = str(hdf5_dtype)

        if xml_dtype == nisarqa.complex32:
            xml_dtype_name = "complex32"
        else:
            xml_dtype_name = str(xml_dtype().dtype)

        if xml_dtype != hdf5_dtype:
            log.error(
                f"dtypes differ: XML: {xml_dtype_name},"
                f" HDF5: {hdf5_dtype_name} - Dataset {xml_dataset.name}"
            )
            flags.xml_and_hdf5_differ = True

    return flags


def attribute_names_check(
    xml_annotation: XMLAnnotation,
    hdf5_annotation: DataAnnotation,
    dataset_name: str,
    name_of_attr_aspect: str,
) -> nisarqa.AttributeAspectStats:
    """
    Compare attribute names between XML and HDF5.

    Parameters
    ----------
    xml_annotation : nisarqa.XMLAnnotation
        An XML annotation to compare.
    hdf5_annotation : nisarqa.DataAnnotation
        An HDF5 annotation to compare.
    dataset_name : str
        The name of the dataset on which both annotations exist.
    name_of_attr_aspect : str
        Name for this attribute aspect, which will be used when constructing
        the returned `stats` object. Example: "Attribute names".

    Returns
    -------
    stats : nisarqa.AttributeAspectStats
        Metrics for attribute names in `xml_annotation` and `hdf5_annotation`.
    """
    log = nisarqa.get_logger()

    # ignore certain attributes present in the XML, such as "lang" and "app"
    ignored_xml_attributes = nisarqa.ignored_xml_annotation_attributes()
    xml_attributes = xml_annotation.attribute_names - ignored_xml_attributes

    ignored_hdf5_attributes = nisarqa.ignored_hdf5_attributes()
    hdf5_attrs = hdf5_annotation.attribute_names - ignored_hdf5_attributes

    total_num = len(hdf5_attrs | xml_attributes)

    # Attributes that exist only in XML but not HDF5.
    xml_unique_attribs = xml_attributes - hdf5_attrs
    num_only_in_xml = len(xml_unique_attribs)
    if num_only_in_xml > 0:
        log.error(
            f"Attributes found in XML but not HDF5: {xml_unique_attribs} -"
            f" Dataset {dataset_name}"
        )

    # Attributes that exist only in HDF5 but not XML.
    hdf5_unique_attribs = hdf5_attrs - xml_attributes
    num_only_in_hdf5 = len(hdf5_unique_attribs)
    if num_only_in_hdf5 > 0:
        log.error(
            f"Attributes found in HDF5 but not XML: {hdf5_unique_attribs} -"
            f" Dataset {dataset_name}"
        )

    num_in_common = len(hdf5_attrs & xml_attributes)
    num_differ = num_only_in_xml + num_only_in_hdf5
    assert total_num == num_in_common + num_differ

    return AttributeAspectStats(
        name_of_attr_aspect=name_of_attr_aspect,
        total_num_attr_aspects_checked=total_num,
        num_improper_in_xml=num_only_in_xml,
        num_improper_in_hdf5=num_only_in_hdf5,
        num_differ=num_differ,
        num_both_pass_checks=num_in_common,
    )


def attribute_units_check(
    xml_annotation: XMLAnnotation,
    hdf5_annotation: DataAnnotation,
    dataset_name: str,
) -> SingleDatasetAspectFlags:
    """
    Check the units listed on two Annotations against each other.

    Parameters
    ----------
    xml_annotation : nisarqa.XMLAnnotation
        An XML annotation to check.
    hdf5_annotation : nisarqa.DataAnnotation
        An HDF5 annotation to check.
    dataset_name : str
        The dataset being checked, for logging.

    Returns
    -------
    stats : nisarqa.SingleDatasetAspectFlags
        Metrics for units attributes in `xml_annotation` and `hdf5_annotation`.
    """

    log = nisarqa.get_logger()

    flags = SingleDatasetAspectFlags()

    if "units" not in xml_annotation.attributes:
        log.error(f"No units detected in XML: Dataset {dataset_name}")
        xml_units = None
        flags.improper_in_xml = True
    else:
        xml_units = str(xml_annotation.attributes["units"])

    if "units" not in hdf5_annotation.attributes:
        log.error(f"No units detected in HDF5: Dataset {dataset_name}")
        hdf5_units = None
        flags.improper_in_hdf5 = True
    else:
        hdf5_units = str(hdf5_annotation.attributes["units"])

    if xml_units == "":
        log.error(f'Empty "units" attribute in XML: Dataset {dataset_name}')
        flags.improper_in_xml = True
    if hdf5_units == "":
        log.error(f'Empty "units" attribute in HDF5: Dataset {dataset_name}')
        flags.improper_in_hdf5 = True

    # Datetime and ISO format strings in `units` fields tend to start with
    # the string "seconds since " - do special datetime/ISO checks for these.
    xml_units_is_iso_str = False
    hdf5_units_is_dt_str = False
    prefix = "seconds since "
    if xml_units is not None and xml_units.startswith(prefix):
        xml_units_is_iso_str = True
        xml_iso_str = xml_units.removeprefix(prefix)
        if not check_iso_format_string(
            iso_format_string=xml_iso_str, dataset_name=dataset_name
        ):
            flags.improper_in_xml = True

    if hdf5_units is not None and hdf5_units.startswith(prefix):
        hdf5_units_is_dt_str = True
        hdf5_datetime_str = hdf5_units.removeprefix(prefix)
        if not check_datetime_string(
            datetime_str=hdf5_datetime_str, dataset_name=dataset_name
        ):
            flags.improper_in_hdf5 = True

    date_time_str_differ = False
    if xml_units_is_iso_str and hdf5_units_is_dt_str:
        # Should either both start with the prefix or both not start with it.
        if xml_units.startswith(prefix) != hdf5_units.startswith(prefix):
            date_time_str_differ = True

    if date_time_str_differ or (xml_units != hdf5_units):
        log.error(
            f"Differing `units` attributes detected for datasets. XML: "
            f'"{xml_units}", HDF5: "{hdf5_units}": Dataset {dataset_name}'
        )
        flags.xml_and_hdf5_differ = True

    return flags


def check_iso_format_string(iso_format_string: str, dataset_name: str) -> bool:
    """
    Compare an ISO format string against the standard ISO format for ISCE3.

    The standard datetime format for ISCE3 XML products specs is set in QA by:
        `nisarqa.NISAR_DATETIME_FORMAT_HUMAN`.
    As of June 2024, this is: "YYYY-mm-ddTHH:MM:SS"

    Parameters
    ----------
    iso_format_string : str
        The ISO format string, e.g. "YYYY-mm-ddTHH:MM:SS".
    dataset_name : str
        Name of dataset associated with `iso_format_string`. (Used for logging.)

    Returns
    -------
    passes : bool
        True if `iso_format_string` is equal to ISCE3 standard, False if not.
    """
    standard_format = nisarqa.NISAR_DATETIME_FORMAT_HUMAN
    if iso_format_string != standard_format:
        nisarqa.get_logger().error(
            f"XML datetime format string '{iso_format_string}' does not match "
            f"ISCE3 standard ISO format '{standard_format}' - "
            f"Dataset {dataset_name}"
        )
        return False
    return True


def check_datetime_string(datetime_str: str, dataset_name: str) -> bool:
    """
    Compare a datetime string against the standard datetime format for ISCE3.

    The standard datetime format for ISCE3 is set by:
        `nisarqa.NISAR_DATETIME_FORMAT_PYTHON`.
    As of June 2024, this is: "%Y-%m-%dT%H:%M:%S"

    Parameters
    ----------
    datetime_str : str
        The datetime string, e.g. "2008-10-12T00:00:00"
    dataset_name : str
        Name of the dataset associated with `datatime_str`. (Used for logging.)

    Returns
    -------
    passes : bool
        True if `datetime_str` conforms to the ISCE3 standard, False if not.
    """
    strptime_format = nisarqa.NISAR_DATETIME_FORMAT_PYTHON

    # Try to read the datetime string with the strptime format.
    # If this fails, then the format and the string don't match.
    try:
        datetime.strptime(datetime_str, strptime_format)
    except Exception:
        nisarqa.get_logger().error(
            f"HDF5 datetime string '{datetime_str}' does not conform to "
            f"ISCE3 standard datetime format '{strptime_format}' - "
            f"Dataset {dataset_name}"
        )
        return False
    return True


def attribute_description_check(
    xml_annotation: XMLAnnotation,
    hdf5_annotation: DataAnnotation,
    dataset_name: str,
) -> SingleDatasetAspectFlags:
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

    Returns
    -------
    stats : nisarqa.SingleDatasetAspectFlags
        Metrics for description attributes in `xml_annotation` and
        `hdf5_annotation`.
    """
    log = nisarqa.get_logger()
    flags = SingleDatasetAspectFlags()

    xml_desc = str(xml_annotation.description)
    hdf5_desc = str(hdf5_annotation.description)

    if xml_desc == "":
        log.error(f"Empty description on XML: Dataset {dataset_name}")
        flags.improper_in_xml = True

    if hdf5_desc == "":
        log.error(
            f"Empty description attribute on HDF5: Dataset {dataset_name}"
        )
        flags.improper_in_hdf5 = True

    if xml_desc != hdf5_desc:
        log.error(
            f'Differing descriptions detected: XML: "{xml_desc}",'
            f' HDF5: "{hdf5_desc}": Dataset {dataset_name}'
        )
        flags.xml_and_hdf5_differ = True

    return flags


__all__ = nisarqa.get_all(__name__, objects_to_skip)
