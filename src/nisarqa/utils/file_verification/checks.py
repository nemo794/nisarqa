from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Tuple

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
    log.info(
        "\tTOTAL NUMBER OF DATASETS IN INTERSECTION OF XML AND HDF5:"
        f" {len(shared_dataset_names)}"
    )
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
class SingleItemMatchesXMLFlags:
    """
    Flags for if a single metadata aspect of a single HDF5 Dataset matches XML.

    Example: The `zeroDopplerTime` Dataset has a "units" Attribute. Create
    an instance of this class to capture if the HDF5 Dataset's "units" is:
        1) found in the HDF5 but missing from the XML,
        2) Improperly formed in the HDF5 (e.g. incorrect datetime format), or
        3) is inconsistent with what the XML says it should be

    Examples of Dataset metadata: an Attribute, the dtype,
    the description, the 'units' Attribute.

    Parameters
    ----------
    missing_in_xml : bool
        True if metadata aspect is found in the HDF5 but missing from the XML.
        False otherwise.
    missing_in_hdf5 : bool
        True if metadata aspect is found in the XML but missing from the HDF5.
        False otherwise.
    hdf5_inconsistent_with_xml : bool
        True if the metadata aspect is found in both the HDF5 and the XML,
        but is inconsistent. False otherwise.

    See Also
    --------
    StatsForAMetadataAspect :
        Class to accumulate instances of SingleItemMatchesXMLFlags.
    """

    missing_in_xml: bool = False
    missing_in_hdf5: bool = False
    hdf5_inconsistent_with_xml: bool = False

    @property
    def hdf5_and_xml_match(self):
        """True if HDF5 Dataset's metadata aspect matches exactly the XML."""
        return not (
            self.missing_in_xml
            | self.missing_in_hdf5
            | self.hdf5_inconsistent_with_xml
        )


@dataclass(frozen=True)
class StatsForAMetadataAspect:
    """
    Number of HDF5 Datasets consistent with XML spec re: a single metadata aspect.

    Examples of a Dataset's metadata aspect: an Attribute, the dtype,
    the description, the 'units' Attribute.

    Parameters
    ----------
    name_of_metadata_aspect : str
        Name of the metadata aspect whose stats are being tallied here.
        Ex: "Attribute name", "units", "dtype", "description"
    total_num_aspects_checked : int
        Total number of instances of `name_of_metadata_aspect` found in
        either HDF5 or XML.
    num_missing_in_xml : int
        Number of instances of `name_of_metadata_aspect` found in HDF5
        but missing from the XML.
    num_missing_from_hdf5 : int
        Number of instances of `name_of_metadata_aspect` found in XML
        but missing from the HDF5.
    num_inconsistent_hdf5_vs_xml : int
        Number of instances of `name_of_metadata_aspect` found in both XML
        and HDF5 but are inconsistent. This includes issues like datetime
        descriptions that include microseconds, or variable length strings
        when fixed-length were expected.
    num_hdf5_and_xml_match : int
        Total number of instances of `name_of_metadata_aspect` that
        are consistent between the HDF5 and the XML.
    """

    name_of_metadata_aspect: str
    total_num_aspects_checked: int = 0
    num_missing_in_xml: int = 0
    num_missing_from_hdf5: int = 0
    num_inconsistent_hdf5_vs_xml: int = 0
    num_hdf5_and_xml_match: int = 0

    def __add__(
        self,
        other: StatsForAMetadataAspect | SingleItemMatchesXMLFlags,
        /,
    ) -> StatsForAMetadataAspect:

        if isinstance(other, StatsForAMetadataAspect):
            if self.name_of_metadata_aspect != other.name_of_metadata_aspect:
                raise ValueError(
                    f"`{self.name_of_metadata_aspect=}` but"
                    f" `{other.name_of_metadata_aspect=}`; they must be identical."
                )
            other_total_num_attrs = other.total_num_aspects_checked
            other_missing_in_xml = other.num_missing_in_xml
            other_missing_in_hdf5 = other.num_missing_from_hdf5
            other_differ = other.num_inconsistent_hdf5_vs_xml
            other_both_pass_checks = other.num_hdf5_and_xml_match
        elif isinstance(other, SingleItemMatchesXMLFlags):
            other_total_num_attrs = 1

            # `True` and `False` become `1` and `0` when used for addition
            other_missing_in_xml = other.missing_in_xml
            other_missing_in_hdf5 = other.missing_in_hdf5
            other_differ = other.hdf5_inconsistent_with_xml
            other_both_pass_checks = other.hdf5_and_xml_match
        else:
            raise TypeError(
                f"The second addend has type `{type(other)}`,"
                " but only StatsForAMetadataAspect or SingleItemMatchesXMLFlags"
                " are supported."
            )

        return StatsForAMetadataAspect(
            name_of_metadata_aspect=self.name_of_metadata_aspect,
            total_num_aspects_checked=self.total_num_aspects_checked
            + other_total_num_attrs,
            num_missing_in_xml=self.num_missing_in_xml + other_missing_in_xml,
            num_missing_from_hdf5=self.num_missing_from_hdf5
            + other_missing_in_hdf5,
            num_inconsistent_hdf5_vs_xml=self.num_inconsistent_hdf5_vs_xml
            + other_differ,
            num_hdf5_and_xml_match=self.num_hdf5_and_xml_match
            + other_both_pass_checks,
        )

    def print_to_log(self) -> None:
        """Record the current status of this instance in the log file."""
        log = nisarqa.get_logger()
        name = self.name_of_metadata_aspect
        fmt_frac = lambda x: f"{x} ({100 * x / total:.1f} %)"

        log.info(f"Comparing HDF5 Dataset {name} vs. XML spec {name}:")

        total = self.total_num_aspects_checked
        log.info(
            f"\t{name}: Total number of instances in union of HDF5 and XML:"
            f" {total}"
        )
        miss_xml = self.num_missing_in_xml
        log.info(
            f"\t{name}: Number of instances in HDF5 but missing from XML:"
            f" {fmt_frac(miss_xml)}"
        )
        miss_hdf5 = self.num_missing_from_hdf5
        log.info(
            f"\t{name}: Number of instances in XML but missing from HDF5:"
            f" {fmt_frac(miss_hdf5)}"
        )
        num_diff = self.num_inconsistent_hdf5_vs_xml
        log.info(
            f"\t{name}: Number of instances in HDF5 that are inconsistent"
            f" with XML spec: {fmt_frac(num_diff)}"
        )

        correct = self.num_hdf5_and_xml_match
        log.info(
            f"\t{name}: Number of instances that are consistent between"
            f" the HDF5 and XML: {fmt_frac(correct)}"
        )


@dataclass(frozen=True)
class AttributeStats:
    """
    Class to hold stats on all metadata aspects of all Datasets.

    Examples of a Dataset's metadata aspect: an Attribute, the dtype,
    the description, the 'units' Attribute.

    Parameters
    ----------
    attr_names_stats : StatsForAMetadataAspect
        Stats on the "Attribute name" metadata aspect.
    dtype_stats : StatsForAMetadataAspect
        Stats on the "dtype" metadata aspect.
    description_stats : StatsForAMetadataAspect
        Stats on the "description" metadata aspect.
    units_stats : StatsForAMetadataAspect
        Stats on the "units" metadata aspect.
    """

    attr_names_stats: StatsForAMetadataAspect = StatsForAMetadataAspect(
        name_of_metadata_aspect="Attribute names"
    )
    dtype_stats: StatsForAMetadataAspect = StatsForAMetadataAspect(
        name_of_metadata_aspect="dtype"
    )
    description_stats: StatsForAMetadataAspect = StatsForAMetadataAspect(
        name_of_metadata_aspect="description"
    )
    units_stats: StatsForAMetadataAspect = StatsForAMetadataAspect(
        name_of_metadata_aspect="units"
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


def compare_hdf5_dataset_to_xml(
    xml_dataset: XMLDataset,
    hdf5_dataset: HDF5Dataset,
) -> AttributeStats:
    """
    Perform checks that compare an HDF5 dataset against the XML spec.

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

    dtype_stats = StatsForAMetadataAspect(name_of_metadata_aspect="dtype")
    dtype_stats += compare_dtypes_xml_hdf5(xml_dataset, hdf5_dataset)

    name_of_attr_name_aspect = "Attribute names"
    attr_names_stats = StatsForAMetadataAspect(
        name_of_metadata_aspect=name_of_attr_name_aspect
    )
    descr_stats = StatsForAMetadataAspect(name_of_metadata_aspect="description")
    units_stats = StatsForAMetadataAspect(name_of_metadata_aspect="units")

    for annotation in xml_dataset.annotations:
        if annotation.attributes["app"] == "io":
            continue

        attr_names_stats += attribute_names_check(
            xml_annotation=annotation,
            hdf5_annotation=hdf5_dataset.annotation,
            dataset_name=hdf5_dataset.name,
            name_of_metadata_aspect=name_of_attr_name_aspect,
        )

        descr_stats += attribute_description_check(
            xml_annotation=annotation,
            hdf5_annotation=hdf5_dataset.annotation,
            dataset_name=hdf5_dataset.name,
        )

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
) -> SingleItemMatchesXMLFlags:
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
    flags : nisarqa.SingleItemMatchesXMLFlags
        Metrics for dtypes in `xml_annotation` and `hdf5_annotation`.
    """
    log = nisarqa.get_logger()
    xml_dtype = xml_dataset.dtype
    hdf5_dtype = hdf5_dataset.dtype

    flags = SingleItemMatchesXMLFlags()

    # If either dtype is None (more likely to be for an XML, but check for both)
    # means the dtype could not be determined. Print error and return.
    if xml_dtype is None:
        log.error(
            f"XML dataset dtype could not be determined: {xml_dataset.name}"
        )
        flags.missing_in_xml = True
    if hdf5_dtype is None:
        log.error(
            f"HDF5 dataset dtype could not be determined: {hdf5_dataset.name}"
        )
        flags.missing_in_hdf5 = True

    # Strings are a special case. When the XML requests a string, the HDF5
    # Dataset should be a byte string with a length specified by the XML file,
    # or an arbitrary length if the XML length is "0".
    if xml_dtype == str:
        if not np.issubdtype(hdf5_dtype, np.bytes_):
            log.error(
                f"XML expects string. HDF5 has type `{hdf5_dtype}`, but"
                f" should be a NumPy byte string. Dataset {hdf5_dataset.name}"
            )
            flags.missing_in_hdf5 = True
        else:
            # NumPy byte string, or a list of numpy bytes strings. Perfect!
            xml_stated_length = xml_dataset.length

            # All strings should have a length in their properties.
            if xml_stated_length is None:
                log.error(
                    f"String given without length: XML dataset {xml_dataset.name}"
                )
                flags.missing_in_xml = True

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
                    flags.hdf5_inconsistent_with_xml = True

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
            flags.hdf5_inconsistent_with_xml = True

    return flags


def attribute_names_check(
    xml_annotation: XMLAnnotation,
    hdf5_annotation: DataAnnotation,
    dataset_name: str,
    name_of_metadata_aspect: str,
) -> nisarqa.StatsForAMetadataAspect:
    """
    Compare attribute names for a single Dataset between XML and HDF5.

    Parameters
    ----------
    xml_annotation : nisarqa.XMLAnnotation
        An XML annotation to compare.
    hdf5_annotation : nisarqa.DataAnnotation
        An HDF5 annotation to compare.
    dataset_name : str
        The name of the dataset on which both annotations exist.
    name_of_metadata_aspect : str
        Name for this attribute aspect, which will be used when constructing
        the returned `stats` object. Example: "Attribute names".

    Returns
    -------
    stats : nisarqa.StatsForAMetadataAspect
        Metrics for attribute names in `xml_annotation` and `hdf5_annotation`.
    """
    log = nisarqa.get_logger()

    # Ignore certain attributes present in the XML, such as "lang" and "app"
    ignored_xml_attributes = nisarqa.ignored_xml_annotation_attributes()
    xml_attributes = xml_annotation.attribute_names - ignored_xml_attributes

    hdf5_attrs = hdf5_annotation.attribute_names

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

    # Here, `num_inconsistent_hdf5_vs_xml` should be zero because attribute
    # names can only be identical in the HDF5 & XML or missing from one
    # or the other.
    num_inconsistent_hdf5_vs_xml = 0

    return StatsForAMetadataAspect(
        name_of_metadata_aspect=name_of_metadata_aspect,
        total_num_aspects_checked=total_num,
        num_missing_in_xml=num_only_in_xml,
        num_missing_from_hdf5=num_only_in_hdf5,
        num_inconsistent_hdf5_vs_xml=num_inconsistent_hdf5_vs_xml,
        num_hdf5_and_xml_match=num_in_common,
    )


def attribute_units_check(
    xml_annotation: XMLAnnotation,
    hdf5_annotation: DataAnnotation,
    dataset_name: str,
) -> SingleItemMatchesXMLFlags:
    """
    Check the units Attribute of an HDF5 dataset against the XML spec.

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
    flags : nisarqa.SingleItemMatchesXMLFlags
        Metrics for units attributes in `xml_annotation` and `hdf5_annotation`.
    """

    log = nisarqa.get_logger()

    flags = SingleItemMatchesXMLFlags()

    # Datetime strings in `units` fields are expected to start with
    # the string "seconds since " - do special checks for these strings
    # because the value of the datetime strings they contain will never match.
    xml_units_has_dt_template_str = False
    hdf5_units_has_dt_str = False
    prefix = "seconds since "

    # Get value of 'units'; perform basic checks
    if "units" in xml_annotation.attributes:
        # `xml_parser.py > element_to_annotation()` already validated the XML.
        # Here, simply use those validated results.
        xml_units = str(xml_annotation.attributes["units"])

        if xml_units.startswith(prefix):
            xml_units_has_dt_template_str = True
    else:
        xml_units = None

    if "units" in hdf5_annotation.attributes:
        hdf5_units = str(hdf5_annotation.attributes["units"])

        if hdf5_units == "":
            log.error(
                f'Empty "units" attribute in HDF5: Dataset {dataset_name}'
            )

        # datetime check for HDF5
        if hdf5_units.startswith(prefix):
            hdf5_units_has_dt_str = True
            hdf5_datetime_str = hdf5_units.removeprefix(prefix)
            if not check_datetime_string(
                datetime_str=hdf5_datetime_str, dataset_name=dataset_name
            ):
                # The XML *should* provide the correct datetime template string,
                # which should follow the NISAR format. So, if the HDF5 does
                # not also follow that NISAR format, flag that here.
                # Warning: Setting this flag introduces a risk of
                # double-counting a problematic "units" attribute.
                flags.hdf5_inconsistent_with_xml = True

    else:
        hdf5_units = None

    log_err = lambda: log.error(
        f"Differing `units` attributes detected for datasets. XML: "
        f'"{xml_units}", HDF5: "{hdf5_units}": Dataset {dataset_name}'
    )

    if xml_units_has_dt_template_str != hdf5_units_has_dt_str:
        # "units" exists in both XML and HDF5, but they do not consistently
        # begin with the prefix "seconds since ". Handle this edge case first.
        log.error(
            f"Differing format of `units` attributes detected for datasets;"
            f" both should start with '{prefix}'. XML: "
            f'"{xml_units}", HDF5: "{hdf5_units}": Dataset {dataset_name}'
        )
        flags.hdf5_inconsistent_with_xml = True
    elif (xml_units is None) and (hdf5_units is not None):
        flags.missing_in_xml = True
        log_err()

    elif (xml_units is not None) and (hdf5_units is None):
        flags.missing_in_hdf5 = True
        log_err()
    elif xml_units != hdf5_units:
        # "units" exists in both XML and HDF5, but they are inconsistent.
        # (The datetime edge case was handled above.)
        flags.hdf5_inconsistent_with_xml = True
        log_err()

    return flags


def check_datetime_template_string(
    datetime_template_string: str, dataset_name: str
) -> bool:
    """
    Compare a string against the datetime template convention for NISAR.

    Logs if there is a discrepancy.

    The standard datetime template for NISAR XML products specs is set in QA by:
        `nisarqa.NISAR_DATETIME_FORMAT_HUMAN`.
    As of June 2024, this is: "YYYY-mm-ddTHH:MM:SS"

    Parameters
    ----------
    datetime_template_string : str
        The datetime template string to be checked.
        Example: "YYYY-mm-ddTHH:MM:SS".
    dataset_name : str
        Name of dataset associated with `datetime_template_string`.
        (Used for logging.)

    Returns
    -------
    passes : bool
         True if `datetime_template_string` matches the NISAR convention,
         False if not.
    """
    standard_format = nisarqa.NISAR_DATETIME_FORMAT_HUMAN
    if datetime_template_string != standard_format:
        nisarqa.get_logger().error(
            f"XML datetime template string '{datetime_template_string}' does"
            f" not match NISAR datetime template convention '{standard_format}'"
            f" - Dataset {dataset_name}"
        )
        return False
    return True


def check_datetime_string(datetime_str: str, dataset_name: str) -> bool:
    """
    Compare a datetime string against the standard datetime format for NISAR.

    The standard datetime format for NISAR is set by:
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
        True if `datetime_str` conforms to the NISAR convention, False if not.
    """
    strptime_format = nisarqa.NISAR_DATETIME_FORMAT_PYTHON

    # Try to read the datetime string with the strptime format.
    # If this fails, then the format and the string don't match.
    try:
        datetime.strptime(datetime_str, strptime_format)
    except Exception:
        nisarqa.get_logger().error(
            f"HDF5 datetime string '{datetime_str}' does not conform to"
            f" NISAR datetime format convention '{strptime_format}' -"
            f" Dataset {dataset_name}"
        )
        return False
    return True


def attribute_description_check(
    xml_annotation: XMLAnnotation,
    hdf5_annotation: DataAnnotation,
    dataset_name: str,
) -> SingleItemMatchesXMLFlags:
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
    flags : nisarqa.SingleItemMatchesXMLFlags
        Metrics for description attributes in `xml_annotation` and
        `hdf5_annotation`.
    """
    log = nisarqa.get_logger()
    flags = SingleItemMatchesXMLFlags()

    xml_desc = str(xml_annotation.description)
    hdf5_desc = str(hdf5_annotation.description)

    if xml_desc == "":
        log.error(f"Empty description on XML: Dataset {dataset_name}")
        flags.missing_in_xml = True

    if hdf5_desc == "":
        log.error(
            f"Empty description attribute on HDF5: Dataset {dataset_name}"
        )
        flags.missing_in_hdf5 = True

    if xml_desc != hdf5_desc:
        log.error(
            f'Differing descriptions detected: XML: "{xml_desc}",'
            f' HDF5: "{hdf5_desc}": Dataset {dataset_name}'
        )
        flags.hdf5_inconsistent_with_xml = True

    return flags


__all__ = nisarqa.get_all(__name__, objects_to_skip)
