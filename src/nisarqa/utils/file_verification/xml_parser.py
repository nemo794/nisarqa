from __future__ import annotations

import os
from collections.abc import Iterable
from xml.etree import ElementTree as ET
from xml.etree.ElementTree import Element

# List of objects from the import statements that
# should not be included when importing this module
import nisarqa
from nisarqa import DataShape, Dimension, XMLAnnotation, XMLDataset

objects_to_skip = nisarqa.get_all(name=__name__)


def get_xml_datasets_and_shapes(
    xml_file: os.PathLike | str,
) -> tuple[list[Element], list[Element]]:
    """
    Get all of the dataset node and shape elements from a valid NISAR product
    spec file.

    Parameters
    ----------
    xml_file : path-like
        The path to the XML file.

    Returns
    -------
    datasets: list[Element]
        All of the datasets node elements held within the XML file.
    shapes: list[Element]
        All of the shape description elements held within the XML file.
    """
    xml_tree = ET.parse(xml_file)

    # This program assumes that the given XML file will contain a root group
    # which contains a hierarchy in which the root element contains "product",
    # "science", and "nodes" elements, one of each nested in that order.
    # The "nodes" element is expected to contain only a list of dataset
    # description elements (herein called "datasets") and the "product"
    # element should contain a list of "shape" elements.
    algorithm_element = xml_tree.getroot()
    product_element = algorithm_element.find("product")
    science_element = product_element.find("science")
    nodes_element = science_element.find("nodes")

    # In the XML, HDF5 dataset descriptions are called "nodes". For the rest of
    # this program, these will be called "datasets" for consistency.
    datasets = nodes_element[:]
    shapes = product_element.findall("shape")

    return datasets, shapes


def elements_to_shapes(
    xml_elements: Iterable[Element],
) -> dict[str, DataShape]:
    """
    Parse a set of "shape" XML elements into DataShape objects.

    Parameters
    ----------
    xml_elements : Iterable[Element]
        The XML elements to parse.

    Returns
    -------
    dict[str, nisarqa.DataShape]
        The DataShape objects generated, addressable by name.
    """
    shapes: dict[str, DataShape] = {}
    for element in xml_elements:
        shape: nisarqa.DataShape = element_to_shape(element)
        shapes[shape.name] = shape
    return shapes


def element_to_shape(xml_element: Element) -> nisarqa.DataShape:
    """
    Process a "shape" XML element into an DataShape object.

    Parameters
    ----------
    xml_element : Element
        The element to be processed.

    Returns
    -------
    nisarqa.DataShape
        The generated DataShape object.
    """
    attributes = xml_element.attrib
    name = attributes["name"]
    order = attributes["order"]

    annotation_elements = xml_element.findall("annotation")
    annotations = parse_annotations(
        annotation_elements=annotation_elements,
        parent_name=name,
    )

    dimension_elements = xml_element.findall("dimension")
    dimensions = parse_dims(dimension_elements)

    return nisarqa.DataShape(
        name=name,
        order=order,
        annotations=annotations,
        dimensions=dimensions,
    )


def parse_dims(
    dimension_elements: Iterable[Element],
) -> list[Dimension]:
    """
    Parse a set of "dimension" XML elements into Dimension objects.

    Parameters
    ----------
    dimension_elements : Iterable[Element]
        The XML elements to parse.

    Returns
    -------
    list[nisarqa.Dimension]
        The Dimension objects generated.
    """
    dimensions: list[Dimension] = []
    for dimension in dimension_elements:
        dimensions.append(element_to_dimension(dimension))
    return dimensions


def element_to_dimension(element: Element) -> Dimension:
    """
    Parse a "dimension" XML element into a Dimension object.

    Parameters
    ----------
    element : Element
        The XML element to parse.

    Returns
    -------
    nisarqa.Dimension
        The generated Dimension object.
    """
    dimension_attribs = element.attrib
    dim_name = dimension_attribs["name"]
    extent = dimension_attribs["extent"]
    return nisarqa.Dimension(name=dim_name, extent=extent)


def elements_to_datasets(
    xml_elements: Iterable[Element],
    shapes: dict[str, DataShape],
) -> dict[str, XMLDataset]:
    """
    Parse a set of XML elements into XMLDataset objects.

    Parameters
    ----------
    xml_elements : Iterable[Element]
        The XML elements to parse.

    Returns
    -------
    dict[str, nisarqa.XMLDataset]
        The XMLDataset objects generated, addressable by name.
    """

    datasets: dict[str, XMLDataset] = {}
    for element in xml_elements:
        dataset: nisarqa.XMLDataset = element_to_xml_dataset(
            element,
            shapes=shapes,
        )
        datasets[dataset.name] = dataset

    return datasets


def element_to_xml_dataset(
    xml_element: Element,
    shapes: dict[str, DataShape],
) -> nisarqa.XMLDataset:
    """
    Process an XML element into an XMLDataset object.

    Parameters
    ----------
    xml_element : Element
        The element to be processed.

    Returns
    -------
    nisarqa.XMLDataset
        The generated XMLDataset object.
    """
    log = nisarqa.get_logger()

    attributes = xml_element.attrib
    name = attributes["name"]
    del attributes["name"]
    shape_name = attributes["shape"]
    del attributes["shape"]
    if shape_name not in shapes:
        log.error(
            f"XML is missing shape {shape_name} associated with this dataset."
            " Could not associate this dataset with a shape - XML dataset"
            f" {name}"
        )
        shape = None
    else:
        shape = shapes[shape_name]

    node_type = xml_element.tag
    if node_type not in ["integer", "real", "string"]:
        log.error(f"unknown type tag: {node_type} - XML dataset {name}")

    annotation_elements = xml_element.findall("annotation")
    annotations = parse_annotations(
        annotation_elements=annotation_elements,
        parent_name=name,
    )

    return nisarqa.XMLDataset(
        name=name,
        properties=attributes,
        annotations=annotations,
        node_type=node_type,
        shape=shape,
    )


def parse_annotations(
    annotation_elements: Iterable[Element],
    parent_name: str,
) -> list[XMLAnnotation]:
    """
    Parse a set of "annotation" XML elements into DataAnnotation objects.

    Parameters
    ----------
    annotation_elements : Iterable[Element]
        The XML elements to parse.

    Returns
    -------
    list[DataAnnotation]
        The DataAnnotation objects generated.
    """
    annotation_objs: list[XMLAnnotation] = []
    for annotation in annotation_elements:
        annotation_objs.append(
            element_to_annotation(
                element=annotation,
                parent_name=parent_name,
            )
        )
    return annotation_objs


def element_to_annotation(
    element: Element, parent_name: str
) -> nisarqa.XMLAnnotation:
    """
    Parse an "annotation" XML element into a DataAnnotation object.

    Parameters
    ----------
    element : Element
        The XML element to parse.

    Returns
    -------
    DataAnnotation
        The generated DataAnnotation object.
    """
    log = nisarqa.get_logger()
    annotation_attribs = element.attrib
    app = annotation_attribs["app"]
    description = element.text
    attributes: dict[str, str] = {}
    for key in annotation_attribs:
        if key == "description":
            log.warning(
                f"{element.tag} contains attribute 'description'. "
                "This tag is deprecated; use the text field to describe "
                f"the element instead - XML Element {parent_name}"
            )
        if key not in ["app", "description"]:
            attribute: str = annotation_attribs[key]
            attributes[key] = attribute
    return nisarqa.XMLAnnotation(
        app=app, attributes=attributes, description=description
    )


__all__ = nisarqa.get_all(__name__, objects_to_skip)
