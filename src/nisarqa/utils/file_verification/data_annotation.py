from __future__ import annotations

from dataclasses import dataclass
from typing import Any

# List of objects from the import statements that
# should not be included when importing this module
import nisarqa

objects_to_skip = nisarqa.get_all(name=__name__)


# Not marked as a dataclass because it needs to contain a mutable type
# (dict, in this case)
class DataAnnotation:
    """A data annotation."""

    attributes: dict[str, Any]
    description: str

    def __init__(self, attributes: dict[str, Any], description: str):
        self.attributes = attributes
        self.description = description

    @property
    def attribute_names(self) -> set[str]:
        return set(self.attributes.keys())

    def __repr__(self) -> str:
        return (
            f"DataAnnotation(attributes={self.attributes}, "
            f'description="{self.description}")'
        )


class XMLAnnotation(DataAnnotation):
    """A data annotation as described in an XML spec file."""

    app: str

    def __init__(self, app: str, attributes: dict[str, Any], description: str):
        super().__init__(attributes=attributes, description=description)
        self.app = app

    @property
    def attribute_names(self) -> set[str]:
        return set(
            filter(
                lambda key: not nisarqa.ignore_xml_annotation_attribute(key),
                self.attributes.keys(),
            )
        )

    @property
    def is_cpx_marker(self) -> bool:
        if self.app == "io":
            if "kwd" in self.attributes.keys():
                if self.attributes["kwd"] == "complex":
                    return True
        return False

    # Hacky method of figuring out if an annotation describes a boolean-valued
    # dataset. Look for the terms "True" and "False" in the annotation description -
    # there is no other unambiguous means of determining if a dataset is a boolean.
    @property
    def is_bool_marker(self) -> bool:
        if self.app != "conformance":
            return False
        if '"True"' in self.description and '"False"' in self.description:
            return True
        if self.description.startswith("Flag to indicate"):
            return True
        return False

    def __repr__(self) -> str:
        return (
            f"XMLAnnotation(values={self.attributes}, "
            f'description="{self.description}", app="{self.app}")'
        )


class HDF5Annotation(DataAnnotation):
    """A data annotation as expressed in an HDF5 file."""

    def __init__(self, attributes: dict[str, Any], description: str):
        super().__init__(attributes=attributes, description=description)


@dataclass
class DataShape:
    """A data shape as described in an XML spec."""

    name: str
    order: str
    annotations: list[XMLAnnotation]
    dimensions: list[Dimension]

    @property
    def is_complex(self) -> bool:
        return any(dimension.is_cpx_marker for dimension in self.dimensions)


@dataclass
class Dimension:
    """A data dimension."""

    name: str
    extent: int

    @property
    def is_cpx_marker(self) -> bool:
        return self.name == "complex"


__all__ = nisarqa.get_all(__name__, objects_to_skip)
