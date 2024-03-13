from __future__ import annotations

from abc import ABC, abstractproperty
from dataclasses import dataclass
from typing import Any

import h5py
import isce3
import numpy as np

# List of objects from the import statements that
# should not be included when importing this module
import nisarqa
from nisarqa import DataShape, XMLAnnotation

objects_to_skip = nisarqa.get_all(name=__name__)


class DatasetDesc(ABC):
    name: str
    properties: dict[str, str]

    @abstractproperty
    def dtype(self) -> np.dtype | None:
        """
        The object's data type, or None if no valid data type could be
        found.
        """
        ...


@dataclass
class HDF5Dataset(DatasetDesc):
    """A dataset as expressed in the form of an h5py dataset."""

    name: str
    annotation: nisarqa.HDF5Annotation
    dataset: h5py.Dataset

    @property
    def value(self) -> Any:
        return self.dataset[:]

    def check(self): ...

    @property
    def dtype(self) -> np.dtype | None:
        if nisarqa.is_complex32(self.dataset):
            return nisarqa.complex32
        return self.dataset.dtype


class XMLDataset(DatasetDesc):
    """
    An XML spec dataset that is intended to describe an HDF5 dataset in a real
    product.
    """

    annotations: list[XMLAnnotation]
    node_type: str
    shape: DataShape | None

    def __init__(
        self,
        name: str,
        properties: dict[str, str],
        annotations: list[XMLAnnotation],
        node_type: str,
        shape: DataShape | None,
    ):
        self.name = name
        self.properties = properties
        self.annotations = annotations
        self.node_type = node_type
        self.shape = shape

    def get_width(self) -> int | None:
        """
        Get the data width of this dataset, if any.

        Returns
        -------
        int | None
            The width, or None if no width was found.
        """
        log = nisarqa.get_logger()
        if not "width" in self.properties.keys():
            log.error(f"Unable to find data width: XML dataset {self.name}")
            return None
        width_str: str = self.properties["width"]
        try:
            width = int(width_str)
        except ValueError:
            log.error(
                f"Unable to recognize data width as int: {width_str}: "
                f"XML dataset {self.name}"
            )
            return None
        return width

    def get_signedness(self) -> bool | None:
        """
        Get the signedness of this dataset, if any.

        Returns
        -------
        bool | None
            True if signed, False if not. If no signedness was found,
            'integer' datasets are assumed to be signed. Otherwise,
            return None.
        """
        log = nisarqa.get_logger()
        if not "signed" in self.properties.keys():
            if self.node_type == "integer":
                return True
            else:
                return None
        signedness_str: str = self.properties["signed"]
        if signedness_str.lower() == "true":
            return True
        if signedness_str.lower() == "false":
            return False
        log.error(
            f"Unrecognized signedness: {signedness_str}: XML dataset"
            f" {self.name}"
        )
        return None

    @property
    def dtype(self) -> np.dtype | None:
        """The data type associated with this dataset."""
        log = nisarqa.get_logger()

        if self.node_type == "string":
            return XMLDataset._get_dtype_str()
        if self.node_type == "integer":
            width = self.get_width()
            signedness = self.get_signedness()
            return self._get_dtype_int(width=width, signed=signedness)
        if self.node_type == "real":
            width = self.get_width()

            # Complex checking for real-valued XML dataset descriptions:
            is_cpx = False

            # Complex checking for non-InSAR products. These products list all complex
            # valued data objects using what this code calls a "complex marker," or
            # an annotation that contains the following:
            #       <annotation app="io" kwd="complex" />
            # This is checked for in the annotation.is_cpx_marker property.
            for annotation in self.annotations:
                if annotation.is_cpx_marker:
                    is_cpx = True
                    break
            # Complex checking for InSAR products. These products list all complex
            # valued data objects using a shape which contains one complex dimension.
            # This dimension contains the following:
            #       <dimension name="complex" extent="2"/>
            # This is checked for in the shape.is_complex property.
            if not is_cpx:
                if self.shape is not None:
                    is_cpx = self.shape.is_complex
                else:
                    log.error(
                        f"Unable to determine dtype: XML dataset {self.name}"
                    )
                    return None

            return self._get_dtype_real(width=width, is_complex=is_cpx)
        log.error(f"Unable to determine dtype: XML dataset {self.name}")
        return None

    def _get_dtype_str() -> np.dtype:
        return str

    def _get_dtype_int(self, width: int, signed: bool) -> np.dtype | None:
        log = nisarqa.get_logger()
        if signed:
            if width == 8:
                return np.int8
            if width == 16:
                return np.int16
            if width == 32:
                return np.int32
            if width == 64:
                return np.int64
            if width == 128:
                return np.int128
            log.error(
                "Unrecognized int width and signedness for integer type: "
                f"width={width}, "
                f"signed={signed} - "
                f"XML dataset {self.name}"
            )
            return None
        if width == 8:
            return np.uint8
        if width == 16:
            return np.uint16
        if width == 32:
            return np.uint32
        if width == 64:
            return np.uint64
        if width == 128:
            return np.uint128
        log.error(
            "Unrecognized int width and signedness for integer type: "
            f"width={width}, signed={signed} - "
            f"XML dataset {self.name}"
        )
        return None

    def _get_dtype_real(
        self,
        width: int | None = None,
        is_complex: bool = False,
    ) -> np.dtype | None:
        log = nisarqa.get_logger()
        if is_complex:
            if width == 16:
                return nisarqa.complex32
            if width == 32:
                return np.complex64
            if width == 64:
                return np.complex128
            log.error(
                f"Unrecognized width value for complex type: {width} - "
                f"XML dataset {self.name}"
            )
            return None
        if width == 16:
            return np.float16
        if width == 32:
            return np.float32
        if width == 64:
            return np.float64
        if width == 128:
            return np.float128
        log.error(
            f"Unrecognized width value for float type: {width} - "
            f"XML dataset {self.name}"
        )
        return None

    def __repr__(self) -> str:
        return (
            "XMLDataset("
            f'name="{self.name}" '
            f"properties={self.properties} "
            f"annotations={self.annotations} "
            f'node_type="{self.node_type}" '
            f"shape={self.shape}"
            ")"
        )


__all__ = nisarqa.get_all(__name__, objects_to_skip)
