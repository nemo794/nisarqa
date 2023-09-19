from __future__ import annotations

import os
from collections.abc import Iterable
from dataclasses import dataclass, field, fields
from typing import ClassVar, Optional, Type, Union

import numpy as np
from numpy.typing import ArrayLike

import nisarqa
from nisarqa import (
    HDF5Attrs,
    HDF5ParamGroup,
    InputFileGroupParamGroup,
    ProductPathGroupParamGroup,
    RootParamGroup,
    WorkflowsParamGroup,
    YamlAttrs,
    YamlParamGroup,
)

objects_to_skip = nisarqa.get_all(__name__)


@dataclass(frozen=True)
class RSLCWorkflowsParamGroup(WorkflowsParamGroup):
    """
    The parameters specifying which RSLC-Caltools QA workflows should be run.

    This corresponds to the `qa: workflows` runconfig group.

    Parameters
    ----------
    validate : bool, optional
        True to run the validate workflow. Default: True
        (inherited from WorkflowsParamGroup class)
    qa_reports : bool, optional
        True to run the QA Reports workflow. Default: True
        (inherited from WorkflowsParamGroup class)
    abs_cal : bool, optional
        True to run the Absolute Radiometric Calibration Factor CalTool workflow
        Default: True
    noise_estimation : bool, optional
        True to run the Noise Estimation Tool (NET) workflow. Default: True
    point_target : bool, optional
        True to run the Point Target Analyzer (PTA) workflow. Default: True
    """

    abs_cal: bool = field(
        default=WorkflowsParamGroup._default_val,
        metadata={
            "yaml_attrs": YamlAttrs(
                name="absolute_radiometric_calibration",
                descr=WorkflowsParamGroup._descr
                % "Absolute Radiometric Calibration calibration tool",
            )
        },
    )

    noise_estimation: bool = field(
        default=WorkflowsParamGroup._default_val,
        metadata={
            "yaml_attrs": YamlAttrs(
                name="noise_estimation",
                descr=WorkflowsParamGroup._descr
                % "Noise Estimator Tool calibration tool",
            )
        },
    )

    point_target: bool = field(
        default=WorkflowsParamGroup._default_val,
        metadata={
            "yaml_attrs": YamlAttrs(
                name="point_target_analyzer",
                descr=WorkflowsParamGroup._descr
                % "Point Target Analyzer calibration tool",
            )
        },
    )

    def __post_init__(self):
        # VALIDATE INPUTS
        super().__post_init__()
        self._check_workflows_arg("abs_cal", self.abs_cal)
        self._check_workflows_arg("noise_estimation", self.noise_estimation)
        self._check_workflows_arg("point_target", self.point_target)


@dataclass(frozen=True)
class DynamicAncillaryFileParamGroup(YamlParamGroup):
    """
    The parameters from the QA Dynamic Ancillary File runconfig group.

    This corresponds to the `groups: dynamic_ancillary_file_group`
    runconfig group.

    Parameters
    ----------
    corner_reflector_file : str
        The input corner reflector file's file name (with path).
        Required for the Absolute Calibration Factor and Point Target
        Analyzer workflows.
    """

    # Required parameter
    corner_reflector_file: str = field(
        metadata={
            "yaml_attrs": YamlAttrs(
                name="corner_reflector_file",
                descr="""Locations of the corner reflectors in the input product.
                Only required if `absolute_radiometric_calibration` or
                `point_target_analyzer` runconfig params are set to True for QA.""",
            )
        }
    )

    def __post_init__(self):
        nisarqa.validate_is_file(
            filepath=self.corner_reflector_file,
            parameter_name="corner_reflector_file",
            extension=".csv",
        )

    @staticmethod
    def get_path_to_group_in_runconfig():
        return ["runconfig", "groups", "dynamic_ancillary_file_group"]


# TODO - move to generic SLC module
@dataclass(frozen=True)
class BackscatterImageParamGroup(YamlParamGroup, HDF5ParamGroup):
    """
    Parameters to generate RSLC or GSLC Backscatter Images and Browse Image.

    This corresponds to the `qa_reports: backscatter_img` runconfig group.

    Parameters
    ----------
    linear_units : bool, optional
        True to compute backscatter image in linear units, False for decibel units.
        Defaults to True.
    nlooks_freqa, nlooks_freqb : iterable of int, None, optional
        Number of looks along each axis of the input array
        for the specified frequency. If None, then nlooks will be computed
        internally based on `longest_side_max`.
    longest_side_max : int, optional
        The maximum number of pixels allowed for the longest side of the final
        2D multilooked browse image.
        Superseded by nlooks_freq* parameters. Defaults to 2048 pixels.
    percentile_for_clipping : pair of float, optional
        Defines the percentile range that the image array will be clipped to
        and that the colormap covers. Must be in the range [0.0, 100.0].
        Defaults to [5.0, 95.0].
    gamma : float, None, optional
        The gamma correction parameter.
        Gamma will be applied as follows:
            array_out = normalized_array ^ gamma
        where normalized_array is a copy of the image with values
        scaled to the range [0,1].
        The image colorbar will be defined with respect to the input
        image values prior to normalization and gamma correction.
        If None, then no normalization, no gamma correction will be applied.
        Default: 0.5
    tile_shape : iterable of int, optional
        User-preferred tile shape for processing images by batches.
        Actual tile shape may be modified by QA to be an integer
        multiple of the number of looks for multilooking, of the
        decimation ratio, etc.
        Format: (num_rows, num_cols)
        -1 to indicate all rows / all columns (respectively).
        Defaults to (1024, 1024).

    Attributes
    ----------
    backscatter_units : Param
        Units of the backscatter image.
        If `linear_units` is True, this will be set to 'linear'.
        If `linear_units` is False, this will be set to 'dB'.
    """

    linear_units: bool = field(
        default=True,
        metadata={
            "yaml_attrs": YamlAttrs(
                name="linear_units",
                descr="""True to compute backscatter in linear units when generating 
                the backscatter image for the browse images and graphical
                summary PDF. False for decibel units.""",
            )
        },
    )

    _nlooks_descr_template: ClassVar[
        str
    ] = """Number of looks along each axis of the Frequency %s
        image arrays for multilooking the backscatter image.
        Format: [<num_rows>, <num_cols>]
        Example: [6,7]
        If not provided, the QA code to compute the nlooks values 
        based on `longest_side_max`.
    """

    nlooks_freqa: Optional[Iterable[int]] = field(
        default=None,
        metadata={
            "yaml_attrs": YamlAttrs(
                name=f"nlooks_freqa", descr=_nlooks_descr_template % "A"
            )
        },
    )

    nlooks_freqb: Optional[Iterable[int]] = field(
        default=None,
        metadata={
            "yaml_attrs": YamlAttrs(
                name="nlooks_freqb", descr=_nlooks_descr_template % "B"
            )
        },
    )

    longest_side_max: int = field(
        default=2048,
        metadata={
            "yaml_attrs": YamlAttrs(
                name="longest_side_max",
                descr="""The maximum number of pixels allowed for the longest side
                of the final 2D multilooked browse image. Defaults to 2048.
                If `nlooks_freq*` parameter(s) is not None, nlooks
                values will take precedence.""",
            )
        },
    )

    percentile_for_clipping: tuple[float, float] = field(
        default=(5.0, 95.0),
        metadata={
            "yaml_attrs": YamlAttrs(
                name="percentile_for_clipping",
                descr="""Percentile range that the image array will be clipped to
                    and that the colormap covers. Must be in range [0.0, 100.0].""",
            ),
            "hdf5_attrs": HDF5Attrs(
                name="backscatterImagePercentileClipped",
                units="unitless",
                descr=(
                    "Percentile range that the image array was clipped to"
                    " and that the colormap covers"
                ),
                group_path=nisarqa.STATS_H5_QA_PROCESSING_GROUP,
            ),
        },
    )

    gamma: Optional[float] = field(
        default=0.5,
        metadata={
            "yaml_attrs": YamlAttrs(
                name="gamma",
                descr="""Gamma correction parameter applied to backscatter and browse image(s).
            Gamma will be applied as follows:
                array_out = normalized_array ^ gamma
            where normalized_array is a copy of the image with values
            scaled to the range [0,1]. 
            The image colorbar will be defined with respect to the input
            image values prior to normalization and gamma correction.
            If None, then no normalization and no gamma correction will be applied.""",
            ),
            "hdf5_attrs": HDF5Attrs(
                name="backscatterImageGammaCorrection",
                units="unitless",
                descr=(
                    "Gamma correction parameter applied to backscatter and"
                    " browse image(s). Dataset will be type float if gamma was"
                    " applied, otherwise it is the string 'None'"
                ),
                group_path=nisarqa.STATS_H5_QA_PROCESSING_GROUP,
            ),
        },
    )

    tile_shape: Iterable[int] = field(
        default=(1024, 1024),
        metadata={
            "yaml_attrs": YamlAttrs(
                name="tile_shape",
                descr="""User-preferred tile shape for processing images by batches.
                Actual tile shape may be modified by QA to be an integer
                multiple of the number of looks for multilooking, of the
                decimation ratio, etc.
                Format: [<num_rows>, <num_cols>]
                -1 to indicate all rows / all columns (respectively).""",
            )
        },
    )

    # Auto-generated attributes, so set init=False and have no default.
    # `backscatter_units` is determined by the `linear_units` attribute.
    backscatter_units: str = field(
        init=False,
        metadata={
            "hdf5_attrs": HDF5Attrs(
                name="backscatterImageUnits",
                units=None,
                descr="""Units of the backscatter image.""",
                group_path=nisarqa.STATS_H5_QA_PROCESSING_GROUP,
            )
        },
    )

    def __post_init__(self):
        # VALIDATE INPUTS

        # validate linear_units
        if not isinstance(self.linear_units, bool):
            raise TypeError(f"`linear_units` must be bool: {self.linear_units}")

        # validate nlooks_freq*
        self._validate_nlooks(self.nlooks_freqa, "A")
        self._validate_nlooks(self.nlooks_freqa, "B")

        # validate longest_side_max
        if not isinstance(self.longest_side_max, int):
            raise TypeError(
                f"longest_side_max must be a int: {self.longest_side_max}"
            )
        if self.longest_side_max <= 0:
            raise ValueError(
                f"`longest_side_max` must be positive: {self.longest_side_max}"
            )

        # validate percentile_for_clipping
        val = self.percentile_for_clipping
        if not isinstance(val, (list, tuple)):
            raise TypeError(
                f"{self.percentile_for_clipping=}, must be a list or tuple."
            )
        if not len(val) == 2:
            raise ValueError(
                f"{self.percentile_for_clipping=} must have a length of two."
            )
        if not all(isinstance(e, float) for e in val):
            raise TypeError(
                f"{self.percentile_for_clipping=} must contain only float."
            )
        if any((e < 0.0 or e > 100.0) for e in val):
            raise ValueError(
                f"{self.percentile_for_clipping=}, must be in range [0.0,"
                " 100.0]"
            )
        if self.percentile_for_clipping[0] >= self.percentile_for_clipping[1]:
            raise ValueError(
                f"{self.percentile_for_clipping=}; values must appear in"
                " increasing order."
            )

        # validate gamma
        if isinstance(self.gamma, float):
            if self.gamma < 0.0:
                raise ValueError(
                    "If `gamma` is a float, it must be non-negative:"
                    f" {self.gamma}"
                )
        elif self.gamma is not None:
            raise TypeError(
                "`gamma` must be a float or None. "
                f"Value: {self.gamma}, Type: {type(self.gamma)}"
            )

        # validate tile_shape
        val = self.tile_shape
        if not isinstance(val, (list, tuple)):
            raise TypeError(f"`tile_shape` must be a list or tuple: {val}")
        if not len(val) == 2:
            raise TypeError(f"`tile_shape` must have a length of two: {val}")
        if not all(isinstance(e, int) for e in val):
            raise TypeError(f"`tile_shape` must contain only integers: {val}")
        if any(e < -1 for e in val):
            raise TypeError(f"Values in `tile_shape` must be >= -1: {val}")

        # SET ATTRIBUTES DEPENDENT UPON INPUT PARAMETERS
        # This dataclass is frozen to ensure that all inputs are validated,
        # so we need to use object.__setattr__()

        # use linear_units to set backscatter_units
        object.__setattr__(
            self, "backscatter_units", "linear" if self.linear_units else "dB"
        )

    @staticmethod
    def get_path_to_group_in_runconfig():
        return ["runconfig", "groups", "qa", "qa_reports", "backscatter_img"]

    @staticmethod
    def _validate_nlooks(nlooks, freq):
        """
        Raise exception if `nlooks` is not a valid input.

        Parameters
        ----------
        nlooks : iterable of int or None
            Number of looks along each axis of the input array
            for the specified frequency.
        freq : str
            The frequency to assign this number of looks to.
            Options: 'A' or 'B'
        """
        if isinstance(nlooks, (list, tuple)):
            if all(isinstance(e, int) for e in nlooks):
                if any((e < 1) for e in nlooks) or not len(nlooks) == 2:
                    raise TypeError(
                        f"nlooks_freq{freq.lower()} must be an int or a "
                        f"sequence of two ints, which are >= 1: {nlooks}"
                    )
        elif nlooks is None:
            # the code will use `longest_side_max` to compute `nlooks` instead.
            pass
        else:
            raise TypeError(
                "`nlooks` must be of type iterable of int, or None: {nlooks}"
            )


@dataclass(frozen=True)
class HistogramParamGroup(YamlParamGroup, HDF5ParamGroup):
    """
    Parameters to generate the RSLC or GSLC Backscatter and Phase Histograms;
    this corresponds to the `qa_reports: histogram` runconfig group.

    Parameters
    ----------
    decimation_ratio : pair of int, optional
        The step size to decimate the input array for computing
        the backscatter and phase histograms.
        For example, (2,3) means every 2nd azimuth line and
        every 3rd range sample will be used to compute the histograms.
        Defaults to (10,10).
        Format: (<azimuth>, <range>)
    backscatter_histogram_bin_edges_range : pair of int or float, optional
        The dB range for the backscatter histogram's bin edges. Endpoint will
        be included. Defaults to [-80.0, 20.0].
        Format: (<starting value>, <endpoint>)
    phs_in_radians : bool, optional
        True to compute phase in radians units, False for degrees units.
        Defaults to True.
    tile_shape : iterable of int, optional
        User-preferred tile shape for processing images by batches.
        Actual tile shape may be modified by QA to be an integer
        multiple of the number of looks for multilooking, of the
        decimation ratio, etc.
        Format: (num_rows, num_cols)
        -1 to indicate all rows / all columns (respectively).

    Attributes
    ----------
    backscatter_bin_edges : numpy.ndarray
        The bin edges (including endpoint) to use when computing
        the backscatter histograms. Will be set to 100 uniformly-spaced bins
        in range `backscatter_histogram_bin_edges_range`, including endpoint.
    phs_bin_edges : numpy.ndarray
        The bin edges (including endpoint) to use when computing
        the phase histograms.
        If `phs_in_radians` is True, this will be set to 100
        uniformly-spaced bins in range [-pi,pi], including endpoint.
        If `phs_in_radians` is False, this will be set to 100
        uniformly-spaced bins in range [-180,180], including endpoint.
    """

    decimation_ratio: Iterable[int] = field(
        default=(10, 10),
        metadata={
            "yaml_attrs": YamlAttrs(
                name="decimation_ratio",
                descr="""Step size to decimate the input array for computing
                the backscatter and phase histograms.
                For example, [2,3] means every 2nd azimuth line and
                every 3rd range sample will be used to compute the histograms.
                Format: [<azimuth>, <range>]""",
            ),
            "hdf5_attrs": HDF5Attrs(
                name="histogramDecimationRatio",
                units="unitless",
                descr=(
                    "Image decimation strides used to compute backscatter"
                    " and phase histograms. Format: [<azimuth>, <range>]"
                ),
                group_path=nisarqa.STATS_H5_QA_PROCESSING_GROUP,
            ),
        },
    )

    backscatter_histogram_bin_edges_range: Iterable[Union[int, float]] = field(
        default=(-80.0, 20.0),
        metadata={
            "yaml_attrs": YamlAttrs(
                name="backscatter_histogram_bin_edges_range",
                descr="""Range in dB for the backscatter histogram's bin edges. Endpoint will
                be included. Format: [<starting value>, <endpoint>]""",
            )
        },
    )

    phs_in_radians: bool = field(
        default=True,
        metadata={
            "yaml_attrs": YamlAttrs(
                name="phs_in_radians",
                descr="""True to compute phase histogram in radians units,
                False for degrees units.""",
            )
        },
    )

    tile_shape: Iterable[int] = field(
        default=(1024, -1),
        metadata={
            "yaml_attrs": YamlAttrs(
                name="tile_shape",
                descr="""User-preferred tile shape for processing images by batches.
                Actual tile shape may be modified by QA to be an integer
                multiple of the number of looks for multilooking, of the
                decimation ratio, etc.
                Format: [<num_rows>, <num_cols>]
                -1 to indicate all rows / all columns (respectively).""",
            )
        },
    )

    # Auto-generated attributes
    # Backscatter Bin Edges (generated from `backscatter_histogram_bin_edges_range`)
    backscatter_bin_edges: ArrayLike = field(
        init=False,
        metadata={
            "hdf5_attrs": HDF5Attrs(
                name="histogramEdgesBackscatter",
                units="dB",
                descr=(
                    "Bin edges (including endpoint) for backscatter histogram"
                ),
                group_path=nisarqa.STATS_H5_QA_PROCESSING_GROUP,
            )
        },
    )

    # Phase bin edges (generated from `phs_in_radians`)
    # Note: `phs_bin_edges` is dependent upon `phs_in_radians` being set
    # first. The value of `phs_bin_edges` can be set in __post_init__,
    # but the contents of the field metadata cannot be modified
    # after initialization. It raises this error:
    #     TypeError: 'mappingproxy' object does not support item assignment
    # So, use a lambda function; this can be called to generate the correct
    # HDF5Attrs when needed, and it does not clutter the dataclass much.
    # Usage: `obj` is an instance of HistogramParamGroup()
    phs_bin_edges: ArrayLike = field(
        init=False,
        metadata={
            "hdf5_attrs_func": lambda obj: (
                HDF5Attrs(
                    name="histogramEdgesPhase",
                    units="radians" if obj.phs_in_radians else "degrees",
                    descr="Bin edges (including endpoint) for phase histogram",
                    group_path=nisarqa.STATS_H5_QA_PROCESSING_GROUP,
                )
                if (isinstance(obj, HistogramParamGroup))
                else nisarqa.raise_(
                    TypeError(
                        f"`obj` is {type(obj)}, but must be type"
                        " HistogramParamGroup"
                    )
                )
            )
        },
    )

    def __post_init__(self):
        # VALIDATE INPUTS

        # validate decimation_ratio
        val = self.decimation_ratio
        if not isinstance(val, (list, tuple)):
            raise TypeError(f"`decimation_ratio` must be list or tuple: {val}")
        if not len(val) == 2:
            raise ValueError(f"`decimation_ratio` must have length of 2: {val}")
        if not all(isinstance(e, int) for e in val):
            raise TypeError(f"`decimation_ratio` must contain integers: {val}")
        if any(e <= 0 for e in val):
            raise ValueError(
                f"`decimation_ratio` must contain positive values: {val}"
            )

        # Validate backscatter_histogram_bin_edges_range
        val = self.backscatter_histogram_bin_edges_range
        if not isinstance(val, (list, tuple)):
            raise TypeError(
                "`backscatter_histogram_bin_edges_range` must"
                f" be a list or tuple: {val}"
            )
        if not len(val) == 2:
            raise ValueError(
                "`backscatter_histogram_bin_edges_range` must"
                f" have a length of two: {val}"
            )
        if not all(isinstance(e, (int, float)) for e in val):
            raise TypeError(
                "`backscatter_histogram_bin_edges_range` must"
                f" contain only int or float values: {val}"
            )
        if val[0] >= val[1]:
            raise ValueError(
                "`backscatter_histogram_bin_edges_range` has format "
                "[<starting value>, <endpoint>] where <starting value> "
                f"must be less than <ending value>: {val}"
            )

        # validate phs_in_radians
        if not isinstance(self.phs_in_radians, bool):
            raise TypeError(f"phs_in_radians` must be bool: {val}")

        # validate tile_shape
        val = self.tile_shape
        if not isinstance(val, (list, tuple)):
            raise TypeError(f"`tile_shape` must be a list or tuple: {val}")
        if not len(val) == 2:
            raise TypeError(f"`tile_shape` must have a length of two: {val}")
        if not all(isinstance(e, int) for e in val):
            raise TypeError(f"`tile_shape` must contain only integers: {val}")
        if any(e < -1 for e in val):
            raise TypeError(f"Values in `tile_shape` must be >= -1: {val}")

        # SET ATTRIBUTES DEPENDENT UPON INPUT PARAMETERS
        # This dataclass is frozen to ensure that all inputs are validated,
        # so we need to use object.__setattr__()

        # Set attributes dependent upon backscatter_histogram_bin_edges_range
        # Backscatter Bin Edges - hardcode to be in decibels
        # 101 bin edges => 100 bins
        object.__setattr__(
            self,
            "backscatter_bin_edges",
            np.linspace(
                self.backscatter_histogram_bin_edges_range[0],
                self.backscatter_histogram_bin_edges_range[1],
                num=101,
                endpoint=True,
            ),
        )

        # Set attributes dependent upon phs_in_radians
        start, stop = (-np.pi, np.pi) if self.phs_in_radians else (-180, 180)
        object.__setattr__(
            self,
            "phs_bin_edges",
            np.linspace(start=start, stop=stop, num=101, endpoint=True),
        )

    @staticmethod
    def get_path_to_group_in_runconfig():
        return ["runconfig", "groups", "qa", "qa_reports", "histogram"]


@dataclass(frozen=True)
class AbsCalParamGroup(YamlParamGroup, HDF5ParamGroup):
    """
    Parameters from the QA-CalTools Absolute Radiometric Calibration
    runconfig group.

    Parameters
    ----------
    attr1 : float, optional
        Placeholder Attribute 1.
    """

    # Attributes
    attr1: float = field(
        default=2.3,
        metadata={
            "yaml_attrs": YamlAttrs(
                name="attr1",
                descr="""Placeholder: Attribute 1 description for runconfig.
            Each new line of text will be a separate line in the runconfig
            template. `attr1` is a non-negative float value.""",
            ),
            "hdf5_attrs": HDF5Attrs(
                name="attribute1",
                units="smoot",
                descr="Description of `attr1` for stats.h5 file",
                group_path=nisarqa.STATS_H5_ABSCAL_PROCESSING_GROUP,
            ),
        },
    )

    def __post_init__(self):
        # validate all attributes in __post_init__

        # validate attr1
        if not isinstance(self.attr1, float):
            raise TypeError(f"`attr1` must be a float: {self.attr1}")
        if self.attr1 < 0:
            raise TypeError(f"`attr1` must be positive: {self.attr1}")

    @staticmethod
    def get_path_to_group_in_runconfig():
        return [
            "runconfig",
            "groups",
            "qa",
            "absolute_radiometric_calibration",
        ]


@dataclass(frozen=True)
class NoiseEstimationParamGroup(YamlParamGroup, HDF5ParamGroup):
    """
    Parameters from the QA-CalTools Noise Estimator (NET) runconfig group.

    Parameters
    ----------
    attr1 : float, optional
        Placeholder Attribute 1.

    Attributes
    ----------
    attr2 : Param
        Placeholder parameter of type bool. This is set based on `attr1`.
    """

    # Attributes for running the NET workflow
    attr1: float = field(
        default=11.9,
        metadata={
            "yaml_attrs": YamlAttrs(
                name="attr1",
                descr=f"""Placeholder: Attribute 1 description for runconfig.
            Each new line of text will be a separate line in the runconfig
            template. The Default value will be auto-appended to this
            description by the QA code during generation of the template.
            `attr1` is a positive float value.""",
            )
        },
    )

    # Auto-generated attributes. Set init=False for auto-generated attributes.
    # attr2 is dependent upon attr1
    attr2: bool = field(
        init=False,
        metadata={
            "hdf5_attrs": HDF5Attrs(
                name="attribute2",
                units="parsecs",
                descr="True if K-run was less than 12.0",
                group_path=nisarqa.STATS_H5_NOISE_EST_PROCESSING_GROUP,
            )
        },
    )

    def __post_init__(self):
        # VALIDATE INPUTS

        # Validate attr1
        if not isinstance(self.attr1, float):
            raise TypeError(f"`attr1` must be a float: {self.attr1}")
        if self.attr1 < 0.0:
            raise TypeError(f"`attr1` must be postive: {self.attr1}")

        # SET ATTRIBUTES DEPENDENT UPON INPUT PARAMETERS
        # This dataclass is frozen to ensure that all inputs are validated,
        # so we need to use object.__setattr__()

        # set attr2 based on attr1
        object.__setattr__(self, "attr2", (self.attr1 < 12.0))

    @staticmethod
    def get_path_to_group_in_runconfig():
        return ["runconfig", "groups", "qa", "noise_estimation"]


@dataclass(frozen=True)
class PointTargetAnalyzerParamGroup(YamlParamGroup, HDF5ParamGroup):
    """
    Parameters from the QA-CalTools Point Target Analyzer runconfig group.

    Parameters
    ----------
    attr1 : float, optional
        Placeholder Attribute 1.
    """

    attr1: float = field(
        default=2300.5,
        metadata={
            "yaml_attrs": YamlAttrs(
                name="attr1",
                descr="""Placeholder: Attribute 1 description for runconfig.
            Each new line of text will be a separate line in the runconfig
            template.
            `attr1` is a non-negative float value.""",
            ),
            "hdf5_attrs": HDF5Attrs(
                name="attribute1",
                units="beard-second",
                descr="Description of `attr1` for stats.h5 file",
                group_path=nisarqa.STATS_H5_PTA_PROCESSING_GROUP,
            ),
        },
    )

    def __post_init__(self):
        # validate attr1
        if not isinstance(self.attr1, float):
            raise TypeError(f"`attr1` must be a float: {self.attr1}")
        if self.attr1 < 0.0:
            raise TypeError(f"`attr1` must be non-negative: {self.attr1}")

    @staticmethod
    def get_path_to_group_in_runconfig():
        return ["runconfig", "groups", "qa", "point_target_analyzer"]


@dataclass
class RSLCRootParamGroup(RootParamGroup):
    """
    Dataclass of all *ParamGroup objects to process QA for NISAR RSLC products.

    `workflows` is the only required parameter; this *ParamGroup contains
    boolean attributes that indicate which QA workflows to run.

    All other parameters are optional, but they each correspond to (at least)
    one of the QA workflows. Based on the workflows set to True in
    `workflows`, certain others of these parameters will become required.

    Parameters
    ----------
    workflows : RSLCWorkflowsParamGroup
        RSLC QA Workflows parameters
    input_f : InputFileGroupParamGroup or None, optional
        Input File Group parameters for RSLC QA
    prodpath : ProductPathGroupParamGroup or None, optional
        Product Path Group parameters for RSLC QA
    backscatter_img : BackscatterImageParamGroup or None, optional
        Backscatter Image Group parameters for RSLC QA
    histogram : HistogramParamGroup or None, optional
        Histogram Group parameters for RSLC or GSLC QA
    anc_files : DynamicAncillaryFileParamGroup or None, optional
        Dynamic Ancillary File Group parameters for RSLC QA-Caltools
    abs_cal : AbsCalParamGroup or None, optional
        Absolute Radiometric Calibration group parameters for RSLC QA-Caltools
    noise_estimation : NoiseEstimationParamGroup or None, optional
        Noise Estimation Tool group parameters for RSLC QA-Caltools
    pta : PointTargetAnalyzerParamGroup or None, optional
        Point Target Analyzer group parameters for RSLC QA-Caltools
    """

    # Shared parameters
    workflows: RSLCWorkflowsParamGroup  # overwrite parent's `workflows` b/c new type

    # QA parameters
    backscatter_img: Optional[BackscatterImageParamGroup] = None
    histogram: Optional[HistogramParamGroup] = None

    # CalTools parameters
    anc_files: Optional[DynamicAncillaryFileParamGroup] = None
    abs_cal: Optional[AbsCalParamGroup] = None
    noise_estimation: Optional[NoiseEstimationParamGroup] = None
    pta: Optional[PointTargetAnalyzerParamGroup] = None

    @staticmethod
    def get_mapping_of_workflows2param_grps(workflows):
        Grp = RootParamGroup.ReqParamGrp  # class object for our named tuple

        flag_any_workflows_true = any(
            [getattr(workflows, field.name) for field in fields(workflows)]
        )

        grps_to_parse = (
            Grp(
                flag_param_grp_req=flag_any_workflows_true,
                root_param_grp_attr_name="input_f",
                param_grp_cls_obj=InputFileGroupParamGroup,
            ),
            Grp(
                flag_param_grp_req=flag_any_workflows_true,
                root_param_grp_attr_name="prodpath",
                param_grp_cls_obj=ProductPathGroupParamGroup,
            ),
            Grp(
                flag_param_grp_req=workflows.qa_reports,
                root_param_grp_attr_name="backscatter_img",
                param_grp_cls_obj=BackscatterImageParamGroup,
            ),
            Grp(
                flag_param_grp_req=workflows.qa_reports,
                root_param_grp_attr_name="histogram",
                param_grp_cls_obj=HistogramParamGroup,
            ),
            Grp(
                flag_param_grp_req=workflows.abs_cal,
                root_param_grp_attr_name="abs_cal",
                param_grp_cls_obj=AbsCalParamGroup,
            ),
            Grp(
                flag_param_grp_req=workflows.abs_cal or workflows.point_target,
                root_param_grp_attr_name="anc_files",
                param_grp_cls_obj=DynamicAncillaryFileParamGroup,
            ),
            Grp(
                flag_param_grp_req=workflows.noise_estimation,
                root_param_grp_attr_name="noise_estimation",
                param_grp_cls_obj=NoiseEstimationParamGroup,
            ),
            Grp(
                flag_param_grp_req=workflows.point_target,
                root_param_grp_attr_name="pta",
                param_grp_cls_obj=PointTargetAnalyzerParamGroup,
            ),
        )

        return grps_to_parse

    @staticmethod
    def get_order_of_groups_in_yaml():
        # This order determines the order
        # the groups will appear in the runconfig.
        return (
            InputFileGroupParamGroup,
            DynamicAncillaryFileParamGroup,
            ProductPathGroupParamGroup,
            RSLCWorkflowsParamGroup,
            BackscatterImageParamGroup,
            HistogramParamGroup,
            AbsCalParamGroup,
            NoiseEstimationParamGroup,
            PointTargetAnalyzerParamGroup,
        )


# TODO - move to nisar_params.py module
def build_root_params(product_type, user_rncfg):
    """
    Build the *RootParamGroup object for the specified product type.

    Parameters
    ----------
    product_type : str
        One of: 'rslc', 'gslc', 'gcov', 'rifg', 'runw', 'gunw', 'roff', 'goff'
    user_rncfg : dict
        A dictionary whose structure matches `product_type`'s QA runconfig
        yaml file and which contains the parameters needed to run its QA SAS.

    Returns
    -------
    root_param_group : RSLCRootParamGroup
        *RootParamGroup object for the specified product type. This will be
        populated with runconfig values where provided,
        and default values for missing runconfig parameters.
    """
    if product_type not in nisarqa.LIST_OF_NISAR_PRODUCTS:
        raise ValueError(
            f"{product_type=} but must one of: {nisarqa.LIST_OF_NISAR_PRODUCTS}"
        )

    if product_type == "rslc":
        workflows_param_cls_obj = RSLCWorkflowsParamGroup
        root_param_class_obj = RSLCRootParamGroup
    elif product_type == "gslc":
        workflows_param_cls_obj = WorkflowsParamGroup
        root_param_class_obj = nisarqa.GSLCRootParamGroup
    elif product_type == "gcov":
        workflows_param_cls_obj = WorkflowsParamGroup
        root_param_class_obj = nisarqa.GCOVRootParamGroup
    else:
        raise NotImplementedError(f"{product_type} code not implemented yet.")

    # Dictionary to hold the *ParamGroup objects. Will be used as
    # kwargs for the *RootParamGroup instance.
    root_inputs = {}

    # Construct *WorkflowsParamGroup dataclass (necessary for all workflows)
    try:
        root_inputs["workflows"] = _get_param_group_instance_from_runcfg(
            param_grp_cls_obj=workflows_param_cls_obj, user_rncfg=user_rncfg
        )

    except KeyError as e:
        raise KeyError("`workflows` group is a required runconfig group") from e
    # If all functionality is off (i.e. all workflows are set to false),
    # then exit early. We will not need any of the other runconfig groups.
    if not root_inputs["workflows"].at_least_one_wkflw_requested():
        raise nisarqa.ExitEarly("All `workflows` were set to False.")

    workflows = root_inputs["workflows"]

    wkflws2params_mapping = (
        root_param_class_obj.get_mapping_of_workflows2param_grps(
            workflows=workflows
        )
    )

    for param_grp in wkflws2params_mapping:
        if param_grp.flag_param_grp_req:
            populated_rncfg_group = _get_param_group_instance_from_runcfg(
                param_grp_cls_obj=param_grp.param_grp_cls_obj,
                user_rncfg=user_rncfg,
            )

            root_inputs[param_grp.root_param_grp_attr_name] = (
                populated_rncfg_group
            )

    # Construct and return *RootParamGroup
    return root_param_class_obj(**root_inputs)


# TODO - move to generic NISAR module
def _get_param_group_instance_from_runcfg(
    param_grp_cls_obj: Type[YamlParamGroup], user_rncfg: Optional[dict] = None
):
    """
    Generate an instance of a YamlParamGroup subclass) object
    where the values from a user runconfig take precedence.

    Parameters
    ----------
    param_grp_cls_obj : Type[YamlParamGroup]
        A class instance of a subclass of YamlParamGroup.
        For example, `HistogramParamGroup`.
    user_rncfg : nested dict, optional
        A dict containing the user's runconfig values that (at minimum)
        correspond to the `param_grp_cls_obj` parameters. (Other values
        will be ignored.) For example, a QA runconfig yaml loaded directly
        into a dict would be a perfect input for `user_rncfg`.
        The nested structure of `user_rncfg` must match the structure
        of the QA runconfig yaml file for this parameter group.
        To see the expected yaml structure for e.g. RSLC, run
        `nisarqa dumpconfig rslc` from the command line.
        If `user_rncfg` contains entries that do not correspond to attributes
        in `param_grp_cls_obj`, they will be ignored.
        If `user_rncfg` is either None, an empty dict, or does not contain
        values for `param_grp_cls_obj` in a nested structure that matches
        the QA runconfig group that corresponds to `param_grp_cls_obj`,
        then an instance with all default values will be returned.

    Returns
    -------
    param_grp_instance : `param_grp_cls_obj` instance
        An instance of `param_grp_cls_obj` that is fully instantiated
        using default values and the arguments provided in `user_rncfg`.
        The values in `user_rncfg` have precedence over the defaults.
    """

    if not user_rncfg:
        # If user_rncfg is None or is an empty dict, then return the default
        return param_grp_cls_obj()

    # Get the runconfig path for this *ParamGroup
    rncfg_path = param_grp_cls_obj.get_path_to_group_in_runconfig()

    try:
        runcfg_grp_dict = nisarqa.get_nested_element_in_dict(
            user_rncfg, rncfg_path
        )
    except KeyError:
        # Group was not found, so construct an instance using all defaults.
        # If a dataclass has a required parameter, this will (correctly)
        # throw another error.
        return param_grp_cls_obj()
    else:
        # Get the relevant yaml runconfig parameters for this ParamGroup
        yaml_names = param_grp_cls_obj.get_dict_of_yaml_names()

        # prune extraneous fields from the runconfig group
        # (aka keep only the runconfig fields that are relevant to QA)
        # The "if..." logic will allow us to skip missing runconfig fields.
        user_input_args = {
            cls_attr_name: runcfg_grp_dict[yaml_name]
            for cls_attr_name, yaml_name in yaml_names.items()
            if yaml_name in runcfg_grp_dict
        }

        return param_grp_cls_obj(**user_input_args)


__all__ = nisarqa.get_all(__name__, objects_to_skip)
