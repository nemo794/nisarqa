from __future__ import annotations

import os
import types
from collections.abc import Sequence
from dataclasses import dataclass, field, fields
from typing import ClassVar, Optional

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
class RIFGInputFileGroupParamGroup(InputFileGroupParamGroup):
    """
    Parameters from the Input File Group runconfig group.

    This corresponds to the `groups: input_file_group` runconfig group.

    Parameters
    ----------
    qa_input_file : str
        The input NISAR product file name (with path).
    """

    # Required parameter - do not set a default
    qa_input_file: str = field(
        metadata={
            "yaml_attrs": YamlAttrs(
                name="qa_rifg_input_file",
                # Copy the description. InputFileGroupParamGroup only has
                # one field, so we'll cheat and access it via the [0] index.
                descr=fields(InputFileGroupParamGroup)[0]
                .metadata["yaml_attrs"]
                .descr,
            )
        }
    )


@dataclass(frozen=True)
class RUNWInputFileGroupParamGroup(InputFileGroupParamGroup):
    """
    Parameters from the Input File Group runconfig group.

    This corresponds to the `groups: input_file_group` runconfig group.

    Parameters
    ----------
    qa_input_file : str
        The input NISAR product file name (with path).
    """

    # Required parameter - do not set a default
    qa_input_file: str = field(
        metadata={
            "yaml_attrs": YamlAttrs(
                name="qa_runw_input_file",
                # Copy the description. InputFileGroupParamGroup only has
                # one field, so we'll cheat and access it via the [0] index.
                descr=fields(InputFileGroupParamGroup)[0]
                .metadata["yaml_attrs"]
                .descr,
            )
        }
    )


@dataclass(frozen=True)
class GUNWInputFileGroupParamGroup(InputFileGroupParamGroup):
    """
    Parameters from the Input File Group runconfig group.

    This corresponds to the `groups: input_file_group` runconfig group.

    Parameters
    ----------
    qa_input_file : str
        The input NISAR product file name (with path).
    """

    # Required parameter - do not set a default
    qa_input_file: str = field(
        metadata={
            "yaml_attrs": YamlAttrs(
                name="qa_gunw_input_file",
                # Copy the description. InputFileGroupParamGroup only has
                # one field, so we'll cheat and access it via the [0] index.
                descr=fields(InputFileGroupParamGroup)[0]
                .metadata["yaml_attrs"]
                .descr,
            )
        }
    )


@dataclass(frozen=True)
class ROFFInputFileGroupParamGroup(InputFileGroupParamGroup):
    """
    Parameters from the Input File Group runconfig group.

    This corresponds to the `groups: input_file_group` runconfig group.

    Parameters
    ----------
    qa_input_file : str
        The input NISAR product file name (with path).
    """

    # Required parameter - do not set a default
    qa_input_file: str = field(
        metadata={
            "yaml_attrs": YamlAttrs(
                name="qa_roff_input_file",
                # Copy the description. InputFileGroupParamGroup only has
                # one field, so we'll cheat and access it via the [0] index.
                descr=fields(InputFileGroupParamGroup)[0]
                .metadata["yaml_attrs"]
                .descr,
            )
        }
    )


@dataclass(frozen=True)
class GOFFInputFileGroupParamGroup(InputFileGroupParamGroup):
    """
    Parameters from the Input File Group runconfig group.

    This corresponds to the `groups: input_file_group` runconfig group.

    Parameters
    ----------
    qa_input_file : str
        The input NISAR product file name (with path).
    """

    # Required parameter - do not set a default
    qa_input_file: str = field(
        metadata={
            "yaml_attrs": YamlAttrs(
                name="qa_goff_input_file",
                # Copy the description. InputFileGroupParamGroup only has
                # one field, so we'll cheat and access it via the [0] index.
                descr=fields(InputFileGroupParamGroup)[0]
                .metadata["yaml_attrs"]
                .descr,
            )
        }
    )


# TODO - move to generic NISAR module (InSAR will need more thought)
@dataclass(frozen=True)
class InSARProductPathGroupParamGroup(ProductPathGroupParamGroup):
    """
    Parameters from the Product Path Group runconfig group.

    This corresponds to the `groups: product_path_group` runconfig group.

    Parameters
    ----------
    qa_output_dir : str, optional
        Filepath to the output directory to store NISAR QA output files.
        Defaults to './qa'.
    """

    qa_output_dir: str = field(
        default="./qa",
        metadata={
            "yaml_attrs": YamlAttrs(
                name="qa_output_dir",
                descr="""Output directory to store all QA output files.
                REQUIRED for QA. NOT REQUIRED if only running Product SAS.
                Because multiple InSAR products can be generated by a single
                ISCE3 runconfig, QA will make new product-specific directories
                here to store the corresponding product output files in.
                Ex: if the output dir is set to './qa' and QA is requested for
                an RIFG product, then QA will create './qa/rifg' and save the
                QA outputs there.""",
            )
        },
    )


@dataclass(frozen=True)
class RIFGProductPathGroupParamGroup(InSARProductPathGroupParamGroup):
    def __post_init__(self):
        # append subdirectory for the insar product to store its outputs
        object.__setattr__(
            self, "qa_output_dir", os.path.join(self.qa_output_dir, "rifg")
        )

        super().__post_init__()


@dataclass(frozen=True)
class RUNWProductPathGroupParamGroup(InSARProductPathGroupParamGroup):
    def __post_init__(self):
        # append subdirectory for the insar product to store its outputs
        object.__setattr__(
            self, "qa_output_dir", os.path.join(self.qa_output_dir, "runw")
        )

        super().__post_init__()


@dataclass(frozen=True)
class GUNWProductPathGroupParamGroup(InSARProductPathGroupParamGroup):
    def __post_init__(self):
        # append subdirectory for the insar product to store its outputs
        object.__setattr__(
            self, "qa_output_dir", os.path.join(self.qa_output_dir, "gunw")
        )

        super().__post_init__()


@dataclass(frozen=True)
class ROFFProductPathGroupParamGroup(InSARProductPathGroupParamGroup):
    def __post_init__(self):
        # append subdirectory for the insar product to store its outputs
        object.__setattr__(
            self, "qa_output_dir", os.path.join(self.qa_output_dir, "roff")
        )

        super().__post_init__()


@dataclass(frozen=True)
class GOFFProductPathGroupParamGroup(InSARProductPathGroupParamGroup):
    def __post_init__(self):
        # append subdirectory for the insar product to store its outputs
        object.__setattr__(
            self, "qa_output_dir", os.path.join(self.qa_output_dir, "goff")
        )

        super().__post_init__()


@dataclass(frozen=True)
class RIFGWorkflowsParamGroup(WorkflowsParamGroup):
    @staticmethod
    def get_path_to_group_in_runconfig():
        return ["runconfig", "groups", "qa", "rifg", "workflows"]


@dataclass(frozen=True)
class RUNWWorkflowsParamGroup(WorkflowsParamGroup):
    @staticmethod
    def get_path_to_group_in_runconfig():
        return ["runconfig", "groups", "qa", "runw", "workflows"]


@dataclass(frozen=True)
class GUNWWorkflowsParamGroup(WorkflowsParamGroup):
    @staticmethod
    def get_path_to_group_in_runconfig():
        return ["runconfig", "groups", "qa", "gunw", "workflows"]


@dataclass(frozen=True)
class ROFFWorkflowsParamGroup(WorkflowsParamGroup):
    @staticmethod
    def get_path_to_group_in_runconfig():
        return ["runconfig", "groups", "qa", "roff", "workflows"]


@dataclass(frozen=True)
class GOFFWorkflowsParamGroup(WorkflowsParamGroup):
    @staticmethod
    def get_path_to_group_in_runconfig():
        return ["runconfig", "groups", "qa", "goff", "workflows"]


@dataclass(frozen=True)
class HSIImageParamGroup(YamlParamGroup, HDF5ParamGroup):
    """
    Parameters to generate HSI Browse Image.

    Parameters
    ----------
    equalize_browse : bool, optional
        True to perform histogram equalization on the Intensity channel
        (the coherence magnitude layer) in the browse image PNG.
        (Browse image is in HSI color space.) PDF report will not be affected.
        See: https://scikit-image.org/docs/stable/auto_examples/color_exposure/plot_equalize.html
        Default is True.
    longest_side_max : int, optional
        The maximum number of pixels allowed for the longest side of the final
        2D multilooked browse image. Defaults to 2048 pixels.
    """

    equalize_browse: bool = field(
        default=True,
        metadata={
            "yaml_attrs": YamlAttrs(
                name="equalize_browse",
                descr="""True to perform histogram equalization on the Intensity channel
        (the coherence magnitude layer) in the browse image PNG.
        (Browse image is in HSI color space.) PDF report will not be affected.
        See: https://scikit-image.org/docs/stable/auto_examples/color_exposure/plot_equalize.html""",
            ),
            "hdf5_attrs": HDF5Attrs(
                name="equalizeBrowse",
                units="1",
                descr=(
                    "If True, histogram equalization was applied to the"
                    " intensity channel (coherence magnitude layer) in the"
                    " HSI browse image. (PDF report is not affected.)"
                ),
                group_path=nisarqa.STATS_H5_QA_PROCESSING_GROUP,
            ),
        },
    )

    longest_side_max: int = field(
        default=2048,
        metadata={
            "yaml_attrs": YamlAttrs(
                name="longest_side_max",
                descr="""The maximum number of pixels allowed for the longest
                side of the final 2D browse image.""",
            )
        },
    )

    def __post_init__(self):
        # VALIDATE INPUTS

        # validate equalize_browse
        if not isinstance(self.equalize_browse, bool):
            raise TypeError(
                f"`equalize_browse` must be bool: {self.equalize_browse}"
            )

        # validate longest_side_max
        if not isinstance(self.longest_side_max, int):
            raise TypeError(
                f"longest_side_max must be a int: {self.longest_side_max}"
            )
        if self.longest_side_max <= 0:
            raise ValueError(
                f"`longest_side_max` must be positive: {self.longest_side_max}"
            )


@dataclass(frozen=True)
class UNWHSIImageParamGroup(HSIImageParamGroup):
    """
    Parameters to generate HSI Browse Image for unwrapped phase image.

    Parameters
    ----------
    equalize_browse : bool, optional
        True to perform histogram equalization on the Intensity channel
        (the coherence magnitude layer) in the browse image PNG.
        (Browse image is in HSI color space.) PDF report will not be affected.
        See: https://scikit-image.org/docs/stable/auto_examples/color_exposure/plot_equalize.html
        Default is True.
    longest_side_max : int, optional
        The maximum number of pixels allowed for the longest side of the final
        2D multilooked browse image. Defaults to 2048 pixels.
    rewrap : float or None, optional
        The multiple of pi to rewrap the unwrapped phase image when generating
        the HSI image(s). If None, no rewrapping will occur.
        Ex: If 3 is provided, the image is rewrapped to the interval [0, 3pi).
    """

    rewrap: Optional[float | int] = field(
        default=7,
        metadata={
            "yaml_attrs": YamlAttrs(
                name="rewrap",
                descr="""The multiple of pi to rewrap the unwrapped phase image
                    when generating the HSI image(s). If None, no rewrapping will occur.
                    Ex: If 3 is provided, the image is rewrapped to the interval [0, 3pi).
                    """,
            ),
            "hdf5_attrs": HDF5Attrs(
                name="HSIImageRewrap",
                units="1",
                descr=(
                    "The multiple of pi for rewrapping the unwrapped phase"
                    " image in the HSI image(s). 'None' if no rewrapping"
                    " occurred. Example: If rewrap=3, the image was rewrapped"
                    " to the interval [0, 3pi)."
                ),
                group_path=nisarqa.STATS_H5_QA_PROCESSING_GROUP,
            ),
        },
    )

    def __post_init__(self):
        # VALIDATE INPUTS

        super().__post_init__()

        # validate rewrap
        if not isinstance(self.rewrap, (float, int)) and (
            self.rewrap is not None
        ):
            raise TypeError(
                f"{self.rewrap=} is {type(self.rewrap)}; "
                "must be float, int, or None."
            )

        if (self.rewrap is not None) and (self.rewrap <= 0):
            raise ValueError(f"{self.rewrap=}; must be a positive value.")


@dataclass(frozen=True)
class RIFGHSIImageParamGroup(HSIImageParamGroup):
    @staticmethod
    def get_path_to_group_in_runconfig():
        return ["runconfig", "groups", "qa", "rifg", "qa_reports", "hsi_img"]


@dataclass(frozen=True)
class RUNWHSIImageParamGroup(UNWHSIImageParamGroup):
    @staticmethod
    def get_path_to_group_in_runconfig():
        return ["runconfig", "groups", "qa", "runw", "qa_reports", "hsi_img"]


@dataclass(frozen=True)
class GUNWHSIImageParamGroup(UNWHSIImageParamGroup):
    @staticmethod
    def get_path_to_group_in_runconfig():
        return ["runconfig", "groups", "qa", "gunw", "qa_reports", "hsi_img"]


@dataclass(frozen=True)
class UNWPhaseImageParamGroup(YamlParamGroup, HDF5ParamGroup):
    """
    Parameters to plot unwrapped phase image.

    Parameters
    ----------
    rewrap : float or None, optional
        The multiple of pi to rewrap the unwrapped phase image.
        If None, no rewrapping will occur.
        Ex: If 3 is provided, the image is rewrapped to the interval [0, 3pi).
    """

    rewrap: Optional[float] = field(
        default=7,
        metadata={
            "yaml_attrs": YamlAttrs(
                name="rewrap",
                descr="""The multiple of pi to rewrap the unwrapped phase image.
                    If None, no rewrapping will occur.
                    Ex: If 3 is provided, the image is rewrapped to the interval [0, 3pi).""",
            ),
            "hdf5_attrs": HDF5Attrs(
                name="phaseImageRewrap",
                units="1",
                descr=(
                    "The multiple of pi for rewrapping the unwrapped phase"
                    " image. 'None' if no rewrapping"
                    " occurred. Example: If rewrap=3, the image was rewrapped"
                    " to the interval [0, 3pi)."
                ),
                group_path=nisarqa.STATS_H5_QA_PROCESSING_GROUP,
            ),
        },
    )

    def __post_init__(self):
        # VALIDATE INPUTS

        # validate rewrap
        if not isinstance(self.rewrap, (float, int)) and (
            self.rewrap is not None
        ):
            raise TypeError(
                f"{self.rewrap=} is {type(self.rewrap)}; "
                "must be float, int, or None."
            )

        if (self.rewrap is not None) and (self.rewrap <= 0):
            raise ValueError(f"{self.rewrap=}; must be a positive value.")


@dataclass(frozen=True)
class RUNWPhaseImageParamGroup(UNWPhaseImageParamGroup):
    @staticmethod
    def get_path_to_group_in_runconfig():
        return ["runconfig", "groups", "qa", "runw", "qa_reports", "phase_img"]


@dataclass(frozen=True)
class GUNWPhaseImageParamGroup(UNWPhaseImageParamGroup):
    @staticmethod
    def get_path_to_group_in_runconfig():
        return ["runconfig", "groups", "qa", "gunw", "qa_reports", "phase_img"]


@dataclass(frozen=True)
class QuiverParamGroup(YamlParamGroup):
    """
    Parameters to generate Quiver Plots with Browse Image.

    Parameters
    ----------
    cbar_min_max : None or pair of float or int, optional
        The vmin and vmax values to generate the quiver plot.
        The magnitude of the offsets is clipped to this interval,
        which (in turn) is used for the interval of the colorbar.
        If None, the range is computed using the min and max magnitudes
        of along track and slant range offset.
        Defaults to None.
    browse_decimation_freqa, browse_decimation_freqb : pair of int or None, optional
        Stride along each axis of the Frequency A (or Frequency B)
        image arrays for decimating the quiver plot image.
        Format: [<num_rows>, <num_cols>]
        Example: [6,7]
        If None, the QA code computes the strides values
        based on `longest_side_max` and by squaring the pixels.
        Defaults to None.
    arrow_density : float, optional
        Number of arrows (vectors) to plot per the longest edge of the raster.
        Defaults to 20.
    arrow_scaling : float or None, optional
        Scales the length of the arrow inversely.
        Number of data units per arrow length unit, e.g., m/s per plot width;
        a smaller scale parameter makes the arrow longer.
        If None, a simple autoscaling algorithm is used, based on the average
        vector length and the number of vectors.
        Defaults to None.
        See: The `scaling` parameter for `matplotlib.axes.Axes.quiver`.
    longest_side_max : int, optional
        The maximum number of pixels allowed for the longest side of the final
        2D decimated browse image. If None, no downscaling will occur
        (other than to form square pixels).
        Defaults to 2048 pixels.
    """

    cbar_min_max: Optional[Sequence[float]] = field(
        default=None,
        metadata={
            "yaml_attrs": YamlAttrs(
                name="colorbar_min_max",
                descr="""The vmin and vmax values to generate the quiver plot.
                The magnitude of the offsets is clipped to this interval,
                which (in turn) is used for the interval of the colorbar.
                If None, the interval is computed using the min and max 
                magnitudes of along track and slant range offset.""",
            )
        },
    )

    _decimation_descr_template: ClassVar[
        str
    ] = """Stride along each axis of the Frequency %s
        image arrays for decimating the quiver plot image for the browse PNG.
        This takes precedence over `longest_side_max`.
        Format: [<num_rows>, <num_cols>]
        Example: [6,7]
        If None, QA-SAS will compute the decimation strides
        based on `longest_side_max` and creating square pixels."""

    browse_decimation_freqa: Optional[Sequence[int]] = field(
        default=None,
        metadata={
            "yaml_attrs": YamlAttrs(
                name="browse_decimation_freqa",
                descr=_decimation_descr_template % "A",
            )
        },
    )

    browse_decimation_freqb: Optional[Sequence[int]] = field(
        default=None,
        metadata={
            "yaml_attrs": YamlAttrs(
                name="browse_decimation_freqb",
                descr=_decimation_descr_template % "B",
            )
        },
    )

    arrow_density: float = field(
        default=20,
        metadata={
            "yaml_attrs": YamlAttrs(
                name="arrow_density",
                descr="""Number of arrows (vectors) to plot per the
                longest edge of the raster.""",
            )
        },
    )

    arrow_scaling: Optional[float] = field(
        default=None,
        metadata={
            "yaml_attrs": YamlAttrs(
                name="arrow_scaling",
                descr="""Scales the length of the arrow inversely.
        Number of data units per arrow length unit, e.g., m/s per plot width;
        a smaller scale parameter makes the arrow longer.
        If None, a simple autoscaling algorithm is used, based on the average
        vector length and the number of vectors.
        See: The `scaling` parameter for `matplotlib.axes.Axes.quiver()`.""",
            )
        },
    )

    longest_side_max: int = field(
        default=2048,
        metadata={
            "yaml_attrs": YamlAttrs(
                name="longest_side_max",
                descr="""The maximum number of pixels allowed for the longest
                side of the final 2D browse image. If `decimation_freqX` is not
                None, then `longest_side_max` will be ignored.""",
            )
        },
    )

    def __post_init__(self):
        # VALIDATE INPUTS

        self._validate_pair_of_numeric(
            param_value=self.cbar_min_max,
            param_name="colorbar_min_max",
            min=None,
            max=None,
            none_is_valid_value=True,
            strictly_increasing=True,
        )

        self._validate_pair_of_numeric(
            param_value=self.browse_decimation_freqa,
            param_name="browse_decimation_freqa",
            min=1,
            max=None,
            none_is_valid_value=True,
            strictly_increasing=False,
        )

        self._validate_pair_of_numeric(
            param_value=self.browse_decimation_freqb,
            param_name="browse_decimation_freqb",
            min=1,
            max=None,
            none_is_valid_value=True,
            strictly_increasing=False,
        )

        if not isinstance(self.arrow_density, (int, float)):
            arrow_density = self.arrow_density
            raise TypeError(
                f"{arrow_density=} and has type {type(arrow_density)}, but must"
                " be an int or float."
            )
        if self.arrow_density < 1:
            arrow_density = self.arrow_density
            raise ValueError(f"{arrow_density=}, but must be greater than 1.")

        if self.arrow_scaling is None:
            pass
        elif isinstance(self.arrow_scaling, (int, float)):
            if self.arrow_scaling <= 0.0:
                arrow_scaling = self.arrow_scaling
                raise ValueError(
                    f"{arrow_scaling=}, but must be greater than 0.0."
                )
        else:
            arrow_scaling = self.arrow_scaling
            raise TypeError(
                f"{arrow_scaling=} and has type {type(arrow_scaling)}, but must"
                " be an int or float or None."
            )

        # validate longest_side_max
        if not isinstance(self.longest_side_max, int):
            longest_side_max = self.longest_side_max
            raise TypeError(
                f"{longest_side_max=} and has type {type(longest_side_max)},"
                " but must be an int."
            )
        if self.longest_side_max <= 0:
            raise ValueError(
                f"`longest_side_max` must be positive: {self.longest_side_max}"
            )


@dataclass(frozen=True)
class ROFFQuiverParamGroup(QuiverParamGroup):
    @staticmethod
    def get_path_to_group_in_runconfig():
        return [
            "runconfig",
            "groups",
            "qa",
            "roff",
            "qa_reports",
            "quiver_plots",
        ]


@dataclass(frozen=True)
class GOFFQuiverParamGroup(QuiverParamGroup):
    @staticmethod
    def get_path_to_group_in_runconfig():
        return [
            "runconfig",
            "groups",
            "qa",
            "goff",
            "qa_reports",
            "quiver_plots",
        ]


@dataclass(frozen=True)
class VarianceLayersParamGroup(YamlParamGroup):
    """
    Parameters to generate Variance Layer Plots for ROFF and GOFF.

    Parameters
    ----------
    cbar_min_max : None or pair of float or int, optional
        The vmin and vmax values to generate the plots
        for the variance layers for ROFF and GOFF.
        The magnitude of these layers is clipped to this interval,
        which (in turn) is used for the interval of the colorbar.
        If None, the interval is computed using the min and max
        magnitudes of along track, slant range, and cross offset variances
        Defaults to [0.0, 0.1].
    """

    cbar_min_max: Optional[Sequence[float]] = field(
        default=(0.0, 0.1),
        metadata={
            "yaml_attrs": YamlAttrs(
                name="colorbar_min_max",
                descr="""The vmin and vmax values to generate the plots
                for the variance layers for ROFF and GOFF.
                The magnitude of these layers is clipped to this interval,
                which (in turn) is used for the interval of the colorbar.
                If None, the interval is computed using the min and max 
                magnitudes of along track, slant range, and cross offset variances.""",
            )
        },
    )

    def __post_init__(self):
        # VALIDATE INPUTS

        self._validate_pair_of_numeric(
            param_value=self.cbar_min_max,
            param_name="colorbar_min_max",
            min=0.0,
            max=None,
            none_is_valid_value=True,
            strictly_increasing=True,
        )


@dataclass(frozen=True)
class ROFFVarianceLayersParamGroup(VarianceLayersParamGroup):
    @staticmethod
    def get_path_to_group_in_runconfig():
        return [
            "runconfig",
            "groups",
            "qa",
            "roff",
            "qa_reports",
            "variance_plots",
        ]


@dataclass(frozen=True)
class GOFFVarianceLayersParamGroup(VarianceLayersParamGroup):
    @staticmethod
    def get_path_to_group_in_runconfig():
        return [
            "runconfig",
            "groups",
            "qa",
            "goff",
            "qa_reports",
            "variance_plots",
        ]


@dataclass
class RIFGRootParamGroup(RootParamGroup):
    """
    Dataclass of all *ParamGroup objects to process QA for NISAR RIFG products.

    `workflows` is the only required parameter; this *ParamGroup contains
    boolean attributes that indicate which QA workflows to run.

    All other parameters are optional, but they each correspond to (at least)
    one of the QA workflows. Based on the workflows set to True in
    `workflows`, certain others of these parameters will become required.

    Parameters
    ----------
    workflows : RIFGWorkflowsParamGroup
        QA Workflows parameters.
    input_f : RIFGInputFileGroupParamGroup or None, optional
        Input File Group parameters.
    prodpath : RIFGProductPathGroupParamGroup or None, optional
        Product Path Group parameters.
    hsi : RIFGHSIImageParamGroup or None, optional
        HSI Image Group parameters.
    """

    workflows: RIFGWorkflowsParamGroup

    # Shared parameters
    input_f: Optional[RIFGInputFileGroupParamGroup] = None
    prodpath: Optional[RIFGProductPathGroupParamGroup] = None

    hsi: Optional[RIFGHSIImageParamGroup] = None

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
                param_grp_cls_obj=RIFGInputFileGroupParamGroup,
            ),
            Grp(
                flag_param_grp_req=flag_any_workflows_true,
                root_param_grp_attr_name="prodpath",
                param_grp_cls_obj=RIFGProductPathGroupParamGroup,
            ),
            Grp(
                flag_param_grp_req=workflows.qa_reports,
                root_param_grp_attr_name="hsi",
                param_grp_cls_obj=RIFGHSIImageParamGroup,
            ),
        )

        return grps_to_parse

    @staticmethod
    def get_order_of_groups_in_yaml():
        # This order determines the order
        # the groups will appear in the runconfig.
        return (
            RIFGInputFileGroupParamGroup,
            RIFGProductPathGroupParamGroup,
            RIFGWorkflowsParamGroup,
            RIFGHSIImageParamGroup,
        )


@dataclass
class RUNWRootParamGroup(RootParamGroup):
    """
    Dataclass of all *ParamGroup objects to process QA for NISAR RUNW products.

    `workflows` is the only required parameter; this *ParamGroup contains
    boolean attributes that indicate which QA workflows to run.

    All other parameters are optional, but they each correspond to (at least)
    one of the QA workflows. Based on the workflows set to True in
    `workflows`, certain others of these parameters will become required.

    Parameters
    ----------
    workflows : RUNWWorkflowsParamGroup
        QA Workflows parameters.
    input_f : RUNWInputFileGroupParamGroup or None, optional
        Input File Group parameters.
    prodpath : RUNWProductPathGroupParamGroup or None, optional
        Product Path Group parameters.
    hsi : RUNWHSIImageParamGroup or None, optional
        HSI Image Group parameters.
    unw_phs_img : RUNWPhaseImageParamGroup or None, optional
        Unwrapped Phase Image Group parameters.
    """

    # Shared parameters
    workflows: RUNWWorkflowsParamGroup
    input_f: Optional[RUNWInputFileGroupParamGroup] = None
    prodpath: Optional[RUNWProductPathGroupParamGroup] = None

    hsi: Optional[RUNWHSIImageParamGroup] = None
    unw_phs_img: Optional[RUNWPhaseImageParamGroup] = None

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
                param_grp_cls_obj=RUNWInputFileGroupParamGroup,
            ),
            Grp(
                flag_param_grp_req=flag_any_workflows_true,
                root_param_grp_attr_name="prodpath",
                param_grp_cls_obj=RUNWProductPathGroupParamGroup,
            ),
            Grp(
                flag_param_grp_req=workflows.qa_reports,
                root_param_grp_attr_name="unw_phs_img",
                param_grp_cls_obj=RUNWPhaseImageParamGroup,
            ),
            Grp(
                flag_param_grp_req=workflows.qa_reports,
                root_param_grp_attr_name="hsi",
                param_grp_cls_obj=RUNWHSIImageParamGroup,
            ),
        )

        return grps_to_parse

    @staticmethod
    def get_order_of_groups_in_yaml():
        # This order determines the order
        # the groups will appear in the runconfig.
        return (
            RUNWInputFileGroupParamGroup,
            RUNWProductPathGroupParamGroup,
            RUNWWorkflowsParamGroup,
            RUNWPhaseImageParamGroup,
            RUNWHSIImageParamGroup,
        )


@dataclass
class GUNWRootParamGroup(RootParamGroup):
    """
    Dataclass of all *ParamGroup objects to process QA for NISAR GUNW products.

    `workflows` is the only required parameter; this *ParamGroup contains
    boolean attributes that indicate which QA workflows to run.

    All other parameters are optional, but they each correspond to (at least)
    one of the QA workflows. Based on the workflows set to True in
    `workflows`, certain others of these parameters will become required.

    Parameters
    ----------
    workflows : GUNWWorkflowsParamGroup
        QA Workflows parameters.
    input_f : GUNWInputFileGroupParamGroup or None, optional
        Input File Group parameters.
    prodpath : GUNWProductPathGroupParamGroup or None, optional
        Product Path Group parameters.
    hsi : GUNWHSIImageParamGroup or None, optional
        HSI Image Group parameters.
    unw_phs_img : GUNWPhaseImageParamGroup or None, optional
        Unwrapped Phase Image Group parameters.
    """

    workflows: GUNWWorkflowsParamGroup

    # Shared parameters
    input_f: Optional[GUNWInputFileGroupParamGroup] = None
    prodpath: Optional[GUNWProductPathGroupParamGroup] = None

    hsi: Optional[GUNWHSIImageParamGroup] = None
    unw_phs_img: Optional[GUNWPhaseImageParamGroup] = None

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
                param_grp_cls_obj=GUNWInputFileGroupParamGroup,
            ),
            Grp(
                flag_param_grp_req=flag_any_workflows_true,
                root_param_grp_attr_name="prodpath",
                param_grp_cls_obj=GUNWProductPathGroupParamGroup,
            ),
            Grp(
                flag_param_grp_req=workflows.qa_reports,
                root_param_grp_attr_name="unw_phs_img",
                param_grp_cls_obj=GUNWPhaseImageParamGroup,
            ),
            Grp(
                flag_param_grp_req=workflows.qa_reports,
                root_param_grp_attr_name="hsi",
                param_grp_cls_obj=GUNWHSIImageParamGroup,
            ),
        )

        return grps_to_parse

    @staticmethod
    def get_order_of_groups_in_yaml():
        # This order determines the order
        # the groups will appear in the runconfig.
        return (
            GUNWInputFileGroupParamGroup,
            GUNWProductPathGroupParamGroup,
            GUNWWorkflowsParamGroup,
            GUNWPhaseImageParamGroup,
            GUNWHSIImageParamGroup,
        )


@dataclass
class ROFFRootParamGroup(RootParamGroup):
    """
    Dataclass of all *ParamGroup objects to process QA for NISAR ROFF products.

    `workflows` is the only required parameter; this *ParamGroup contains
    boolean attributes that indicate which QA workflows to run.

    All other parameters are optional, but they each correspond to (at least)
    one of the QA workflows. Based on the workflows set to True in
    `workflows`, certain others of these parameters will become required.

    Parameters
    ----------
    workflows : ROFFWorkflowsParamGroup
        QA Workflows parameters.
    input_f : ROFFInputFileGroupParamGroup or None, optional
        Input File Group parameters.
    prodpath : ROFFProductPathGroupParamGroup or None, optional
        Product Path Group parameters.
    quiver : ROFFQuiverParamGroup or None, optional
        Quiver plots and browse image group parameters.
    variances : ROFFVarianceLayersParamGroup or None, optional
        Parameters for *Variance layers' plots.
    """

    workflows: ROFFWorkflowsParamGroup

    # Shared parameters
    input_f: Optional[ROFFInputFileGroupParamGroup] = None
    prodpath: Optional[ROFFProductPathGroupParamGroup] = None

    quiver: Optional[ROFFQuiverParamGroup] = None
    variances: Optional[ROFFVarianceLayersParamGroup] = None

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
                param_grp_cls_obj=ROFFInputFileGroupParamGroup,
            ),
            Grp(
                flag_param_grp_req=flag_any_workflows_true,
                root_param_grp_attr_name="prodpath",
                param_grp_cls_obj=ROFFProductPathGroupParamGroup,
            ),
            Grp(
                flag_param_grp_req=workflows.qa_reports,
                root_param_grp_attr_name="quiver",
                param_grp_cls_obj=ROFFQuiverParamGroup,
            ),
            Grp(
                flag_param_grp_req=workflows.qa_reports,
                root_param_grp_attr_name="variances",
                param_grp_cls_obj=ROFFVarianceLayersParamGroup,
            ),
        )

        return grps_to_parse

    @staticmethod
    def get_order_of_groups_in_yaml():
        # This order determines the order
        # the groups will appear in the runconfig.
        return (
            ROFFInputFileGroupParamGroup,
            ROFFProductPathGroupParamGroup,
            ROFFWorkflowsParamGroup,
            ROFFQuiverParamGroup,
            ROFFVarianceLayersParamGroup,
        )


@dataclass
class GOFFRootParamGroup(RootParamGroup):
    """
    Dataclass of all *ParamGroup objects to process QA for NISAR GOFF products.

    `workflows` is the only required parameter; this *ParamGroup contains
    boolean attributes that indicate which QA workflows to run.

    All other parameters are optional, but they each correspond to (at least)
    one of the QA workflows. Based on the workflows set to True in
    `workflows`, certain others of these parameters will become required.

    Parameters
    ----------
    workflows : GOFFWorkflowsParamGroup
        QA Workflows parameters.
    input_f : GOFFInputFileGroupParamGroup or None, optional
        Input File Group parameters.
    prodpath : GOFFProductPathGroupParamGroup or None, optional
        Product Path Group parameters.
    quiver : GOFFQuiverParamGroup or None, optional
        Quiver plots and browse image group parameters.
    variances : GOFFVarianceLayersParamGroup or None, optional
        Parameters for *Variance layers' plots.
    """

    workflows: GOFFWorkflowsParamGroup

    # Shared parameters
    input_f: Optional[GOFFInputFileGroupParamGroup] = None
    prodpath: Optional[GOFFProductPathGroupParamGroup] = None

    quiver: Optional[GOFFQuiverParamGroup] = None
    variances: Optional[GOFFVarianceLayersParamGroup] = None

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
                param_grp_cls_obj=GOFFInputFileGroupParamGroup,
            ),
            Grp(
                flag_param_grp_req=flag_any_workflows_true,
                root_param_grp_attr_name="prodpath",
                param_grp_cls_obj=GOFFProductPathGroupParamGroup,
            ),
            Grp(
                flag_param_grp_req=workflows.qa_reports,
                root_param_grp_attr_name="quiver",
                param_grp_cls_obj=GOFFQuiverParamGroup,
            ),
            Grp(
                flag_param_grp_req=workflows.qa_reports,
                root_param_grp_attr_name="variances",
                param_grp_cls_obj=GOFFVarianceLayersParamGroup,
            ),
        )

        return grps_to_parse

    @staticmethod
    def get_order_of_groups_in_yaml():
        # This order determines the order
        # the groups will appear in the runconfig.
        return (
            GOFFInputFileGroupParamGroup,
            GOFFProductPathGroupParamGroup,
            GOFFWorkflowsParamGroup,
            GOFFQuiverParamGroup,
            GOFFVarianceLayersParamGroup,
        )


__all__ = nisarqa.get_all(__name__, objects_to_skip)
