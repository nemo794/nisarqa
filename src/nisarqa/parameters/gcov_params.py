from dataclasses import dataclass, field, fields
from typing import Optional

import nisarqa
from nisarqa.parameters.nisar_params import *
# TODO Remove the rslc_caltools_params imports after re-org of code
from nisarqa.parameters.rslc_caltools_params import (
    BackscatterImageParamGroup, HistogramParamGroup, InputFileGroupParamGroup,
    ProductPathGroupParamGroup)

objects_to_skip = nisarqa.get_all(__name__)


@dataclass(frozen=True)
class GCOVBackscatterImageParamGroup(BackscatterImageParamGroup):
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
    percentile_for_clipping : float, optional
        Defines the percentile range that the image array will be clipped to
        and that the colormap covers. Must be in the range [0.0, 100.0].
        Defaults to [0.0, 95.0].
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

    percentile_for_clipping: float = field(
        default=(0.0, 95.0),  # overwrite parent class' default value
        metadata=BackscatterImageParamGroup.get_attribute_metadata(
            attribute_name="percentile_for_clipping"
        ),
    )


@dataclass
class GCOVRootParamGroup(RootParamGroup):
    """
    Dataclass of all *ParamGroup objects to process QA for NISAR GCOV products.

    `workflows` is the only required parameter; this *ParamGroup contains
    boolean attributes that indicate which QA workflows to run.

    All other parameters are optional, but they each correspond to (at least)
    one of the QA workflows. Based on the workflows set to True in
    `workflows`, certain others of these parameters will become required.

    Parameters
    ----------
    workflows : WorkflowsParamGroup
        QA Workflows parameters
    input_f : InputFileGroupParamGroup or None, optional
        Input File Group parameters for QA
    prodpath : ProductPathGroupParamGroup or None, optional
        Product Path Group parameters for QA
    backscatter_img : GCOVBackscatterImageParamGroup or None, optional
        Covariance Term Magnitude Image Group parameters for GCOV QA
    histogram : GCOVHistogramParamGroup or None, optional
        Histogram Group parameters for GCOV QA
    """

    # Shared parameters
    input_f: Optional[InputFileGroupParamGroup] = None
    prodpath: Optional[ProductPathGroupParamGroup] = None

    # QA parameters
    backscatter_img: Optional[GCOVBackscatterImageParamGroup] = None
    histogram: Optional[HistogramParamGroup] = None

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
                param_grp_cls_obj=GCOVBackscatterImageParamGroup,
            ),
            Grp(
                flag_param_grp_req=workflows.qa_reports,
                root_param_grp_attr_name="histogram",
                param_grp_cls_obj=HistogramParamGroup,
            ),
        )

        return grps_to_parse

    @staticmethod
    def get_order_of_groups_in_yaml():
        # This order determines the order
        # the groups will appear in the runconfig.
        return (
            InputFileGroupParamGroup,
            ProductPathGroupParamGroup,
            WorkflowsParamGroup,
            GCOVBackscatterImageParamGroup,
            HistogramParamGroup,
        )


__all__ = nisarqa.get_all(__name__, objects_to_skip)
