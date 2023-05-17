from dataclasses import dataclass, field, fields
from typing import Optional

import nisarqa
from nisarqa.parameters.nisar_params import *
# TODO Remove the rslc_caltools_params imports after re-org of code
from nisarqa.parameters.rslc_caltools_params import (
    HistogramParamGroup, InputFileGroupParamGroup, PowerImageParamGroup,
    ProductPathGroupParamGroup)

objects_to_skip = nisarqa.get_all(__name__)


@dataclass(frozen=True)
class GCOVPowerImageParamGroup(PowerImageParamGroup):
    """
    Parameters to generate RSLC or GSLC Power Images and Browse Image.

    This corresponds to the `qa_reports: power_image` runconfig group.

    Parameters
    ----------
    linear_units : bool, optional
        True to compute power image in linear units, False for decibel units.
        Defaults to True.
    input_raster_represents_power : bool, optional
        The input dataset rasters associated with these Power Image parameters
        should have their pixel values represent either magnitude or power,
        where power = magnitude^2
        If `input_raster_represents_power` is `True`, then QA SAS assumes
        that the input data already represents power. If `False`, then
        QA SAS will square the input data values to convert from magnitude
        to power.
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
    pow_units : Param
        Units of the power image.
        If `linear_units` is True, this will be set to 'linear'.
        If `linear_units` is False, this will be set to 'dB'.
    """

    percentile_for_clipping: float = field(
        default=(0.0, 95.0),  # overwrite parent class' default value
        metadata=PowerImageParamGroup.get_attribute_metadata(
            attribute_name="percentile_for_clipping"
        ),
    )

    input_raster_represents_power: bool = field(
        default=True,  # overwrite parent class' default value
        metadata=PowerImageParamGroup.get_attribute_metadata(
            attribute_name="input_raster_represents_power"
        ),
    )

@dataclass(frozen=True)
class GCOVHistogramParamGroup(HistogramParamGroup):
    """
    Parameters to generate the RSLC or GSLC Power and Phase Histograms;
    this corresponds to the `qa_reports: histogram` runconfig group.

    Parameters
    ----------
    decimation_ratio : pair of int, optional
        The step size to decimate the input array for computing
        the power and phase histograms.
        For example, (2,3) means every 2nd azimuth line and
        every 3rd range sample will be used to compute the histograms.
        Defaults to (10,10).
        Format: (<azimuth>, <range>)
    input_raster_represents_power : bool, optional
        The input dataset rasters associated with these histogram parameters
        should have their pixel values represent either magnitude or power,
        with terms defined as power = magnitude^2 .
        (In practice, this is computed as power = abs(<pixel value>)^2 .)
        If `input_raster_represents_power` is `True`, then QA SAS assumes
        that the input data already has already been squared.
        If `False`, then QA SAS will handle the full computation to power.
        Defaults to True.
    pow_histogram_bin_edges_range : pair of int or float, optional
        The dB range for the power histogram's bin edges. Endpoint will
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
    pow_bin_edges : numpy.ndarray
        The bin edges (including endpoint) to use when computing
        the power histograms. Will be set to 100 uniformly-spaced bins
        in range `pow_histogram_bin_edges_range`, including endpoint.
    phs_bin_edges : numpy.ndarray
        The bin edges (including endpoint) to use when computing
        the phase histograms.
        If `phs_in_radians` is True, this will be set to 100
        uniformly-spaced bins in range [-pi,pi], including endpoint.
        If `phs_in_radians` is False, this will be set to 100
        uniformly-spaced bins in range [-180,180], including endpoint.
    """
    input_raster_represents_power: bool = field(
        default=True,  # overwrite parent class' default value
        metadata=HistogramParamGroup.get_attribute_metadata(
            attribute_name="input_raster_represents_power"
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
    power_img : GCOVPowerImageParamGroup or None, optional
        Covariance Term Magnitude Image Group parameters for GCOV QA
    histogram : GCOVHistogramParamGroup or None, optional
        Histogram Group parameters for GCOV QA
    """

    # Shared parameters
    input_f: Optional[InputFileGroupParamGroup] = None
    prodpath: Optional[ProductPathGroupParamGroup] = None

    # QA parameters
    power_img: Optional[GCOVPowerImageParamGroup] = None
    histogram: Optional[GCOVHistogramParamGroup] = None

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
                root_param_grp_attr_name="power_img",
                param_grp_cls_obj=GCOVPowerImageParamGroup,
            ),
            Grp(
                flag_param_grp_req=workflows.qa_reports,
                root_param_grp_attr_name="histogram",
                param_grp_cls_obj=GCOVHistogramParamGroup,
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
            GCOVPowerImageParamGroup,
            GCOVHistogramParamGroup,
        )


__all__ = nisarqa.get_all(__name__, objects_to_skip)
