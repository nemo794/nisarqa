from dataclasses import dataclass, fields
from typing import Optional

import nisarqa
from nisarqa.parameters.nisar_params import *

# TODO Remove the rslc_caltools_params imports after re-org of code
from nisarqa.parameters.rslc_caltools_params import (
    InputFileGroupParamGroup,
    ProductPathGroupParamGroup,
    SLCHistogramParamGroup,
    SLCPowerImageParamGroup,
)

objects_to_skip = nisarqa.get_all(__name__)


@dataclass
class GSLCRootParamGroup(RootParamGroup):
    """
    Dataclass of all *ParamGroup objects to process QA for NISAR GSLC products.

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
    power_img : SLCPowerImageParamGroup or None, optional
        Power Image Group parameters for SLC QA
    """

    # Shared parameters
    input_f: Optional[InputFileGroupParamGroup] = None
    prodpath: Optional[ProductPathGroupParamGroup] = None

    # QA parameters
    power_img: Optional[SLCPowerImageParamGroup] = None
    histogram: Optional[SLCHistogramParamGroup] = None

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
                param_grp_cls_obj=SLCPowerImageParamGroup,
            ),
            Grp(
                flag_param_grp_req=workflows.qa_reports,
                root_param_grp_attr_name="histogram",
                param_grp_cls_obj=nisarqa.SLCHistogramParamGroup,
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
            SLCPowerImageParamGroup,
            SLCHistogramParamGroup,
        )


__all__ = nisarqa.get_all(__name__, objects_to_skip)
