from dataclasses import dataclass, fields
from typing import Optional

import nisarqa
from nisarqa import (
    BackscatterImageParamGroup,
    HistogramParamGroup,
    InputFileGroupParamGroup,
    ProductPathGroupParamGroup,
    RootParamGroup,
    ValidationGroupParamGroup,
    WorkflowsParamGroup,
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
    validation : ValidationGroupParamGroup or None, optional
        Validation Group parameters for QA
    backscatter_img : BackscatterImageParamGroup or None, optional
        Backscatter Image Group parameters for SLC QA
    histogram : HistogramParamGroup or None, optional
        Histogram Group parameters for RSLC or GSLC QA
    """

    # QA parameters
    backscatter_img: Optional[BackscatterImageParamGroup] = None
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
                flag_param_grp_req=workflows.validate,
                root_param_grp_attr_name="validation",
                param_grp_cls_obj=ValidationGroupParamGroup,
            ),
            Grp(
                flag_param_grp_req=workflows.qa_reports,
                root_param_grp_attr_name="backscatter_img",
                param_grp_cls_obj=BackscatterImageParamGroup,
            ),
            Grp(
                flag_param_grp_req=workflows.qa_reports,
                root_param_grp_attr_name="histogram",
                param_grp_cls_obj=nisarqa.HistogramParamGroup,
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
            ValidationGroupParamGroup,
            BackscatterImageParamGroup,
            HistogramParamGroup,
        )


__all__ = nisarqa.get_all(__name__, objects_to_skip)
