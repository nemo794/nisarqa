import os
import sys
from dataclasses import dataclass, field, fields
from typing import ClassVar, Iterable, Optional, Type, Union

import nisarqa
import numpy as np
from nisarqa.parameters.nisar_params import *
from nisarqa.parameters.rslc_caltools_params import *  # TODO Remove this import after re-org of code
from numpy.typing import ArrayLike
from ruamel.yaml import YAML
from ruamel.yaml import CommentedMap as CM

objects_to_skip = nisarqa.get_all(__name__)



@dataclass
class GSLCRootParamGroup(RootParamGroup):
    '''
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
    '''

    # Shared parameters
    input_f: Optional[InputFileGroupParamGroup] = None
    prodpath: Optional[ProductPathGroupParamGroup] = None

    # QA parameters
    power_img: Optional[SLCPowerImageParamGroup] = None


    @staticmethod
    def get_mapping_of_workflows2param_grps(workflows):
        Grp = RootParamGroup.ReqParamGrp  # class object for our named tuple

        flag_any_workflows_true = any([getattr(workflows, field.name) \
                            for field in fields(workflows)])

        grps_to_parse = (
            Grp(flag_param_grp_req=flag_any_workflows_true, 
                root_param_grp_attr_name='input_f',
                param_grp_cls_obj=InputFileGroupParamGroup),

            Grp(flag_param_grp_req=flag_any_workflows_true, 
                root_param_grp_attr_name='prodpath',
                param_grp_cls_obj=ProductPathGroupParamGroup),

            Grp(flag_param_grp_req=workflows.qa_reports, 
                root_param_grp_attr_name='power_img',
                param_grp_cls_obj=SLCPowerImageParamGroup)
            )

        return grps_to_parse


    @staticmethod
    def get_order_of_groups_in_yaml():
        # This order determines the order
        # the groups will appear in the runconfig.
        return (InputFileGroupParamGroup,
                ProductPathGroupParamGroup,
                WorkflowsParamGroup,
                SLCPowerImageParamGroup
                )


__all__ = nisarqa.get_all(__name__, objects_to_skip)
