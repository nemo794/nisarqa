from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Optional

import nisarqa
from ruamel.yaml import CommentedMap as CM
from ruamel.yaml import CommentedSeq as CS


@dataclass
class Param:
    '''
    Data structure to hold information about a QA input parameter.

    Parameters
    ----------
    name : str
        Name of the parameter. It is suggested that this be the exact
        name of the parameter in a QA runconfig and formatted in snakecase
        and no spaces.
    value : Any
        The value assigned to parameter `name`
    units : str
        Units of `ds_data`; will be stored in a `units` attribute
        for the new Dataset.
        For NISAR datasets, use this convention:
            - If the values have dimensions, use CF-compliant names (e.g. 'meters')
            - If the values are numeric but dimensionless (e.g. ratios),
                set `ds_units` to 'unitless'
            - If the values are inherently descriptive and have no units
                (e.g. a file name, or a list of frequency names such as ['A', 'B']),
                then set `ds_units` to None so that no units attribute
                is created.
        Defaults to None (no units attribute will be created)
    short_descr : str, optional
        Short one-line description of this parameter. Preferably,
        this should meet CF conventions; can be used for the stats.h5 file.
        Defaults to None.
    long_descr : str, optional
        Long description of the parameter describing the type, defaults,
        usage, examples, etc. This is what would appear in a docstring
        or runconfig. If not provided, or if `None` is provided,
        will defaults to the `short_descr`.
    '''
    name: str
    val: Any
    units: Optional[str] = None
    short_descr: Optional[str] = None
    long_descr: Optional[str] = None

    def __post_init__(self):
        if self.long_descr == None:
            self.long_descr = self.short_descr


@dataclass
class BaseParams(ABC):
    '''Abstract Base Class for creating *Params dataclasses.'''

    @staticmethod
    @abstractmethod
    def get_path_to_group_in_runconfig():
        '''
        Return the path to the group in the runconfig.

        For example, the dynamic ancillary file group in the NISAR runconfig
        is located via this nesting path:
              runconfig > groups > dynamic_ancillary_file_group
        So, get_path_to_group_in_runconfig() will return
              ['runconfig', 'groups', 'dynamic_ancillary_file_group']
        The first entry is a root group, and the final entry is 
        the name of the group.

        Returns
        -------
        path_to_group : list of str
            The path in the runconfig file from root to group for this
            Params dataclass.
            (See docstring for example.)
        '''
        pass

    @staticmethod
    @abstractmethod
    def populate_runcfg(yaml_obj):
        '''Update the provided ruamel.yaml object with select attributes
        (parameters) of this instance of the dataclass for use in a
        NISAR product QA runconfig file.

        Only default values will be used.

        Parameters
        ----------
        runconfig_cm : ruamel.yaml.comments.CommentedMap
            The base commented map; will be updated with the attributes
            from this dataclass that are in the QA runconfig file

        Notes
        -----
        Reference: https://stackoverflow.com/questions/56471040/add-a-comment-in-list-element-in-ruamel-yaml
        '''
        pass

    @abstractmethod
    def write_params_to_h5(self, h5_file, bands=('LSAR')):
        '''Update the provided HDF5 file handle with select attributes
        (parameters) of this instance of the dataclass.
        '''
        pass


    def add_param_to_cm(self, params_cm, param_attr):
        '''
        Add a Param attribute to a Commented Map.

        This can be used to add a new parameter into a group for a runconfig.

        Parameters
        ----------
        params_cm : ruamel.yaml.comments.CommentedMap
            Commented Map for a parameter group in the runconfig.
            Will be updated to include `param_attr`.
        param_attr : Param
            Param object that will be added to the `params_cm` Commented Map
        '''
        if not isinstance(param_attr, Param):
            raise ValueError(f'`param_attr` is type {type(param_attr)} '
                              'but must be type Param')

        # set indentation for displaying the comments correctly in the yaml
        comment_indent = len(self.get_path_to_group_in_runconfig()) * 4

        # To have ruamel.yaml display list values as a list in the runconfig,
        # use CommentedSeq
        # https://stackoverflow.com/questions/56937691/making-yaml-ruamel-yaml-always-dump-lists-inline
        if isinstance(param_attr.val, (list, tuple)):
            seq = CS()
            seq.fa.set_flow_style()
            for item in param_attr.val:
                seq.append(item)
            val = seq
        else:
            val = param_attr.val

        # Add attribute to the group
        name = param_attr.name
        params_cm[name] = val
        comment = param_attr.long_descr
        for line in nisarqa.multi_line_string_iter(comment):
            params_cm.yaml_set_comment_before_after_key(
                name, before=line, indent=comment_indent)


    def add_param_group_to_runconfig(self, yaml_cm, params_cm):
        '''
        Add a new group of parameters to a yaml Commented Map
        along the nested directory structure specified by 
        `self.get_path_to_group_in_runconfig()`.
        
        In the process, this function will check that the nested structure
        specified in get_path_to_group_in_runconfig() exists.
        If not, the missing commented maps are created.

        Parameters
        ----------
        yaml_cm : ruamel.yaml.comments.CommentedMap
            Commented Map object that holds the runconfig information.
            This will be updated to include `param_cm` added 
            along the path specified by `get_path_to_group_in_runconfig()`.
        params_cm : ruamel.yaml.comments.CommentedMap
            A Commented Map populated with the parameters, etc. to be
            added to `yaml_cm` in the group specified by 
            `get_path_to_group_in_runconfig()`.
        '''
        path = self.get_path_to_group_in_runconfig()

        # start at root Commented Map
        parent_cm = yaml_cm

        # Traverse to the direct parent CM group for this params_cm.
        # Ensure all parent groups for this params_cm exist
        for group in path[:-1]:
            # If group does not exist, create it
            if group not in parent_cm:
                parent_cm[group] = CM()

                # For readability, add a newline before this new group
                parent_cm.yaml_set_comment_before_after_key(group, before='\n')

            # Move into the next nested group
            parent_cm = parent_cm[group]
        
        # Attach new params_cm group into the runconfig yaml
        parent_cm[path[-1]] = params_cm

        # For readability, add a newline before the new params_cm group
        parent_cm.yaml_set_comment_before_after_key(path[-1], before='\n')
