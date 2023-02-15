import inspect
from abc import ABC, abstractmethod
from dataclasses import dataclass, fields
from typing import ClassVar, Generic, TypeVar

import nisarqa
from ruamel.yaml import CommentedMap as CM
from ruamel.yaml import CommentedSeq as CS

objects_to_skip = nisarqa.get_all(name=__name__)


# 1. Param (just a value) + YamlAttrs
# 2. Param + HDF5Attrs
# 3. Param + YamlAttrs + HDF5Attrs

# Param
# YamlParam
# HDF5Param
# YamlHDF5Param

# # --------------------
T = TypeVar("T")


@dataclass
class YamlAttrs:
    '''
    Dataclass to hold information about a parameter for the runconfig yaml.

    The intention is for this dataclass to be used in conjunction with 


    Parameters
    ----------
    name : str
        Name of the parameter. It is suggested that this be the exact
        name of the parameter in a QA runconfig and formatted in snakecase
        and no spaces.
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
    descr: str


@dataclass
class HDF5Attrs:
    name: str
    units: str
    descr: str
    path: str


@dataclass
class Param(Generic[T]):
    '''
    Data structure to hold the parameter value of a QA input parameter.

    This dataclass can be used in conjunction with `YamlAttrs`, `HDF5Attrs`,
    etc. to collect metadata-like attributes about the QA input parameter
    in the `YamlParam`, `HDF5Param`, etc. dataclasses.


    Parameters
    ----------
    val : Any
        The value assigned to parameter `name`
    '''
    val: T


@dataclass
class YamlParam(Param[T]):
    '''
    Data structure to hold information about a QA input parameter
    as related to the runconfig yaml.

    Parameters
    ----------
    val : Any
        The value assigned to parameter `name`.
        This parameter is inherited from the Param base class.
    yaml_attrs : YamlAttrs
        Attributes of `val` that will be used for reading/writing the
        runconfig yaml file.
    '''
    yaml_attrs: YamlAttrs


@dataclass
class HDF5Param(Param[T]):
    '''
    Data structure to hold information about a QA input parameter
    as related to the output STATS.h5 HDF5 file.

    If `val` is used in both the runconfig yaml and the output STATS.h5
    file, it is suggested to use both 



    Parameters
    ----------
    val : Any
        The value assigned to parameter `name`.
        This parameter is inherited from the Param base class.
    hdf5_attrs : HDF5Attrs
        Attributes of `val` that will be used for writing the
        output STATS.h5 HDF5 file.
    '''
    hdf5_attrs: HDF5Attrs


@dataclass
class YamlHDF5Param(YamlParam[T], HDF5Param[T]):
    '''
    Data structure to hold information about a QA input parameter
    when the QA input parameter appears in both the runconfig yaml and
    the output STATS.h5 HDF5 file.

    Parameters
    ----------
    val : Any
        The value assigned to this parameter.
        `val` should only be defined in the Param base class; `val` should
        not be overloaded in YamlParam nor HDF5Param.
    yaml_attrs : YamlAttrs
        Attributes of `val` that will be used for reading/writing the
        runconfig yaml file.
    hdf5_attrs : HDF5Attrs
        Attributes of `val` that will be used for writing the
        output STATS.h5 HDF5 file.
    '''
    # This is a composition dataclass; it composes the two base classes
    # into one dataclass.

    pass    

# @dataclass
# class YamlHDF5Param(YamlParam[T], HDF5Param[T]):
#     '''
#     Data structure to hold information about a QA input parameter
#     when the QA input parameter appears in both the runconfig yaml and
#     the output STATS.h5 HDF5 file.

#     Parameters
#     ----------
#     val : Any
#         The value assigned to this parameter.
#         `val` must be the same in both YamlParam and HDF5Param.
#     yaml_attrs : YamlAttrs
#         Attributes of `val` that will be used for reading/writing the
#         runconfig yaml file.
#     hdf5_attrs : HDF5Attrs
#         Attributes of `val` that will be used for writing the
#         output STATS.h5 HDF5 file.
#     '''
#     # This is a composition dataclass; it composes the two base classes
#     # into one dataclass.
#     pass    


# @dataclass
# class Param(Generic[T]):
#     """
#     """
#     val: T
#     yaml_attr: Optional[YamlAttrs]
#     hdf5_attr: Optional[HDF5Attrs]


# @dataclass
# class YamlHDF5Param(Generic[T]):
#     """
#     """
#     val: T
#     yaml_attr: YamlAttrs
#     hdf5_attr: HDF5Attrs

# class YamlParam(YamlHDF5Param):
#     yaml_attr: YamlAttrs  # required
#     hdf5_attr: = None


# @dataclass
# class YamlAttrs:
#     name: str
#     descr: str


# class YamlParam(Param):
#     yaml_name: str
#     val: T
#     yaml_descr: Optional[str] = None


# class HDF5Param(YamlParam):
#     h5_name:
#     units: Optional[str] = None
#     long_descr:


# def foo(x: Any) -> Any:
#     """
#     """
#     return x


# def bar(x: T) -> T:



# @dataclass
# class Param:
#     '''
#     Data structure to hold information about a QA input parameter.

#     Parameters
#     ----------
#     name : str
#         Name of the parameter. It is suggested that this be the exact
#         name of the parameter in a QA runconfig and formatted in snakecase
#         and no spaces.
#     value : Any
#         The value assigned to parameter `name`
#     units : str
#         Units of `ds_data`; will be stored in a `units` attribute
#         for the new Dataset.
#         For NISAR datasets, use this convention:
#             - If the values have dimensions, use CF-compliant names (e.g. 'meters')
#             - If the values are numeric but dimensionless (e.g. ratios),
#                 set `ds_units` to 'unitless'
#             - If the values are inherently descriptive and have no units
#                 (e.g. a file name, or a list of frequency names such as ['A', 'B']),
#                 then set `ds_units` to None so that no units attribute
#                 is created.
#         Defaults to None (no units attribute will be created)
#     short_descr : str, optional
#         Short one-line description of this parameter. Preferably,
#         this should meet CF conventions; can be used for the stats.h5 file.
#         Defaults to None.
#     long_descr : str, optional
#         Long description of the parameter describing the type, defaults,
#         usage, examples, etc. This is what would appear in a docstring
#         or runconfig. If not provided, or if `None` is provided,
#         will defaults to the `short_descr`.
#     '''
#     name: str
#     val: Any
#     units: Optional[str] = None
#     short_descr: Optional[str] = None
#     long_descr: Optional[str] = None

#     def __post_init__(self):
#         if self.long_descr == None:
#             self.long_descr = self.short_descr


class YamlParamGroup(ABC):
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
        '''
        pass


    @classmethod
    def get_default_arg_for_yaml(cls, attr_name: str):
        '''
        Get this class' default argument for its attribute `attr_name`
        
        Parameters
        ----------
        attr_name : str
            Name of an attribute of this class.

        Returns
        -------
        default_arg : Any
            The default value for `attr_name`. If there is no default value,
            then an instance of RequiredParam is returned.
            TODO - what is the best value to return if there is no default?
        '''

        # Get the default argument for the given parameter.
        signature = inspect.signature(cls.__init__)
        default_arg = signature.parameters[attr_name].default
        
        # If there is no default value, then assign it to an instance of RequiredParamClass()
        if default_arg is inspect.Parameter.empty:
            default_arg = RequiredParam()

        return default_arg


    @classmethod
    def populate_runcfg(cls, runconfig_cm, indent_size=4):
        '''Update the provided ruamel.yaml object with select attributes
        (parameters) of this instance of the dataclass for use in a
        NISAR product QA runconfig file.

        Only default values will be used.

        Parameters
        ----------
        runconfig_cm : ruamel.yaml.comments.CommentedMap
            The base commented map; will be updated with the attributes
            from this dataclass that are in the QA runconfig file
        indent_size : int, optional
            Number of spaces per indent. Default: 4.

        Notes
        -----
        Reference: https://stackoverflow.com/questions/56471040/add-a-comment-in-list-element-in-ruamel-yaml
        '''
        # Create a default instance of this class

        # ruamel.yaml requires an argument value to create each parameter.
        # But, the *ParamGroup dataclasses do not have default values
        # for their required parameters. So, find these required parameters,
        # and then override their property setter to skip the input
        # verification and assign an empty string to be their default
        # value for the sake of generating the default runconfig template
        # file; the empty string appears as a blank in the output runconfig
        # yaml file.
        required_params = {}
        for field in fields(cls):
            def_val = cls.get_default_arg_for_yaml(attr_name=field.name)
            if isinstance(def_val, RequiredParam):
                required_params[field.name] = ''

        # Unpack any required params when creating the default class instance.
        # (If there are no required parameters, the dictionary will be empty,
        # and an instance of the class with all default values will still be
        # created.)
        default = cls(**required_params)

        '''
        if required_params:  # if `required_params` is not empty
            # Override just the property setters in the base class;
            # keep all other defaults, descriptions, etc.
            def get_dict_attr(obj, attr):
                for obj in [obj] + obj.__class__.mro():
                    if attr in obj.__dict__:
                        return obj.__dict__[attr]
                raise AttributeError
            print("LALA: ", cls)

            class OverrideCls(cls):

                # class property(fget=None, fset=None, fdel=None, doc=None)

                def __init__(self):
                    for param in required_params:
                        
                        wrapped_param = getattr(cls, f'_{param}_2_param')('')


                        # @cls.qa_input_file.setter
                        # def qa_input_file(self, attr):
                        #     cls._qa_input_file = attr

                    #     # object.__setattr__(cls, f'_{param}', required_params[param])

                    #     x_property = get_dict_attr(self, param)
                    #     print("x_property: ", x_property)
                    #     # setattr(self, param, property(lambda sub_self: super(type(sub_self), sub_self).qa_input_file))
                    #     # new_param = cls._qa_input_file_2_param('')
                    #     # raise Exception('x_property: ', x_property)
                    #     x_setter = getattr(x_property, 'setter')
                    #     print("x_setter: ", x_setter)
                    #     setattr(OverrideCls, param,
                    #                   x_setter(lambda sub_self, val: 
                    #                             super(type(sub_self), 
                    #                             type(sub_self)).qa_input_file.fset(sub_self, val)))
                    # print(required_params)
                    super().__init__(self, **required_params)

            default = OverrideCls()
        else:
            default = cls()
        '''

        # build yaml params group
        params_cm = CM()

        # Add all attributes from this dataclass to the group
        for field in fields(cls):
            attr = getattr(default, field.name)

            if hasattr(attr, 'yaml_attrs'):
                default.add_param_to_cm(params_cm=params_cm,
                                        param_attr=attr,
                                        indent=indent_size)

        # Add the new parameter group to the runconfig
        default.add_param_group_to_runconfig(runconfig_cm, params_cm)


    def add_param_to_cm(self, params_cm, param_attr, indent=4):
        '''
        Add a Param attribute to a Commented Map.

        This can be used to add a new parameter into a group for a runconfig.

        Parameters
        ----------
        params_cm : ruamel.yaml.comments.CommentedMap
            Commented Map for a parameter group in the runconfig.
            Will be updated to include `param_attr`.
        param_attr : YamlParam
            YamlParam object that will be added to the `params_cm` Commented Map
        '''
        # set indentation for displaying the comments correctly in the yaml
        comment_indent = len(self.get_path_to_group_in_runconfig()) * indent

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
        name = param_attr.yaml_attrs.name
        params_cm[name] = val
        comment = param_attr.yaml_attrs.descr
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


@dataclass
class HDF5ParamGroup:
    '''Class for parameters that will be stored in the output HDF5 file.'''

    path_to_processing_group_in_stats_h5: ClassVar[str] = \
                                                '/science/%s/QA/processing'


    def write_params_to_h5(self, h5_file, bands=('LSAR')):
        '''
        Update `h5_file` with the attributes of this dataclass
        that are a subclass of HDF5Param.

        Parameters
        ----------
        h5_file : h5py.File
            Handle to an HDF5 file where the parameter metadata
            should be saved
        bands : iterable of str, optional
            Sequence of the band names. Ex: ('SSAR', 'LSAR')
            Defaults to ('LSAR')
        '''

        # Add all attributes from this dataclass to the group
        for band in bands:
            for field in fields(self):
                attr = getattr(self, field.name)
                if issubclass(type(attr), nisarqa.parameters.nisar_params.HDF5Param):
                    
                    if attr.val is None:
                        val = 'None'
                    else:
                        val = attr.val

                    nisarqa.create_dataset_in_h5group(
                        h5_file=h5_file,
                        grp_path=attr.hdf5_attrs.path % band,
                        ds_name=attr.hdf5_attrs.name,
                        ds_data=val,
                        ds_units=attr.hdf5_attrs.units,
                        ds_description=attr.hdf5_attrs.descr)


class YamlHDF5ParamGroup(YamlParamGroup, HDF5ParamGroup):
    '''Abstract Base Class for creating *Params dataclasses.'''

    pass

class RequiredParam:
    '''Sentinel value to indicate that a runconfig param is a required param, 
    and hence there is no default value.
    '''
    pass


__all__ = nisarqa.get_all(__name__, objects_to_skip)
