import dataclasses
import inspect
import warnings
from abc import ABC, abstractmethod
from dataclasses import MISSING, dataclass, fields
from typing import ClassVar, Generic, TypeVar

import nisarqa
from ruamel.yaml import CommentedMap as CM
from ruamel.yaml import CommentedSeq as CS

objects_to_skip = nisarqa.get_all(name=__name__)

@dataclass
class YamlAttrs:
    '''
    Dataclass to hold information about a parameter for the runconfig yaml.

    Parameters
    ----------
    name : str
        Name of the parameter parameter in a QA runconfig. Should be
        formatted in snakecase and no spaces.
    descr : str
        Detailed description of the yaml parameter, such as would appear
        in a runconfig. Should describe what the parameter is, its type,
        requirements, usage, examples, etc.
    '''
    name: str
    descr: str


@dataclass
class HDF5Attrs:
    '''
    Dataclass to hold information about a parameter for the stats.h5 file.

    Parameters
    ----------
    name : str
        Name of the HDF5 dataset as it should output to the stats.h5 file.
        It is suggested that this be formatted in camelCase with no spaces
        and the first letter lowercase. Acronyms like "RIFG" can be uppercase.
    units : str
        Units for this dataset; will be stored in a `units` attribute
        for the new Dataset.
        For NISAR datasets, use this convention:
            - If the values have dimensions, use CF-compliant names (e.g. 'meters')
            - If the values are numeric but dimensionless (e.g. ratios),
                set `units` to 'unitless'
            - If the values are inherently descriptive and have no units
                (e.g. a file name, or a list of frequency names such as ['A', 'B']),
                then set `units` to None so that no units attribute
                is created.
        Defaults to None
    descr : str
        Short one-line description of this parameter. Preferably,
        this should meet CF conventions; can be used for the stats.h5 file.
    '''
    name: str
    units: str
    descr: str
    path: str


class YamlParamGroup(ABC):
    '''Abstract Base Class for creating *Params dataclasses.'''

    @abstractmethod
    def __post_init__(self):
        '''
        Validate all input parameter arguments, and set any auto-generated
        attributes.
        '''
        pass

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
    def get_list_of_yaml_names(cls):
        '''
        For all attributes in this dataclass that appear in the runconfig,
        return a list of their names as they appear in the runconfig.

        This list will be parsed from each attribute's field metadata;
        specifically, it will be parsed from the YamlAttrs `name` metadata.

        Returns
        -------
        yaml_names : list of str
            Name of the class attribute to parse the HDF5Attrs
            `units` value from
        '''
        yaml_names = []
        for field in fields(cls):
            if 'yaml_attrs' in field.metadata:
                yaml_names.append(field.metadata['yaml_attrs'].name)

        if yaml_names:
            return yaml_names
        else:
            # Sanity check - list is still empty
            warnings.warn(f'None of the attributes in {cls.__name__}'
                            ' contain info for an YamlAttrs object')
            return yaml_names


    @classmethod
    def populate_runcfg(cls, runconfig_cm, indent=4):
        '''Update the provided ruamel.yaml object with select attributes
        (parameters) of this instance of the dataclass for use in a
        NISAR product QA runconfig file.

        Only default values will be used.

        Parameters
        ----------
        runconfig_cm : ruamel.yaml.comments.CommentedMap
            The base commented map; will be updated with the attributes
            from this dataclass that are in the QA runconfig file
        indent : int, optional
            Number of spaces per indent. Default: 4.

        Notes
        -----
        Reference: https://stackoverflow.com/questions/56471040/add-a-comment-in-list-element-in-ruamel-yaml
        '''
        # build yaml params group
        params_cm = CM()

        # Add all attributes from this dataclass to the group
        for field in fields(cls):
            if 'yaml_attrs' in field.metadata:

                if field.default == dataclasses.MISSING:
                    msg = 'REQUIRED'

                    # in ruamel.yaml, providing None as a value causes
                    # the output .yaml to have an empty field argument.
                    # This indicates to a user that there is no default value.
                    val = None
                else:
                    msg = f'Default: {field.default}'
                    val = field.default

                yaml_attrs = field.metadata['yaml_attrs']
                cls.add_param_to_cm(params_cm=params_cm,
                                    name=yaml_attrs.name,
                                    val=val,
                                    comment=f'\n{yaml_attrs.descr}\n{msg}',
                                    indent=indent)

        if not params_cm:  # No attributes were added
            warnings.warn(f'{cls.__name__} is a subclass of YamlParamGroup'
                          ' but does not have any attributes whose'
                          ' dataclasses.field metadata contains "yaml_attrs"')
        else:
            # Add the new parameter group to the runconfig
            cls.add_param_group_to_runconfig(runconfig_cm, params_cm)


    @classmethod
    def add_param_to_cm(cls, params_cm, name, val, comment=None, indent=4):
        '''
        Add a new attribute to a Commented Map.

        This can be used to add a new parameter into a group for a runconfig.

        Parameters
        ----------
        params_cm : ruamel.yaml.comments.CommentedMap
            Commented Map for a parameter group in the runconfig.
            Will be updated to include `param_attr`.
        name : str
            Parameter name, as it should appear in `params_cm`
        val : Any
            Parameter value, as it should appear in `params_cm`
        name : str
            Parameter name, as it should appear in `params_cm`
        comment : str or None, optional
            Parameter comment (description), as it should appear in `params_cm`.
            If None, then no comment will be added. Defaults to None.
        indent : int, optional
            Number of spaces of an indent. Defaults to 4.
        '''
        # set indentation for displaying the comments correctly in the yaml
        comment_indent = len(cls.get_path_to_group_in_runconfig()) * indent

        # To have ruamel.yaml display list values as a list in the runconfig,
        # use CommentedSeq
        # https://stackoverflow.com/questions/56937691/making-yaml-ruamel-yaml-always-dump-lists-inline
        if isinstance(val, (list, tuple)):
            seq = CS()
            seq.fa.set_flow_style()
            for item in val:
                seq.append(item)
            val = seq

        # Add parameter to the group
        params_cm[name] = val

        # If provided, add comment
        if comment is not None:
            for line in nisarqa.multi_line_string_iter(comment):
                params_cm.yaml_set_comment_before_after_key(
                    name, before=line, indent=comment_indent)


    @classmethod
    def add_param_group_to_runconfig(cls, yaml_cm, params_cm):
        '''
        Add a new group of parameters to a yaml Commented Map
        along the nested directory structure specified by 
        `get_path_to_group_in_runconfig()`.
        
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
        path = cls.get_path_to_group_in_runconfig()

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


@dataclass(frozen=True)
class HDF5ParamGroup:
    '''Class for parameters that will be stored in the output HDF5 file.'''

    def __post_init__(self):
        '''
        Validate all input parameter arguments, and set any auto-generated
        attributes.
        '''
        pass


    def get_units_from_hdf5_metadata(self, attribute_name):
        '''
        Return the value of the input attribute's HDF5Attrs `units` metadata.

        Parameters
        ----------
        attribute_name : str
            Name of the class attribute to parse the HDF5Attrs
            `units` value from
        '''
        for field in fields(self):
            if field.name == attribute_name:
                if 'hdf5_attrs' in field.metadata:
                    return field.metadata['hdf5_attrs'].units
                elif 'hdf5_attrs_func' in field.metadata:
                    return field.metadata['hdf5_attrs_func'](self).units
                else:
                    raise TypeError(f'The field metadata for `{attribute_name}`'
                                    ' does not contain info for an HDF5Attrs'
                                    ' object')

        # If the request field was not found, raise an error        
        raise KeyError(
            f'`{attribute_name}` is not an attribute of this dataclass.')


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

        # Flag -- the intention is to help assist developers to set up
        # child classes are set up correctly
        found_at_least_one_hdf5_attr = False

        # Add all attributes from this dataclass to the group
        for band in bands:
            for field in fields(self):
                if ('hdf5_attrs' in field.metadata) or \
                    ('hdf5_attrs_func' in field.metadata):

                    attr = getattr(self, field.name)
                    
                    # Create filler data to stand in for Python's Nonetype.
                    # TODO - Geoff - is 'None' the correct filler value for use in stats.h5?
                    val = 'None' if attr is None else attr

                    if 'hdf5_attrs' in field.metadata:
                        hdf5_attrs = field.metadata['hdf5_attrs']
                    else:
                        hdf5_attrs = field.metadata['hdf5_attrs_func'](self)

                    nisarqa.create_dataset_in_h5group(
                        h5_file=h5_file,
                        grp_path=hdf5_attrs.path % band,
                        ds_name=hdf5_attrs.name,
                        ds_data=val,
                        ds_units=hdf5_attrs.units,
                        ds_description=hdf5_attrs.descr)
                    
                    found_at_least_one_hdf5_attr = True

        if not found_at_least_one_hdf5_attr:
            # No attributes in this class were added to stats.h5.
            # Since this class is specifically for HDF5 parameters, this
            # is probably cause for concern because something was not
            # set up (coded) properly elsewhere.
            warnings.warn(f'{self.__name__} is a subclass of HDF5ParamGroup'
                        ' but does not have any attributes whose'
                        ' dataclasses.field metadata contains \'hdf5_attrs\''
                        ' or \'hdf5_attrs_func\'')


@dataclass(frozen=True)
class YamlHDF5ParamGroup(YamlParamGroup, HDF5ParamGroup):
    '''Abstract Base Class for creating *Params dataclasses.'''
    pass


__all__ = nisarqa.get_all(__name__, objects_to_skip)
