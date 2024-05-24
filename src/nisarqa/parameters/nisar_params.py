from __future__ import annotations

import collections
import dataclasses
import os
import sys
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import ClassVar, Optional

import h5py
from ruamel.yaml import YAML, CommentedMap, CommentedSeq

import nisarqa

objects_to_skip = nisarqa.get_all(name=__name__)


@dataclass
class YamlAttrs:
    """
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
    """

    name: str
    descr: str


@dataclass
class HDF5Attrs:
    """
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
        NISAR datasets use this convention:
            - If values have dimensions, use CF- and UDUNITS-compliant names.
              Units should be spelled out:
                  Correct: "meters"
                  Incorrect: "m"
              Units should favor math symbols:
                  Correct: "meters / second ^ 2"
                  Incorrect: "meters per second squared"
            - If values are numeric but dimensionless (e.g. ratios),
              set `ds_units` to "1" (the string "1").
            - If values are inherently descriptive and have no units
              (e.g. a file name, or a list of frequency names like: ['A', 'B']),
              then set `ds_units` to None so that no units attribute
              is created.
    descr : str
        Short one-line description of this parameter. Preferably,
        this should meet CF conventions; can be used for the stats.h5 file.
    group_path : str
        The HDF5 path to parent group for the `name` dataset
    """

    name: str
    units: str
    descr: str
    group_path: str


class YamlParamGroup(ABC):
    """Abstract Base Class for creating *Params dataclasses."""

    @staticmethod
    @abstractmethod
    def get_path_to_group_in_runconfig():
        """
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
        """
        pass

    @classmethod
    def get_dict_of_yaml_names(cls):
        """
        For all attributes in this class that are also in the runconfig,
        return a dict mapping their class attribute name to their name
        as it appears in the runconfig.

        This will be parsed from each attribute's field metadata;
        specifically, it will be parsed from the 'yaml_attrs' key's
        YamlAttrs `name` attribute.

        Returns
        -------
        yaml_names : dict
            Dict mapping class attribute names to runconfig yaml names.
            Format: {<class_attribute_name> : <YamlAttrs object .name>}
            Types:  {<str> : <str>}
        """
        yaml_names = {}
        for field in fields(cls):
            if "yaml_attrs" in field.metadata:
                yaml_names[field.name] = field.metadata["yaml_attrs"].name

        if not yaml_names:
            # Sanity check - dict is still empty
            nisarqa.get_logger().warning(
                f"None of the attributes in {cls.__name__}"
                " contain info for an YamlAttrs object"
            )

        return yaml_names

    @classmethod
    def get_attribute_metadata(cls, attribute_name):
        """
        Get the attribute's field metadata.

        Returns
        -------
        metadata : dict
            Dict mapping class attribute names to runconfig yaml names.
            Format: {<class_attribute_name> : <YamlAttrs object .name>}
            Types:  {<str> : <str>}
        """
        for field in fields(cls):
            if attribute_name == field.name:
                return field.metadata
        else:
            raise ValueError(
                f"{attribute_name=} is not an attribute in {cls.__name__}"
            )

    @classmethod
    def populate_runcfg(cls, runconfig_cm, indent=4):
        """
        Update the provided ruamel.yaml object with select attributes
        (parameters) of this dataclass object for use in a NISAR product
        QA runconfig file.

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
        """
        # build yaml params group
        params_cm = CommentedMap()

        # Add all attributes from this dataclass to the group
        for field in fields(cls):
            if "yaml_attrs" in field.metadata:
                if field.default == dataclasses.MISSING:
                    msg = "REQUIRED"

                    # in ruamel.yaml, providing None as a value causes
                    # the output .yaml to have an empty field argument.
                    # This indicates to a user that there is no default value.
                    val = None
                else:
                    msg = f"Default: {field.default}"
                    val = field.default

                yaml_attrs = field.metadata["yaml_attrs"]
                cls.add_param_to_cm(
                    params_cm=params_cm,
                    name=yaml_attrs.name,
                    val=val,
                    comment=f"\n{yaml_attrs.descr}\n{msg}",
                    indent=indent,
                )

        if not params_cm:  # No attributes were added
            nisarqa.get_logger().warning(
                f"{cls.__name__} is a subclass of YamlParamGroup"
                " but does not have any attributes whose"
                ' dataclasses.field metadata contains "yaml_attrs"'
            )
        else:
            # Add the new parameter group to the runconfig
            cls.add_param_group_to_runconfig(runconfig_cm, params_cm)

    @classmethod
    def add_param_to_cm(cls, params_cm, name, val, comment=None, indent=4):
        """
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
        comment : str or None, optional
            Parameter comment (description), as it should appear in `params_cm`.
            If None, then no comment will be added. Defaults to None.
        indent : int, optional
            Number of spaces of an indent. Defaults to 4.
        """
        # set indentation for displaying the comments correctly in the yaml
        comment_indent = len(cls.get_path_to_group_in_runconfig()) * indent

        # To have ruamel.yaml display list values as a list in the runconfig,
        # use CommentedSeq
        # https://stackoverflow.com/questions/56937691/making-yaml-ruamel-yaml-always-dump-lists-inline
        if isinstance(val, (list, tuple)):
            seq = CommentedSeq()
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
                    name, before=line, indent=comment_indent
                )

    @classmethod
    def add_param_group_to_runconfig(cls, yaml_cm, params_cm):
        """
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
        """
        path = cls.get_path_to_group_in_runconfig()

        # start at root Commented Map
        parent_cm = yaml_cm

        # Traverse to the direct parent CM group for this params_cm.
        # Ensure all parent groups for this params_cm exist
        for group in path[:-1]:
            # If group does not exist, create it
            if group not in parent_cm:
                parent_cm[group] = CommentedMap()

                # For readability, add a newline before this new group
                parent_cm.yaml_set_comment_before_after_key(group, before="\n")

            # Move into the next nested group
            parent_cm = parent_cm[group]

        # Attach new params_cm group into the runconfig yaml
        parent_cm[path[-1]] = params_cm

        # For readability, add a newline before the new params_cm group
        parent_cm.yaml_set_comment_before_after_key(path[-1], before="\n")

    @staticmethod
    def _validate_pair_of_numeric(
        param_value: Optional[Sequence[int]],
        param_name: str,
        min: Optional[int | float] = None,
        max: Optional[int | float] = None,
        none_is_valid_value: bool = False,
        strictly_increasing: bool = False,
    ) -> None:
        """
        Raise exception if `param_value` is not a valid input.

        Parameters
        ----------
        param_value : None or pair of int or float
            Sequence of two int or float value, or None.
        param_name : str
            Name of this parameter. Will be used for the error message.
        min, max : None or int or float, optional
            The minimum or maximum allowed values (respectively) for each value in
            `param_value`. `param_value` may not be outside this range.
        none_is_valid_value : bool, optional
            True if `None` is a valid value. Defaults to False.
        strictly_increasing : bool, optional
            True if `input_value[0]` must be less than `input_value[1]`.
            Defaults to False.
        """
        if param_value is None:
            if none_is_valid_value:
                return
            else:
                raise TypeError(
                    f"`{param_name}` is None, but must be a pair of numeric."
                )

        if not isinstance(param_value, Sequence):
            msg = f"`{param_name}` must be a sequence"
            if none_is_valid_value:
                msg += " or None."
            raise TypeError(msg)

        if len(param_value) != 2:
            raise ValueError(
                f"{param_name}={param_value}; must have a length of two."
            )

        if not all(isinstance(e, (float, int)) for e in param_value):
            raise TypeError(
                f"{param_name}={param_value}; must contain only float or int."
            )

        if (min is not None) and (any((e < min) for e in param_value)):
            raise ValueError(
                f"{param_name}={param_value}; must be in range [{min}, {max}]."
            )

        if (max is not None) and (any((e > max) for e in param_value)):
            raise ValueError(
                f"{param_name}={param_value}; must be in range [{min}, {max}]."
            )

        if strictly_increasing:
            if param_value[0] >= param_value[1]:
                raise ValueError(
                    f"{param_name}={param_value}; values must be"
                    " strictly increasing."
                )


class HDF5ParamGroup:
    """Class for parameters that will be stored in the output HDF5 file."""

    def get_units_from_hdf5_metadata(self, attribute_name):
        """
        Return the value of the input attribute's HDF5Attrs `units` metadata.

        Parameters
        ----------
        attribute_name : str
            Name of the class attribute to parse the HDF5Attrs
            `units` value from

        Returns
        -------
        units : str
            The value of the `units` attribute of the HDF5Attrs metadata
            for this instance's attribute `attribute_name`.
        """
        for field in fields(self):
            if field.name == attribute_name:
                if "hdf5_attrs" in field.metadata:
                    return field.metadata["hdf5_attrs"].units
                elif "hdf5_attrs_func" in field.metadata:
                    return field.metadata["hdf5_attrs_func"](self).units
                else:
                    raise TypeError(
                        f"The field metadata for `{attribute_name}`"
                        " does not contain info for an HDF5Attrs"
                        " object"
                    )

        # If the request field was not found, raise an error
        raise KeyError(
            f"`{attribute_name}` is not an attribute of this dataclass."
        )

    def write_params_to_h5(self, h5_file, band):
        """
        Update `h5_file` with the attributes of this dataclass
        that are a subclass of HDF5Param.

        Parameters
        ----------
        h5_file : h5py.File
            Handle to an HDF5 file where the parameter metadata
            should be saved
        band : str
            The letter of the band. Ex: "L" or "S".
        """

        # Flag -- the intention is to help assist developers and ensure
        # child classes are set up correctly
        found_at_least_one_hdf5_attr = False

        # Add all attributes from this dataclass to the group
        for field in fields(self):
            if ("hdf5_attrs" in field.metadata) or (
                "hdf5_attrs_func" in field.metadata
            ):
                attr = getattr(self, field.name)

                # Create filler data to stand in for Python's Nonetype.
                # This is for cases such as gamma=None (not a float)
                val = "None" if attr is None else attr

                if "hdf5_attrs" in field.metadata:
                    hdf5_attrs = field.metadata["hdf5_attrs"]
                else:
                    hdf5_attrs = field.metadata["hdf5_attrs_func"](self)

                nisarqa.create_dataset_in_h5group(
                    h5_file=h5_file,
                    grp_path=hdf5_attrs.group_path % band,
                    ds_name=hdf5_attrs.name,
                    ds_data=val,
                    ds_units=hdf5_attrs.units,
                    ds_description=hdf5_attrs.descr,
                )

                found_at_least_one_hdf5_attr = True

        if not found_at_least_one_hdf5_attr:
            # No attributes in this class were added to stats.h5.
            # Since this class is specifically for HDF5 parameters, this
            # is probably cause for concern because something was not
            # set up (coded) properly elsewhere.
            nisarqa.get_logger().warning(
                f"{self.__name__} is a subclass of HDF5ParamGroup"
                " but does not have any attributes whose"
                " dataclasses.field metadata contains 'hdf5_attrs'"
                " or 'hdf5_attrs_func'"
            )


@dataclass(frozen=True)
class ThresholdParamGroup(YamlParamGroup):
    """
    Abstract Base Class for creating *Params dataclasses with thresholds.

    Parameters
    ----------
    nan_threshold, inf_threshold, fill_threshold, near_zero_threshold,
        total_invalid_threshold : float, optional
        Threshold value for alerting users to possible malformed datasets.
        If the percentage of NaN-, Inf-, fill-, near-zero-, or total
        invalid-valued pixels is above the respective threshold, it will be
        logged as an error and an exception raised.
        "Total invalid pixels" is the sum of NaN, Inf, & fill pixels.
        If `zero_is_invalid` is True, near-zero-valued pixels are also
        included in the total count of invalid pixels.
        Each threshold should be between 0 and 100.
        Setting a threshold to -1 causes that threshold to be effectively
        ignored (e.g. any percent of NaN values is fine; it will be noted as
        info, and will not trigger a QA failure).
        Setting a threshold to 0 triggers a QA failure if the dataset
        contains any pixels with that value.
        All default to `nisarqa.STATISTICS_THRESHOLD_PERCENTAGE`.
    epsilon : float, optional
        Absolute tolerance for determining if a raster pixel is 'almost zero'.
        Defaults to 1e-6.
    zero_is_invalid: bool, optional
        True if near-zero pixels should be counted towards the
        total number of invalid pixels. False to exclude them.
        If False, consider setting `near_zero_threshold` to -1.
        Note: fill values are always considered invalid, so if a raster's
        fill value is zero, then zeros will still be included in the total.
        Defaults to True.

    Notes
    -----
    When subclassing, to update the default for a particular threshold value,
    suggest using the class method `get_field_with_updated_default()`.
    """

    _threshold_descr_template: ClassVar[str] = (
        "Percent of %s pixels per total raster area allowed"
    )

    nan_threshold: float = field(
        default=nisarqa.STATISTICS_THRESHOLD_PERCENTAGE,
        metadata={
            "yaml_attrs": YamlAttrs(
                name="nan_threshold",
                # This description will be printed first in the runconfig,
                # so have it describe all of the threshold parameters.
                # For subsequent parameters, use a short version.
                # (This helps keep the runconfig shorter and easier to read.)
                descr=f"""*** Threshold Percentage Parameters ***
        Check for malformed datasets using these threshold percentages.
        An exception is raised if the percent of pixels with the corresponding value
        per the total raster area exceeds the threshold percentage provided.
        Thresholds must be in the interval [0, 100], or -1 to be ignored.
        (-1 logs the percentages as INFO, but will not trigger a QA failure.)

        {_threshold_descr_template % 'NaN'}""",
            )
        },
    )

    inf_threshold: float = field(
        default=nisarqa.STATISTICS_THRESHOLD_PERCENTAGE,
        metadata={
            "yaml_attrs": YamlAttrs(
                name="inf_threshold",
                descr=_threshold_descr_template % "+/- Inf",
            )
        },
    )

    fill_threshold: float = field(
        default=nisarqa.STATISTICS_THRESHOLD_PERCENTAGE,
        metadata={
            "yaml_attrs": YamlAttrs(
                name="fill_threshold",
                descr=_threshold_descr_template % "Fill-valued",
            )
        },
    )

    near_zero_threshold: float = field(
        default=nisarqa.STATISTICS_THRESHOLD_PERCENTAGE,
        metadata={
            "yaml_attrs": YamlAttrs(
                name="near_zero_threshold",
                descr=_threshold_descr_template % "Near-zero",
            )
        },
    )

    total_invalid_threshold: float = field(
        default=nisarqa.STATISTICS_THRESHOLD_PERCENTAGE,
        metadata={
            "yaml_attrs": YamlAttrs(
                name="total_invalid_threshold",
                descr=(
                    f"{_threshold_descr_template % 'total invalid'}\n"
                    "'Total invalid pixels' is the sum of NaN, Inf, fill,"
                    " & (if `zero_is_invalid` is True) near-zero pixels."
                ),
            )
        },
    )

    epsilon: float = field(
        default=1e-6,
        metadata={
            "yaml_attrs": YamlAttrs(
                name="epsilon",
                descr=(
                    "Absolute tolerance for determining if a raster pixel"
                    " is 'near zero'."
                ),
            ),
        },
    )

    zero_is_invalid: bool = field(
        default=True,
        metadata={
            "yaml_attrs": YamlAttrs(
                name="zero_is_invalid",
                descr="""True if near-zero pixels should be counted towards the
                    total number of invalid pixels. False to exclude them.
                    If False, consider setting `near_zero_threshold` to -1.
                    Note: fill values are always considered invalid, so if a raster's
                    fill value is zero, then zeros will still be included in the total.""",
            ),
        },
    )

    def __post_init__(self):
        # VALIDATE INPUTS

        for name, thresh in zip(
            (
                "nan_threshold",
                "near_zero_threshold",
                "fill_threshold",
                "inf_threshold",
                "total_invalid_threshold",
            ),
            (
                self.nan_threshold,
                self.near_zero_threshold,
                self.fill_threshold,
                self.inf_threshold,
                self.total_invalid_threshold,
            ),
        ):
            if thresh != -1:
                try:
                    nisarqa.verify_valid_percent(thresh)
                except ValueError:
                    raise ValueError(
                        f"`{name}` is {thresh}, must be in range [0.0, 100.0]"
                        " or -1."
                    )

        if self.epsilon < 0:
            raise ValueError(f"`{self.epsilon=}`, must be >= 0.")

    @classmethod
    def get_field_with_updated_default(
        self, param_name: str, default: float
    ) -> dataclasses.Field:
        """
        Return the Field object for a class parameter with an updated default.

        Parameters
        ----------
        param_name : str
            One of the class parameters of `ThresholdParamGroup`.
        default : float
            The desired default value for the parameter.

        Returns
        -------
        updated_field : dataclasses.Field
            A Field object with the updated default value. All other
            aspects of `ThresholdParamGroup` class' version of the Field
            object remain unchanged (for example, the `metadata` which contains
            the description is unchanged).

        Examples
        --------
        When subclassing, override the parameter like this:
        ```
        nan_threshold: float = (
            nisarqa.ThresholdParamGroup.get_field_with_updated_default(
                param_name="nan_threshold", default=0
            )
        )
        ```
        """

        for f in fields(self):
            if f.name == param_name:
                metadata = f.metadata
                return field(default=default, metadata=metadata)
        else:
            raise ValueError(
                f"`{param_name=}`, must be a parameter in `ThresholdParamGroup`"
            )


@dataclass(frozen=True)
class ZeroIsValidThresholdParamGroup(ThresholdParamGroup):
    """
    `ThresholdParamGroup` but with defaults of zeros as valid and all-zeros ok.

    Parameters
    ----------
    nan_threshold, inf_threshold, fill_threshold, near_zero_threshold,
        total_invalid_threshold : float, optional
        Threshold values for alerting users to possible malformed datasets.
        See `ThresholdParamGroup` docstring for complete description.
        Default for NaN, Inf, fill, and total thresholds:
            `nisarqa.STATISTICS_THRESHOLD_PERCENTAGE`.
        Default for near-zero threshold: -1.
    epsilon : float, optional
        Absolute tolerance for determining if a raster pixel is 'almost zero'.
        Defaults to 1e-6.
    zero_is_invalid: bool, optional
        True if near-zero pixels should be counted towards the
        total number of invalid pixels. False to exclude them.
        If False, consider setting `near_zero_threshold` to -1.
        Note: Fill values are always considered invalid. So, if a raster's
        fill value is zero, then zeros will still be included in the total.
        Defaults to False.
    """

    near_zero_threshold: float = (
        ThresholdParamGroup.get_field_with_updated_default(
            param_name="near_zero_threshold", default=-1
        )
    )

    zero_is_invalid: float = ThresholdParamGroup.get_field_with_updated_default(
        param_name="zero_is_invalid", default=False
    )


@dataclass(frozen=True)
class Threshold99ParamGroup(ThresholdParamGroup):
    """
    ABC for creating *Params dataclasses with default thresholds of 99%.

    Typically, geocoded L2 products should use defaults of 99% for thresholds

    Parameters
    ----------
    nan_threshold, inf_threshold, fill_threshold, near_zero_threshold,
        total_invalid_threshold : float, optional
        Threshold values for alerting users to possible malformed datasets.
        See `ThresholdParamGroup` docstring for complete description.
        All thresholds default to 99.0%.
    epsilon : float, optional
        Absolute tolerance for determining if a raster pixel is 'almost zero'.
        Defaults to 1e-6.
    zero_is_invalid: bool, optional
        True if near-zero pixels should be counted towards the
        total number of invalid pixels. False to exclude them.
        If False, consider setting `near_zero_threshold` to -1.
        Note: fill values are always considered invalid, so if a raster's
        fill value is zero, then zeros will still be included in the total.
        Defaults to True.

    Notes
    -----
    When subclassing, to update the default for a particular threshold value,
    suggest using the class method `get_field_with_updated_default()`.
    """

    nan_threshold: float = ThresholdParamGroup.get_field_with_updated_default(
        param_name="nan_threshold", default=99.0
    )
    inf_threshold: float = ThresholdParamGroup.get_field_with_updated_default(
        param_name="inf_threshold", default=99.0
    )
    fill_threshold: float = ThresholdParamGroup.get_field_with_updated_default(
        param_name="fill_threshold", default=99.0
    )
    near_zero_threshold: float = (
        ThresholdParamGroup.get_field_with_updated_default(
            param_name="near_zero_threshold", default=99.0
        )
    )
    total_invalid_threshold: float = (
        ThresholdParamGroup.get_field_with_updated_default(
            param_name="total_invalid_threshold", default=99.0
        )
    )


@dataclass(frozen=True)
class ZeroIsValidThreshold99ParamGroup(Threshold99ParamGroup):
    """
    `Threshold99ParamGroup` but w/ defaults of zeros as valid and all-zeros ok.

    Parameters
    ----------
    nan_threshold, inf_threshold, fill_threshold, near_zero_threshold,
        total_invalid_threshold : float, optional
        Threshold values for alerting users to possible malformed datasets.
        See `ThresholdParamGroup` docstring for complete description.
        Default for NaN, Inf, fill, and total thresholds: 99.0%
        Default for near-zero threshold: -1.
    epsilon : float, optional
        Absolute tolerance for determining if a raster pixel is 'almost zero'.
        Defaults to 1e-6.
    zero_is_invalid: bool, optional
        True if near-zero pixels should be counted towards the
        total number of invalid pixels. False to exclude them.
        If False, consider setting `near_zero_threshold` to -1.
        Note: Fill values are always considered invalid. So, if a raster's
        fill value is zero, then zeros will still be included in the total.
        Defaults to False.
    """

    near_zero_threshold: float = (
        ThresholdParamGroup.get_field_with_updated_default(
            param_name="near_zero_threshold", default=-1
        )
    )

    zero_is_invalid: float = ThresholdParamGroup.get_field_with_updated_default(
        param_name="zero_is_invalid", default=False
    )


@dataclass(frozen=True)
class WorkflowsParamGroup(YamlParamGroup):
    """
    The parameters specifying which QA workflows should be run.

    This corresponds to the `workflows` runconfig group.

    Attributes
    ----------
    validate : bool, optional
        True to run the validate workflow. Default: True
    qa_reports : bool, optional
        True to run the QA Reports workflow. Default: True
    """

    # default value for all workflows
    _default_val: ClassVar[bool] = True

    # Generic description for all workflows
    _descr: ClassVar[str] = f"Flag to run %s."

    validate: bool = field(
        default=_default_val,
        metadata={
            "yaml_attrs": YamlAttrs(
                name="validate",
                descr=_descr % "`validate` workflow to validate the\n"
                " input file against its product spec",
            )
        },
    )

    qa_reports: bool = field(
        default=_default_val,
        metadata={
            "yaml_attrs": YamlAttrs(
                name="qa_reports",
                descr=_descr % "`qa_reports` workflow to generate a\n"
                "PDF report, geolocated browse image, compute statistics\n"
                "on the input file, etc.",
            )
        },
    )

    def __post_init__(self):
        self._check_workflows_arg("validate", self.validate)
        self._check_workflows_arg("qa_reports", self.qa_reports)

    @staticmethod
    def get_path_to_group_in_runconfig():
        return ["runconfig", "groups", "qa", "workflows"]

    @staticmethod
    def _check_workflows_arg(attr_name, val):
        """
        Validate that `val` is of the correct type for the
        WorkflowsParamGroup's attribute `attr_name`.

        Parameters
        ----------
        attr_name : str
            The name of the attribute of WorkflowsParamGroup for `attr`
        val : bool
            Argument value for `attr_name`.
        """
        # Validate `val`
        if not isinstance(val, bool):
            raise TypeError(
                f"`{attr_name}` must be of type bool. It is {type(val)}"
            )

    def at_least_one_wkflw_requested(self):
        """
        Return True if at least one of this instance's workflow attributes
        (e.g. `validate` or `qa_reports`) is True.

        Returns
        -------
        at_least_one_true_wkflw : bool
            True if at least one of the workflow field attributes in this
            instance is True. False is all workflows are set to False.
        """
        return any(getattr(self, field.name) for field in fields(self))


@dataclass(frozen=True)
class InputFileGroupParamGroup(YamlParamGroup):
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
                name="qa_input_file",
                descr="""Filename of the input file for QA.
                REQUIRED for QA. NOT REQUIRED if only running Product SAS.
                If Product SAS and QA SAS are run back-to-back,
                this field should be identical to `sas_output_file`.
                Otherwise, this field should contain the filename of the single
                NISAR product for QA to process.""",
            )
        }
    )

    def __post_init__(self):
        # VALIDATE INPUTS
        nisarqa.validate_is_file(
            filepath=self.qa_input_file,
            parameter_name="qa_input_file",
            extension=".h5",
        )

    @staticmethod
    def get_path_to_group_in_runconfig():
        return ["runconfig", "groups", "input_file_group"]


@dataclass(frozen=True)
class ProductPathGroupParamGroup(YamlParamGroup):
    """
    Parameters from the Product Path Group runconfig group.

    This corresponds to the `groups: product_path_group` runconfig group.

    Parameters
    ----------
    qa_output_dir : str, optional
        Filepath to the output directory to store NISAR QA output files.
        Defaults to './qa'
    """

    qa_output_dir: str = field(
        default="./qa",
        metadata={
            "yaml_attrs": YamlAttrs(
                name="qa_output_dir",
                descr="""Output directory to store all QA output files.""",
            )
        },
    )

    def __post_init__(self):
        # VALIDATE INPUTS

        if not isinstance(self.qa_output_dir, str):
            raise TypeError(f"`qa_output_dir` must be a str")

        # If this directory does not exist, make it.
        if not os.path.isdir(self.qa_output_dir):
            log = nisarqa.get_logger()
            log.info(f"Creating QA output directory: {self.qa_output_dir}")
            os.makedirs(self.qa_output_dir, exist_ok=True)

    @staticmethod
    def get_path_to_group_in_runconfig():
        return ["runconfig", "groups", "product_path_group"]


@dataclass(frozen=True)
class ValidationGroupParamGroup(YamlParamGroup):
    """
    Parameters from the Validation Group runconfig group.

    Parameters
    ----------
    metadata_cubes_fail_if_all_nan : str
        True to raise an exception if one or more metadata cubes contains
        all non-finite (e.g. Nan, +/- Inf) values, or if one or more
        z-dimension height layers in a 3D cube has all non-finite values.
        False to quiet the exception (although it will still be logged).
        Defaults to True.
    """

    metadata_cubes_fail_if_all_nan: bool = field(
        default=True,
        metadata={
            "yaml_attrs": YamlAttrs(
                name="metadata_cubes_fail_if_all_nan",
                descr="""True to raise an exception if one or more metadata cubes contains
                all non-finite (e.g. Nan, +/- Inf) values, or if one or more
                z-dimension height layers in a 3D cube has all non-finite values.
                False to quiet the exception (although it will still be logged).""",
            )
        },
    )

    def __post_init__(self):
        # VALIDATE INPUTS
        if not isinstance(self.metadata_cubes_fail_if_all_nan, bool):
            raise TypeError(
                f"`{self.metadata_cubes_fail_if_all_nan=}`, must be bool."
            )

    @staticmethod
    def get_path_to_group_in_runconfig():
        return ["runconfig", "groups", "qa", "validation"]


@dataclass
class RootParamGroup(ABC):
    """Abstract Base Class for all NISAR Products' *RootParamGroup"""

    workflows: WorkflowsParamGroup
    input_f: Optional[InputFileGroupParamGroup] = None
    prodpath: Optional[ProductPathGroupParamGroup] = None
    validation: Optional[ValidationGroupParamGroup] = None

    # Create a namedtuple which maps the workflows requested
    # in `workflows` to their corresponding *RootParamGroup attribute(s)
    # and runconfig groups.
    # Structure of each namedtuple:
    # (<bool of whether the requested workflows require this *ParamGroup>,
    #     <str name of the *RootParamGroup attribute to store the *ParamGroup>,
    #         <class object for the corresponding *ParamGroup>)
    ReqParamGrp: ClassVar[collections.namedtuple] = collections.namedtuple(
        typename="ReqParamGrp",
        field_names=[
            "flag_param_grp_req",
            "root_param_grp_attr_name",
            "param_grp_cls_obj",
        ],
    )

    def __post_init__(self):
        # Ensure that the minimum parameters were provided
        msg = (
            "`%s` parameter of type `%s` is required for the requested "
            "QA workflow(s)."
        )

        mapping_of_req_wkflws2params = (
            self.get_mapping_of_workflows2param_grps_from_self()
        )

        for param_grp in mapping_of_req_wkflws2params:
            if param_grp.flag_param_grp_req:
                attr = getattr(self, param_grp.root_param_grp_attr_name)
                if not isinstance(attr, param_grp.param_grp_cls_obj):
                    raise TypeError(
                        msg
                        % (
                            param_grp.root_param_grp_attr_name,
                            str(param_grp.param_grp_cls_obj),
                        )
                    )

    def get_mapping_of_workflows2param_grps_from_self(self):
        """Wrapper to call `get_mapping_of_workflows2param_grps` on
        the current instance of *RootParamGroup"""
        return self.get_mapping_of_workflows2param_grps(self.workflows)

    @staticmethod
    @abstractmethod
    def get_mapping_of_workflows2param_grps(workflows):
        """
        Return a tuple of tuples which map the workflows requested in this
        class' WorkflowsParamGroup attribute to their corresponding
        *RootParamGroup attribute(s) and runconfig groups.

        Parameters
        ----------
        workflows : WorkflowsParamGroup
            An instance of WorkflowsParamGroup with attributes that correspond
            to this product's *RootParams object.

        Returns
        -------
        grps_to_parse : Tuple of RootParamGroup.ReqParamGrp
            This tuple maps the workflows requested in the WorkflowsParamGroup
            attribute to their corresponding *RootParamGroup attribute(s)
            and runconfig groups.
            Structure of each inner named tuple:

            (<bool of whether the requested workflows require this *ParamGroup>,
                <str name of the *RootParamGroup attribute to store the *ParamGroup>,
                    <class object for the corresponding *ParamGroup>)
        """
        pass

    @staticmethod
    @abstractmethod
    def get_order_of_groups_in_yaml():
        """
        Return the order that parameter groups should appear in the output
        runconfig template file.

        Returns
        -------
        param_group_class_objects : tuple of *ParamGroup class objects
            Tuple containing the *ParamGroup class objects for the
            fields in this *RootParamGroup class. The order that these
            *ParamGroup objects appear in the tuple is the order that
            they will appear in the output runconfig yaml template file.

        Examples
        --------
        A call to `RSLCRootParamGroup.get_order_of_groups_in_yaml()`
        would return**:

            (InputFileGroupParamGroup,
            DynamicAncillaryFileParamGroup,
            ProductPathGroupParamGroup,
            RSLCWorkflowsParamGroup,
            BackscatterImageParamGroup,
            HistogramParamGroup,
            AbsCalParamGroup,
            NESZParamGroup,
            PointTargetAnalyzerParamGroup
            )

        ** Exact RSLC groups are subject to change, based on code updates
           and new features added to RSLC QA-SAS.
        """
        pass

    @classmethod
    def dump_runconfig_template(cls, indent=4):
        """Output the runconfig template (with default values) to stdout.

        Parameters
        ----------
        indent : int, optional
            Number of spaces of an indent. Defaults to 4.
        """

        # Build a ruamel yaml object that contains the runconfig structure
        yaml = YAML()

        # Here, the `mapping` parameter sets the size of an indent for the
        # mapping keys (aka the variable names) in the output yaml file. But,
        # it does not set the indent for the in-line comments in the output
        # yaml file; the indent spacing for inline comments will need to be
        # set later while creating the commented maps.
        # Re: `sequence` and `offset` parameters -- At the time of writing,
        # the current QA implementation of `add_param_to_cm` specifies that
        # lists should always be dumped inline, which means that these
        # `sequence` and `offset` parameters are a moot point. However,
        # should that underlying implementation change, settings sequence=4,
        # offset=2 results in nicely-indented yaml files.
        # Ref: https://yaml.readthedocs.io/en/latest/detail.html#indentation-of-block-sequences
        yaml.indent(mapping=indent, sequence=indent, offset=max(indent - 2, 0))

        runconfig_cm = CommentedMap()

        # Populate the yaml object. This order determines the order
        # the groups will appear in the runconfig.
        param_group_class_objects = cls.get_order_of_groups_in_yaml()

        for param_grp in param_group_class_objects:
            param_grp.populate_runcfg(runconfig_cm, indent=indent)

        # output to console. Let user stream that into a file.
        yaml.dump(runconfig_cm, sys.stdout)

    def save_processing_params_to_stats_h5(
        self, h5_file: h5py.File, band: str
    ) -> None:
        """
        Populate the HDF5 file's processing group with QA processing parameters.

        This function updates the "processing" group in `h5_file` with each
        parameter in this instance of *RootParams that contains `hdf5_attrs`
        metadata. It also adds the QA software version.

        Parameters
        ----------
        h5_file : h5py.File
            Handle to an h5 file where the processing metadata should be saved.
        band : str
            The letter of the band. Ex: "L" or "S".
        """
        for params_obj in fields(self):
            po = getattr(self, params_obj.name)
            # If a workflow was not requested, its RootParams attribute
            # will be None, so there will be no params to add to the h5 file
            if po is not None:
                if issubclass(type(po), HDF5ParamGroup):
                    po.write_params_to_h5(h5_file, band=band)

        # Add QA version to stats file
        nisarqa.create_dataset_in_h5group(
            h5_file=h5_file,
            grp_path=nisarqa.STATS_H5_QA_PROCESSING_GROUP % band,
            ds_name="QASoftwareVersion",
            ds_data=nisarqa.__version__,
            ds_description="QA software version used for processing",
            ds_units=None,
        )

    def log_parameters(self):
        """Log the parameter values for this instance of *RootParamGroup."""
        log = nisarqa.get_logger()

        log.debug(
            "QA processing parameters, per runconfig and defaults (runconfig"
            " has precedence)"
        )

        # Iterate through each *ParamGroup attribute in the *RootParamGroup
        for root_group_attr in fields(self):
            param_group_obj = getattr(self, root_group_attr.name)

            # Iterate through each attribute in the *ParamGroup
            if param_group_obj is not None:
                # One of the `workflows` set to `True` required this group,
                # so this *ParamGroup was instantiated and its values logged.

                # Use the path in the runconfig to identify the group
                rncfg_grp_path = (
                    param_group_obj.get_path_to_group_in_runconfig()
                )
                rncfg_grp_path = "/".join(rncfg_grp_path)
                log.debug(
                    "  Final Input Parameters corresponding to Runconfig"
                    f" group: {rncfg_grp_path}"
                )

                # Show the final value assigned to the parameter
                for param in fields(param_group_obj):
                    po2 = getattr(param_group_obj, param.name)
                    log.debug(f"    {param.name}: {po2}")
            else:
                log.debug(
                    "  Per `workflows`, runconfig group for"
                    f" {root_group_attr.name} not required."
                )

    def get_output_dir(self) -> Path:
        """
        Returns the filepath to the output directory.

        Returns
        -------
        filepath : Path
            The filepath to the QA output directory.
        """
        if self.prodpath is None:
            raise ValueError("Output directory not provided via runconfig.")

        return Path(self.prodpath.qa_output_dir)

    def _get_input_file_basename_stripped(self) -> str:
        """
        Returns the input file's basename with the extension stripped.

        This can be used for creating the QA output filenames.

        Returns
        -------
        basename : str
            The input file's basename with the extension removed.
            For example, if the input filename is "/home/qa/my_goff.h5",
            then `basename` will be "my_goff".
        """
        if self.input_f is None:
            raise ValueError("Input filename not provided via runconfig.")

        return self.input_f.qa_input_file

    def get_browse_png_filename(self) -> Path:
        """Return the browse image filename as a Path object. Does not include
        the filepath.
        """
        # # For R3.3, QA should not use the input filename for the output files.
        # # However, if/when this change occurs, here is the new code to use:
        #
        # return Path(f{self._get_input_file_basename_stripped}.png")

        return Path("BROWSE.png")

    def get_kml_browse_filename(self) -> Path:
        """Return the browse kml filename as a Path object. Does not include
        the filepath.
        """
        # # For R3.3, QA should not use the input filename for the output files.
        # # However, if/when this change occurs, here is the new code to use:
        #
        # return Path(f{self._get_input_file_basename_stripped}.kml")

        return Path("BROWSE.kml")

    def get_summary_csv_filename(self) -> Path:
        """Return the Pass/Fail checks summary csv filename as a Path object.
        Does not include the filepath.
        """
        # # For R3.3, QA should not use the input filename for the output files.
        # # However, if/when this change occurs, here is the new code to use:
        #
        # return Path(f{self._get_input_file_basename_stripped}_QA_SUMMARY.csv")

        return Path("SUMMARY.csv")

    def get_report_pdf_filename(self) -> Path:
        """Return the reports PDF filename as a Path object. Does not include
        the filepath.
        """
        # # For R3.3, QA should not use the input filename for the output files.
        # # However, if/when this change occurs, here is the new code to use:
        #
        # return Path(f{self._get_input_file_basename_stripped}_QA_REPORT.pdf")

        return Path("REPORT.pdf")

    def get_stats_h5_filename(self) -> Path:
        """Return the stats HDF5 filename as a Path object. Does not include
        the filepath.
        """
        # # For R3.3, QA should not use the input filename for the output files.
        # # However, if/when this change occurs, here is the new code to use:
        #
        # return Path(f{self._get_input_file_basename_stripped}_QA_STATS.h5")

        return Path("STATS.h5")

    def get_log_filename(self) -> Path:
        """Return the log TXT filename as a Path object. Does not include
        the filepath.
        """
        # # For R3.3, QA should not use the input filename for the output files.
        # # However, if/when this change occurs, here is the new code to use:
        #
        # return Path(f{self._get_input_file_basename_stripped}_QA_LOG.txt")

        return Path("LOG.txt")


__all__ = nisarqa.get_all(__name__, objects_to_skip)
