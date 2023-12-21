from __future__ import annotations

import csv
import logging
import os
from dataclasses import InitVar, dataclass

# import nisarqa

# objects_to_skip = nisarqa.get_all(name=__name__)


@dataclass
class GetSummary:
    """The global SUMMARY.csv writer for PASS/FAIL checks."""

    csv_file: InitVar[str | os.PathLike | None] = None

    def __post_init__(self, csv_file):
        logger = self._get_summary_logger()
        if not logger.handlers:
            # First time GetSummary is instantiated; set up the internal logger
            if csv_file is None:
                raise ValueError(
                    f" `{csv_file=}` but must be provided the first time that"
                    " an instance of GetSummary() is generated. For all"
                    " subsequent calls, user should use the default of None."
                )
            csv_file = os.fspath(csv_file)
            self._setup_summary_csv(csv_file=csv_file)

        else:
            # Logger was already set up
            if csv_file is not None:
                raise ValueError(
                    f" `{csv_file=}` but can only be provided the first time"
                    " that an instance of GetSummary() is generated. For all"
                    " subsequent calls, please use the default of None."
                )

    @staticmethod
    def _get_summary_logger() -> logging.Logger:
        """Get the underlying Logger for the SUMMARY CSV file."""
        return logging.getLogger("SUMMARY")

    def _setup_summary_csv(self, csv_file: str | os.PathLike) -> None:
        """
        Setup the SUMMARY CSV file with correct filepath and formatting.

        Parameters
        ----------
        csv_file : path-like
            Filepath (with basepath) to the SUMMARY file.
        """

        # Internal to the GetSummary class, we'll use the Python `logging`
        # module to handle formatting, file location, etc..
        # The `logging` module will also allow global access, so we will not
        # need to pass a logger object around in the rest of the code.
        # If QA SAS exits early, a partial output of the SUMMARY CSV will have
        # been generated and written, too.

        # Get the SUMMARY logger
        summary = self._get_summary_logger()

        # The summary file should only be set up once during the execution of QA.
        # If any handlers exist, this means it was previously set up. Bad!
        if summary.handlers:
            raise RuntimeError(
                " `GetSummary._setup_summary_csv(..)` can only be called"
                " once during the execution of QA SAS. This is at least the"
                " second time it was called."
            )

        # Input validation
        if not isinstance(csv_file, (str, os.PathLike)):
            raise TypeError(
                f"`{csv_file=}` and has type {type(csv_file)}, but must be"
                " path-like."
            )

        # Clean the filepath
        csv_file = os.fspath(csv_file)

        # Setup the SUMMARY logger

        # Set minimum log level for the root logger; this sets the minimum
        # possible log level for all handlers. (It typically defaults to WARNING.)
        # Later, set the minimum log level for individual handlers.
        summary.setLevel(logging.DEBUG)

        # Create a formatter
        fmt = logging.Formatter(
            "%(tool)s,%(description)s,%(result)s,%(threshold)s,%(actual)s,%(notes)s"
        )

        # Write messages to the specified file. Since this function will only
        # be called the first time, open in "w" mode for a fresh file.
        handler = logging.FileHandler(filename=csv_file, mode="w")
        handler.setLevel(logging.DEBUG)
        handler.setFormatter(fmt)
        summary.addHandler(handler)

        # Write the header row of the CSV
        # Kludge: `_make_extra()` requires `result` and `tool` to be one of
        # a few acceptable options. This is to ensure that other checks
        # adhere to a strict guidelines to maintain uniformity.
        # The only exception is the header row, so we'll use a hack:
        extra = self._make_extra(
            tool="QA",  # Kludge; will be corrected below
            description="Check",
            result="PASS",  # Kludge; will be corrected below
            threshold="Threshold",
            actual="Actual",
            notes="Notes",
        )
        extra["result"] = "Result"
        extra["tool"] = "Tool"

        self._write_to_csv(extra=extra)

    def _validate_result(self, result: str) -> str:
        """
        Validates that `result` is either 'PASS' or 'FAIL'.

        Parameters
        ----------
        result : str
            Either 'PASS' or 'FAIL'.

        Returns
        -------
        out : str
            Same as `result`, assuming `result` passes validation.

        Raises
        ------
        ValueError : If `result` is neither 'PASS' nor 'FAIL'.
        """
        result = self._validate_string(result)
        if result not in ("PASS", "FAIL"):
            raise ValueError(f"`{result=}`, must be either 'PASS' or 'FAIL'.")
        return result

    def _validate_tool(self, tool: str) -> str:
        """
        Validates that `tool` is one of "QA", "AbsCal", "PTA", "NESZ".

        Parameters
        ----------
        tool : str
            One of "QA", "AbsCal", "PTA", "NESZ".

        Returns
        -------
        out : str
            Same as `tool`, assuming `tool` passes validation.

        Raises
        ------
        ValueError : If `tool` is not one of "QA", "AbsCal", "PTA", "NESZ".
        """
        tool = self._validate_string(tool)
        tools = ("QA", "AbsCal", "PTA", "NESZ")
        if tool not in tools:
            raise ValueError(f"`{tool=}`, must be one of {tools}.")
        return tool

    @staticmethod
    def _validate_string(my_str: str) -> str:
        """Validates that `my_str` is a string; if so, returns `my_str`."""
        if isinstance(my_str, str):
            return my_str
        else:
            raise TypeError(
                f"`{my_str=}` and has type {type(my_str)}, must be type string."
            )

    def _make_extra(
        self,
        description: str,
        result: str,
        threshold="",
        actual="",
        notes="",
        tool: str = "QA",
    ) -> dict[str, str]:
        """Make the `extra` dictionary that will populate a row of the CSV."""

        # The keys in `extra` must key must match exactly the user-defined
        # fields in the underlying logger's format string.
        # This formatter string is set in `_setup_summary_csv()`.
        extra = {
            "tool": self._validate_tool(tool),
            "description": self._validate_string(description),
            "result": self._validate_result(result),
            "threshold": self._validate_string(threshold),
            "actual": self._validate_string(actual),
            "notes": self._validate_string(notes),
        }
        return extra

    def _write_to_csv(self, extra: dict[str, str]) -> None:
        """
        Write the contents of `extra` in a new row to the CSV file.

        Parameters
        ----------
        extra : dict[str, str]
            A dictionary which is used to populate the new row in the CSV.
            It must contain the identical keys to the dictionary returned
            by `_make_extra()`. Any additional keys will be ignored.
            Missing keys will result in an error.
        """
        summary = self._get_summary_logger()

        # `logging` requires the `msg` keyword, but since the formatter string
        # does not include place for `msg`, it will be ignored.
        # Use "" as dummy value.
        summary.info(msg="", extra=extra)

    def check_can_open_input_file(self, result: str) -> None:
        extra = self._make_extra(
            description="Able to open NISAR input file?", result=result
        )
        self._write_to_csv(extra)

    def check_statistics_within_threshold(
        self, result: str, notes: str
    ) -> None:
        extra = self._make_extra(
            description="Statistics within acceptable threshold?",
            result=result,
            notes=notes,
        )
        self._write_to_csv(extra)


# Example usage:
summary = GetSummary("./test_log.txt")

summary.check_can_open_input_file(result="PASS")

sum2 = GetSummary()

summary.check_statistics_within_threshold(result="FAIL", notes="RSLC_L_A_HH")

summary.check_can_open_input_file(result="FAIL")


# __all__ = nisarqa.get_all(__name__, objects_to_skip)
