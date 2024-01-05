from __future__ import annotations

import logging
import os
from dataclasses import InitVar, dataclass

import nisarqa

objects_to_skip = nisarqa.get_all(name=__name__)


@dataclass
class GetSummary:
    """
    The global SUMMARY.csv writer for PASS/FAIL checks.

    This class can be called from anywhere in the nisarqa package, and it
    will always write the pass/fail checks to the same SUMMARY.csv file.

    Notes
    -----
    Looking under the hood, this class is a wrapper around the Python
    logging module, using a logger called "SUMMARY". By using the logging
    module, we some nice features for free:
        * The logger can be set up once at the beginning of QA, and then
          be called from anywhere in the nisarqa package while still
          retaining the same formatting, output file, etc.
        * In the event of a crash, whatever was already written to the
          SUMMARY.csv file will still be there. (We will not lose it.)
        * The logger enforces a strict format, ensuring that we do not
          accidentally write a row that is missing the spot for an entry
    Unfortunately, by using the logger and writing as we go, certain things
    are much trickier:
        * We cannot have a list of checks, and then ensure that each check
          actually occurred.
        * The order of the checks is dependent upon the order of execution
          of the source code; if functions in the source code get reordered,
          then the order that the checks are printed in the SUMMARY.csv

    There are pros and cons to both approaches. The first approach provides
    protection against crashes and is less invasive to integrate into
    the existing code, hence choosing that approach for now.
    """

    csv_file: InitVar[str | os.PathLike | None] = None

    def __post_init__(self, csv_file: str | os.PathLike | None) -> None:
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
        # Kludge: `_construct_extra_dict()` requires `result` and `tool` to be
        # one of a set of acceptable options. This is to ensure that PASS/FAIL checks
        # adhere to strict guidelines to maintain uniformity.
        # The only exception is the header row, so we'll use a hack:
        extra = self._construct_extra_dict(
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
        Validate that `result` is either 'PASS' or 'FAIL'.

        Parameters
        ----------
        result : str
            Either 'PASS' or 'FAIL'.

        Raises
        ------
        ValueError
            If `result` is neither 'PASS' nor 'FAIL'.
        """
        self._validate_string(result)
        if result not in ("PASS", "FAIL"):
            raise ValueError(f"`{result=}`, must be either 'PASS' or 'FAIL'.")

    def _validate_description(self, description: str) -> str:
        """
        Check that `description` is a string and not too long.

        Parameters
        ----------
        description : str
            The description field for a summary check.
        """
        self._validate_string(description)
        if len(description) > 88:
            nisarqa.get_logger().warning(
                f"{description=}, and has length {len(description)}. Consider"
                " making it more concise."
            )

    def _validate_tool(self, tool: str) -> str:
        """
        Validate that `tool` is one of "QA", "AbsCal", "PTA", "NESZ".

        Parameters
        ----------
        tool : str
            One of "QA", "AbsCal", "PTA", "NESZ".

        Raises
        ------
        ValueError : If `tool` is not one of "QA", "AbsCal", "PTA", "NESZ".
        """
        self._validate_string(tool)
        tools = ("QA", "AbsCal", "PTA", "NESZ")
        if tool not in tools:
            raise ValueError(f"`{tool=}`, must be one of {tools}.")

    @staticmethod
    def _validate_string(s: str) -> str:
        """Validate that `s` is a string and does not contain commas."""
        if not isinstance(s, str):
            raise TypeError(f"`{s=}` and has type {type(s)}, must be type str.")

        if "," in s:
            raise ValueError(
                "Given string contains a comma (','). This is used in a"
                " a CSV file with comma delimiters, so there cannot be"
                f" additional commas. Provided string: '{s}'"
            )

    def _construct_extra_dict(
        self,
        description: str,
        result: str,
        threshold: str = "",
        actual: str = "",
        notes: str = "",
        tool: str = "QA",
    ) -> dict[str, str]:
        """Make the `extra` dictionary that will populate a row of the CSV."""

        self._validate_tool(tool)
        self._validate_description(description)
        self._validate_result(result)
        self._validate_string(threshold)
        self._validate_string(actual)
        self._validate_string(notes)

        # The keys in `extra` must match exactly the user-defined
        # fields in the underlying logger's format string.
        # This formatter string is set in `_setup_summary_csv()`.
        extra = {
            "tool": tool,
            "description": description,
            "result": result,
            "threshold": threshold,
            "actual": actual,
            "notes": notes,
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
            by `_construct_extra_dict()`. Any additional keys will be ignored.
            Missing keys will result in an error.
        """
        summary = self._get_summary_logger()

        # `logging` requires the `msg` keyword, but since the formatter string
        # does not include place for `msg`, it will be ignored.
        # Use "" as dummy value.
        summary.info(msg="", extra=extra)

    def check(
        self,
        description: str,
        result: str,
        threshold: str = "",
        actual: str = "",
        notes: str = "",
        tool: str = "QA",
    ) -> None:
        """
        Write a PASS/FAIL check to the SUMMARY CSV file.

        Parameters
        ----------
        description : str
            The PASS/FAIL check. Note: this should be a short, unabiguous
            phrase or question that has a response of "PASS" or "FAIL".
            Example 1: "Able to open NISAR input file?"
            Example 2: "Percentage of invalid pixels under threshold?"
        result : str
            The result of the check. Must be one of: "PASS" or "FAIL"
        threshold : str, optional
            If a threshold value was used to determine the outcome of the
            PASS/FAIL check, note that value here. Defaults to the empty
            string `""`, meaning that no threshold was used.
        actual : str, optional
            The quantity determined from the input NISAR product which was used
            to compare against the threshold value, etc. in order to
            determine the outcome of the PASS/FAIL check. Defaults to the empty
            string `""`, meaning that no meaningful actual value was used.
            Example 1: the check "Able to open NISAR input file?" would not
                involve an quantative value, so leave this as the empty string.
            Example 2: the check "All statistics under acceptable threshold?"
                might set `actual` to 10.4, to indicate that 10.4% of computed metrics were below the threshold.
        notes : str, optional
            Additional notes to better describe the check. For example,
            this would be a good place for the name of the raster being checked.
        tool : str, optional
            Short name for the QA workflow generating the summary check.
            One of: "QA", "PTA", "AbsCal", "NESZ". Defaults to "QA".
        """
        extra = self._construct_extra_dict(
            description=description,
            result=result,
            threshold=threshold,
            actual=actual,
            notes=notes,
            tool=tool,
        )

        self._write_to_csv(extra)

    def check_can_open_input_file(self, result: str) -> None:
        """Check: 'Able to open input NISAR file?'"""
        self.check(description="Able to open input NISAR file?", result=result)

    def check_invalid_pixels_within_threshold(
        self, result: str, threshold: str, actual: str, notes: str = ""
    ) -> None:
        """Check: '% Non-finite and/or 'fill' pixels under threshold?'"""
        self.check(
            description="% Non-finite and/or 'fill' pixels under threshold?",
            threshold=threshold,
            actual=actual,
            result=result,
            notes=notes,
        )

    def check_nan_pixels_within_threshold(
        self, result: str, threshold: str, actual: str, notes: str = ""
    ) -> None:
        """Check: '% NaN pixels under threshold?'"""
        self.check(
            description="% NaN pixels under threshold?",
            threshold=threshold,
            actual=actual,
            result=result,
            notes=notes,
        )

    def check_QA_completed_no_exceptions(self, result: str) -> None:
        """Check: 'QA SAS completed with no exceptions?'"""
        self.check(
            description="QA completed with no exceptions?",
            result=result,
        )


__all__ = nisarqa.get_all(__name__, objects_to_skip)
