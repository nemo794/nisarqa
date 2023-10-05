import os
import warnings

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from PIL import Image

import nisarqa

objects_to_skip = nisarqa.get_all(name=__name__)

LOG_TXT = """PLACEHOLDER FILE -- NOT ACTUAL LOG OUTPUTS
<time-tag>, <log level>, <workflow>, <module>, <error code>, <error location>, <logged message>
PLACEHOLDER FILE -- NOT ACTUAL LOG OUTPUTS
"""

SUMMARY_CSV = """Tool,Check Description,Result,Threshold,Actual,Notes
QA,Able to open NISAR input file?,PASS,,,PLACEHOLDER
"""

KML_FILE = """<?xml version="1.0" encoding="UTF-8"?>
<kml xmlns:gx="http://www.google.com/kml/ext/2.2">
  <Document>
    <name>PLACEHOLDER ONLY</name>
    <Folder>
      <name>PLACEHOLDER ONLY</name>
      <GroundOverlay>
        <name>PLACEHOLDER ONLY</name>
        <Icon>
          <href>BROWSE.png</href>
        </Icon>
        <gx:LatLonQuad>
          <coordinates>-122.766411,33.135963 -120.086220,33.537392 -119.730453,31.742420 -122.355446,31.338583</coordinates>
        </gx:LatLonQuad>
      </GroundOverlay>
    </Folder>
  </Document>
</kml>
"""


def output_stub_files(output_dir, stub_files="all", input_file=None):
    """This function outputs stub files for the NISAR QA L1/L2 products.

    Parameters
    ----------
    output_dir : str
        Filepath for the output directory to place output stub files
    stub_files : str, List of str, optional
        Which file(s) to save into the output directory. Options:
            'browse_png'
            'browse_kml'
            'summary_csv'
            'log_txt'
            'stats_h5'
            'report_pdf'
            'all'
        If 'all' is selected, then all six of the stub files will be generated
        and saved. If a single string is provided, only that file will be
        output. To save a subset of the available stub files, provide them
        as a list of strings.
        Ex: 'all', 'browse_png', or ['summary_csv', 'log_txt'] are valid inputs
    input_file : str or None, optional
        The input NISAR product HDF5 file. Only required/used for saving
        the stub stats_h5 file.

    """
    ## Validate inputs
    opts = [
        "browse_png",
        "browse_kml",
        "summary_csv",
        "log_txt",
        "stats_h5",
        "report_pdf",
    ]
    if stub_files == "all":
        stub_files = opts

    if isinstance(stub_files, str):
        stub_files = [stub_files]

    # ensure that the inputs are a subset of the valid options
    assert set(stub_files) <= set(
        opts
    ), "invalid input for argument `stub_files`"

    if "stats_h5" in stub_files:
        assert input_file is not None, (
            "to generate a stub STATS.h5, a valid NISAR product input file must"
            " be provided."
        )
        assert os.path.isfile(
            input_file
        ), f"`input_file` is not a valid file: {input_file}"
        assert input_file.endswith(
            ".h5"
        ), f"`input_file` must have the extension .h5: {input_file}"

    # If output directory does not exist, make it.
    os.makedirs(output_dir, exist_ok=True)

    ## Save stub files
    # Save geolocation stub file
    if "browse_kml" in stub_files:
        with open(os.path.join(output_dir, "BROWSE.kml"), "w") as f:
            f.write(KML_FILE)

    # Save summary.csv stub file
    if "summary_csv" in stub_files:
        with open(os.path.join(output_dir, "SUMMARY.csv"), "w") as f:
            f.write(SUMMARY_CSV)

    # Save Log file stub file
    if "log_txt" in stub_files:
        with open(os.path.join(output_dir, "LOG.txt"), "w") as f:
            f.write(LOG_TXT)

    # Save stats.h5 stub file
    if "stats_h5" in stub_files:
        stats_file = os.path.join(output_dir, "STATS.h5")

        with nisarqa.open_h5_file(
            input_file, mode="r"
        ) as in_file, nisarqa.open_h5_file(stats_file, mode="w") as stats_h5:
            for band in nisarqa.NISAR_BANDS:
                grp_path = f"/science/{band}SAR/identification"

                if grp_path in in_file:
                    # Copy identification metadata from input file to stats.h5
                    in_file.copy(in_file[grp_path], stats_h5, grp_path)

    # Save browse image stub file and PDF stub file
    # Create a roughly 2048x2048 pixels^2 RGB image
    # (ASF allows for in-exact dimensions, so let's test that.)
    # (Current plan is for all NISAR products to generate RGBA browse images)
    imarray = np.random.randint(
        low=0, high=256, size=(1800, 2000, 4), dtype=np.uint8
    )

    # Make all pixels opaque by setting the alpha channel to 255
    imarray[:, :, 3] = 255

    # Make a subset of the pixels transparent by setting alpha channel to 0
    imarray[500:900, 500:900, 3] = 0

    if "browse_png" in stub_files:
        im = Image.fromarray(imarray).convert("RGBA")
        datas = im.getdata()
        newData = []
        for item in datas:
            newData.append(item)
        im.putdata(newData)
        im.save(os.path.join(output_dir, "BROWSE.png"))

    if "report_pdf" in stub_files:
        # Save image into a .pdf
        with PdfPages(os.path.join(output_dir, "REPORT.pdf")) as f:
            # Instantiate the figure object
            fig = plt.figure()
            ax = plt.gca()

            # Plot the img_arr image.
            ax_img = ax.imshow(X=imarray, cmap=plt.cm.ocean)
            plt.colorbar(ax_img, ax=ax)

            plt.xlabel("Placeholder x-axis label")
            plt.ylabel("Placeholder y-axis label")
            plt.title(
                "PLACEHOLDER IMAGE - NOT REPRESENTATIVE OF ACTUAL NISAR PRODUCT"
            )

            # Make sure axes labels do not get cut off
            fig.tight_layout()

            # Append figure to the output PDF
            f.savefig(fig)

            # Close the figure
            plt.close(fig)


def get_input_file(user_rncfg, in_file_param="qa_input_file"):
    """
    Parse input file name from the given runconfig.

    Parameters
    ----------
    user_rncfg : dict
        The user runconfig file, parsed into a Python dictionary.
    in_file_param : str
        The name of the runconfig parameter designating the input file.

    Returns
    -------
    input_file : str
        The argument value of the `in_file_param` in `user_rncfg`.
    """

    rncfg_path = ("runconfig", "groups", "input_file_group")
    try:
        params_dict = nisarqa.get_nested_element_in_dict(user_rncfg, rncfg_path)
    except KeyError as e:
        raise KeyError(
            "`input_file_group` is a required runconfig group"
        ) from e
    try:
        input_file = params_dict[in_file_param]
    except KeyError as e:
        raise KeyError(
            f"`{in_file_param}` is a required parameter for QA"
        ) from e

    return input_file


def get_output_dir(user_rncfg):
    """Parse output directory from the given runconfig.

    Parameters
    ----------
    user_rncfg : dict
        The user runconfig file, parsed into a Python dictionary.

    Returns
    -------
    output_dir : str
        The argument value of the `product_path_group > qa_output_dir` parameter
        in `user_rncfg`. If a value is not found, will default to './qa'
    """

    rncfg_path = ("runconfig", "groups", "product_path_group")
    output_dir = "./qa"

    try:
        params_dict = nisarqa.get_nested_element_in_dict(user_rncfg, rncfg_path)
    except KeyError:
        # group not found in runconfig. Use defaults.
        warnings.warn(
            "`product_path_group` not found in runconfig. "
            "Using default output directory."
        )
    else:
        try:
            output_dir = params_dict["qa_output_dir"]

        except KeyError:
            # parameter not found in runconfig. Use defaults.
            warnings.warn(
                "`qa_output_dir` not found in runconfig. "
                "Using default output directory."
            )

    return output_dir


def get_workflows(
    user_rncfg, rncfg_path=("runconfig", "groups", "qa", "workflows")
):
    """
    Parse workflows group from the given runconfig path.

    Parameters
    ----------
    user_rncfg : dict
        The user runconfig file, parsed into a Python dictionary.
    rncfg_path : sequence of str
        The nested path in the runconfig to a specific `workflows` group.
        Example for RSLC runconfig: ('runconfig','groups','qa','workflows')

    Returns
    -------
    validate : bool
        The argument value of the `workflows > validate` parameter
        in `user_rncfg`.
    qa_reports : bool
        The argument value of the `workflows > qa_reports` parameter
        in `user_rncfg`.
    """

    validate = False
    qa_reports = False
    try:
        params_dict = nisarqa.get_nested_element_in_dict(user_rncfg, rncfg_path)
    except KeyError:
        # group not found in runconfig. Use defaults.
        warnings.warn("`workflows` not found in runconfig. Using defaults.")
    else:
        try:
            validate = params_dict["validate"]
        except KeyError:
            # parameter not found in runconfig. Use default.
            warnings.warn(
                "`validate` not found in runconfig. "
                "Using default `validate` setting."
            )
        try:
            qa_reports = params_dict["qa_reports"]
        except KeyError:
            # parameter not found in runconfig. Use default.
            warnings.warn(
                "`qa_reports` not found in runconfig. "
                "Using default `qa_reports` setting."
            )

    return validate, qa_reports


def verify_insar(user_rncfg, product):
    """
    Parse the runconfig and generate stub outputs for InSAR products.

    Parameters
    ----------
    user_rncfg : dict
        A dictionary whose structure matches this product's QA runconfig
        YAML file and which contains the parameters needed to run its QA SAS.
    product : str
        InSAR product name
        Options: 'rifg','runw','gunw','roff','goff'
    """

    assert product in ("rifg", "runw", "gunw", "roff", "goff")

    # Step 1: Get workflows flags for the requested product
    wkflw_path = ("runconfig", "groups", "qa", product, "workflows")
    validate, qa_reports = get_workflows(user_rncfg, rncfg_path=wkflw_path)

    # Step 2: If any workflows flags are true, run QA
    if not (validate or qa_reports):
        # Early exit if no workflows are requested
        print("No `workflows` requested, so no QA outputs will be generated")
        return

    # Step 3: "Run QA"
    in_file_param = f"qa_{product}_input_file"
    input_file = get_input_file(user_rncfg, in_file_param=in_file_param)
    output_dir = get_output_dir(user_rncfg)

    # add subdirectory for the insar product to store its outputs
    output_dir = os.path.join(output_dir, product)

    if qa_reports:
        # output stub files
        output_stub_files(output_dir, stub_files="all", input_file=input_file)
    elif validate:
        output_stub_files(
            output_dir,
            stub_files=["summary_csv", "log_txt"],
            input_file=input_file,
        )


__all__ = nisarqa.get_all(__name__, objects_to_skip)
