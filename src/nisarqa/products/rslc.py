import functools
import numbers
import os
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime

import h5py
import numpy as np
import numpy.typing as npt
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import FuncFormatter
from PIL import Image

import nisarqa

# List of objects from the import statements that
# should not be included when importing this module
objects_to_skip = nisarqa.get_all(name=__name__)


def verify_rslc(user_rncfg):
    """
    Verify an RSLC product based on the input file, parameters, etc.
    specified in the input runconfig file.

    This is the main function for running the entire QA workflow. It will
    run based on the options supplied in the input runconfig file.
    The input runconfig file must follow the standard RSLC QA runconfig
    format. Run the command line command 'nisar_qa dumpconfig rslc'
    for an example template with default parameters (where available).

    Parameters
    ----------
    user_rncfg : dict
        A dictionary whose structure matches an this product's QA runconfig
        yaml file and which contains the parameters needed to run its QA SAS.
    """

    # Build the RSLCRootParamGroup parameters per the runconfig
    try:
        root_params = nisarqa.build_root_params(
            product_type="rslc", user_rncfg=user_rncfg
        )
    except nisarqa.ExitEarly:
        # No workflows were requested. Exit early.
        print(
            "All `workflows` set to `False` in the runconfig, "
            "so no QA outputs will be generated. This is not an error."
        )
        return

    # Start logger
    # TODO get logger from Brian's code and implement here
    # For now, output the stub log file.
    nisarqa.output_stub_files(
        output_dir=root_params.get_output_dir(), stub_files="log_txt"
    )

    # Log the values of the parameters.
    # Currently, this prints to stdout. Once the logger is implemented,
    # it should log the values directly to the log file.
    root_params.log_parameters()

    # For readibility, store output filenames in variables.
    # Depending on which workflows are set to True, not all filename
    # variables will be used.
    input_file = root_params.input_f.qa_input_file
    browse_file_png = (
        root_params.get_output_dir() / root_params.get_browse_png_filename()
    )
    browse_file_kml = (
        root_params.get_output_dir() / root_params.get_kml_browse_filename()
    )
    report_file = (
        root_params.get_output_dir() / root_params.get_report_pdf_filename()
    )
    stats_file = (
        root_params.get_output_dir() / root_params.get_stats_h5_filename()
    )
    summary_file = (
        root_params.get_output_dir() / root_params.get_summary_csv_filename()
    )

    print(f"Starting Quality Assurance for input file: {input_file}")

    # Begin QA workflows

    # Run validate first because it checks the product spec
    if root_params.workflows.validate:
        print(f"Beginning input file validation...")
        # TODO Validate file structure
        # (After this, we can assume the file structure for all
        # subsequent accesses to it)
        # NOTE: Refer to the original 'get_bands()' to check that in_file
        # contains metadata, swaths, Identification groups, and that it
        # is SLC/RSLC compliant. These should trigger a fatal error!
        # NOTE: Refer to the original get_freq_pol() for the verification
        # checks. This could trigger a fatal error!

        # These reports will be saved to the SUMMARY.csv file.
        # For now, output the stub file
        nisarqa.output_stub_files(
            output_dir=root_params.get_output_dir(),
            stub_files="summary_csv",
        )
        print(
            f"Input file validation PASS/FAIL checks saved to {summary_file}"
        )
        print(f"Input file validation complete.")

    # If running these workflows, save the processing parameters and
    # identification group to STATS.h5
    if (
        root_params.workflows.qa_reports
        or root_params.workflows.abs_cal
        or root_params.workflows.noise_estimation
        or root_params.workflows.point_target
    ):
        # This is the first time opening the STATS.h5 file for RSLC
        # workflow, so open in 'w' mode.
        # After this, always open STATS.h5 in 'r+' mode.
        with nisarqa.open_h5_file(
            input_file, mode="r"
        ) as in_file, nisarqa.open_h5_file(stats_file, mode="w") as stats_h5:
            pols = nisarqa.rslc.get_pols(in_file)

            # Save the processing parameters to the stats.h5 file
            # Note: If only the validate workflow is requested,
            # this will do nothing.
            root_params.save_params_to_stats_file(
                h5_file=stats_h5, bands=tuple(pols.keys())
            )
            print(f"QA Processing Parameters saved to {stats_file}")

            # Copy the Product identification group to STATS.h5
            nisarqa.rslc.save_NISAR_identification_group_to_h5(
                nisar_h5=in_file, stats_h5=stats_h5
            )
            print(f"Input file Identification group copied to {stats_file}")

    if root_params.workflows.qa_reports:
        print(f"Beginning `qa_reports` processing...")
        # TODO qa_reports will add to the SUMMARY.csv file.
        # For now, make sure that the stub file is output
        if not os.path.isfile(summary_file):
            nisarqa.output_stub_files(
                output_dir=root_params.get_output_dir(),
                stub_files="summary_csv",
            )

        # Due to complexities with ISCE3, the HDF5 input file cannot
        # simultaneously be open as an h5py file handle for QA and also be
        # passed to / opened by ISCE3 to be used for generating the KML.
        # So, create the KML separately.
        nisarqa.write_latlonquad_to_kml(
            llq=nisarqa.get_latlonquad(input_file),
            output_dir=root_params.get_output_dir(),
            kml_filename=root_params.get_kml_browse_filename(),
            png_filename=root_params.get_browse_png_filename(),
        )
        print("Processing of browse image kml complete.")
        print(f"Browse image kml file saved to {browse_file_kml}")

        with nisarqa.open_h5_file(
            input_file, mode="r"
        ) as in_file, nisarqa.open_h5_file(
            stats_file, mode="r+"
        ) as stats_h5, PdfPages(
            report_file
        ) as report_pdf:
            # Note: `pols` contains references to datasets in the open input file.
            # All processing with `pols` must be done within this context manager,
            # or the references will be closed and inaccessible.
            pols = nisarqa.rslc.get_pols(in_file)

            # Save frequency/polarization info to stats file
            save_nisar_freq_metadata_to_h5(stats_h5=stats_h5, pols=pols)

            input_raster_represents_power = False
            name_of_backscatter_content = (
                r"RSLC Backscatter Coefficient ($\beta^0$)"
            )

            # Generate the RSLC Backscatter Image and Browse Image
            process_backscatter_imgs_and_browse(
                pols=pols,
                params=root_params.backscatter_img,
                product_type="rslc",
                stats_h5=stats_h5,
                report_pdf=report_pdf,
                plot_title_prefix=name_of_backscatter_content,
                input_raster_represents_power=input_raster_represents_power,
                browse_filename=browse_file_png,
            )
            print("Processing of backscatter images complete.")
            print(f"Browse image PNG file saved to {browse_file_png}")

            # Generate the RSLC Power and Phase Histograms
            process_backscatter_and_phase_histograms(
                pols=pols,
                params=root_params.histogram,
                stats_h5=stats_h5,
                report_pdf=report_pdf,
                plot_title_prefix=name_of_backscatter_content,
                input_raster_represents_power=input_raster_represents_power,
            )
            print("Processing of backscatter and phase histograms complete.")

            # Process Interferograms

            # Generate Spectra

            # Check for invalid values

            # Compute metrics for stats.h5

            print(f"PDF reports saved to {report_file}")
            print(f"HDF5 statistics saved to {stats_file}")
            print(f"CSV Summary PASS/FAIL checks saved to {summary_file}")
            print("`qa_reports` processing complete.")

    if root_params.workflows.abs_cal:
        print("Beginning Absolute Radiometric Calibration CalTool...")

        # Run Absolute Radiometric Calibration tool
        nisarqa.caltools.run_abscal_tool(
            abscal_params=root_params.abs_cal,
            dyn_anc_params=root_params.anc_files,
            input_filename=input_file,
            stats_filename=stats_file,
        )
        print(
            f"Absolute Radiometric Calibration CalTool results saved to {stats_file}"
        )
        print("Absolute Radiometric Calibration CalTool complete.")

    if root_params.workflows.noise_estimation:
        print("Beginning Noise Estimation Tool CalTool...")

        # Run NET tool
        nisarqa.caltools.run_noise_estimation_tool(
            params=root_params.noise_estimation,
            input_filename=input_file,
            stats_filename=stats_file,
        )
        print(f"Noise Estimation Tool CalTool results saved to {stats_file}")
        print("Noise Estimation Tool CalTool complete.")

    if root_params.workflows.point_target:
        print("Beginning Point Target Analyzer CalTool...")

        # Run Point Target Analyzer tool
        nisarqa.caltools.run_pta_tool(
            pta_params=root_params.pta,
            dyn_anc_params=root_params.anc_files,
            input_filename=input_file,
            stats_filename=stats_file,
        )
        print(f"Point Target Analyzer CalTool results saved to {stats_file}")
        print("Point Target Analyzer CalTool complete.")

    print(
        "Successful completion of QA SAS. Check log file for validation warnings and errors."
    )


# TODO - move to generic NISAR module
def save_NISAR_identification_group_to_h5(nisar_h5, stats_h5):
    """
    Copy the identification group from the input NISAR file
    to the STATS.h5 file.

    For each band in `nisar_h5`, this function will recursively copy
    all available items in the `nisar_h5` group
    '/science/<band>/identification' to the group
    '/science/<band>/identification/*' in `stats_h5`.

    Parameters
    ----------
    nisar_h5 : h5py.File
        Handle to the input NISAR product h5 file
    stats_h5 : h5py.File
        Handle to an h5 file where the identification metadata
        should be saved
    """

    for band in nisar_h5["/science"]:
        src_grp_path = f"/science/{band}/identification"
        dest_grp_path = nisarqa.STATS_H5_IDENTIFICATION_GROUP % band

        if dest_grp_path in stats_h5:
            # The identification group already exists, so copy each
            # dataset, etc. individually
            for item in nisar_h5[src_grp_path]:
                item_path = f"{dest_grp_path}/{item}"
                nisar_h5.copy(nisar_h5[item_path], stats_h5, item_path)
        else:
            # Copy entire identification metadata from input file to stats.h5
            nisar_h5.copy(nisar_h5[src_grp_path], stats_h5, dest_grp_path)


# TODO - move to generic NISAR module
def save_nisar_freq_metadata_to_h5(stats_h5, pols):
    """
    Populate the `stats_h5` HDF5 file with a list of each available
    frequency's polarizations.

    If `pols` contains values for Frequency A, then this dataset will
    be created in `stats_h5`:
        /science/<band>/QA/data/frequencyA/listOfPolarizations

    If `pols` contains values for Frequency B, then this dataset will
    be created in `stats_h5`:
        /science/<band>/QA/data/frequencyB/listOfPolarizations

    * Note: The paths are pulled from the global nisarqa.STATS_H5_QA_FREQ_GROUP.
    If the value of that global changes, then the path for the
    `listOfPolarizations` dataset(s) will change accordingly.

    Parameters
    ----------
    stats_h5 : h5py.File
        Handle to an h5 file where the list(s) of polarizations should be saved
    pols : nested dict of RadarRaster
        Nested dict of RadarRaster objects, where each object represents
        a polarization dataset.
        Format: pols[<band>][<freq>][<pol>] -> a RadarRaster
        Ex: pols['LSAR']['A']['HH'] -> the HH dataset, stored in a RadarRaster object
    """

    # Populate data group's metadata
    for band in pols:
        for freq in pols[band]:
            list_of_pols = list(pols[band][freq])
            grp_path = nisarqa.STATS_H5_QA_FREQ_GROUP % (band, freq)
            nisarqa.create_dataset_in_h5group(
                h5_file=stats_h5,
                grp_path=grp_path,
                ds_name="listOfPolarizations",
                ds_data=list_of_pols,
                ds_description=f"Polarizations for Frequency {freq} "
                "discovered in input NISAR product by QA code",
            )


class ComplexFloat16Decoder(object):
    """Wrapper to read in NISAR product datasets that are '<c4' type,
    which raise an TypeError if accessed naively by h5py.

    Indexing operatations always return data converted to numpy.complex64.

    Parameters
    ----------
    h5dataset : h5py.Dataset
        Dataset to be stored. Dataset should have type '<c4'.

    Notes
    -----
    The ComplexFloat16Decoder class is an example of what the NumPy folks call a 'duck array',
    i.e. a class that exports some subset of np.ndarray's API so that it can be used
    as a drop-in replacement for np.ndarray in some cases. This is different from an
    'array_like' object, which is simply an object that can be converted to a numpy array.
    Reference: https://numpy.org/neps/nep-0022-ndarray-duck-typing-overview.html
    """

    def __init__(self, h5dataset):
        self._dataset = h5dataset
        self._dtype = np.complex64

    def __getitem__(self, key):
        # Have h5py convert to the desired dtype on the fly when reading in data
        return self.read_c4_dataset_as_c8(self.dataset, key)

    @staticmethod
    def read_c4_dataset_as_c8(ds: h5py.Dataset, key=np.s_[...]):
        """
        Read a complex float16 HDF5 dataset as a numpy.complex64 array.
        Avoids h5py/numpy dtype bugs and uses numpy float16 -> float32 conversions
        which are about 10x faster than HDF5 ones.
        """
        # This avoids h5py exception:
        # TypeError: data type '<c4' not understood
        # Also note this syntax changed in h5py 3.0 and was deprecated in 3.6, see
        # https://docs.h5py.org/en/stable/whatsnew/3.6.html

        complex32 = np.dtype([("r", np.float16), ("i", np.float16)])
        z = ds.astype(complex32)[key]

        # Define a similar datatype for complex64 to be sure we cast safely.
        complex64 = np.dtype([("r", np.float32), ("i", np.float32)])

        # Cast safely and then view as native complex64 numpy dtype.
        return z.astype(complex64).view(np.complex64)

    @property
    def dataset(self):
        return self._dataset

    @property
    def dtype(self):
        return self._dtype

    @property
    def shape(self):
        return self.dataset.shape

    @property
    def ndim(self):
        return self.dataset.ndim


# TODO - move to raster.py
@dataclass
class Raster:
    """
    Raster image dataset base class.

    Parameters
    ----------
    data : array_like
        Raster data to be stored. Can be a numpy.ndarray, h5py.Dataset, etc.
    name : str
        Name for the dataset
    band : str
        Name of the band for `img`, e.g. 'LSAR'
    freq : str
        Name of the frequency for `img`, e.g. 'A' or 'B'
    pol : str
        Name of the polarization for `img`, e.g. 'HH' or 'HV'
    """

    # Raster data. Could be a numpy.ndarray, h5py.Dataset, etc.
    data: npt.ArrayLike

    # identifying name of this Raster; can be used for logging
    # e.g. 'LSAR_A_HH'
    name: str

    band: str
    freq: str
    pol: str


# TODO - move to raster.py
@dataclass
class SARRaster(ABC, Raster):
    """Abstract Base Class for SAR Raster dataclasses."""

    @property
    @abstractmethod
    def y_axis_spacing(self):
        """Pixel Spacing in Y direction (azimuth for radar domain rasters)"""
        pass

    @property
    @abstractmethod
    def x_axis_spacing(self):
        """Pixel Spacing in X direction (range for radar domain rasters)"""
        pass


# TODO - move to raster.py
@dataclass
class RadarRaster(SARRaster):
    """
    A Raster with attributes specific to Radar products.

    The attributes specified here are based on the needs of the QA code
    for generating and labeling plots, etc.

    Parameters
    ----------
    data : array_like
        Raster data to be stored.
    name : str
        Name for the dataset
    band : str
        name of the band for `img`, e.g. 'LSAR'
    freq : str
        name of the frequency for `img`, e.g. 'A' or 'B'
    pol : str
        name of the polarization for `img`, e.g. 'HH' or 'HV'
    az_spacing : float
        Azimuth spacing of pixels of input array
    az_start : float
        The start time of the observation for this RSLC Raster.
        This corresponds to the upper edge of the top pixels.
    az_stop : float
        The stopping time of the observation for this RSLC Raster.
        This corresponds to the lower side of the bottom pixels.
    range_spacing : float
        Range spacing of pixels of input array
    rng_start : float
        Start (near) distance of the range of input array
        This corresponds to the left side of the left-most pixels.
    rng_stop : float
        End (far) distance of the range of input array
        This corresponds to the right side of the right-most pixels.
    epoch : str
        The start of the epoch for this observation,
        in the format 'YYYY-MM-DD HH:MM:SS'

    Notes
    -----
    If data is NISAR HDF5 dataset, suggest initializing using
    the class method init_from_nisar_h5_product(..).
    """

    # Attributes of the input array
    az_spacing: float
    az_start: float
    az_stop: float

    range_spacing: float
    rng_start: float
    rng_stop: float

    epoch: str

    @property
    def y_axis_spacing(self):
        return self.az_spacing

    @property
    def x_axis_spacing(self):
        return self.range_spacing

    @classmethod
    def init_from_nisar_h5_product(cls, h5_file, band, freq, pol):
        """
        Initialize an RadarRaster object for the given
        band-freq-pol image in the input NISAR Radar domain HDF5 file.

        NISAR product type must be one of: 'RSLC', 'SLC', 'RIFG', 'RUNW', 'ROFF'
        If the product type is 'RSLC' or 'SLC', then the image dataset
        will be stored as a ComplexFloat16Decoder instance; this will allow
        significantly faster access to the data.

        Parameters
        ----------
        h5_file : h5py.File
            File handle to a valid NISAR product hdf5 file.
            Polarization images must be located in the h5 file in the path:
            /science/<band>/<product name>/swaths/frequency<freq>/<pol>
            or they will not be found. This is the file structure
            as determined from the NISAR Product Spec.
        band : str
            name of the band for `img`, e.g. 'LSAR'
        freq : str
            name of the frequency for `img`, e.g. 'A' or 'B'
        pol : str
            name of the polarization for `img`, e.g. 'HH' or 'HV'

        Raises
        ------
        DatasetNotFoundError
            If the file does not contain an image dataset for the given
            band-freq-pol combination, a DatasetNotFoundError
            exception will be thrown.

        Notes
        -----
        The `name` attribute will be populated with a string
        of the format: <product type>_<band>_<freq>_<pol>
        """

        product = nisarqa.get_NISAR_product_type(h5_file)

        if product not in ("RSLC", "SLC", "RIFG", "RUNW", "ROFF"):
            # self.logger.log_message(logging_base.LogFilterError, 'Invalid file structure.')
            raise nisarqa.InvalidNISARProductError

        # Hardcoded paths to various groups in the NISAR RSLC h5 file.
        # These paths are determined by the RSLC .xml product spec
        swaths_path = f"/science/{band}/{product}/swaths"
        freq_path = f"{swaths_path}/frequency{freq}"
        pol_path = f"{freq_path}/{pol}"

        swaths_group = h5_file[swaths_path]
        freq_group = h5_file[freq_path]

        if pol_path in h5_file:
            pol_group = h5_file[pol_path]
            # self.logger.log_message(logging_base.LogFilterInfo,
            #                         'Found image %s' % band_freq_pol_str)
            pass
        else:
            # self.logger.log_message(logging_base.LogFilterInfo,
            #                         'Image %s not present' % band_freq_pol_str)
            raise nisarqa.DatasetNotFoundError

        # Get dataset object
        # Most Radar Doppler NISAR products should be directly readible
        # by h5py, numpy, etc. as complex64, float, etc. The exception is RSLC.
        # RSLC Product Spec says that NISAR RSLC files should be complex32,
        # which requires special handling to read and access.
        # As of h5py 3.8.0, h5py gained the ability to read complex32
        # datasets, however numpy and other downstream packages do not
        # necessarily have that flexibility.
        if nisarqa.is_complex32(pol_group):
            # If the input RSLC product has dtype complex32, then we'll need
            # to use ComplexFloat16Decoder.
            if product == "RSLC":
                # The RSLC dataset is complex32. h5py >= 3.8 can read these
                # but numpy cannot yet. So, use the ComplexFloat16Decoder.
                dataset = ComplexFloat16Decoder(pol_group)
                print(
                    "(PASS) PASS/FAIL Check: Product raster dtype conforms"
                    " to RSLC Product Spec dtype of complex32."
                )
            else:
                raise TypeError(
                    f"Input dataset is for a {product} product and "
                    "has dtype complex32. As of R3.3, of the "
                    "radar-doppler NISAR products, only RSLC "
                    "products can have dtype complex32."
                )
        else:
            # Use h5py's standard reader
            dataset = pol_group

            if product == "SLC":
                print(
                    "(FAIL) PASS/FAIL Check: Product raster dtype conforms "
                    "to RSLC Product Spec dtype of complex32."
                )
            else:
                # TODO - for RIFG, RUNW, and ROFF, confirm that this
                # next print statement is, in fact, true.
                print(
                    "(PASS) PASS/FAIL Check: Product raster dtype conforms "
                    f"to {product} Product Spec dtype."
                )

        # From the xml Product Spec, sceneCenterAlongTrackSpacing is the
        # 'Nominal along track spacing in meters between consecutive lines
        # near mid swath of the RSLC image.'
        az_spacing = freq_group["sceneCenterAlongTrackSpacing"][...]

        # Get Azimuth (y-axis) tick range + label
        # path in h5 file: /science/LSAR/RSLC/swaths/zeroDopplerTime
        # For NISAR, radar-domain grids are referenced by the center of the
        # pixel, so +/- half the distance of the pixel's side to capture
        # the entire range.
        az_start = float(swaths_group["zeroDopplerTime"][0]) - 0.5 * az_spacing
        az_stop = float(swaths_group["zeroDopplerTime"][-1]) + 0.5 * az_spacing

        # From the xml Product Spec, sceneCenterGroundRangeSpacing is the
        # 'Nominal ground range spacing in meters between consecutive pixels
        # near mid swath of the RSLC image.'

        range_spacing = freq_group["sceneCenterGroundRangeSpacing"][...]

        # Range in meters (units are specified as meters in the product spec)
        # For NISAR, radar-domain grids are referenced by the center of the
        # pixel, so +/- half the distance of the pixel's side to capture
        # the entire range.
        rng_start = float(freq_group["slantRange"][0]) - 0.5 * range_spacing
        rng_stop = float(freq_group["slantRange"][-1]) + 0.5 * range_spacing

        # output of the next line has format: 'seconds since YYYY-MM-DD HH:MM:SS'
        sec_since_epoch = (
            swaths_group["zeroDopplerTime"].attrs["units"].decode("utf-8")
        )

        # Sanity Check
        format_data = "seconds since %Y-%m-%d %H:%M:%S"
        try:
            datetime.strptime(sec_since_epoch, format_data)
        except ValueError:
            warnings.warn(
                f"Invalid epoch format in input file: {sec_since_epoch}",
                RuntimeWarning,
            )
            # This text should appear in the REPORT.pdf to make it obvious:
            epoch = "INVALID EPOCH"
        else:
            epoch = sec_since_epoch.replace("seconds since ", "").strip()

        return cls(
            data=dataset,
            name=f"{product.upper()}_{band}_{freq}_{pol}",
            band=band,
            freq=freq,
            pol=pol,
            az_spacing=az_spacing,
            az_start=az_start,
            az_stop=az_stop,
            range_spacing=range_spacing,
            rng_start=rng_start,
            rng_stop=rng_stop,
            epoch=epoch,
        )


# TODO - move to generic
def get_pols(h5_file):
    """
    Locate the available bands, frequencies, and polarizations
    in the input HDF5 file.

    Parameters
    ----------
    h5_file : h5py.File
        Handle to the input product h5 file

    Returns
    -------
    pols : nested dict of *Raster
        Nested dict of *Raster objects, where each object represents
        a polarization dataset in `h5_file`.
        If the input product is in radar domain, *Raster means RadarRaster.
        If the input product is geocoded, *Raster means GeoRaster.
        Format: pols[<band>][<freq>][<pol>] -> a *Raster
        Ex: pols['LSAR']['A']['HH'] -> the HH dataset, stored in a *Raster object
    """
    product_type = nisarqa.get_NISAR_product_type(h5_file=h5_file)

    if product_type.startswith("G"):
        # geocoded product
        swaths_or_grids = "grids"
        raster_cls = nisarqa.GeoRaster
    else:
        # radar domain
        assert product_type.startswith("R") or (product_type == "SLC")
        swaths_or_grids = "swaths"
        raster_cls = RadarRaster

    # Discover images in input file and populate the `pols` dictionary
    pols = {}
    path = "/science"
    for band in h5_file[path]:
        pols[band] = {}
        path += f"/{band}/{product_type}/{swaths_or_grids}"

        for freq in nisarqa.NISAR_FREQS:
            path = f"/science/{band}/{product_type}/{swaths_or_grids}/frequency{freq}"
            if path not in h5_file:
                continue

            pols[band][freq] = {}

            for pol in nisarqa.get_possible_pols(product_type.lower()):
                try:
                    raster = raster_cls.init_from_nisar_h5_product(
                        h5_file, band, freq, pol
                    )

                except nisarqa.DatasetNotFoundError:
                    # RadarRaster could not be created, which means that the
                    # input file did not contain am image with the current
                    # `band`, `freq`, and `pol` combination.
                    continue

                pols[band][freq][pol] = raster

    # Sanity Check - if a band/freq does not have any polarizations,
    # this is a validation error. This check should be handled during
    # the validation process before this function was called,
    # not the quality process, so raise an error.
    # In the future, this step might be moved earlier in the
    # processing, and changed to be handled via: 'log the error
    # and remove the band from the dictionary'
    for band in pols.keys():
        for freq in pols[band].keys():
            # Empty dictionaries evaluate to False in Python
            if not pols[band][freq]:
                raise ValueError(
                    f"Provided input file does not have any polarizations"
                    f" included under band {band}, frequency {freq}."
                )

    return pols


def select_layers_for_slc_browse(pols):
    """
    Assign the polarization layers in the input file to grayscale or
    RGBA channels for the Browse Image.

    See `Notes` for details on the possible NISAR modes and assigned channels
    for LSAR band.
    SSAR is currently only minimally supported, so only a grayscale image
    will be created. Prioritization order to select the freq/pol to use:
        For frequency: Freq A then Freq B.
        For polarization: 'HH', then 'VV', then the first polarization found.


    Parameters
    ----------
    pols : nested dict of RadarRaster
        Nested dict of RadarRaster objects, where each object represents
        a polarization dataset in `h5_file`.
        Format: pols[<band>][<freq>][<pol>] -> a RadarRaster
        Ex: pols['LSAR']['A']['HH'] -> the HH dataset, stored
                                       in a RadarRaster object

    Returns
    -------
    layers_for_browse : dict
        A dictionary containing the channel assignments. Its structure is:

        layers_for_browse['band']  : str
                                        Either 'LSAR' or 'SSAR'
        layers_for_browse['A']     : list of str, optional
                                        List of the Freq A polarization(s)
                                        required to create the browse image.
                                        A subset of:
                                           ['HH','HV','VV','RH','RV','LH','LV']
        layers_for_browse['B']     : list of str, optional
                                        List of the Freq B polarizations
                                        required to create the browse image.
                                        A subset of ['HH','VV']

    Notes
    -----
    Possible modes for L-Band, as of Feb 2023:
        Single Pol      SP HH:      20+5, 40+5, 77
        Single Pol      SP VV:      5, 40

        Dual Pol        DP HH/HV:   77, 40+5, 20+5
        Dual Pol        DP VV/VH:   5, 77, 20+5, 40+5
        Quasi Quad Pol  QQ:         20+20, 20+5, 40+5, 5+5

        Quad Pol        QP:         20+5, 40+5

        Quasi Dual Pol  QD HH/VV:   5+5
        Compact Pol     CP RH/RV:   20+20           # an experimental mode

    Single Pol (SP) Assignment:
        - Freq A CoPol
        else:
        - Freq B CoPol
    DP and QQ Assignment:
        - Freq A: Red=HH, Green=HV, Blue=HH
    QP Assignment:
        - Freq A: Red=HH, Green=HV, Blue=VV
    QD Assignment:
        - Freq A: Red=HH, Blue=HH
        - Freq B: Green=VV
    CP Assignment:
        - Freq A: Grayscale of one pol image, with
                  Prioritization order: ['RH','RV','LH','LV']
    """

    layers_for_browse = {}

    # Determine which band to use. LSAR has priority over SSAR.
    for b in ("LSAR", "SSAR"):
        if b in pols:
            layers_for_browse["band"] = b
            band = b
            break
    else:
        raise ValueError(
            f'Only "LSAR" and "SSAR" bands are supported: {list(pols)}'
        )

    # Check that the correct frequencies are available
    if not set(pols[band].keys()).issubset({"A", "B"}):
        raise ValueError(
            f"`pols['{band}']` contains {set(pols[band].keys())}"
            ", but must be a subset of {'A', 'B'}"
        )

    # Get the frequency sub-band containing science mode data.
    # This is always frequency A if present, otherwise B.
    freq = "A" if ("A" in pols[band]) else "B"

    # SSAR is not fully supported by QA, so just make a simple grayscale
    if band == "SSAR":
        # Prioritize Co-Pol
        if "HH" in pols[band][freq]:
            layers_for_browse[freq] = ["HH"]
        elif "VV" in pols[band][freq]:
            layers_for_browse[freq] = ["VV"]
        else:
            # Take the first available Cross-Pol
            layers_for_browse[freq] = [pols[band][freq][0]]

        return layers_for_browse

    # The input file contains LSAR data. Will need to make
    # grayscale/RGB channel assignments

    # Get the available polarizations
    available_pols = list(pols[band][freq])
    n_pols = len(available_pols)

    if freq == "B":
        # This means only Freq B has data; this only occurs in Single Pol case.
        if n_pols > 1:
            raise ValueError(
                "When only Freq B is present, then only "
                f"single-pol mode supported. Freq{freq}: {available_pols}"
            )

        layers_for_browse["B"] = available_pols

    else:  # freq A exists
        if available_pols[0].startswith("R") or available_pols[0].startswith(
            "L"
        ):
            # Compact Pol. This is not a planned mode for LSAR,
            # and there is no test data, so simply make a grayscale image.

            # Per the Prioritization Order, use first available polarization
            for pol in ["RH", "RV", "LH", "LV"]:
                if pol in available_pols:
                    layers_for_browse["A"] = [pol]
                    break

            assert len(layers_for_browse["A"]) == 1

        elif n_pols == 1:  # Horizontal/Vertical transmit
            if "B" in pols[band]:
                # Freq A has one pol image, and Freq B exists.
                if set(available_pols) == set(pols[band]["B"]):
                    # A's polarization image is identical to B's pol image,
                    # which means that this is a single-pol observation mode
                    # where both frequency bands were active
                    layers_for_browse["A"] = available_pols

                elif len(pols[band]["B"]) == 1:
                    # Quasi Dual Pol -- Freq A has HH, Freq B has VV
                    assert "HH" in pols[band]["A"]
                    assert "VV" in pols[band]["B"]

                    layers_for_browse["A"] = available_pols
                    layers_for_browse["B"] = ["VV"]

                else:
                    # There is/are polarization image(s) for both A and B.
                    # But, they are not representative of any of the current
                    # observation modes for NISAR.
                    raise ValueError(
                        f"Freq A contains 1 polarization {available_pols}, but"
                        f' Freq B contains polarization(s) {pols[band]["B"]}.'
                        " This setup does not match any known NISAR"
                        " observation mode."
                    )
            else:
                # Single Pol
                layers_for_browse["A"] = available_pols

        elif n_pols in (2, 4):  # Horizontal/Vertical transmit
            # dual-pol, quad-pol, or Quasi-Quad pol

            # HH has priority over VV
            if "HH" in available_pols and "HV" in available_pols:
                layers_for_browse["A"] = ["HH", "HV"]
                if n_pols == 4:
                    # quad pol
                    layers_for_browse["A"].append("VV")

            elif "VV" in available_pols and "VH" in available_pols:
                # If there is only 'VV', then this granule must be dual-pol
                assert n_pols == 2
                layers_for_browse["A"] = ["VV", "VH"]

            else:
                raise ValueError(
                    "For dual-pol, quad-pol, and quasi-quad modes, "
                    "the input product must contain at least one "
                    "of HH+HV and/or VV+VH channels. Instead got: "
                    f"{available_pols}"
                )
        else:
            raise ValueError(
                f"Input product's band {band} contains {n_pols} "
                "polarization images, but only 1, 2, or 4 "
                "are supported."
            )

    # Sanity Check
    if ("A" not in layers_for_browse) and ("B" not in layers_for_browse):
        raise ValueError(
            "Current Mode (configuration) of the NISAR input file"
            " not supported for browse image."
        )

    return layers_for_browse


def process_backscatter_imgs_and_browse(
    pols,
    params,
    stats_h5,
    report_pdf,
    product_type,
    input_raster_represents_power=False,
    plot_title_prefix="Backscatter Coefficient",
    browse_filename="BROWSE.png",
):
    """
    Generate Backscatter Image plots for the `report_pdf` and
    corresponding browse image product.

    Parameters
    ----------
    pols : nested dict of RadarRaster or GeoRaster
        Nested dict of RadarRaster or GeoRaster objects, where each
        object represents a polarization dataset.
        Format: pols[<band>][<freq>][<pol>] -> a RadarRaster
        Ex: pols['LSAR']['A']['HH'] -> the HH dataset, stored in a
                                       RadarRaster or GeoRaster object
    params : BackscatterImageParamGroup
        A dataclass containing the parameters for processing
        and outputting backscatter image(s) and browse image.
    stats_h5 : h5py.File
        The output file to save QA metrics, etc. to
    report_pdf : PdfPages
        The output pdf file to append the backscatter image plot to
    product_type : str
        One of "rslc", "gslc", "slc", or "gcov".
    input_raster_represents_power : bool, optional
        The input dataset rasters associated with these histogram parameters
        should have their pixel values represent either power or root power.
        If `True`, then QA SAS assumes the input data already represents
        power and uses the pixels' magnitudes for computations.
        If `False`, then QA SAS assumes the input data represents root power
        aka magnitude and will handle the full computation to power using
        the formula:  power = abs(<magnitude>)^2 .
        Defaults to False (root power).
    plot_title_prefix : str
        Prefix for the title of the backscatter plots.
        Suggestions: "RSLC Backscatter Coefficient (beta-0)" or
        "GCOV Backscatter Coefficient (gamma-0)".
        Defaults to "Backscatter Coefficient".
    browse_filename : str, optional
        Filename (with path) for the browse image PNG.
        Defaults to 'BROWSE.png'
    """

    if product_type.endswith("slc"):
        layers_for_browse_func = select_layers_for_slc_browse
        save_browse_img_func = save_slc_browse_img
    elif product_type == "gcov":
        layers_for_browse_func = (
            nisarqa.products.gcov.select_layers_for_gcov_browse
        )
        save_browse_img_func = nisarqa.products.gcov.save_gcov_browse_img
    else:
        raise NotImplementedError(f"{product_type=} not implemented")

    # Select which layers will be needed for the browse image.
    # Multilooking takes a long time, but each multilooked polarization image
    # should be less than ~4 MB (per the current requirements for NISAR),
    # so it's ok to store the necessary multilooked Backscatter Images in memory.
    # to combine them later into the Browse image. The memory costs are
    # less than the costs for re-computing the multilooking.
    layers_for_browse = layers_for_browse_func(pols)

    # At the end of the loop below, the keys of this dict should exactly
    # match the set of TxRx polarizations needed to form the browse image
    pol_imgs_for_browse = {}

    # Process each image in the dataset

    for band in pols:
        for freq in pols[band]:
            for pol in pols[band][freq]:
                img = pols[band][freq][pol]

                # Input validation
                if not isinstance(img, (RadarRaster, nisarqa.GeoRaster)):
                    raise TypeError(
                        "`pols` must contain objects of type "
                        f"RadarRaster or GeoRaster. Current type: {type(img)}"
                    )

                multilooked_img = get_multilooked_backscatter_img(
                    img=img,
                    params=params,
                    stats_h5=stats_h5,
                    input_raster_represents_power=input_raster_represents_power,
                )

                corrected_img, orig_vmin, orig_vmax = apply_image_correction(
                    img_arr=multilooked_img, params=params
                )

                if params.gamma is not None:
                    inverse_func = functools.partial(
                        invert_gamma_correction,
                        gamma=params.gamma,
                        vmin=orig_vmin,
                        vmax=orig_vmax,
                    )

                    colorbar_formatter = FuncFormatter(
                        lambda x, pos: "{:.3f}".format(inverse_func(x))
                    )

                else:
                    colorbar_formatter = None

                if isinstance(img, RadarRaster):
                    img2pdf_func = save_rslc_backscatter_img_to_pdf
                else:  # is a GeoRaster
                    img2pdf_func = (
                        nisarqa.gslc.save_geocoded_backscatter_img_to_pdf
                    )

                img2pdf_func(
                    img_arr=corrected_img,
                    img=img,
                    params=params,
                    report_pdf=report_pdf,
                    colorbar_formatter=colorbar_formatter,
                    plot_title_prefix=f"Multilooked {plot_title_prefix}",
                )

                # If this backscatter image is needed to construct the browse image...
                if (
                    band == layers_for_browse["band"]
                    and freq in layers_for_browse
                    and pol in layers_for_browse[freq]
                ):
                    # ...keep the multilooked, color-corrected image
                    pol_imgs_for_browse[pol] = corrected_img

    # Construct the browse image
    save_browse_img_func(
        pol_imgs=pol_imgs_for_browse, filepath=browse_filename
    )


# TODO - move to generic location
def get_multilooked_backscatter_img(
    img, params, stats_h5, input_raster_represents_power=False
):
    """
    Generate the multilooked Backscatter Image array for a single
    polarization image.

    Parameters
    ----------
    img : RadarRaster or GeoRaster
        The raster to be processed
    params : BackscatterImageParamGroup
        A structure containing the parameters for processing
        and outputting the backscatter image(s).
    stats_h5 : h5py.File
        The output file to save QA metrics, etc. to
    input_raster_represents_power : bool, optional
        The input dataset rasters associated with these histogram parameters
        should have their pixel values represent either power or root power.
        If `True`, then QA SAS assumes the input data already represents
        power and uses the pixels' magnitudes for computations.
        If `False`, then QA SAS assumes the input data represents root power
        aka magnitude and will handle the full computation to power using
        the formula:  power = abs(<magnitude>)^2 .
        Defaults to False (root power).

    Returns
    -------
    out_img : numpy.ndarray
        The multilooked Backscatter Image
    """

    nlooks_freqa_arg = params.nlooks_freqa
    nlooks_freqb_arg = params.nlooks_freqb

    # Get the window size for multilooking
    if (img.freq == "A" and nlooks_freqa_arg is None) or (
        img.freq == "B" and nlooks_freqb_arg is None
    ):
        nlooks = nisarqa.compute_square_pixel_nlooks(
            img.data.shape,
            sample_spacing=(
                np.abs(img.y_axis_spacing),
                np.abs(img.x_axis_spacing),
            ),
            longest_side_max=params.longest_side_max,
        )

    elif img.freq == "A":
        nlooks = nlooks_freqa_arg
    elif img.freq == "B":
        nlooks = nlooks_freqb_arg
    else:
        raise ValueError(
            f"frequency is '{img.freq}', but only 'A' or 'B' are valid options."
        )

    # Save the final nlooks to the HDF5 dataset
    grp_path = nisarqa.STATS_H5_QA_PROCESSING_GROUP % img.band
    dataset_name = f"backscatterImageNlooksFreq{img.freq.upper()}"

    if isinstance(img, RadarRaster):
        axes = "[<azimuth>,<range>]"
    elif isinstance(img, nisarqa.GeoRaster):
        axes = "[<Y direction>,<X direction>]"
    else:
        raise TypeError(
            "Input `img` must be RadarRaster or GeoRaster. "
            f"It is {type(img)}"
        )

    # Create the nlooks dataset
    if dataset_name in stats_h5[grp_path]:
        assert tuple(stats_h5[grp_path][dataset_name][...]) == tuple(nlooks)
    else:
        nisarqa.create_dataset_in_h5group(
            h5_file=stats_h5,
            grp_path=grp_path,
            ds_name=dataset_name,
            ds_data=nlooks,
            ds_units="unitless",
            ds_description=f"Number of looks along {axes} axes of "
            f"Frequency {img.freq.upper()} image arrays "
            "for multilooking the backscatter and browse images.",
        )

    print(f"\nMultilooking Image {img.name} with shape: {img.data.shape}")
    print("Y direction (azimuth) ground spacing: ", img.y_axis_spacing)
    print("X direction (range) ground spacing: ", img.x_axis_spacing)
    print("Beginning Multilooking with nlooks window shape: ", nlooks)

    # Multilook
    out_img = nisarqa.compute_multilooked_backscatter_by_tiling(
        arr=img.data,
        nlooks=nlooks,
        input_raster_represents_power=input_raster_represents_power,
        tile_shape=params.tile_shape,
    )

    print(f"Multilooking Complete. Multilooked image shape: {out_img.shape}")

    return out_img


def apply_image_correction(img_arr, params):
    """
    Apply image correction in `img_arr` as specified in `params`.

    Image correction is applied in the following order:
        Step 1: Per `params.percentile_for_clipping`, clip the image array's outliers
        Step 2: Per `params.linear_units`, convert from linear units to dB
        Step 3: Per `params.gamma`, apply gamma correction

    Parameters
    ----------
    img_arr : numpy.ndarray
        2D image array to have image correction applied to.
        For example, for RSLC this is the multilooked image array.
    params : BackscatterImageParamGroup
        A structure containing the parameters for processing
        and outputting the backscatter image(s).

    Returns
    -------
    out_img : numpy.ndarray
        2D image array. If any image correction was specified via `params`
        and applied to `img_arr`, this returned array will include that
        image correction.
    vmin, vmax : float
        The min and max of the image array (excluding Nan), as computed
        after Step 2 but before Step 3. These can be used to set
        colorbar tick mark values; by computing vmin and vmax prior to
        gamma correction, the tick marks values retain physical meaning.
    """

    # Step 1: Clip the image array's outliers
    img_arr = clip_array(
        img_arr, percentile_range=params.percentile_for_clipping
    )

    # Step 2: Convert from linear units to dB
    if not params.linear_units:
        img_arr = nisarqa.pow2db(img_arr)

    # Get the vmin and vmax prior to applying gamma correction.
    # These can later be used for setting the colorbar's
    # tick mark values.
    vmin = np.nanmin(img_arr)
    vmax = np.nanmax(img_arr)

    # Step 3: Apply gamma correction
    if params.gamma is not None:
        img_arr = apply_gamma_correction(img_arr, gamma=params.gamma)

    return img_arr, vmin, vmax


def save_rslc_backscatter_img_to_pdf(
    img_arr,
    img,
    params,
    report_pdf,
    plot_title_prefix="Backscatter Coefficient",
    colorbar_formatter=None,
):
    """
    Annotate and save a RSLC Backscatter Image to `report_pdf`.

    Parameters
    ----------
    img_arr : numpy.ndarray
        2D image array to be saved. All image correction, multilooking, etc.
        needs to have previously been applied
    img : RadarRaster
        The RadarRaster object that corresponds to `img`. The metadata
        from this will be used for annotating the image plot.
    params : BackscatterImageParamGroup
        A structure containing the parameters for processing
        and outputting the backscatter image(s).
    report_pdf : PdfPages
        The output pdf file to append the backscatter image plot to
    plot_title_prefix : str, optional
        Prefix for the title of the backscatter plots.
        Suggestions: "RSLC Backscatter Coefficient (beta-0)" or
        "GCOV Backscatter Coefficient (gamma-0)".
        Defaults to "Backscatter Coefficient".
    colorbar_formatter : matplotlib.ticker.FuncFormatter or None, optional
        Tick formatter function to define how the numeric value
        associated with each tick on the colorbar axis is formatted
        as a string. This function must take exactly two arguments:
        `x` for the tick value and `pos` for the tick position,
        and must return a `str`. The `pos` argument is used
        internally by matplotlib.
        If None, then default tick values will be used. Defaults to None.
        See: https://matplotlib.org/2.0.2/examples/pylab_examples/custom_ticker1.html
    """

    # Plot and Save Backscatter Image to graphical summary pdf
    title = f"{plot_title_prefix}\n(scale={params.backscatter_units}%s)\n{img.name}"
    if params.gamma is None:
        title = title % ""
    else:
        title = title % rf", $\gamma$-correction={params.gamma}"

    # Get Azimuth (y-axis) label
    az_title = f"Zero Doppler Time\n(seconds since {img.epoch})"

    # Get Range (x-axis) labels and scale
    rng_title = "Slant Range (km)"

    img2pdf(
        img_arr=img_arr,
        title=title,
        ylim=[img.az_start, img.az_stop],
        xlim=[nisarqa.m2km(img.rng_start), nisarqa.m2km(img.rng_stop)],
        colorbar_formatter=colorbar_formatter,
        ylabel=az_title,
        xlabel=rng_title,
        plots_pdf=report_pdf,
    )


def clip_array(arr, percentile_range=(0.0, 100.0)):
    """
    Clip input array to the provided percentile range.

    NaN values are excluded from the computation of the percentile.

    Parameters
    ----------
    arr : array_like
        Input array
    percentile_range : pair of numeric, optional
        Defines the percentile range of the `arr`
        that the colormap covers. Must be in the range [0.0, 100.0],
        inclusive.
        Defaults to (0.0, 100.0) (no clipping).

    Returns
    -------
    out_img : numpy.ndarray
        A copy of the input array with the values outside of the
        range defined by `percentile_range` clipped.
    """
    for p in percentile_range:
        nisarqa.verify_valid_percentile(p)
    if len(percentile_range) != 2:
        raise ValueError(f"{percentile_range=} must have length of 2")

    # Get the value of the e.g. 5th percentile and the 95th percentile
    vmin, vmax = np.nanpercentile(arr, percentile_range)

    # Clip the image data and return
    return np.clip(arr, a_min=vmin, a_max=vmax)


def apply_gamma_correction(img_arr, gamma):
    """
    Apply gamma correction to the input array.

    Function will normalize the array and apply gamma correction.
    The returned output array will remain in range [0,1].

    Parameters
    ----------
    img_arr : array_like
        Input array
    gamma : float
        The gamma correction parameter.
        Gamma will be applied as follows:
            array_out = normalized_array ^ gamma
        where normalized_array is a copy of `img_arr` with values
        scaled to the range [0,1].

    Returns
    -------
    out_img : numpy.ndarray
        Copy of `img_arr` with the specified gamma correction applied.
        Due to normalization, values in `out_img` will be in range [0,1].

    Also See
    --------
    invert_gamma_correction : inverts this function
    """
    # Normalize to range [0,1]
    out_img = nisarqa.normalize(img_arr)

    # Apply gamma correction
    out_img = np.power(out_img, gamma)

    return out_img


def invert_gamma_correction(img_arr, gamma, vmin, vmax):
    """
    Invert the gamma correction to the input array.

    Function will normalize the array and apply gamma correction.
    The returned output array will remain in range [0,1].

    Parameters
    ----------
    img_arr : array_like
        Input array
    gamma : float
        The gamma correction parameter.
        Gamma will be inverted as follows:
            array_out = img_arr ^ (1/gamma)
        The array will then be rescaled as follows, to "undo" normalization:
            array_out = (array_out * (vmax - vmin)) + vmin
    vmin, vmax : float
        The min and max of the source image array BEFORE gamma correction
        was applied.

    Returns
    -------
    out : numpy.ndarray
        Copy of `img_arr` with the specified gamma correction inverted
        and scaled to range [<vmin>, <vmax>]

    Also See
    --------
    apply_gamma_correction : inverts this function
    """
    # Invert the power
    out = np.power(img_arr, 1 / gamma)

    # Invert the normalization
    out = (out * (vmax - vmin)) + vmin

    return out


def save_slc_browse_img(pol_imgs, filepath):
    """
    Save the given polarization images to a RGB or Grayscale PNG with
    transparency.

    Dimensions of the output PNG (in pixels) will be the same as the dimensions
    of the input polarization image array(s). (No scaling will occur.)
    Non-finite values will be made transparent.

    Color Channels will be assigned per the following pseudocode:

        If pol_imgs.keys() contains only one image, then:
            grayscale = <that image>
        If pol_imgs.keys() is ['HH','HV','VV'], then:
            red = 'HH'
            green = 'HV'
            blue = 'VV'
        If pol_imgs.keys() is ['HH','HV'], then:
            red = 'HH'
            green = 'HV'
            blue = 'HH'
        If pol_imgs.keys() is ['HH','VV'], then:
            red = 'HH'
            green = 'VV'
            blue = 'HH'
        If pol_imgs.keys() is ['VV','VH'], then:
            red = 'VV'
            green = 'VH'
            blue = 'VV'
        Otherwise, one image in `pol_imgs` will be output as grayscale.

    Parameters
    ----------
    pol_imgs : dict of numpy.ndarray
        Dictionary of 2D array(s) that will be mapped to specific color
        channel(s) for the output browse PNG.
        If there are multiple image arrays, they must have identical shape.
        Format of dictionary:
            pol_imgs[<polarization>] : <2D numpy.ndarray image>, where
                <polarization> must be a subset of: 'HH', 'HV', 'VV', 'VH',
                                                    'RH', 'RV', 'LV', 'LH',
        Example:
            pol_imgs['HH'] : <2D numpy.ndarray image>
            pol_imgs['VV'] : <2D numpy.ndarray image>
    filepath : str
        Full filepath for where to save the browse image PNG.

    Notes
    -----
    Provided image array(s) must previously be image-corrected. This
    function will take the image array(s) as-is and will not apply additional
    image correction processing to them. This function directly combines
    the image(s) into a single browse image.

    If there are multiple input images, they must be thoughtfully prepared and
    standardized relative to each other prior to use by this function.
    For example, trying to combine a Freq A 20 MHz image
    and a Freq B 5 MHz image into the same output browse image might not go
    well, unless the image arrays were properly prepared and standardized
    in advance.
    """

    # WLOG, get the shape of the image arrays
    # They should all be the same shape; the check for this is below.
    arbitrary_img = next(iter(pol_imgs.values()))
    img_2D_shape = np.shape(arbitrary_img)
    for img in pol_imgs.values():
        # Input validation check
        if np.shape(img) != img_2D_shape:
            raise ValueError(
                "All image arrays in `pol_imgs` must have the same shape."
            )

    # Assign color channels
    set_of_pol_imgs = set(pol_imgs)

    if set_of_pol_imgs == {"HH", "HV", "VV"}:
        # Quad Pol
        red = pol_imgs["HH"]
        green = pol_imgs["HV"]
        blue = pol_imgs["VV"]
    elif set_of_pol_imgs == {"HH", "HV"}:
        # dual pol horizontal transmit, or quasi-quad
        red = pol_imgs["HH"]
        green = pol_imgs["HV"]
        blue = pol_imgs["HH"]
    elif set_of_pol_imgs == {"HH", "VV"}:
        # quasi-dual mode
        red = pol_imgs["HH"]
        green = pol_imgs["VV"]
        blue = pol_imgs["HH"]
    elif set_of_pol_imgs == {"VV", "VH"}:
        # dual-pol only, vertical transmit
        red = pol_imgs["VV"]
        green = pol_imgs["VH"]
        blue = pol_imgs["VV"]
    else:
        # If we get into this "else" statement, then
        # either there is only one image provided (e.g. single pol),
        # or the images provided are not one of the expected cases.
        # Either way, WLOG plot one of the image(s) in `pol_imgs`.
        gray_img = pol_imgs.popitem()[1]
        plot_to_grayscale_png(img_arr=gray_img, filepath=filepath)

        # This `else` is a catch-all clause. Return early, so that
        # we do not try to plot to RGB
        return

    plot_to_rgb_png(red=red, green=green, blue=blue, filepath=filepath)


def plot_to_grayscale_png(img_arr, filepath):
    """
    Save the image array to a 1-channel grayscale PNG with transparency.

    Finite pixels will have their values scaled to 1-255. Non-finite pixels
    will be set to 0 and made to appear transparent in the PNG.
    The pixel value of 0 is reserved for the transparent pixels.

    Parameters
    ----------
    img_arr : array_like
        2D Image to plot
    filepath : str
        Full filepath the browse image product.

    Notes
    -----
    This function does not add a full alpha channel to the output png.
    It instead uses "cheap transparency" (palette-based transparency)
    to keep file size smaller.
    See: http://www.libpng.org/pub/png/book/chapter08.html#png.ch08.div.5.4
    """

    # Only use 2D arrays
    if len(np.shape(img_arr)) != 2:
        raise ValueError("Input image array must be 2D.")

    img_arr, transparency_val = prep_arr_for_png_with_transparency(img_arr)

    # Save as grayscale image using PIL.Image. 'L' is grayscale mode.
    # (Pyplot only saves png's as RGB, even if cmap=plt.cm.gray)
    im = Image.fromarray(img_arr, mode="L")
    im.save(filepath, transparency=transparency_val)  # default = 72 dpi


def plot_to_rgb_png(red, green, blue, filepath):
    """
    Combine and save RGB channel arrays to a browse PNG with transparency.

    Finite pixels will have their values scaled to 1-255. Non-finite pixels
    will be set to 0 and made to appear transparent in the PNG.
    The pixel value of 0 is reserved for the transparent pixels.

    Parameters
    ----------
    red, green, blue : numpy.ndarray
        2D arrays that will be mapped to the red, green, and blue
        channels (respectively) for the PNG. These three arrays must have
        identical shape.
    filepath : str
        Full filepath for where to save the browse image PNG.

    Notes
    -----
    This function does not add a full alpha channel to the output png.
    It instead uses "cheap transparency" (palette-based transparency)
    to keep file size smaller.
    See: http://www.libpng.org/pub/png/book/chapter08.html#png.ch08.div.5.4
    """

    # Only use 2D arrays
    for arr in (red, green, blue):
        if len(np.shape(arr)) != 2:
            raise ValueError("Input image array must be 2D.")

    # Concatenate into uint8 RGB array.
    nrow, ncol = np.shape(red)
    rgb_arr = np.zeros((nrow, ncol, 3), dtype=np.uint8)

    # transparency_val will be the same from all calls to this function;
    # only need to capture it once.
    rgb_arr[:, :, 0], transparency_val = prep_arr_for_png_with_transparency(
        red
    )
    rgb_arr[:, :, 1] = prep_arr_for_png_with_transparency(green)[0]
    rgb_arr[:, :, 2] = prep_arr_for_png_with_transparency(blue)[0]

    # make a tuple with length 3, where each entry denotes the transparent
    # value for R, G, and B channels (respectively)
    transparency_val = (transparency_val,) * 3

    im = Image.fromarray(rgb_arr, mode="RGB")
    im.save(filepath, transparency=transparency_val)  # default = 72 dpi


def prep_arr_for_png_with_transparency(img_arr):
    """
    Prepare a 2D image array for use in a uint8 PNG with palette-based
    transparency.

    Normalizes and then scales the array values to 1-255. Non-finite pixels
    (aka invalid pixels) are set to 0.

    Parameters
    ----------
    img_arr : array_like
        2D Image to plot

    Returns
    -------
    out : numpy.ndarray with dtype numpy.uint8
        Copy of the input image array that has been prepared for use in
        a PNG file.
        Input image array values were normalized to [0,1] and then
        scaled to [1,255]. Non-finite pixels are set to 0.
    transparency_value : int
        The pixel value denoting non-finite (invalid) pixels. This is currently always 0.

    Notes
    -----
    For PNGs with palette-based transparency, one value in 0-255 will need
    to be assigned to be the fill value (i.e. the value that will appear
    as transparent). For unsigned integer data, it's conventional to use
    the largest representable value. (For signed integer data you usually
    want the most negative value.)
    However, when using RGB mode + palette-based transparency in Python's
    PIL library, if a pixel in only e.g. one color channel is invalid,
    but the corresponding pixel in other channels is valid, then the
    resulting PNG image will make the color for the first channel appear
    dominant. For example, for a given pixel in an RGB image. If a red
    channel's value for that pixel is 255 (invalid), while the green and
    blue channels' values are 123 and 67 (valid), then in the output RGB
    that pixel will appear bright red -- even if the `transparency` parameter
    is assigned correctly. So, instead we'll use 0 to represent invalid
    pixels, so that the resulting PNG "looks" more representative of the
    underlying data.
    """

    # Normalize to range [0,1]. If the array is already normalized,
    # this should have no impact.
    out = nisarqa.normalize(img_arr)

    # After normalization to range [0,1], scale to 1-255 for unsigned int8
    # Reserve the value 0 for use as the transparency value.
    #   out = (<normalized array> * (target_max - target_min)) + target_min
    out = (np.uint8(out * (255 - 1))) + 1

    # Set transparency value so that the "alpha" is added to the image
    transparency_value = 0

    # Denote invalid pixels with 255, so that they output as transparent
    out[~np.isfinite(img_arr)] = transparency_value

    return out, transparency_value


# TODO - move to generic plotting.py
def img2pdf(
    img_arr,
    plots_pdf,
    title=None,
    xlim=None,
    ylim=None,
    colorbar_formatter=None,
    xlabel=None,
    ylabel=None,
):
    """
    Plot the image array in grayscale, add a colorbar, and append to the pdf.

    Parameters
    ----------
    img_arr : array_like
        Image to plot in grayscale
    plots_pdf : PdfPages
        The output pdf file to append the backscatter image plot to
    title : str, optional
        The full title for the plot
    xlim, ylim : sequence of numeric, optional
        Lower and upper limits for the axes ticks for the plot.
        Format: xlim=[<x-axis lower limit>, <x-axis upper limit>],
                ylim=[<y-axis lower limit>, <y-axis upper limit>]
    colorbar_formatter : matplotlib.ticker.FuncFormatter or None, optional
        Tick formatter function to define how the numeric value
        associated with each tick on the colorbar axis is formatted
        as a string. `FuncFormatter`s take exactly two arguments:
        `x` for the tick value and `pos` for the tick position,
        and must return a `str`. The `pos` argument is used
        internally by matplotlib.
        If None, then default tick values will be used. Defaults to None.
        See: https://matplotlib.org/2.0.2/examples/pylab_examples/custom_ticker1.html
        (Wrapping the function with FuncFormatter is optional.)
    xlabel, ylabel : str, optional
        Axes labels for the x-axis and y-axis (respectively)
    """

    # Instantiate the figure object
    # (Need to instantiate it outside of the plotting function
    # in order to later modify the plot for saving purposes.)
    f = plt.figure()
    ax = plt.gca()

    # Plot the img_arr image.
    ax_img = ax.imshow(X=img_arr, cmap=plt.cm.gray)

    # Add Colorbar
    cbar = plt.colorbar(ax_img, ax=ax)

    if colorbar_formatter is not None:
        cbar.ax.yaxis.set_major_formatter(colorbar_formatter)

    ## Label the plot

    # If xlim or ylim are not provided, let matplotlib auto-assign the ticks.
    # Otherwise, dynamically calculate and set the ticks w/ labels for
    # the x-axis and/or y-axis.
    # (Attempts to set the limits by using the `extent` argument for
    # matplotlib.imshow() caused significantly distorted images.
    # So, compute and set the ticks w/ labels manually.)
    if xlim is not None or ylim is not None:
        img_arr_shape = np.shape(img_arr)

        # Set the density of the ticks on the figure
        ticks_per_inch = 2.5

        # Get full figure size in inches
        fig_w, fig_h = f.get_size_inches()
        W = img_arr_shape[1]
        H = img_arr_shape[0]

        # Update variables to the actual, displayed image size
        # (The actual image will have a different aspect ratio
        # than the matplotlib figure window's aspect ratio.
        # But, altering the matplotlib figure window's aspect ratio
        # will lead to inconsistently-sized pages in the output .pdf)
        if H / W >= fig_h / fig_w:
            # image will be limited by its height, so
            # it will not use the full width of the figure
            fig_w = W * (fig_h / H)
        else:
            # image will be limited by its width, so
            # it will not use the full height of the figure
            fig_h = H * (fig_w / W)

    if xlim is not None:
        # Compute num of xticks to use
        num_xticks = int(ticks_per_inch * fig_w)

        # Always have a minimum of 2 labeled ticks
        num_xticks = num_xticks if num_xticks >= 2 else 2

        # Specify where we want the ticks, in pixel locations.
        xticks = np.linspace(0, img_arr_shape[1], num_xticks)
        ax.set_xticks(xticks)

        # Specify what those pixel locations correspond to in data coordinates.
        # By default, np.linspace is inclusive of the endpoint
        xticklabels = [
            "{:.1f}".format(i)
            for i in np.linspace(start=xlim[0], stop=xlim[1], num=num_xticks)
        ]
        ax.set_xticklabels(xticklabels)

        plt.xticks(rotation=45)

    if ylim is not None:
        # Compute num of yticks to use
        num_yticks = int(ticks_per_inch * fig_h)

        # Always have a minimum of 2 labeled ticks
        if num_yticks < 2:
            num_yticks = 2

        # Specify where we want the ticks, in pixel locations.
        yticks = np.linspace(0, img_arr_shape[0], num_yticks)
        ax.set_yticks(yticks)

        # Specify what those pixel locations correspond to in data coordinates.
        # By default, np.linspace is inclusive of the endpoint
        yticklabels = [
            "{:.1f}".format(i)
            for i in np.linspace(start=ylim[0], stop=ylim[1], num=num_yticks)
        ]
        ax.set_yticklabels(yticklabels)

    # Label the Axes
    if xlabel is not None:
        plt.xlabel(xlabel)
    if ylabel is not None:
        plt.ylabel(ylabel)

    # Add title
    if title is not None:
        plt.title(title)

    # Make sure axes labels do not get cut off
    f.tight_layout()

    # Append figure to the output .pdf
    plots_pdf.savefig(f)

    # Close the plot
    plt.close(f)


def process_backscatter_and_phase_histograms(
    pols,
    params,
    stats_h5,
    report_pdf,
    plot_title_prefix="Backscatter Coefficient",
    input_raster_represents_power=False,
):
    """
    Generate the Backscatter and Phase Histograms and save their plots
    to the graphical summary .pdf file.

    Backscatter histogram will be computed in decibel units.
    Phase histogram defaults to being computed in radians,
    configurable to be computed in degrees by setting
    `params.phs_in_radians` to False.
    NaN values will be excluded from Histograms.

    Parameters
    ----------
    pols : nested dict of RadarRaster or GeoRaster
        Nested dict of *Raster objects, where each object represents
        a polarization dataset in `h5_file`.
        Format: pols[<band>][<freq>][<pol>] -> a RadarRaster or GeoRaster
        Ex: pols['LSAR']['A']['HH'] -> the HH dataset, stored
                                       in a RadarRaster object
    params : HistogramParams
        A structure containing the parameters for processing
        and outputting the backscatter and phase histograms.
    stats_h5 : h5py.File
        The output file to save QA metrics, etc. to
    report_pdf : PdfPages
        The output pdf file to append the backscatter image plot to
    plot_title_prefix : str, optional
        Prefix for the title of the backscatter plots.
        Suggestions: "RSLC Backscatter Coefficient (beta-0)" or
        "GCOV Backscatter Coefficient (gamma-0)".
        Defaults to "Backscatter Coefficient".
    input_raster_represents_power : bool, optional
        The input dataset rasters associated with these histogram parameters
        should have their pixel values represent either power or root power.
        If `True`, then QA SAS assumes the input data already represents
        power and uses the pixels' magnitudes for computations.
        If `False`, then QA SAS assumes the input data represents root power
        aka magnitude and will handle the full computation to power using
        the formula:  power = abs(<magnitude>)^2 .
        Defaults to False (root power).
    """

    # Generate and store the histograms
    for band in pols:
        for freq in pols[band]:
            generate_backscatter_image_histogram_single_freq(
                pol=pols[band][freq],
                band=band,
                freq=freq,
                params=params,
                stats_h5=stats_h5,
                report_pdf=report_pdf,
                input_raster_represents_power=input_raster_represents_power,
                plot_title_prefix=plot_title_prefix,
            )

            generate_phase_histogram_single_freq(
                pol=pols[band][freq],
                band=band,
                freq=freq,
                params=params,
                stats_h5=stats_h5,
                report_pdf=report_pdf,
            )


def generate_backscatter_image_histogram_single_freq(
    pol,
    band,
    freq,
    params,
    stats_h5,
    report_pdf,
    plot_title_prefix="Backscatter Coefficient",
    input_raster_represents_power=False,
):
    """
    Generate Backscatter Image Histogram for a single frequency.

    The histogram's plot will be appended to the graphical
    summary file `report_pdf`, and its data will be
    stored in the statistics .h5 file `stats_h5`.
    Backscatter histogram will be computed in decibel units.

    Parameters
    ----------
    pol : dict of RadarRaster or GeoRaster
        dict of *Raster objects for the given `band`
        and `freq`. Each key is a polarization (e.g. 'HH'
        or 'HV'), and each key's item is the corresponding
        RadarRaster instance.
        Ex: pol['HH'] -> the HH dataset, stored
                         in a RadarRaster or GeoRaster object
    band : str
        Band name for the histograms to be processed,
        e.g. 'LSAR'
    freq : str
        Frequency name for the histograms to be processed,
        e.g. 'A' or 'B'
    params : HistogramParamGroup
        A structure containing the parameters for processing
        and outputting the histograms.
    stats_h5 : h5py.File
        The output file to save QA metrics, etc. to
    report_pdf : PdfPages
        The output pdf file to append the backscatter image plot to
    plot_title_prefix : str
        Prefix for the title of the backscatter plots.
        Suggestions: "RSLC Backscatter Coefficient (beta-0)" or
        "GCOV Backscatter Coefficient (gamma-0)"
        Defaults to "Backscatter Coefficient"
    input_raster_represents_power : bool, optional
        The input dataset rasters associated with these histogram parameters
        should have their pixel values represent either power or root power.
        If `True`, then QA SAS assumes the input data already represents
        power and uses the pixels' magnitudes for computations.
        If `False`, then QA SAS assumes the input data represents root power
        aka magnitude and will handle the full computation to power using
        the formula:  power = abs(<magnitude>)^2 .
        Defaults to False (root power).
    """

    print(f"Generating Backscatter Image Histograms for Frequency {freq}...")

    # Open one figure+axes.
    # Each band+frequency will have a distinct plot, with all of the
    # polarization images for that setup plotted together on the same plot.
    fig, ax = plt.subplots(nrows=1, ncols=1)

    # Use custom cycler for accessibility
    ax.set_prop_cycle(nisarqa.CUSTOM_CYCLER)

    def img_prep(arr):
        # Convert to backscatter.
        # For Backscatter Histogram, do not mask out zeros.
        power = (
            np.abs(arr)
            if input_raster_represents_power
            else nisarqa.arr2pow(arr)
        )

        return nisarqa.pow2db(power)

    for pol_name, pol_data in pol.items():
        # Get histogram probability density
        hist_density = nisarqa.compute_histogram_by_tiling(
            arr=pol_data.data,
            bin_edges=params.backscatter_bin_edges,
            data_prep_func=img_prep,
            density=True,
            decimation_ratio=params.decimation_ratio,
            tile_shape=params.tile_shape,
        )

        # Save to stats.h5 file
        grp_path = f"{nisarqa.STATS_H5_QA_FREQ_GROUP}/{pol_name}/" % (
            band,
            freq,
        )

        # Save Backscatter Histogram Counts to HDF5 file
        backscatter_units = params.get_units_from_hdf5_metadata(
            "backscatter_bin_edges"
        )

        nisarqa.create_dataset_in_h5group(
            h5_file=stats_h5,
            grp_path=grp_path,
            ds_name="backscatterHistogramDensity",
            ds_data=hist_density,
            ds_units=f"1/{backscatter_units}",
            ds_description="Normalized density of the backscatter image histogram",
        )

        # Add backscatter histogram density to the figure
        add_hist_to_axis(
            ax,
            counts=hist_density,
            edges=params.backscatter_bin_edges,
            label=pol_name,
        )

    # Label the Backscatter Image Figure
    title = f"{plot_title_prefix} Histograms\n{band} Frequency {freq}"
    ax.set_title(title)

    ax.legend(loc="upper right")
    ax.set_xlabel(f"Backscatter ({backscatter_units})")
    ax.set_ylabel(f"Density (1/{backscatter_units})")

    # Per ADT, let the top limit float for Backscatter Histogram
    ax.set_ylim(bottom=0.0)
    ax.grid()

    # Save complete plots to graphical summary pdf file
    report_pdf.savefig(fig)

    # Close figure
    plt.close(fig)

    print(f"Backscatter Image Histograms for Frequency {freq} complete.")


def generate_phase_histogram_single_freq(
    pol, band, freq, params, stats_h5, report_pdf
):
    """
    Generate Phase Histograms for a single frequency.

    The histograms' plots will be appended to the graphical
    summary file `report_pdf`, and their data will be
    stored in the statistics .h5 file `stats_h5`.
    Phase histogram defaults to being computed in radians,
    configurable to be computed in degrees per `params.phs_in_radians`.
    NOTE: Only if the dtype of a polarization raster is complex-valued
    (e.g. complex32) will it be included in the Phase histogram(s).
    NaN values will be excluded from the histograms.

    Parameters
    ----------
    pol : dict of RadarRaster or GeoRaster
        dict of *Raster objects for the given `band`
        and `freq`. Each key is a polarization (e.g. 'HH'
        or 'HV'), and each key's item is the corresponding
        RadarRaster instance.
        Ex: pol['HH'] -> the HH dataset, stored
                         in a RadarRaster or GeoRaster object
    band : str
        Band name for the histograms to be processed,
        e.g. 'LSAR'
    freq : str
        Frequency name for the histograms to be processed,
        e.g. 'A' or 'B'
    params : HistogramParamGroup
        A structure containing the parameters for processing
        and outputting the backscatter and phase histograms.
    stats_h5 : h5py.File
        The output file to save QA metrics, etc. to
    report_pdf : PdfPages
        The output pdf file to append the backscatter image plot to
    """

    # flag for if any phase histogram densities are generated
    # (We expect this flag to be set to True if any polarization contains
    # phase information. But for example, if a GCOV product only has
    # on-diagonal terms which are real-valued and lack phase information,
    # this will remain False.)
    save_phase_histogram = False

    print(f"Generating Phase Histograms for Frequency {freq}...")

    # Open one figure+axes.
    # Each band+frequency will have a distinct plot, with all of the
    # polarization images for that setup plotted together on the same plot.
    fig, ax = plt.subplots(nrows=1, ncols=1)

    # Use custom cycler for accessibility
    ax.set_prop_cycle(nisarqa.CUSTOM_CYCLER)

    def img_prep(arr):
        # Remove zero values (and nans) in case of 0 magnitude vectors, etc.
        # Note: There will be no need to clip phase values; the output of
        # np.angle() is always in the range (-pi, pi] (or (-180, 180]).
        if params.phs_in_radians:
            return np.angle(arr[np.abs(arr) >= 1.0e-05], deg=False)
        else:
            # phase in degrees
            return np.angle(arr[np.abs(arr) >= 1.0e-05], deg=True)

    for pol_name, pol_data in pol.items():
        # Only create phase histograms for complex datasets. Examples of
        # complex datasets include RSLC, GSLC, and GCOV off-diagonal rasters.
        # Note: Need to use `np.issubdtype` instead of `np.iscomplexobj`
        # due to e.g. RSLC and GSLC datasets of type ComplexFloat16Decoder.
        if not np.issubdtype(pol_data.data, np.complexfloating):
            continue

        save_phase_histogram = True

        # Get histogram probability densities
        hist_density = nisarqa.compute_histogram_by_tiling(
            arr=pol_data.data,
            bin_edges=params.phs_bin_edges,
            data_prep_func=img_prep,
            density=True,
            decimation_ratio=params.decimation_ratio,
            tile_shape=params.tile_shape,
        )

        # Save to stats.h5 file
        freq_path = nisarqa.STATS_H5_QA_FREQ_GROUP % (band, freq)
        grp_path = f"{freq_path}/{pol_name}/"

        phs_units = params.get_units_from_hdf5_metadata("phs_bin_edges")

        nisarqa.create_dataset_in_h5group(
            h5_file=stats_h5,
            grp_path=grp_path,
            ds_name="phaseHistogramDensity",
            ds_data=hist_density,
            ds_units=f"1/{phs_units}",
            ds_description="Normalized density of the phase histogram",
        )

        # Add phase histogram density to the figure
        add_hist_to_axis(
            ax,
            counts=hist_density,
            edges=params.phs_bin_edges,
            label=pol_name,
        )

    # Label and output the Phase Histogram Figure
    if save_phase_histogram:
        ax.set_title(f"{band} Frequency {freq} Phase Histograms")
        ax.legend(loc="upper right")
        ax.set_xlabel(f"Phase ({phs_units})")
        ax.set_ylabel(f"Density (1/{phs_units})")
        if params.phs_in_radians:
            ax.set_ylim(bottom=0.0, top=0.5)
        else:
            ax.set_ylim(bottom=0.0, top=0.01)
        ax.grid()

        # Save complete plots to graphical summary pdf file
        report_pdf.savefig(fig)

        # Close figure
        plt.close(fig)

    else:
        # Remove unused dataset from STATS.h5 because no phase histogram was
        # generated.

        # Get param attribute for the extraneous group
        metadata = nisarqa.HistogramParamGroup.get_attribute_metadata(
            "phs_bin_edges"
        )

        # Get the instance of the HDF5Attrs object for this parameter
        hdf5_attrs_instance = metadata["hdf5_attrs_func"](params)

        # Form the path in output STATS.h5 file to the group to be deleted
        path = hdf5_attrs_instance.group_path % band
        path += f"/{hdf5_attrs_instance.name}"

        # Delete the unnecessary dataset
        if path in stats_h5:
            del stats_h5[path]

    print(f"Histograms for Frequency {freq} complete.")


def add_hist_to_axis(axis, counts, edges, label):
    """Add the plot of the given counts and edges to the
    axis object. Points will be centered in each bin,
    and the plot will be denoted `label` in the legend.
    """
    bin_centers = 0.5 * (edges[:-1] + edges[1:])
    axis.plot(bin_centers, counts, label=label)


__all__ = nisarqa.get_all(__name__, objects_to_skip)
