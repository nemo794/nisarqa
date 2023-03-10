import os
from dataclasses import dataclass, fields

import h5py
import nisarqa
import numpy as np
import numpy.typing as npt
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from PIL import Image

# List of objects from the import statements that
# should not be included when importing this module
objects_to_skip = nisarqa.get_all(name=__name__)

def verify_rslc(runconfig_file):
    '''
    Verify an RSLC product based on the input file, parameters, etc.
    specified in the input runconfig file.

    This is the main function for running the entire QA workflow. It will
    run based on the options supplied in the input runconfig file.
    The input runconfig file must follow the standard RSLC QA runconfig
    format. Run the command line command 'nisar_qa dumpconfig rslc'
    for an example template with default parameters (where available).

    Parameters
    ----------
    runconfig_file : str
        Full filename for an existing RSLC QA runconfig file
    '''

    # Parse the runconfig file
    rslc_params = nisarqa.parse_rslc_runconfig(runconfig_file)
    output_dir = rslc_params.prodpath.qa_output_dir.val

    print('QA Processing parameters, per runconfig and defaults (runconfig has precedence)')

    rslc_params_names = {
        'input_f': 'Input File Group',
        'prodpath': 'Product Path Group',
        'anc_files': 'Dynamic Ancillary File',
        'workflows': 'Workflows',
        'power_img': 'Power Image',
        'histogram': 'Histogram',
        'abs_cal': 'Absolute Calibration Factor',
        'nesz': 'NESZ',
        'pta': 'Point Target Analyzer'
        }

    for params_obj in fields(rslc_params):
        grp_name = rslc_params_names[params_obj.name]
        print(f'  {grp_name} Parameters:')

        po = getattr(rslc_params, params_obj.name)
        if po is not None:
            for param in fields(po):
                po2 = getattr(po, param.name)
                if isinstance(po2, bool):
                    print(f'    {param.name}: {po2}')
                else:
                    print(f'    {param.name}: {po2.val}')

    # Start logger
    # TODO get logger from Brian's code and implement here
    # For now, output the stub log file.
    nisarqa.output_stub_files(output_dir=output_dir, stub_files='log_txt')

    # Create file paths for output files ()
    input_file = rslc_params.input_f.qa_input_file.val
    msg = f'Starting Quality Assurance for input file: {input_file}' \
            f'\nOutputs to be generated:'
    if rslc_params.workflows.validate or rslc_params.workflows.qa_reports:
        summary_file = os.path.join(output_dir, 'SUMMARY.csv')
        msg += f'\n\tSummary file: {summary_file}'

    if rslc_params.workflows.qa_reports or \
        rslc_params.workflows.absolute_calibration_factor or \
        rslc_params.workflows.nesz or \
        rslc_params.workflows.point_target_analyzer:
    
        stats_file = os.path.join(output_dir, 'STATS.h5')
        msg += f'\n\tMetrics file: {stats_file}'

    if rslc_params.workflows.qa_reports:
        report_file = os.path.join(output_dir, 'REPORT.pdf')
        browse_image = os.path.join(output_dir, 'BROWSE.png')
        browse_kml = os.path.join(output_dir, 'BROWSE.kml')

        msg += f'\n\tReport file: {report_file}' \
               f'\n\tBrowse Image: {browse_image}' \
               f'\n\tBrowse Image Geolocation file: {browse_kml}'
    print(msg)

    # Parse the file's bands, frequencies, and polarizations.
    # Save data to STATS.h5
    with nisarqa.open_h5_file(input_file, mode='r') as in_file:

        # Note: `pols` contains references to datasets in the open input file.
        # All processing with `pols` must be done within this context manager,
        # or the references will be closed and inaccessible.
        bands, freqs, pols = nisarqa.rslc.get_bands_freqs_pols(in_file)

        # If running these workflows, save the processing parameters and
        # identification group to STATS.h5
        if rslc_params.workflows.qa_reports or \
            rslc_params.workflows.absolute_calibration_factor or \
            rslc_params.workflows.nesz or \
            rslc_params.workflows.point_target_analyzer:

            # This is the first time opening the STATS.h5 file for RSLC
            # workflow, so open in 'w' mode.
            # After this, always open STATS.h5 in 'r+' mode.
            with nisarqa.open_h5_file(stats_file, mode='w') as stats_h5:

                # Save the processing parameters to the stats.h5 file
                # Note: If only the validate workflow is requested,
                # this will do nothing.
                rslc_params.save_params_to_stats_file(h5_file=stats_h5,
                                                      bands=bands)

                # Copy the Product identification group to STATS.h5
                nisarqa.rslc.save_NISAR_identification_group_to_h5(
                        nisar_h5=in_file,
                        stats_h5=stats_h5)

        # Run the requested workflows
        if rslc_params.workflows.validate:
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
            nisarqa.output_stub_files(output_dir=output_dir,
                                    stub_files='summary_csv')

        if rslc_params.workflows.qa_reports:

            # TODO qa_reports will add to the SUMMARY.csv file.
            # For now, make sure that the stub file is output
            if not os.path.isfile(summary_file):
                nisarqa.output_stub_files(output_dir=output_dir,
                                        stub_files='summary_csv')

            # TODO qa_reports will create the BROWSE.kml file.
            # For now, make sure that the stub file is output
            nisarqa.output_stub_files(output_dir=output_dir,
                                    stub_files='browse_kml')

            with nisarqa.open_h5_file(stats_file, mode='r+') as stats_h5, \
                PdfPages(report_file) as report_pdf:

                # Save product info to stats file
                save_nisar_freq_metadata_to_h5(stats_h5=stats_h5,
                                               path_to_group='/QA/data',
                                               pols=pols)

                # Generate the RSLC Power Image
                process_power_images(pols=pols,
                                     params=rslc_params.power_img,
                                     stats_h5=stats_h5,
                                     report_pdf=report_pdf,
                                     browse_filename=browse_image)

                # Generate the RSLC Power and Phase Histograms
                process_power_and_phase_histograms(pols=pols,
                                                   params=rslc_params.histogram,
                                                   stats_h5=stats_h5,
                                                   report_pdf=report_pdf)

                # Process Interferograms

                # Generate Spectra

                # Check for invalid values

                # Compute metrics for stats.h5

    if rslc_params.workflows.absolute_calibration_factor:
        msg = f'Running Absolute Calibration Factor CalTool: {input_file}'
        print(msg)
        # logger.log_message(logging_base.LogFilterInfo, msg)

        # Run Absolute Calibration Factor tool
        nisarqa.caltools.run_abscal_tool(abscal_params=rslc_params.abs_cal,
                                         dyn_anc_params=rslc_params.anc_files,
                                         input_filename=input_file,
                                         stats_filename=stats_file)

    if rslc_params.workflows.nesz:
        msg = f'Running NESZ CalTool: {input_file}'
        print(msg)
        # logger.log_message(logging_base.LogFilterInfo, msg)

        # Run NESZ tool
        nisarqa.caltools.run_nesz_tool(params=rslc_params.nesz,
                                       input_filename=input_file,
                                       stats_filename=stats_file)

    if rslc_params.workflows.point_target_analyzer:
        msg = f'Running Point Target Analyzer CalTool: {input_file}'
        print(msg)
        # logger.log_message(logging_base.LogFilterInfo, msg)

        # Run Point Target Analyzer tool
        nisarqa.caltools.run_pta_tool(pta_params=rslc_params.pta,
                                      dyn_anc_params=rslc_params.anc_files,
                                      input_filename=input_file,
                                      stats_filename=stats_file)

    print('Successful completion. Check log file for validation warnings and errors.')


def save_NISAR_identification_group_to_h5(nisar_h5, stats_h5):
    '''
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
    '''

    for band in nisar_h5['/science']:
        grp_path = f'/science/{band}/identification'

        if 'identification' in stats_h5[f'/science/{band}']:
            # If the identification group already exists, copy each
            # dataset, etc. individually
            for item in nisar_h5[grp_path]:
                item_path = f'{grp_path}/{item}'
                nisar_h5.copy(nisar_h5[item_path], stats_h5, item_path)
        else:
            # Copy entire identification metadata from input file to stats.h5
            nisar_h5.copy(nisar_h5[grp_path], stats_h5, grp_path)


def save_nisar_freq_metadata_to_h5(stats_h5,
                                   path_to_group,
                                   pols):
    '''
    Populate the `stats_h5` HDF5 file with a list of each available
    frequency's polarizations.

    If `pols` contains values for Frequency A, then this dataset will
    be created in `stats_h5`:
        /science/<band>/<path_to_group>/frequencyA/listOfPolarizations
    
    If `pols` contains values for Frequency B, then this dataset will
    be created in `stats_h5`:
        /science/<band>/<path_to_group>/frequencyB/listOfPolarizations

    Parameters
    ----------
    stats_h5 : h5py.File
        Handle to an h5 file where the identification metadata
        should be saved
    path_to_group : str
        Internal path in `stats_h5` to the HDF5 group where
        the polarization data will be stored.
        That this will be appended to the root '/science/<band>'
        Example: if `path_to_group` is '/QA/data', and if only
        LSAR is an available band in `pols`, then the
        group will be '/science/LSAR/QA/data'
    pols : nested dict of RSLCRasterQA
        Nested dict of RSLCRasterQA objects, where each object represents
        a polarization dataset.
        Format: pols[<band>][<freq>][<pol>] -> a RSLCRasterQA
        Ex: pols['LSAR']['A']['HH'] -> the HH dataset, stored in a RSLCRasterQA object
    '''

    # Populate data group's metadata
    for band in pols:
        for freq in pols[band]:
            list_of_pols = list(pols[band][freq])
            grp_path = \
                f'/science/{band}/{path_to_group.lstrip("/")}/frequency{freq}'
            nisarqa.create_dataset_in_h5group(
                h5_file=stats_h5,
                grp_path=grp_path,
                ds_name='listOfPolarizations',
                ds_data=list_of_pols,
                ds_description=f'Polarizations for Frequency {freq} ' \
                    'discovered in input NISAR product by QA code')


class ComplexFloat16Decoder(object):
    '''Wrapper to read in NISAR product datasets that are '<c4' type,
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
    '''

    def __init__(self, h5dataset):
        self._dataset = h5dataset
        self._dtype = np.complex64

    def __getitem__(self, key):
        # Have h5py convert to the desired dtype on the fly when reading in data
        return self.read_c4_dataset_as_c8(self.dataset, key)

    @staticmethod
    def read_c4_dataset_as_c8(ds: h5py.Dataset, key=np.s_[...]):
        '''
        Read a complex float16 HDF5 dataset as a numpy.complex64 array.
        Avoids h5py/numpy dtype bugs and uses numpy float16 -> float32 conversions
        which are about 10x faster than HDF5 ones.
        '''
        # This avoids h5py exception:
        # TypeError: data type '<c4' not understood
        # Also note this syntax changed in h5py 3.0 and was deprecated in 3.6, see
        # https://docs.h5py.org/en/stable/whatsnew/3.6.html

        complex32 = np.dtype([('r', np.float16), ('i', np.float16)])
        z = ds.astype(complex32)[key]

        # Define a similar datatype for complex64 to be sure we cast safely.
        complex64 = np.dtype([('r', np.float32), ('i', np.float32)])

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


@dataclass
class RSLCRaster:
    '''
    RSLC image dataset with name.
    
    Parameters
    ----------
    data : array_like
        Raster data to be stored.
    name : str
        Name for the dataset

    Notes
    -----
    If data is an HDF5 dataset, suggest initializing using
    the class method from_h5dataset(..).
    '''

    # Raster data
    data: npt.ArrayLike

    # identifying name of this Raster; can be used for logging
    # e.g. 'LSAR_A_HH'
    name: str


    @classmethod
    def from_h5dataset(cls, h5dataset, name):
        '''
        Initialize an RSLCRaster object for a HDF5 dataset
        that needs to be decoded via a specific dtype.

        This will store the dataset as a ComplexFloat16Decoder
        object instead of a standard Arraylike object.

        Parameters
        ----------
        h5dataset : h5py.Dataset
            Raster data to be stored.
        name : str
            Name for the dataset
        '''
        data = ComplexFloat16Decoder(h5dataset)
        return cls(data, name)


@dataclass
class RSLCRasterQA(RSLCRaster):
    '''
    An RSLCRaster with additional attributes specific to the QA Code.
    
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
        The start time of the observation for this
        RSLC Raster
    az_stop : float
        The stopping time of the observation for this
        RSLC Raster
    range_spacing : float
        Range spacing of pixels of input array
    rng_start : float
        Start (near) distance of the range of input array
    rng_stop : float
        End (far) distance of the range of input array
    epoch : str
        The start of the epoch for this observation,
        in the format 'YYYY-MM-DD HH:MM:SS'

    Notes
    -----
    If data is an HDF5 dataset, suggest initializing using
    the class method from_h5dataset(..).
    '''

    # Attributes of the input array
    band: str
    freq: str
    pol: str

    az_spacing: float
    az_start: float
    az_stop: float

    range_spacing: float
    rng_start: float
    rng_stop: float

    epoch: str


    @classmethod
    def from_h5dataset(cls,
                       h5dataset, band, freq, pol,
                       az_spacing, az_start, az_stop,
                       range_spacing, rng_start, rng_stop,
                       epoch):
        '''
        Initialize an RSLCRasterQA object for a HDF5 dataset
        that needs to be decoded via a specific dtype.

        This will store the dataset as a ComplexFloat16Decoder
        object instead of a standard Arraylike object.

        Parameters
        ----------
        h5dataset : h5py.Dataset
            Raster data to be stored.
        band : str
            name of the band for `img`, e.g. 'LSAR'
        freq : str
            name of the frequency for `img`, e.g. 'A' or 'B'
        pol : str
            name of the polarization for `img`, e.g. 'HH' or 'HV'
        az_spacing : float
            Azimuth spacing of pixels of input array
        az_start : float
            The start time of the observation for this
            RSLC Raster
        az_stop : float
            The stopping time of the observation for this
            RSLC Raster
        range_spacing : float
            Range spacing of pixels of input array
        rng_start : float
            Start (near) distance of the range of input array
        rng_stop : float
            End (far) distance of the range of input array
        epoch : str
            The reference epoch for this observation,
            in the format 'YYYY-MM-DD HH:MM:SS'

        Notes
        -----
        Unlike the default constructor for RSLCRasterQA, this
        function does not include a user-specified input 
        parameter `name`. Instead, to maintain consistency of
        the format of the `name` for each NISAR RSLC QA image,
        the `name` attribute will be populated with a string
        of the format: <band>_<freq>_<pol>
        '''
        data = ComplexFloat16Decoder(h5dataset)

        # Format the name
        name = f'{band}_{freq}_{pol}'

        # Note the order of the positional arguments being passed;
        # the attributes of the parent class must be passed first.
        return cls(data, name, band, freq, pol, 
                   az_spacing, az_start, az_stop,
                   range_spacing, rng_start, rng_stop,
                   epoch)

    @classmethod
    def from_nisar_rslc_h5_dataset(cls,
                       h5_file, band, freq, pol):
        '''
        Initialize an RSLCRasterQA object for the given 
        band-freq-pol image in the input NISAR RSLC HDF5 file.
        
        This will store the dataset as a ComplexFloat16Decoder
        object instead of a standard Arraylike object. If the file 
        does not contain an image dataset for the given 
        band-freq-pol combination, a DatasetNotFoundError
        exception will be thrown.

        Parameters
        ----------
        h5_file : h5py.File
            File handle to a valid NISAR RSLC hdf5 file.
            Polarization images must be located in the h5 file in the path: 
            /science/<band>/RSLC/swaths/frequency<freq>/<pol>
            or they will not be found. This is the file structure
            as determined from the NISAR Product Spec.
        band : str
            name of the band for `img`, e.g. 'LSAR'
        freq : str
            name of the frequency for `img`, e.g. 'A' or 'B'
        pol : str
            name of the polarization for `img`, e.g. 'HH' or 'HV'

        Notes
        -----
        The `name` attribute will be populated with a string
        of the format: <band>_<freq>_<pol>
        '''

        # check if this is an RSLC or and SLC file.
        if f'/science/{band}/RSLC' in h5_file:
            slc_type = 'RSLC'
        elif f'/science/{band}/SLC' in h5_file:
            # TODO - The UAVSAR test datasets were created with only the 'SLC'
            # filepath. New NISAR RSLC Products should only contain 'RSLC' file paths.
            # Once the test datasets have been updated to 'RSLC', then remove this
            # warning, and raise a fatal error.
            print('WARNING!! This product uses the deprecated "SLC" group. Update to "RSLC".')

            slc_type = 'SLC'
        else:
            # self.logger.log_message(logging_base.LogFilterError, 'Invalid file structure.')
            raise nisarqa.DatasetNotFoundError

        # Hardcoded paths to various groups in the NISAR RSLC h5 file.
        # These paths are determined by the RSLC .xml product spec
        swaths_path = f'/science/{band}/{slc_type}/swaths'
        freq_path = f'{swaths_path}/frequency{freq}/'
        pol_path = f'{freq_path}/{pol}'
        band_freq_pol_str = f'{band}_{freq}_{pol}'

        if pol_path in h5_file:
            # self.logger.log_message(logging_base.LogFilterInfo, 
            #                         'Found image %s' % band_freq_pol_str)
            pass
        else:
            # self.logger.log_message(logging_base.LogFilterInfo, 
            #                         'Image %s not present' % band_freq_pol_str)
            return None

        # From the xml Product Spec, sceneCenterAlongTrackSpacing is the 
        # 'Nominal along track spacing in meters between consecutive lines 
        # near mid swath of the RSLC image.'
        az_spacing = h5_file[freq_path]['sceneCenterAlongTrackSpacing'][...]

        # Get Azimuth (y-axis) tick range + label
        # path in h5 file: /science/LSAR/RSLC/swaths/zeroDopplerTime
        az_start = float(h5_file[swaths_path]['zeroDopplerTime'][0])
        az_stop =  float(h5_file[swaths_path]['zeroDopplerTime'][-1])

        # From the xml Product Spec, sceneCenterGroundRangeSpacing is the 
        # 'Nominal ground range spacing in meters between consecutive pixels
        # near mid swath of the RSLC image.'
        range_spacing = h5_file[freq_path]['sceneCenterGroundRangeSpacing'][...]

        # Range in meters (units are specified as meters in the product spec)
        rng_start = float(h5_file[freq_path]['slantRange'][0])
        rng_stop = float(h5_file[freq_path]['slantRange'][-1])

        # output of the next line will have the format: 'seconds since YYYY-MM-DD HH:MM:SS'
        sec_since_epoch = h5_file[swaths_path]['zeroDopplerTime'].attrs['units'].decode('utf-8')
        epoch = sec_since_epoch.replace('seconds since ', '').strip()

        return RSLCRasterQA.from_h5dataset(h5_file[pol_path],
                                   band=band,
                                   freq=freq,
                                   pol=pol,
                                   az_spacing=az_spacing,
                                   az_start=az_start,
                                   az_stop=az_stop,
                                   range_spacing=range_spacing,
                                   rng_start=rng_start,
                                   rng_stop=rng_stop,
                                   epoch=epoch)


def get_bands_freqs_pols(h5_file):
    '''
    Locate the available bands, frequencies, and polarizations
    in the input HDF5 file.

    Parameters
    ----------
    h5_file : h5py.File
        Handle to the input product h5 file

    Returns
    -------
    bands : dict of h5py Groups
        Dict of the h5py Groups for each band in `h5_file`,
        where the keys are the available bands (i.e. 'SSAR' or 'LSAR').
        Format: bands[<band>] -> a h5py Group
        Ex: bands['LSAR'] -> the h5py Group for LSAR
    freqs : nested dict of h5py Groups
        Nested dict of the h5py Groups for each freq in `h5_file`,
        where the keys are the available bands-freqs (i.e. 'LSAR B' or 'SSAR A').
        Format: freqs[<band>][<freq>] -> a h5py Group
        Ex: freqs['LSAR']['A'] -> the h5py Group for LSAR's FrequencyA
    pols : nested dict of RSLCRasterQA
        Nested dict of RSLCRasterQA objects, where each object represents
        a polarization dataset in `h5_file`.
        Format: pols[<band>][<freq>][<pol>] -> a RSLCRasterQA
        Ex: pols['LSAR']['A']['HH'] -> the HH dataset, stored in a RSLCRasterQA object

    '''

    bands = _get_bands(h5_file)
    freqs = _get_freqs(h5_file, bands)
    pols = _get_pols(h5_file, freqs)

    return bands, freqs, pols


def _get_bands(h5_file):
    '''
    Finds the available bands in the input file
    and stores their paths in a nested dictionary.

    Parameters
    ----------
    h5_file : h5py.File
        File handle to a valid NISAR RSLC hdf5 file.
        Bands must be located in the h5 file in the path: /science/<band>
        or they will not be found.

    Returns
    -------
    bands : dict of h5py Groups
        Dict of the h5py Groups for each band in `h5_file`,
        where the keys are the available bands (i.e. 'SSAR' and/or 'LSAR').
        Format: bands[<band>] -> a h5py Group
        Ex: bands['LSAR'] -> the h5py Group for LSAR
    '''

    bands = {}
    for band in nisarqa.BANDS:
        path = f'/science/{band}'
        if path in h5_file:
            # self.logger.log_message(logging_base.LogFilterInfo, 'Found band %s' % band)
            bands[band] = h5_file[path]
        else:
            # self.logger.log_message(logging_base.LogFilterInfo, '%s not present' % band)
            pass
    return bands


def _get_freqs(h5_file, bands):
    '''
    Finds the available frequencies in the input file
    and stores their paths in a nested dictionary.

    Parameters
    ----------
    h5_file : h5py.File
        File handle to a valid NISAR RSLC hdf5 file.
        Frequencies must be located in the h5 file in the path: 
        /science/<band>/RSLC/swaths/frequency<freq>
        or they will not be found.
    bands : list_like
        An iterable of the bands in `h5_file`.

    Returns
    -------
    freqs : nested dict of h5py Groups
        Dict of the h5py Groups for each freq in `h5_file`,
        where the keys are the available bands-freqs (i.e. 'LSAR B' or 'SSAR A').
        Format: freqs[<band>][<freq>] -> a h5py Group
        Ex: freqs['LSAR']['A'] -> the h5py Group for LSAR's FrequencyA

    See Also
    --------
    get_bands : function to generate the `bands` input argument
    '''

    freqs = {}
    for band in bands.keys():
        freqs[band] = {}
        for freq in nisarqa.RSLC_FREQS:
            path = f'/science/{band}/RSLC/swaths/frequency{freq}'
            if path in h5_file:
                # self.logger.log_message(logging_base.LogFilterInfo, 'Found band %s' % band)
                freqs[band][freq] = h5_file[path]

            # TODO - The original test datasets were created with only the 'SLC'
            # filepath. New NISAR RSLC Products should only contain 'RSLC' file paths.
            # Once the test datasets have been updated to 'RSLC', then remove this
            # warning, and raise a fatal error.
            elif path.replace('RSLC', 'SLC') in h5_file:
                freqs[band][freq] = h5_file[path.replace('RSLC', 'SLC')]
                print('WARNING!! This product uses the deprecated "SLC" group. Update to "RSLC".')
            else:
                # self.logger.log_message(logging_base.LogFilterInfo, '%s not present' % band)
                pass

    # Sanity Check - if a band does not have any frequencies, this is a validation error.
    # This check should be handled during the validation process before this function was called,
    # not the quality process, so raise an error.
    # In the future, this step might be moved earlier in the processing, and changed to
    # be handled via: 'log the error and remove the band from the dictionary' 
    for band in freqs.keys():
        # Empty dictionaries evaluate to False in Python
        if not freqs[band]:
            raise ValueError(f'Provided input file\'s band {band} does not '
                              'contain any frequency groups.')

    return freqs


def _get_pols(h5_file, freqs):
    '''
    Finds the available polarization rasters in the input file
    and stores their paths in a nested dictionary.

    Parameters
    ----------
    h5_file : h5py.File
        File handle to a valid NISAR RSLC hdf5 file.
        frequencies must be located in the h5 file in the path: 
        /science/<band>/RSLC/swaths/frequency<freq>/<pol>
        or they will not be found.
    freqs : nested dict of h5py Groups
        Dict of the h5py Groups for each freq in `h5_file`,
        where the keys are the available bands-freqs (i.e. 'LSAR B' or 'SSAR A').
        Format: freqs[<band>][<freq>] -> a h5py Group
        Ex: freqs['LSAR']['A'] -> the h5py Group for LSAR's FrequencyA

    Returns
    -------
    pols : nested dict of RSLCRasterQA
        Nested dict of RSLCRasterQA objects, where each object represents
        a polarization dataset in `h5_file`.
        Format: pols[<band>][<freq>][<pol>] -> a RSLCRasterQA
        Ex: pols['LSAR']['A']['HH'] -> the HH dataset, stored in a RSLCRasterQA object

    See Also
    --------
    get_freqs : function to generate the `freqs` input argument
    '''

    # Discover images in dataset and populate the dictionary
    pols = {}
    for band in freqs:
        pols[band] = {}
        for freq in freqs[band]:
            pols[band][freq] = {}
            for pol in nisarqa.RSLC_POLS:

                try:
                    tmp_RSLCRasterQA = \
                        RSLCRasterQA.from_nisar_rslc_h5_dataset(h5_file, band, freq, pol)
                except nisarqa.DatasetNotFoundError:
                    # RSLC Raster QA could not be created, which means that the
                    # input file did not contain am image with the current
                    # `band`, `freq`, and `pol` combination.
                    continue

                if isinstance(tmp_RSLCRasterQA, RSLCRasterQA):
                    pols[band][freq][pol] = tmp_RSLCRasterQA

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
                raise ValueError(f'Provided input file does not have any polarizations'
                            f' included under band {band}, frequency {freq}.')

    return pols


def _layer_selection_for_browse(pols):
    '''
    Assign the polarization layers in the input file to grayscale or
    RGBA channels for the Browse Image.

    See `Notes` for details on the possible NISAR modes and assigned channels
    for LSAR band.
    SSAR is currently only minimally supported, so only the first polarization
    found should be used to create a grayscale image.

    
    Parameters
    ----------
    pols : nested dict of RSLCRasterQA
        Nested dict of RSLCRasterQA objects, where each object represents
        a polarization dataset in `h5_file`.
        Format: pols[<band>][<freq>][<pol>] -> a RSLCRasterQA
        Ex: pols['LSAR']['A']['HH'] -> the HH dataset, stored 
                                       in a RSLCRasterQA object

    Returns
    -------
    layers_for_browse : dict
        A dictionary containing the channel assignments. Its structure is:

        layers_for_browse['band']  : str
                                        Either 'LSAR' or 'SSAR'
        layers_for_browse['A']     : list of str, optional
                                        List of the Freq A polarizations
                                        required to create the browse image.
                                        A subset of ['HH','HV','VV','RH', 'LH']
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
        - Freq A: Red=RH, Blue=RH (or LH?)
        - Freq B: Green=RV (or LV?)
    '''

    layers_for_browse = {}
    # Determine which band to use. LSAR has priority over SSAR.

    bands = list(pols)
    if 'LSAR' in bands:
        layers_for_browse['band'] = 'LSAR'
    elif 'SSAR' in bands:
        layers_for_browse['band'] = 'SSAR'
    else:
        raise ValueError(f'Only "LSAR" and "SSAR" bands are supported: {band}')

    band = bands[0]

    # Check that the correct frequencies are available
    if not set(pols[band].keys()).issubset({'A', 'B'}):
        raise ValueError(f'`pols["{band}"]` must contain only "A" '
                         f'and/or "B": {pols.keys()}')

    # Get the frequency. A has priority over B.
    if 'A' in pols[band]:
        freq = 'A'
    else:
        freq = 'B'

    # SSAR is not fully supported by QA, so just make a simple grayscale
    if band == 'SSAR':
        # Prioritize Co-Pol
        if 'HH' in pols[band][freq]:
            layers_for_browse[freq] = ['HH']
        elif 'VV' in pols[band][freq]:
            layers_for_browse[freq] = ['VV']
        else:
            # Take the first available Cross-Pol
            layers_for_browse[freq] = [pols[band][freq][0]]

        return layers_for_browse


    # The input file contains LSAR data. Will need to make
    # grayscale/RGB channel assignments

    # Get the available polarizations
    available_pols = list(pols[band][freq])
    n_pols = len(available_pols)

    if freq == 'B':
        # This means only Freq B has data; this only occurs in Single Pol case.
        if n_pols > 1:
            raise ValueError('When only Freq B is present, then only '
                    f'single-pol mode supported. Freq{freq}: {available_pols}')

        layers_for_browse['B'] = available_pols

    else:  # freq A exists
        if n_pols == 1:

            if ('B' in pols[band]) and \
                (available_pols == list(pols[band]['B'])):

                # A's polarization image is identical to B's pol image,
                # which only occurs for Quasi Dual Pol
                layers_for_browse['A'] = available_pols
                layers_for_browse['B'] = available_pols

            else:
                # Single Pol
                layers_for_browse['A'] = available_pols

        elif n_pols in (2,4):
            # dual-pol, quad-pol, or Quasi-Quad pol

            # HH has priority over VV
            if 'HH' in available_pols and 'HV' in available_pols:
                layers_for_browse['A'] = ['HH', 'HV']
                if n_pols == 4:
                    # quad pol
                    layers_for_browse['A'].append('VV')

            elif 'VV' in available_pols and 'VH' in available_pols:
                # If there is only 'VV', then this granule must be dual-pol
                layers_for_browse['A'] = ['VV', 'VH']
            
            else:
                raise ValueError('For dual-pol, quad-pol, and quasi-quad, '
                                 'polarizations must have the same Tx '
                                 f'polarization: {available_pols}')

    # Sanity Check
    if ('A' not in layers_for_browse) and ('B' not in layers_for_browse):
        raise ValueError('Current Mode (configuration) of the NISAR input file'
                         ' not supported for browse image.')

    return layers_for_browse


def process_power_images(pols, params, stats_h5, report_pdf,
                         browse_filename='./BROWSE.png'):
    '''
    Generate the RSLC Power Image plots for the `report_pdf` and
    corresponding browse image product.

    Parameters
    ----------
    pols : nested dict of RSLCRasterQA
        Nested dict of RSLCRasterQA objects, where each object represents
        a polarization dataset in `h5_file`.
        Format: pols[<band>][<freq>][<pol>] -> a RSLCRasterQA
        Ex: pols['LSAR']['A']['HH'] -> the HH dataset, stored 
                                       in a RSLCRasterQA object
    params : RSLCPowerImageParams
        A dataclass containing the parameters for processing
        and outputting the power image(s).
    stats_h5 : h5py.File
        The output file to save QA metrics, etc. to
    report_pdf : PdfPages
        The output pdf file to append the power image plot to
    browse_filename : str, optional
        Filename (with path) for the browse image PNG.
        Defaults to './BROWSE.png'
    '''

    # Select which layers will be needed for the browse image.
    # Multilooking takes a long time, but each multilooked polarization image
    # should be less than ~4 MB given the current requirements for NISAR,
    # so it's ok to store the needed, multilooked Power Images in memory.
    layers_for_browse = _layer_selection_for_browse(pols)
    browse_pol_imgs = {}

    # Process each image in the dataset

    for band in pols:
        for freq in pols[band]:
            if freq == 'B':
                continue
            for pol in pols[band][freq]:
                img = pols[band][freq][pol]

                multilooked_img = get_multilooked_power_image(
                                            img=img,
                                            params=params,
                                            stats_h5=stats_h5)

                corrected_img = save_single_power_image_to_pdf(
                                            img_arr=multilooked_img,
                                            img=img,
                                            params=params,
                                            report_pdf=report_pdf)

                # If this power image is needed to construct the browse image...
                if band == layers_for_browse['band'] and \
                    freq in layers_for_browse and \
                        pol in layers_for_browse[freq]:
                    
                    # ...keep the multilooked, color-corrected image

                    if not params.linear_units.val:
                        # Browse image must be linear (not dB).
                        # Redo color correction, but without dB correction.
                        corrected_img = clip_array(
                                            multilooked_img,
                                            middle_percentile= \
                                                params.middle_percentile.val)
                        
                        if params.gamma.val is not None:
                            corrected_img = apply_gamma_correction(
                                            corrected_img,
                                            gamma=params.gamma.val)

                    browse_pol_imgs[pol] = corrected_img

    # Construct the browse image
    _save_rslc_browse_img(pol_imgs=browse_pol_imgs, 
                          params=params, 
                          filepath=browse_filename)


def get_multilooked_power_image(img,
                                params,
                                stats_h5):
    '''
    Generate the multilooked RSLC Power Image array for a single RSLC
    polarization image.

    Parameters
    ----------
    img : RSLCRasterQA
        The RSLC raster to be processed
    params : RSLCPowerImageParams
        A structure containing the parameters for processing
        and outputting the power image(s).
    stats_h5 : h5py.File
        The output file to save QA metrics, etc. to

    Returns
    -------
    out_img : numpy.ndarray
        The multilooked Power Image
    '''

    nlooks_freqa_arg = params.nlooks_freqa.val
    nlooks_freqb_arg = params.nlooks_freqb.val

    # Get the window size for multilooking
    if (img.freq == 'A' and nlooks_freqa_arg is None) or \
        (img.freq == 'B' and nlooks_freqb_arg is None):

        nlooks = nisarqa.compute_square_pixel_nlooks(
                    img.data.shape,
                    sample_spacing=(img.az_spacing, img.range_spacing),
                    num_mpix=params.num_mpix.val)

    elif img.freq == 'A':
        nlooks = nlooks_freqa_arg
    elif img.freq == 'B':
        nlooks = nlooks_freqb_arg
    else:
        raise ValueError(f'frequency is "{img.freq}", but only "A" or "B" '
                          'are valid options.')

    print(f'\nMultilooking Image {img.name} with shape: {img.data.shape}')
    print('sceneCenterAlongTrackSpacing: ', img.az_spacing)
    print('sceneCenterGroundRangeSpacing: ', img.range_spacing)
    print('Beginning Multilooking with nlooks window shape: ', nlooks)

    # Multilook
    out_img = nisarqa.compute_multilooked_power_by_tiling(
                                            arr=img.data,
                                            nlooks=nlooks,
                                            tile_shape=params.tile_shape.val)

    print(f'Multilooking Complete. Multilooked image shape: {out_img.shape}')

    return out_img


def save_single_power_image_to_pdf(img_arr, img, params, report_pdf):
    '''
    Annotate and save a RSLC Power Image to `report_pdf`.

    Parameters
    ----------
    img_arr : numpy.ndarray
        RSLC 2D image array to be saved. This has typically been multilooked
        to the correct size.
    img : RSLCRasterQA
        The RSLCRasterQA object that corresponds to `img`. The metadata
        from this will be used for annotating the image plot.
    params : RSLCPowerImageParams
        A structure containing the parameters for processing
        and outputting the power image(s).
    report_pdf : PdfPages
        The output pdf file to append the power image plot to

    Returns
    -------
    out_img : numpy.ndarray
        2D image array that was saved to the PDF. If any image correction
        was specified via `params` and applied to `img_arr`, this returned
        array will include that image correction.
    '''

    # Apply image correction to the multilooked array

    # Step 1: Clip the image array's outliers
    img_arr = clip_array(img_arr, middle_percentile=params.middle_percentile.val)

    # Step 2: Convert from linear units to dB
    if not params.linear_units.val:
        img_arr = nisarqa.pow2db(img_arr)

    # Step 3: Apply gamma correction
    if params.gamma.val is not None:
        # Get the vmin and vmax prior to applying gamma correction.
        # These will later be used for setting the colorbar's
        # tick mark values.
        vmin = np.min(img_arr)
        vmax = np.max(img_arr)

        img_arr = apply_gamma_correction(img_arr, gamma=params.gamma.val)

    # Plot and Save Power Image to graphical summary pdf
    title = f'RSLC Multilooked Power ({params.pow_units.val}%s)\n{img.name}'
    if params.gamma.val is None:
        title = title % ''
    else:
        title = title % fr', $\gamma$={params.gamma.val}'

    # Get Azimuth (y-axis) label
    az_title = f'Zero Doppler Time\n(seconds since {img.epoch})'

    # Get Range (x-axis) labels and scale
    # Convert range from meters to km
    rng_start_km = img.rng_start/1000.
    rng_stop_km = img.rng_stop/1000.
    rng_title = 'Slant Range (km)'

    # Define the formatter function to invert the gamma correction
    # and produce the colorbar labels with values that match
    # the underlying, pre-gamma corrected data.
    # See: https://matplotlib.org/2.0.2/examples/pylab_examples/custom_ticker1.html
    if params.gamma.val is not None:
        def inverse_gamma_correction(x, pos):
            '''
            FuncFormatter to invert the gamma correction values
            and return the "true" value of the data for a 
            given tick.

            FuncFormatter functions must take two arguments: 
            `x` for the tick value and `pos` for the tick position,
            and must return a str. The `pos` argument is used
            internally by matplotlib.
            '''
            # Invert the power
            val = np.power(x, 1 / params.gamma.val)

            # Invert the normalization
            val = (val * (vmax - vmin)) + vmin

            return '{:.2f}'.format(val)

        colorbar_formatter = inverse_gamma_correction
        
    else:
        colorbar_formatter = None

    plot2pdf(img_arr=img_arr,
             title=title,
             ylim=[img.az_start, img.az_stop],
             xlim=[rng_start_km, rng_stop_km],
             colorbar_formatter=colorbar_formatter,
             ylabel=az_title,
             xlabel=rng_title,
             plots_pdf=report_pdf
             )
    
    return img_arr


def clip_array(arr, middle_percentile=100.0):
    '''
    Clip input array to the middle percentile.

    Parameters
    ----------
    arr : array_like
        Input array
    middle_percentile : numeric, optional
        Defines the middle percentile range of the `arr` 
        that the colormap covers. Must be in the range [0, 100].
        Defaults to 100.0.

    Returns
    -------
    out_img : numpy.ndarray
        A copy of the input array with the values outside of the
        range defined by `middle_percentile` clipped.
    '''
    # Clip the image data
    vmin, vmax = calc_vmin_vmax(arr, middle_percentile=middle_percentile)
    out_arr = np.clip(arr, a_min=vmin, a_max=vmax)

    return out_arr


def apply_gamma_correction(img_arr, gamma):
    '''
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
    '''
    # Normalize to range [0,1]
    out_img = nisarqa.normalize(img_arr)

    # Apply gamma correction
    out_img = np.power(out_img, gamma)

    return out_img


def plot_to_grayscale_png(img_arr, filepath, valid_arr=None):
    '''
    Save the image array to a 1-channel grayscale browse image png.

    Browse image pixels values will be scaled from 0-254. The pixel value
    of 255 is reserved to denote transparent pixels.
    If `valid_arr` is not provided or is `None`, then output png will
    not have transparency. If `valid_arr` is provided, output png
    will have transparency. 

    Parameters
    ----------
    img_arr : array_like
        2D Image to plot
    filepath : str
        Full filepath the browse image product.
    valid_arr : Boolean_like array_like, optional
        2D array of same shape as `img_arr`.
        Valid pixels should be denoted with a `1` or `True`, indicating that
        these pixels will be opaque and appear in the output .png image.
        Invalid pixels should be denoted with a `0` or `False`, indicating
        that these pixels should appear as transparent in the output .png.
        Defaults to `None`, meaning no transparency will be applied.

    Notes
    -----
    This function does not add a full alpha channel to the output png.
    It instead uses "cheap transparency" (palette-based transparency)
    to keep file size smaller.
    See: http://www.libpng.org/pub/png/book/chapter08.html#png.ch08.div.5.4
    '''

    # Only use 2D arrays
    if len(np.shape(img_arr)) != 2:
        raise ValueError('Input image array must be 2D.')

    if valid_arr is not None:
        if np.shape(valid_arr) != np.shape(img_arr):
            raise ValueError('Input valid pixels array has shape'
                             f' {np.shape(valid_arr)}, but it must'
                             ' match the input image array shape which is'
                             f' {np.shape(img_arr)}')

    img_arr, transparency_val = \
        _prep_arr_for_png_with_transparency(img_arr, valid_arr)

    # Save as grayscale image using PIL.Image. 'L' is grayscale mode.
    # (Pyplot only saves png's as RGB, even if cmap=plt.cm.gray)
    im = Image.fromarray(img_arr, mode='L')
    im.save(filepath, transparency=transparency_val)  # default = 72 dpi


def _save_rslc_browse_img(pol_imgs, params, filepath):
    '''
    Process and save a RSLC Power Image to a browse PNG.

    Output browse image will be keep the same pixel dimensions as the
    input polarization image(s). Non-finite values will be made transparent.

    Color Channels will be assigned as follows:

        If pol_imgs.keys() contains only one image, then:
            gray = <that image>
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
                <polarization> must be a subset of: 'HH', 'HV', 'VV', 'VH'
        Example:
            pol_imgs['HH'] : <2D numpy.ndarray image>
            pol_imgs['VV'] : <2D numpy.ndarray image>
    params : RSLCPowerImageParams
        A structure containing the parameters for processing
        and outputting the power image(s).
    filepath : str
        Full filepath for where to save the browse image PNG.

    Notes
    -----
    If there are multiple input images, they must be thoughtfully prepared and
    standardized to each other prior to being passed into this function. This
    function directly combines the images into the same browse image.
    For example, trying to combine a Freq A 20 MHz image
    and a Freq B 5 MHz image into the same output browse image might not go
    well, unless the image arrays were properly prepared and coordinated
    in advance.
    '''

    # Create the valid pixels mask. True for valid pixels. False for invalid.
    # If a pixel is invalid in any polarization image, it should be invalid
    # for all images. Otherwise, only e.g. one channel might have a missing
    # pixel, which would cause the final image to have have "flecks"
    valid_arr = np.full(np.shape(set(pol_imgs).pop()), True, dtype=bool)
    print(valid_arr)
    for img in pol_imgs.values():
        valid_arr &= np.isfinite(img)

    set_of_pol_imgs = set(pol_imgs)

    if set_of_pol_imgs == {'HH','HV','VV'}:
        # Quad Pol
        red = pol_imgs['HH']
        green = pol_imgs['HV']
        blue = pol_imgs['VV']
    elif set_of_pol_imgs == {'HH','HV'}:
        # dual pol horizontal transmit, or quasi-quad
        red = pol_imgs['HH']
        green = pol_imgs['HV']
        blue = pol_imgs['HH']
    elif set_of_pol_imgs == {'HH','VV'}:
        # quasi-dual mode
        red = pol_imgs['HH']
        green = pol_imgs['VV']
        blue = pol_imgs['HH']
    elif set_of_pol_imgs == {'VV','VH'}:
        # dual-pol only, vertical transmit
        red = pol_imgs['VV']
        green = pol_imgs['VH']
        blue = pol_imgs['VV']
    else:
        # If we get into this "else" statement, then
        # either there is only one image provided (e.g. single pol),
        # or the images provided are not one of the expected cases.
        # Either way, WLOG plot one of the image(s) in `pol_imgs`.
        plot_to_grayscale_png(img_arr=set(pol_imgs).pop(),
                              filepath=filepath,
                              transparency_arr=valid_arr)

        # This `else` is a catch-all clause. Return early, so that 
        # we do not try to plot to RGB
        return

    plot_to_rgb_png(red=red,
                    green=green,
                    blue=blue,
                    filepath=filepath,
                    valid_arr=valid_arr)


def plot_to_rgb_png(red, green, blue, filepath, valid_arr):
    '''
    Image correct and save a RSLC Power Image to a browse PNG.

    Parameters
    ----------
    red, green, blue : numpy.ndarray
        2D arrays that will be mapped to the red, green, and blue
        channels (respectively) for the PNG. These four arrays must have
        identical shape.
    filepath : str
        Full filepath for where to save the browse image PNG.
    valid_arr : Boolean_like array_like, optional
        2D array of same shape as `img_arr`.
        Valid pixels should be denoted with a `1` or `True`, indicating that
        these pixels will be opaque and appear in the output .png image.
        Invalid pixels should be denoted with a `0` or `False`, indicating
        that these pixels should appear as transparent in the output .png.
        Defaults to no transparency applied.

    Notes
    -----
    This function does not add a full alpha channel to the output png.
    It instead uses "cheap transparency" (palette-based transparency)
    to keep file size smaller.
    See: http://www.libpng.org/pub/png/book/chapter08.html#png.ch08.div.5.4
    '''

    # Only use 2D arrays
    for arr in (red, green, blue):
        if len(np.shape(arr)) != 2:
            raise ValueError('Input image array must be 2D.')

    if valid_arr is not None:
        if np.shape(valid_arr) != np.shape(red):  # WLOG use red layer
            raise ValueError('Input valid pixels array has shape'
                             f' {np.shape(valid_arr)}, but it must'
                             ' match the input image array shape which is'
                             f' {np.shape(red)}')

    # Concatenate into RGB array
    rgb_arr = np.zeros((np.shape(red)[0], np.shape(red)[1], 3))

    # transparency_val will be the same from all calls to this function;
    # only need to capture it once.
    rgb_arr[:,:,0], transparency_val = \
                        _prep_arr_for_png_with_transparency(red, valid_arr)
    rgb_arr[:,:,1] = _prep_arr_for_png_with_transparency(green, valid_arr)
    rgb_arr[:,:,2] = _prep_arr_for_png_with_transparency(blue, valid_arr)

    if valid_arr is not None:
        # make a tuple with length 3, where each entry denotes the transparent
        # value for R, G, and B channels (respectively)
        transparency_val = ((transparency_val, ) * 3)

    im = Image.fromarray(rgb_arr, mode='RGB')
    im.save(filepath, transparency=transparency_val)  # default = 72 dpi    


def _prep_arr_for_png_with_transparency(img_arr, valid_arr):
    '''
    Prepare a 2D image array for use in a uint8 PNG.
    
    Normalizes a scales the array values to 0-254. If `valid_arr` is provided,
    will set each invalid pixel to 255.

    Parameters
    ----------
    img_arr : array_like
        2D Image to plot
    valid_arr : Boolean_like array_like, optional
        2D array of same shape as `img_arr`.
        Valid pixels should be denoted with a `1` or `True`.
        Invalid pixels should be denoted with a `0` or `False`.
        All invalid pixels will be set to 255 in the output array.
        Defaults to None, meaning all pixels are valid.
    
    Returns
    -------
    out : array_like
        Copy of the input image array that has been prepared for use in
        a PNG file.
        Input image array values were normalized to [0,1] and then
        scaled to [0,254]. If a `valid_arr` was provided, invalid
        pixels are set to 255.
    transparency_value : int or None
        If `valid_arr` was not provided, then `None` will be returned.
        If `valid_arr` was provided, then the number denoting invalid
        pixels (255) will be returned.
    '''

    # Normalize to range [0,1]. If the array is already normalized,
    # this should have no impact.
    out = nisarqa.normalize(img_arr)

    # After normalization to range [0,1], scale to 0-254 for unsigned int8
    # The value 255 will be (possibly) later used as the transparency value.
    out = np.uint8(out * 254)

    if valid_arr is None:
        transparency_value = None

    else:

        # Update transparency value so that the "alpha" is added to the image
        transparency_value = 255

        # Denote invalid pixels with 255, so that they output as transparent
        out = np.where(out, valid_arr.astype(bool),
                            out, transparency_value)
    
    return out, transparency_value



def plot2pdf(img_arr,
             plots_pdf,
             title=None,
             xlim=None,
             ylim=None,
             colorbar_formatter=None,
             xlabel=None,
             ylabel=None
             ):
    '''
    Plot the clipped image array and append it to the pdf.

    Parameters
    ----------
    img_arr : array_like
        Image to plot
    plots_pdf : PdfPages
        The output pdf file to append the power image plot to
    title : str, optional
        The full title for the plot
    xlim, ylim : sequence of numeric, optional
        Lower and upper limits for the axes ticks for the plot.
        Format: xlim=[<x-axis lower limit>, <x-axis upper limit>], 
                ylim=[<y-axis lower limit>, <y-axis upper limit>]
    colorbar_formatter : function, optional
        Tick formatter function to define how the numeric value 
        associated with each tick on the colorbar axis is formatted
        as a string. This function must take exactly two arguments: 
        `x` for the tick value and `pos` for the tick position,
        and must return a `str`. The `pos` argument is used
        internally by matplotlib.
        See: https://matplotlib.org/2.0.2/examples/pylab_examples/custom_ticker1.html
        (Wrapping the function with FuncFormatter is optional.)
    xlabel, ylabel : str, optional
        Axes labels for the x-axis and y-axis (respectively)
    '''

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
        if H/W >= fig_h/fig_w:
            # image will be limited by its height, so
            # it will not use the full width of the figure
            fig_w = W * (fig_h/H)
        else:
            # image will be limited by its width, so
            # it will not use the full height of the figure
            fig_h = H * (fig_w/W)

    if xlim is not None:

        # Compute num of xticks to use
        num_xticks = int(ticks_per_inch * fig_w)

        # Always have a minimum of 2 labeled ticks
        num_xticks = num_xticks if num_xticks >=2 else 2

        # Specify where we want the ticks, in pixel locations.
        xticks = np.linspace(0,img_arr_shape[1], num_xticks)
        ax.set_xticks(xticks)

        # Specify what those pixel locations correspond to in data coordinates.
        # By default, np.linspace is inclusive of the endpoint
        xticklabels = ['{:.1f}'.format(i) for i in np.linspace(start=xlim[0],
                                              stop=xlim[1],
                                              num=num_xticks)]
        ax.set_xticklabels(xticklabels)

        plt.xticks(rotation=45)

    if ylim is not None:
        
        # Compute num of yticks to use
        num_yticks = int(ticks_per_inch * fig_h)

        # Always have a minimum of 2 labeled ticks
        if num_yticks < 2:
            num_yticks = 2

        # Specify where we want the ticks, in pixel locations.
        yticks = np.linspace(0,img_arr_shape[0], num_yticks)
        ax.set_yticks(yticks)

        # Specify what those pixel locations correspond to in data coordinates.
        # By default, np.linspace is inclusive of the endpoint
        yticklabels = ['{:.1f}'.format(i) for i in np.linspace(start=ylim[0],
                                                        stop=ylim[1],
                                                        num=num_yticks)]
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


def calc_vmin_vmax(data_in, middle_percentile=100.0):
    '''
    Calculate the values of vmin and vmax for the 
    input array using the given middle percentile.

    For example, if `middle_percentile` is 95.0, then this will
    return the value of the 2.5th percentile and the 97.5th 
    percentile.

    Parameters
    ----------
    data_in : array_like
        Input array
    middle_percentile : numeric
        Defines the middle percentile range of the `data_in`. 
        Must be in the range [0, 100].
        Defaults to 100.0.

    Returns
    -------
    vmin, vmax : numeric
        The lower and upper values (respectively) of the middle 
        percentile.

    '''
    nisarqa.verify_valid_percentile(middle_percentile)

    fraction = 0.5*(1.0 - middle_percentile/100.0)

    # Get the value of the e.g. 0.025 quantile and the 0.975 quantile
    vmin, vmax = np.quantile(data_in, [fraction, 1-fraction])

    return vmin, vmax


def process_power_and_phase_histograms(pols, params, stats_h5, report_pdf):
    '''
    Generate the RSLC Power Histograms and save their plots
    to the graphical summary .pdf file.

    Power histogram will be computed in decibel units.
    Phase histogram defaults to being computed in radians, 
    configurable to be computed in degrees by setting
    `params.phs_in_radians` to False.

    Parameters
    ----------
    pols : nested dict of RSLCRaster
        Nested dict of RSLCRaster objects, where each object represents
        a polarization dataset in `h5_file`.
        Format: pols[<band>][<freq>][<pol>] -> a RSLCRaster
        Ex: pols['LSAR']['A']['HH'] -> the HH dataset, stored 
                                       in a RSLCRaster object
    params : RSLCHistogramParams
        A structure containing the parameters for processing
        and outputting the power and phase histograms.
    stats_h5 : h5py.File
        The output file to save QA metrics, etc. to
    report_pdf : PdfPages
        The output pdf file to append the power image plot to
    '''

    # Generate and store the histograms
    for band in pols:
        for freq in pols[band]:
            generate_histogram_single_freq(
                pol=pols[band][freq],
                band=band,
                freq=freq,
                params=params,
                stats_h5=stats_h5,
                report_pdf=report_pdf)


def generate_histogram_single_freq(pol, band, freq, 
                                    params, stats_h5, report_pdf):
    '''
    Generate the RSLC Power Histograms for a single frequency.
    
    The histograms' plots will be appended to the graphical
    summary file `report_pdf`, and their data will be
    stored in the statistics .h5 file `stats_h5`.
    Power histogram will be computed in decibel units.
    Phase histogram defaults to being computed in radians, 
    configurable to be computed in degrees.

    Parameters
    ----------
    pol : dict of RSLCRaster
        dict of RSLCRaster objects for the given `band`
        and `freq`. Each key is a polarization (e.g. 'HH'
        or 'HV'), and each key's item is the corresponding
        RSLCRaster instance.
        Ex: pol['HH'] -> the HH dataset, stored 
                         in a RSLCRaster object
    band : str
        Band name for the histograms to be processed,
        e.g. 'LSAR'
    freq : str
        Frequency name for the histograms to be processed,
        e.g. 'A' or 'B'
    params : RSLCHistogramParams
        A structure containing the parameters for processing
        and outputting the power and phase histograms.
    stats_h5 : h5py.File
        The output file to save QA metrics, etc. to
    report_pdf : PdfPages
        The output pdf file to append the power image plot to
    '''

    print(f'Generating Histograms for Frequency {freq}...')

    # Open separate figures for each of the power and phase histograms.
    # Each band+frequency will have a distinct plot, with all of the
    # polarization images for that setup plotted together on the same plot.
    pow_fig, pow_ax = plt.subplots(nrows=1, ncols=1)
    phs_fig, phs_ax = plt.subplots(nrows=1, ncols=1)

    # Use custom cycler for accessibility
    pow_ax.set_prop_cycle(nisarqa.CUSTOM_CYCLER)
    phs_ax.set_prop_cycle(nisarqa.CUSTOM_CYCLER)

    for pol_name, pol_data in pol.items():
        # Get histogram probability densities
        pow_hist_density, phs_hist_density = \
                nisarqa.compute_power_and_phase_histograms_by_tiling(
                            arr=pol_data.data,
                            pow_bin_edges=params.pow_bin_edges.val,
                            phs_bin_edges=params.phs_bin_edges.val,
                            phs_in_radians=params.phs_in_radians.val,
                            decimation_ratio=params.decimation_ratio.val,
                            tile_shape=params.tile_shape.val,
                            density=True)

        # Save to stats.h5 file
        grp_path = f'/science/{band}/QA/data/frequency{freq}/{pol_name}/'
        pow_units = params.pow_bin_edges.units
        phs_units = params.phs_bin_edges.units

        nisarqa.create_dataset_in_h5group(
            h5_file=stats_h5,
            grp_path=grp_path,
            ds_name='powerHistogramDensity',
            ds_data=pow_hist_density,
            ds_units=f'1/{pow_units}',
            ds_description='Normalized density of the power histogram')

        nisarqa.create_dataset_in_h5group(
            h5_file=stats_h5,
            grp_path=grp_path,
            ds_name='phaseHistogramDensity',
            ds_data=phs_hist_density,
            ds_units=f'1/{phs_units}',
            ds_description='Normalized density of the phase histogram')

        # Add these densities to the figures
        add_hist_to_axis(pow_ax,
                         counts=pow_hist_density, 
                         edges=params.pow_bin_edges.val,
                         label=pol_name)

        add_hist_to_axis(phs_ax,
                         counts=phs_hist_density,
                         edges=params.phs_bin_edges.val,
                         label=pol_name)

    # Label the Power Figure
    title = f'{band} Frequency {freq} Power Histograms'
    pow_ax.set_title(title)

    pow_ax.legend(loc='upper right')
    pow_ax.set_xlabel(f'RSLC Power ({pow_units})')
    pow_ax.set_ylabel(f'Density (1/{pow_units})')

    # Per ADT, let the top limit float for Power Spectra
    pow_ax.set_ylim(bottom=0.0)
    pow_ax.grid()

    # Label the Phase Figure
    phs_ax.set_title(f'{band} Frequency {freq} Phase Histograms')
    phs_ax.legend(loc='upper right')
    phs_ax.set_xlabel(f'RSLC Phase ({phs_units})')
    phs_ax.set_ylabel(f'Density (1/{phs_units})')
    if params.phs_in_radians.val:
        phs_ax.set_ylim(bottom=0.0, top=0.5)
    else:
        phs_ax.set_ylim(bottom=0.0, top=0.01)
    phs_ax.grid()

    # Save complete plots to graphical summary pdf file
    report_pdf.savefig(pow_fig)
    report_pdf.savefig(phs_fig)

    # Close all figures
    plt.close(pow_fig)
    plt.close(phs_fig)

    print(f'Histograms for Frequency {freq} complete.')


def add_hist_to_axis(axis, counts, edges, label):
    '''Add the plot of the given counts and edges to the
    axis object. Points will be centered in each bin,
    and the plot will be denoted `label` in the legend.
    '''
    bin_centers = 0.5 * (edges[:-1] + edges[1:])
    axis.plot(bin_centers, counts, label=label)


__all__ = nisarqa.get_all(__name__, objects_to_skip)
