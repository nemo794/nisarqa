import os

import h5py
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from PIL import Image

LOG_TXT = '''PLACEHOLDER FILE ONLY -- NOT ACTUAL LOG OUTPUTS
2022-05-02 18:51:40,466, INFO, QA, misc, 100000 '/projects/QualityAssurance/verify_rslc.py':73, "N/A: Successfully parsed XML file /projects/QualityAssurance/xml/nisar_L1_RSLC.xml"
2022-05-02 18:51:40,469, INFO, QA, misc, 100000 '/projects/QualityAssurance/verify_rslc.py':79, "N/A: Opening file /home/niemoell/dat/fromJoanne_05022022/rslc_REE_testarea134/output_rslc/rslc.h5 with xml spec /projects/QualityAssurance/xml/nisar_L1_RSLC.xml"
2022-05-02 18:51:40,472, INFO, QA, misc, 100000 '/projects/QualityAssurance/quality/NISARFile.py':37, "N/A: Opening file /home/niemoell/dat/fromJoanne_05022022/rslc_REE_testarea134/output_rslc/rslc.h5"
2022-05-02 18:51:40,479, INFO, QA, misc, 100000 '/projects/QualityAssurance/quality/NISARFile.py':65, "N/A: LSAR Start time 2023-07-01T00:08:13.501873000"
2022-05-02 18:51:40,482, INFO, QA, misc, 100000 '/projects/QualityAssurance/quality/NISARFile.py':65, "N/A: SSAR Start time 9999-99-99T99:99:99"
2022-05-02 18:51:40,485, INFO, QA, misc, 100000 '/projects/QualityAssurance/quality/SLCFile.py':67, "N/A: Found band LSAR"
2022-05-02 18:51:40,488, INFO, QA, misc, 100000 '/projects/QualityAssurance/quality/HdfGroup.py':23, "N/A: Initializing HDF Group LSAR Swath with /science/LSAR/RSLC/swaths"
2022-05-02 18:51:40,491, INFO, QA, misc, 100000 '/projects/QualityAssurance/quality/HdfGroup.py':23, "N/A: Initializing HDF Group LSAR Metadata with /science/LSAR/RSLC/metadata"
2022-05-02 18:51:40,494, INFO, QA, misc, 100000 '/projects/QualityAssurance/quality/HdfGroup.py':23, "N/A: Initializing HDF Group LSAR Identification with /science/LSAR/identification/"
2022-05-02 18:51:40,497, INFO, QA, misc, 100000 '/projects/QualityAssurance/quality/SLCFile.py':64, "N/A: SSAR not present"
2022-05-02 18:51:40,500, INFO, QA, misc, 100000 '/projects/QualityAssurance/quality/NISARFile.py':84, "N/A: Found LSAR FrequencyA"
2022-05-02 18:51:40,504, INFO, QA, misc, 100000 '/projects/QualityAssurance/quality/NISARFile.py':84, "N/A: Found LSAR FrequencyB"
2022-05-02 18:51:40,509, INFO, QA, misc, 100000 '/projects/QualityAssurance/quality/NISARFile.py':178, "N/A: Checking polarization  in <class 'h5py._hl.group.Group'>"
2022-05-02 18:51:40,513, INFO, QA, misc, 100000 '/projects/QualityAssurance/quality/NISARFile.py':178, "N/A: Checking polarization  in <class 'h5py._hl.group.Group'>"
2022-05-02 18:51:40,518, DEBUG, QA, misc, 101000 '/projects/QualityAssurance/quality/NISARFile.py':298, "N/A: nsubswaths <class 'int'> = 1"
2022-05-02 18:51:40,522, DEBUG, QA, misc, 101000 '/projects/QualityAssurance/quality/NISARFile.py':298, "N/A: nsubswaths <class 'int'> = 1"
2022-05-02 18:51:40,525, INFO, QA, misc, 100000 '/projects/QualityAssurance/quality/NISARFile.py':311, "N/A: LSAR: no_look=['frequencyA/validSamplesSubSwath2', 'frequencyA/validSamplesSubSwath3', 'frequencyA/validSamplesSubSwath4', 'frequencyA/HHHH', 'frequencyA/HHHV', 'frequencyA/HHVH', 'frequencyA/HHVV', 'frequencyA/HVHH', 'frequencyA/HVHV', 'frequencyA/HVVH', 'frequencyA/HVVV', 'frequencyA/VHHH', 'frequencyA/VHHV', 'frequencyA/VHVH', 'frequencyA/VHVV', 'frequencyA/VVHH', 'frequencyA/VVHV', 'frequencyA/VVVH', 'frequencyA/VVVV', 'frequencyA/RHRH', 'frequencyA/RHRV', 'frequencyA/RVRH', 'frequencyA/RVRV', 'frequencyA/VH', 'frequencyA/VV', 'frequencyA/RH', 'frequencyA/RV', 'frequencyB/validSamplesSubSwath2', 'frequencyB/validSamplesSubSwath3', 'frequencyB/validSamplesSubSwath4', 'frequencyB/HHHH', 'frequencyB/HHHV', 'frequencyB/HHVH', 'frequencyB/HHVV', 'frequencyB/HVHH', 'frequencyB/HVHV', 'frequencyB/HVVH', 'frequencyB/HVVV', 'frequencyB/VHHH', 'frequencyB/VHHV', 'frequencyB/VHVH', 'frequencyB/VHVV', 'frequencyB/VVHH', 'frequencyB/VVHV', 'frequencyB/VVVH', 'frequencyB/VVVV', 'frequencyB/RHRH', 'frequencyB/RHRV', 'frequencyB/RVRH', 'frequencyB/RVRV', 'frequencyB/VH', 'frequencyB/VV', 'frequencyB/RH', 'frequencyB/RV']"
2022-05-02 18:51:40,542, WARNING, QA, misc, 102000 '/projects/QualityAssurance/quality/NISARFile.py':327, "Warning: LSAR Metadata missing 1 fields: processingInformation/parameters/azimuthChirpWeighting"
2022-05-02 18:51:40,554, WARNING, QA, misc, 102000 '/projects/QualityAssurance/quality/NISARFile.py':384, "Warning: Invalid Orbit Number: 0"
2022-05-02 18:51:40,560, INFO, QA, misc, 100000 '/projects/QualityAssurance/quality/NISARFile.py':544, "N/A: Checking frequencies for flist: {'A': <HDF5 group "/science/LSAR/RSLC/swaths/frequencyA" (15 members)>, 'B': <HDF5 group "/science/LSAR/RSLC/swaths/frequencyB" (15 members)>}"
2022-05-02 18:51:40,567, INFO, QA, misc, 100000 '/projects/QualityAssurance/quality/NISARFile.py':485, "N/A: Start time b'2023-07-01T00:08:13.501873000', End time b'2023-07-01T00:08:21.501001000'"
2022-05-02 19:00:52,490, INFO, QA, misc, 100000 '/projects/QualityAssurance/quality/NISARFile.py':445, "N/A: Checking image LSAR A HH for zeros and NaNs"
2022-05-02 19:00:52,496, WARNING, QA, misc, 102000 '/projects/QualityAssurance/quality/NISARFile.py':462, "Warning: LSAR A_HH has 348233 Zeros=0.1%"
2022-05-02 19:01:05,199, INFO, QA, misc, 100000 '/projects/QualityAssurance/quality/NISARFile.py':445, "N/A: Checking image LSAR A HV for zeros and NaNs"
2022-05-02 19:01:05,205, WARNING, QA, misc, 102000 '/projects/QualityAssurance/quality/NISARFile.py':462, "Warning: LSAR A_HV has 349618 Zeros=0.1%"
2022-05-02 19:01:08,444, INFO, QA, misc, 100000 '/projects/QualityAssurance/quality/NISARFile.py':445, "N/A: Checking image LSAR B HH for zeros and NaNs"
2022-05-02 19:01:08,450, WARNING, QA, misc, 102000 '/projects/QualityAssurance/quality/NISARFile.py':462, "Warning: LSAR B_HH has 2131840 Zeros=2.2%"
2022-05-02 19:01:11,668, INFO, QA, misc, 100000 '/projects/QualityAssurance/quality/NISARFile.py':445, "N/A: Checking image LSAR B HV for zeros and NaNs"
2022-05-02 19:01:11,674, WARNING, QA, misc, 102000 '/projects/QualityAssurance/quality/NISARFile.py':462, "Warning: LSAR B_HV has 2134675 Zeros=2.2%"
2022-05-02 19:01:11,849, INFO, QA, misc, 100000 '/projects/QualityAssurance/quality/SLCFile.py':197, "N/A: Found 4 images: dict_keys(['LSAR A HH', 'LSAR A HV', 'LSAR B HH', 'LSAR B HV'])"
2022-05-02 19:03:42,604, INFO, QA, misc, 100000 '/projects/QualityAssurance/quality/SLCFile.py':246, "N/A: Looking at 0-th image: LSAR A HH"
2022-05-02 19:03:53,893, INFO, QA, misc, 100000 '/projects/QualityAssurance/quality/SLCFile.py':246, "N/A: Looking at 1-th image: LSAR A HV"
2022-05-02 19:04:04,618, INFO, QA, misc, 100000 '/projects/QualityAssurance/quality/SLCFile.py':246, "N/A: Looking at 2-th image: LSAR B HH"
2022-05-02 19:04:07,374, INFO, QA, misc, 100000 '/projects/QualityAssurance/quality/SLCFile.py':246, "N/A: Looking at 3-th image: LSAR B HV"
2022-05-02 19:04:10,059, DEBUG, QA, misc, 101000 '/projects/QualityAssurance/quality/SLCFile.py':266, "N/A: bounds_linear 0.2"
2022-05-02 19:09:44,166, INFO, QA, misc, 100000 '/projects/QualityAssurance/quality/SLCFile.py':376, "N/A: File stats.h5 mode r+"
2022-05-02 19:09:45,065, INFO, QA, misc, 100000 '/projects/QualityAssurance/verify_rslc.py':143, "N/A: Runtime = 1084 seconds"
2022-05-02 19:24:21,158, INFO, QA, misc, 100000 '/projects/QualityAssurance/verify_rslc.py':73, "N/A: Successfully parsed XML file /projects/QualityAssurance/xml/nisar_L1_RSLC.xml"
2022-05-02 19:24:21,161, INFO, QA, misc, 100000 '/projects/QualityAssurance/verify_rslc.py':79, "N/A: Opening file /home/niemoell/dat/fromJoanne_05022022/rslc_REE_testarea134/output_rslc/rslc.h5 with xml spec /projects/QualityAssurance/xml/nisar_L1_RSLC.xml"
2022-05-02 19:24:21,165, INFO, QA, misc, 100000 '/projects/QualityAssurance/quality/NISARFile.py':37, "N/A: Opening file /home/niemoell/dat/fromJoanne_05022022/rslc_REE_testarea134/output_rslc/rslc.h5"
2022-05-02 19:24:21,172, INFO, QA, misc, 100000 '/projects/QualityAssurance/quality/NISARFile.py':65, "N/A: LSAR Start time 2023-07-01T00:08:13.501873000"
2022-05-02 19:24:21,174, INFO, QA, misc, 100000 '/projects/QualityAssurance/quality/NISARFile.py':65, "N/A: SSAR Start time 9999-99-99T99:99:99"
2022-05-02 19:24:21,177, INFO, QA, misc, 100000 '/projects/QualityAssurance/quality/SLCFile.py':67, "N/A: Found band LSAR"
2022-05-02 19:24:21,180, INFO, QA, misc, 100000 '/projects/QualityAssurance/quality/HdfGroup.py':23, "N/A: Initializing HDF Group LSAR Swath with /science/LSAR/RSLC/swaths"
2022-05-02 19:24:21,183, INFO, QA, misc, 100000 '/projects/QualityAssurance/quality/HdfGroup.py':23, "N/A: Initializing HDF Group LSAR Metadata with /science/LSAR/RSLC/metadata"
2022-05-02 19:24:21,186, INFO, QA, misc, 100000 '/projects/QualityAssurance/quality/HdfGroup.py':23, "N/A: Initializing HDF Group LSAR Identification with /science/LSAR/identification/"
2022-05-02 19:24:21,189, INFO, QA, misc, 100000 '/projects/QualityAssurance/quality/SLCFile.py':64, "N/A: SSAR not present"
2022-05-02 19:24:21,192, INFO, QA, misc, 100000 '/projects/QualityAssurance/quality/NISARFile.py':84, "N/A: Found LSAR FrequencyA"
2022-05-02 19:24:21,196, INFO, QA, misc, 100000 '/projects/QualityAssurance/quality/NISARFile.py':84, "N/A: Found LSAR FrequencyB"
2022-05-02 19:24:21,202, INFO, QA, misc, 100000 '/projects/QualityAssurance/quality/NISARFile.py':178, "N/A: Checking polarization  in <class 'h5py._hl.group.Group'>"
2022-05-02 19:24:21,205, INFO, QA, misc, 100000 '/projects/QualityAssurance/quality/NISARFile.py':178, "N/A: Checking polarization  in <class 'h5py._hl.group.Group'>"
2022-05-02 19:24:21,211, DEBUG, QA, misc, 101000 '/projects/QualityAssurance/quality/NISARFile.py':298, "N/A: nsubswaths <class 'int'> = 1"
2022-05-02 19:24:21,215, DEBUG, QA, misc, 101000 '/projects/QualityAssurance/quality/NISARFile.py':298, "N/A: nsubswaths <class 'int'> = 1"
2022-05-02 19:24:21,218, INFO, QA, misc, 100000 '/projects/QualityAssurance/quality/NISARFile.py':311, "N/A: LSAR: no_look=['frequencyA/validSamplesSubSwath2', 'frequencyA/validSamplesSubSwath3', 'frequencyA/validSamplesSubSwath4', 'frequencyA/HHHH', 'frequencyA/HHHV', 'frequencyA/HHVH', 'frequencyA/HHVV', 'frequencyA/HVHH', 'frequencyA/HVHV', 'frequencyA/HVVH', 'frequencyA/HVVV', 'frequencyA/VHHH', 'frequencyA/VHHV', 'frequencyA/VHVH', 'frequencyA/VHVV', 'frequencyA/VVHH', 'frequencyA/VVHV', 'frequencyA/VVVH', 'frequencyA/VVVV', 'frequencyA/RHRH', 'frequencyA/RHRV', 'frequencyA/RVRH', 'frequencyA/RVRV', 'frequencyA/VH', 'frequencyA/VV', 'frequencyA/RH', 'frequencyA/RV', 'frequencyB/validSamplesSubSwath2', 'frequencyB/validSamplesSubSwath3', 'frequencyB/validSamplesSubSwath4', 'frequencyB/HHHH', 'frequencyB/HHHV', 'frequencyB/HHVH', 'frequencyB/HHVV', 'frequencyB/HVHH', 'frequencyB/HVHV', 'frequencyB/HVVH', 'frequencyB/HVVV', 'frequencyB/VHHH', 'frequencyB/VHHV', 'frequencyB/VHVH', 'frequencyB/VHVV', 'frequencyB/VVHH', 'frequencyB/VVHV', 'frequencyB/VVVH', 'frequencyB/VVVV', 'frequencyB/RHRH', 'frequencyB/RHRV', 'frequencyB/RVRH', 'frequencyB/RVRV', 'frequencyB/VH', 'frequencyB/VV', 'frequencyB/RH', 'frequencyB/RV']"
2022-05-02 19:24:21,236, WARNING, QA, misc, 102000 '/projects/QualityAssurance/quality/NISARFile.py':327, "Warning: LSAR Metadata missing 1 fields: processingInformation/parameters/azimuthChirpWeighting"
2022-05-02 19:24:21,249, WARNING, QA, misc, 102000 '/projects/QualityAssurance/quality/NISARFile.py':384, "Warning: Invalid Orbit Number: 0"
2022-05-02 19:24:21,253, INFO, QA, misc, 100000 '/projects/QualityAssurance/quality/NISARFile.py':544, "N/A: Checking frequencies for flist: {'A': <HDF5 group "/science/LSAR/RSLC/swaths/frequencyA" (15 members)>, 'B': <HDF5 group "/science/LSAR/RSLC/swaths/frequencyB" (15 members)>}"
2022-05-02 19:24:21,261, INFO, QA, misc, 100000 '/projects/QualityAssurance/quality/NISARFile.py':485, "N/A: Start time b'2023-07-01T00:08:13.501873000', End time b'2023-07-01T00:08:21.501001000'"
2022-05-02 19:33:28,390, INFO, QA, misc, 100000 '/projects/QualityAssurance/quality/NISARFile.py':445, "N/A: Checking image LSAR A HH for zeros and NaNs"
2022-05-02 19:33:28,396, WARNING, QA, misc, 102000 '/projects/QualityAssurance/quality/NISARFile.py':462, "Warning: LSAR A_HH has 348233 Zeros=0.1%"
2022-05-02 19:33:41,694, INFO, QA, misc, 100000 '/projects/QualityAssurance/quality/NISARFile.py':445, "N/A: Checking image LSAR A HV for zeros and NaNs"
2022-05-02 19:33:41,700, WARNING, QA, misc, 102000 '/projects/QualityAssurance/quality/NISARFile.py':462, "Warning: LSAR A_HV has 349618 Zeros=0.1%"
2022-05-02 19:33:45,085, INFO, QA, misc, 100000 '/projects/QualityAssurance/quality/NISARFile.py':445, "N/A: Checking image LSAR B HH for zeros and NaNs"
2022-05-02 19:33:45,091, WARNING, QA, misc, 102000 '/projects/QualityAssurance/quality/NISARFile.py':462, "Warning: LSAR B_HH has 2131840 Zeros=2.2%"
2022-05-02 19:33:48,479, INFO, QA, misc, 100000 '/projects/QualityAssurance/quality/NISARFile.py':445, "N/A: Checking image LSAR B HV for zeros and NaNs"
2022-05-02 19:33:48,485, WARNING, QA, misc, 102000 '/projects/QualityAssurance/quality/NISARFile.py':462, "Warning: LSAR B_HV has 2134675 Zeros=2.2%"
2022-05-02 19:33:48,660, INFO, QA, misc, 100000 '/projects/QualityAssurance/quality/SLCFile.py':197, "N/A: Found 4 images: dict_keys(['LSAR A HH', 'LSAR A HV', 'LSAR B HH', 'LSAR B HV'])"
2022-05-02 19:36:18,204, INFO, QA, misc, 100000 '/projects/QualityAssurance/quality/SLCFile.py':246, "N/A: Looking at 0-th image: LSAR A HH"
2022-05-02 19:36:29,492, INFO, QA, misc, 100000 '/projects/QualityAssurance/quality/SLCFile.py':246, "N/A: Looking at 1-th image: LSAR A HV"
2022-05-02 19:36:40,237, INFO, QA, misc, 100000 '/projects/QualityAssurance/quality/SLCFile.py':246, "N/A: Looking at 2-th image: LSAR B HH"
2022-05-02 19:36:43,002, INFO, QA, misc, 100000 '/projects/QualityAssurance/quality/SLCFile.py':246, "N/A: Looking at 3-th image: LSAR B HV"
2022-05-02 19:36:45,699, DEBUG, QA, misc, 101000 '/projects/QualityAssurance/quality/SLCFile.py':266, "N/A: bounds_linear 0.2"
2022-05-02 19:42:09,117, INFO, QA, misc, 100000 '/projects/QualityAssurance/quality/SLCFile.py':376, "N/A: File stats.h5 mode r+"
2022-05-02 19:42:09,972, INFO, QA, misc, 100000 '/projects/QualityAssurance/verify_rslc.py':143, "N/A: Runtime = 1068 seconds"
PLACEHOLDER FILE ONLY -- NOT ACTUAL LOG OUTPUTS
'''

SUMMARY_CSV = '''Tool,Check Description,Result,Threshold,Actual,Notes
QA,Able to open NISAR input file?,PASS,,,PLACEHOLDER
QA,Input file validation successful?,FAIL,,,PLACEHOLDER Invalid Start/End Time(s).
QA,0 missing fields per product spec?,FAIL,0,4,PLACEHOLDER Missing: "/science/LSAR/RSLC/swaths/frequencyA/validSamplesSubSwath3".
QA,Only LSAR and/or SSAR band found?,PASS,,,PLACEHOLDER Found: LSAR.
QA,Only Frequency A and/or B found?,PASS,,,PLACEHOLDER Found: A.
QA,> 0 valid images found?,PASS,1,2,PLACEHOLDERFound: LSAR_A_HH. LSAR_A_HV.
QA,Contains 4 subswath bounds for FreqA?,FAIL,4,3,PLACEHOLDER		
QA,Contains 4 subswath bounds for FreqB?,PASS,n/a,n/a,PLACEHOLDER FreqB not present in file.
QA,% invalid pixels < 80%?,PASS,80%,1.7%,PLACEHOLDER # invalid pixels: 250/15000.
QA,% zero pixels < 98%?,FAIL,98%,93.9%,PLACEHOLDER # zero pixels: 14090/15000.
'''

KML_FILE = '''<?xml version="1.0" encoding="UTF-8"?>
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
'''

def output_stub_files(output_dir, stub_files='all', input_file=None):
    '''This function outputs stub files for the NISAR QA L1/L2 products.
    
    Parameters
    ----------
    output_dir : str
        Filepath for the output directory to place output stub files
    file_type : str, List of str, optional
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
    input_file : str, optional
        The input NISAR product HDF5 file. Only required/used for saving
        the stub stats_h5 file.

    '''
    ## Validate inputs
    opts = ['browse_png', 'browse_kml', 'summary_csv', 'log_txt', 
            'stats_h5', 'report_pdf']
    if stub_files == 'all':
        stub_files = opts

    if isinstance(stub_files, str):
        stub_files = [stub_files]

    # ensure that the inputs are a subset of the valid options
    assert set(stub_files) <= set(opts), 'invalid input for argument `stub_files`'

    if 'stats_h5' in stub_files:
        assert input_file is not None, 'to generate a stub STATS.h5, a valid NISAR product input file must be provided.'
        assert os.path.isfile(input_file), f'`input_file` is not a valid file: {input_file}'
        assert input_file.endswith('.h5'), f'`input_file` must have the extension .h5: {input_file}'

    # If output directory does not exist, make it.
    os.makedirs(output_dir, exist_ok=True)

    ## Save stub files
    # Save geolocation stub file
    if 'browse_kml' in stub_files:
        with open(os.path.join(output_dir, 'BROWSE.kml'), 'w') as f:
            f.write(KML_FILE)

    # Save summary.csv stub file
    if 'summary_csv' in stub_files:
        with open(os.path.join(output_dir, 'SUMMARY.csv'), 'w') as f:
            f.write(SUMMARY_CSV)

    # Save Log file stub file
    if 'log_txt' in stub_files:
        with open(os.path.join(output_dir, 'LOG.txt'), 'w') as f:
            f.write(LOG_TXT)

    # Save stats.h5 stub file
    if 'stats_h5' in stub_files:
        # TODO use context manager
        nisar_h5 = h5py.File(input_file, 'r')
        stats_h5_file = h5py.File(os.path.join(output_dir, 'STATS.h5'), "w")

        grp_path = os.path.join('/science/LSAR/identification')

        # Copy identification metadata from input file to stats.h5
        nisar_h5.copy(nisar_h5[f'/science/LSAR/identification'],
                      stats_h5_file, grp_path)

        # Save filename for this input NISAR product
        grp = stats_h5_file.require_group(grp_path)
        ds = grp.create_dataset('NISARProductFilename', 
                                data=os.path.basename(nisar_h5.filename))
        ds.attrs.create(name='description',
                        data='Input NISAR product filename',
                        dtype=f'<S{len("Input NISAR product filename")}')

        nisar_h5.close()
        stats_h5_file.close()

    # Save browse image stub file and pdf stub file
    # Create a roughly 2048x2048 pixels^2 RGB image
    # (ASF allows for in-exact dimensions, so let's test that.)
    # (The current plan is for all NISAR products to generate RGB browse images)
    imarray = np.random.randint(low=0, high=256, 
                                size=(1800,2000,4), dtype=np.uint8)

    # Make all pixels opaque by setting the alpha channel to 255
    imarray[:,:,3] = 255

    # Make a subset of the pixels transparent by setting alpha channel to 0
    imarray[500:900,500:900,3] = 0

    if 'browse_png' in stub_files:
        im = Image.fromarray(imarray).convert('RGBA')
        datas = im.getdata()
        newData = []
        for item in datas:
            newData.append(item)
        im.putdata(newData)
        im.save(os.path.join(output_dir, 'BROWSE.png'))

    if 'report_pdf' in stub_files:
        # Save image into a .pdf
        with PdfPages(os.path.join(output_dir, 'REPORT.pdf')) as f:
            # Instantiate the figure object
            fig = plt.figure()
            ax = plt.gca()

            # Plot the img_arr image.
            ax_img = ax.imshow(X=imarray, cmap=plt.cm.ocean)
            plt.colorbar(ax_img, ax=ax)

            plt.xlabel('Placeholder x-axis label')
            plt.ylabel('Placeholder y-axis label')
            plt.title('PLACEHOLDER IMAGE - NOT REPRESENTATIVE OF ACTUAL NISAR PRODUCT')

            # Make sure axes labels do not get cut off
            fig.tight_layout()

            # Append figure to the output .pdf
            f.savefig(fig)

            # Close the figure
            plt.close(fig)

# Manually create the __all__ attribute.
__all__ = ['output_stub_files']
