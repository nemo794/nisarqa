import h5py
import numpy
from scipy import constants

import os, os.path
import shutil
import time

TEST_DIR_IN = "test_data_inputs"
TEST_DIR_OUT = "test_data"
GSLC_FILE = "GSLC_lowRes.h5"
RSLC_FILE = "rslc_ree1.h5"
RSLC_FILE2 = "SanAnd_05024_18038_006_180730_L090_CX_129_03_CFloat16.h5"
GCOV_FILE = "SanAnd_05024_18038_006_180730_L090_CX_129_05_L2GCOV.h5"
GUNW_FILE = "GUNW_SanAndreas_utm_reduced.h5"

complex32 = numpy.dtype([("real", numpy.float16), ("imag", numpy.float16)])

def fix_rslc(hfile):

    time1 = numpy.array(["2020-04-30T00:00:00:000001"], dtype="S26")
    time2 = numpy.array(["2020-04-30T23:59:59:000001"], dtype="S26")
    product = numpy.array(["RSLC"], dtype="S4")
    cycle = numpy.array([1], dtype=numpy.int64)

    id = hfile["/science/LSAR/identification"]
    id["absoluteOrbitNumber"][...] = 100
    del id["zeroDopplerStartTime"]
    del id["zeroDopplerEndTime"]
    del id["productType"]
    id.create_dataset("zeroDopplerStartTime", data=time1)
    id.create_dataset("zeroDopplerEndTime", data=time2)
    id.create_dataset("productType", data=product)
    id.create_dataset("cycleNumber", data=cycle)

def repack_rslc(hfile_in, hfile_out, frequency, polarization, dname):

    id_out = hfile_out.create_group("/science/LSAR/")
    meta_out = hfile_out.create_group("/science/LSAR/%s" % dname)
    freq_out = hfile_out.create_group("/science/LSAR/%s/swaths/frequency%s" % (dname, frequency))

    swaths_out = hfile_out["/science/LSAR/%s/swaths" % dname]
    hfile_in.copy("/science/LSAR/SLC/metadata", meta_out)
    hfile_in.copy("/science/LSAR/identification", id_out)
    hfile_in.copy("/science/LSAR/%s/swaths/zeroDopplerTime" % dname, swaths_out)
    hfile_in.copy("/science/LSAR/%s/swaths/zeroDopplerTimeSpacing" % dname, swaths_out)

    for key in hfile_in["/science/LSAR/%s/swaths/frequency%s" % (dname, frequency)].keys():
        if (key != key.upper()):
            hfile_in.copy("/science/LSAR/%s/swaths/frequency%s/%s" % (dname, frequency, key), freq_out)

    del hfile_out["/science/LSAR/identification/listOfFrequencies"]
    del hfile_out["/science/LSAR/%s/swaths/frequency%s/listOfPolarizations" % (dname, frequency)]
    hfile_out.create_dataset("/science/LSAR/identification/listOfFrequencies", \
                             data=numpy.array(["%s" % frequency], dtype="S1"))
    hfile_out.create_dataset("/science/LSAR/%s/swaths/frequency%s/listOfPolarizations" % (dname, frequency), \
                             data=numpy.array(["%s" % polarization], dtype="S2"))
    
            
    return
    
    for f in flist:
        if (f != frequency):
            del hfile["/science/LSAR/%s/swaths/frequency%s" % (dname, f)]
        else:
            plist = hfile["/science/LSAR/%s/swaths/frequency%s/listOfPolarizations" % (dname, f)][...]
            plist = [p.decode("utf-8") for p in plist]
            assert(polarization in plist)

            for p in plist:
                if (p != polarization):
                    del hfile["/science/LSAR/%s/swaths/frequency%s/%s" % (dname, f, p)]

def resize_rslc(hfile, frequency, polarization, dname, nlines, nsmp):

    data_name = "/science/LSAR/%s/swaths/frequency%s/%s" % (dname, frequency, polarization)
    time_name = "/science/LSAR/%s/swaths/zeroDopplerTime" % dname
    slant_name = "/science/LSAR/%s/swaths/frequency%s/slantRange" % (dname, frequency)

    hdata = hfile[data_name]
    with hdata.astype(numpy.complex64):
        data_old = hdata[...]
    assert( (nlines < data_old.shape[0]) & (nsmp < data_old.shape[1]) )
    data_new = data_old[0:nlines, 0:nsmp]

    del hfile[data_name]
    hfile.create_dataset(data_name, data=data_new)

    time_old = hfile[time_name][...]
    time_new = time_old[0:nlines]
    del hfile[time_name]
    hfile.create_dataset(time_name, data=time_new)

    slant_old = hfile[slant_name][...]
    slant_new = slant_old[0:nlines]
    del hfile[slant_name]
    hfile.create_dataset(slant_name, data=slant_new)
                    
def fix_gcov(hfile):

    polA = numpy.array(["HHHH", "HVHV"], dtype="S4")
    polB = numpy.array(["HHHH", "HVHV"], dtype="S4")

    del hfile["/science/LSAR/GCOV/grids/frequencyA/listOfPolarizations"]
    del hfile["/science/LSAR/GCOV/grids/frequencyB/listOfPolarizations"]

    hfile.create_dataset("/science/LSAR/GCOV/grids/frequencyA/listOfPolarizations", data=polA)
    hfile.create_dataset("/science/LSAR/GCOV/grids/frequencyB/listOfPolarizations", data=polB)    
    
def remove_zeroes_complex(data):

    mask = numpy.where( (data.real == 0.0) & (data.imag == 0.0), True, False)
    data.real[mask] = 1.0
    data.imag[mask] = 1.0

    return data

def remove_nans_real(hfile):

    for f in ("A", "B"):
        for p in ("HHHH", "HVHV"):
            data = hfile["/science/LSAR/GCOV/grids/frequency%s/%s" % (f, p)][...]
            mask = numpy.isnan(data)
            hfile["/science/LSAR/GCOV/grids/frequency%s/%s" % (f, p)][mask] = 1.0

def remove_nans_negatives_real(hfile):

    for f in ("A", "B"):
        for p in ("HHHH", "HVHV"):
            data = hfile["/science/LSAR/GCOV/grids/frequency%s/%s" % (f, p)][...]
            mask = numpy.isnan(data) | numpy.where(data < 0.0, True, False)
            hfile["/science/LSAR/GCOV/grids/frequency%s/%s" % (f, p)][mask] = 0.0

            

def write_field(hfile, dname, data_in, dtype=complex32):

    if (dtype is complex32):
        print("Writing c32 field %s" % dname)
        real = data_in.real.astype(numpy.float16)
        imag = data_in.imag.astype(numpy.float16)
        data_out = real + imag*1j
    else:
        data_out = data_in

    try:
        del hfile[dname]
    except KeyError:
        pass
    hfile.create_dataset(dname, data=data_out)
    
def gslc_wrong_size(file_in, file_out):

    shutil.copyfile(os.path.join(TEST_DIR_IN, file_in), os.path.join(TEST_DIR_OUT, file_out))

    hfile = h5py.File(os.path.join(TEST_DIR_OUT, file_out), "r+")

    ycoord = hfile["/science/LSAR/GSLC/grids/frequencyA/yCoordinates"][...]
    del hfile["/science/LSAR/GSLC/grids/frequencyA/yCoordinates"]
    hfile.create_dataset("/science/LSAR/GSLC/grids/frequencyA/yCoordinates", data=ycoord[0:ycoord.size-100])
    hfile.close()

def gslc_missing_band(file_in, file_out):

    shutil.copyfile(os.path.join(TEST_DIR_IN, file_in), os.path.join(TEST_DIR_OUT, file_out))

    hfile = h5py.File(os.path.join(TEST_DIR_OUT, file_out), "r+")

    del hfile["/science/LSAR/GSLC/metadata"]
    hfile.close()
    
def gslc_wrong_frequency1(file_in, file_out):

    shutil.copyfile(os.path.join(TEST_DIR_IN, file_in), os.path.join(TEST_DIR_OUT, file_out))

    hfile = h5py.File(os.path.join(TEST_DIR_OUT, file_out), "r+")

    flist = hfile["/science/LSAR/identification/listOfFrequencies"][...]
    flist[1] = "C"
    hfile["/science/LSAR/identification/listOfFrequencies"][...] = flist

    hfile.close()

def gslc_wrong_frequency2(file_in, file_out):

    shutil.copyfile(os.path.join(TEST_DIR_IN, file_in), os.path.join(TEST_DIR_OUT, file_out))

    hfile = h5py.File(os.path.join(TEST_DIR_OUT, file_out), "r+")

    del hfile["/science/LSAR/GSLC/grids/frequencyB"]
    hfile.close()

def gslc_wrong_frequency3(file_in, file_out):

    shutil.copyfile(os.path.join(TEST_DIR_IN, file_in), os.path.join(TEST_DIR_OUT, file_out))

    hfile = h5py.File(os.path.join(TEST_DIR_OUT, file_out), "r+")

    del hfile["/science/LSAR/identification/listOfFrequencies"]
    freq = numpy.array(["B"], dtype="S1")
    hfile.create_dataset("/science/LSAR/identification/listOfFrequencies", data=freq)
    hfile.close()
    
def gcov_wrong_frequency1(file_in, file_out):

    shutil.copyfile(os.path.join(TEST_DIR_IN, file_in), os.path.join(TEST_DIR_OUT, file_out))

    hfile = h5py.File(os.path.join(TEST_DIR_OUT, file_out), "r+")

    flist = hfile["/science/LSAR/identification/listOfFrequencies"][...]
    flist[1] = "D"
    hfile["/science/LSAR/identification/listOfFrequencies"][...] = flist

    hfile.close()

def gcov_wrong_frequency2(file_in, file_out):

    shutil.copyfile(os.path.join(TEST_DIR_IN, file_in), os.path.join(TEST_DIR_OUT, file_out))

    hfile = h5py.File(os.path.join(TEST_DIR_OUT, file_out), "r+")

    del hfile["/science/LSAR/GCOV/grids/frequencyA"]
    hfile.close()

def gcov_wrong_frequency3(file_in, file_out):

    shutil.copyfile(os.path.join(TEST_DIR_IN, file_in), os.path.join(TEST_DIR_OUT, file_out))

    hfile = h5py.File(os.path.join(TEST_DIR_OUT, file_out), "r+")

    del hfile["/science/LSAR/identification/listOfFrequencies"]
    freq = numpy.array(["A"], dtype="S1")
    hfile.create_dataset("/science/LSAR/identification/listOfFrequencies", data=freq)
    hfile.close()
        
def gslc_wrong_polarizations1(file_in, file_out):

    shutil.copyfile(os.path.join(TEST_DIR_IN, file_in), os.path.join(TEST_DIR_OUT, file_out))

    hfile = h5py.File(os.path.join(TEST_DIR_OUT, file_out), "r+")

    pol = hfile["/science/LSAR/GSLC/grids/frequencyA/listOfPolarizations"][...]
    pol[1] = "XY"
    hfile["/science/LSAR/GSLC/grids/frequencyA/listOfPolarizations"][...] = pol
    hfile.close()

def gslc_wrong_polarizations2(file_in, file_out):

    shutil.copyfile(os.path.join(TEST_DIR_IN, file_in), os.path.join(TEST_DIR_OUT, file_out))

    hfile = h5py.File(os.path.join(TEST_DIR_OUT, file_out), "r+")

    del hfile["/science/LSAR/GSLC/grids/frequencyA/HV"]
    hfile.close()

def gslc_wrong_polarizations3(file_in, file_out):

    shutil.copyfile(os.path.join(TEST_DIR_IN, file_in), os.path.join(TEST_DIR_OUT, file_out))

    hfile = h5py.File(os.path.join(TEST_DIR_OUT, file_out), "r+")

    pol = numpy.array(["VV"], dtype="S2")
    del hfile["/science/LSAR/GSLC/grids/frequencyB/listOfPolarizations"]
    hfile.create_dataset("/science/LSAR/GSLC/grids/frequencyB/listOfPolarizations", data=pol)
    hfile.close()
    
def gcov_wrong_polarizations1(file_in, file_out):

    shutil.copyfile(os.path.join(TEST_DIR_IN, file_in), os.path.join(TEST_DIR_OUT, file_out))

    hfile = h5py.File(os.path.join(TEST_DIR_OUT, file_out), "r+")

    del hfile["/science/LSAR/GCOV/grids/frequencyA/listOfPolarizations"]
    hfile.close()

def gcov_wrong_polarizations2(file_in, file_out):

    shutil.copyfile(os.path.join(TEST_DIR_IN, file_in), os.path.join(TEST_DIR_OUT, file_out))

    hfile = h5py.File(os.path.join(TEST_DIR_OUT, file_out), "r+")
    del hfile["/science/LSAR/GCOV/grids/frequencyA/listOfPolarizations"]
    pol = numpy.array(["HH"], dtype="S2")
    hfile.create_dataset("/science/LSAR/GCOV/grids/frequencyA/listOfPolarizations", data=pol)
    hfile.close()

def gcov_wrong_polarizations3(file_in, file_out):

    shutil.copyfile(os.path.join(TEST_DIR_IN, file_in), os.path.join(TEST_DIR_OUT, file_out))

    hfile = h5py.File(os.path.join(TEST_DIR_OUT, file_out), "r+")

    del hfile["/science/LSAR/GCOV/grids/frequencyA/listOfPolarizations"]
    pol = numpy.array(["HH", "HV", "VH"], dtype="S2")
    hfile.create_dataset("/science/LSAR/GCOV/grids/frequencyA/listOfPolarizations", data=pol)
    hfile.close()

def gcov_wrong_polarizations4(file_in, file_out):

    shutil.copyfile(os.path.join(TEST_DIR_IN, file_in), os.path.join(TEST_DIR_OUT, file_out))

    hfile = h5py.File(os.path.join(TEST_DIR_OUT, file_out), "r+")

    hfile.move("/science/LSAR/GCOV/grids/frequencyA/HHHH", "/science/LSAR/GCOV/grids/frequencyA/HHVVH")
    hfile.close()

def gcov_wrong_polarizations5(file_in, file_out):

    shutil.copyfile(os.path.join(TEST_DIR_IN, file_in), os.path.join(TEST_DIR_OUT, file_out))

    hfile = h5py.File(os.path.join(TEST_DIR_OUT, file_out), "r+")

    hfile.move("/science/LSAR/GCOV/grids/frequencyA/HHHH", "/science/LSAR/GCOV/grids/frequencyA/HHRV")
    hfile.close()

def gcov_wrong_polarizations6(file_in, file_out):

    shutil.copyfile(os.path.join(TEST_DIR_IN, file_in), os.path.join(TEST_DIR_OUT, file_out))

    hfile = h5py.File(os.path.join(TEST_DIR_OUT, file_out), "r+")

    hfile.move("/science/LSAR/GCOV/grids/frequencyA/HHHH", "/science/LSAR/GCOV/grids/frequencyA/RHRZ")
    hfile.close()

def rslc_missing_none(file_in, file_out):

    shutil.copyfile(os.path.join(TEST_DIR_IN, file_in), os.path.join(TEST_DIR_OUT, file_out))

    hfile = h5py.File(os.path.join(TEST_DIR_OUT, file_out), "r+")
    fix_rslc(hfile)
    hfile.close()

def rslc_missing_one(file_in, file_out):

    shutil.copyfile(os.path.join(TEST_DIR_IN, file_in), os.path.join(TEST_DIR_OUT, file_out))

    hfile = h5py.File(os.path.join(TEST_DIR_OUT, file_out), "r+")

    fix_rslc(hfile)
    del hfile["/science/LSAR/identification/absoluteOrbitNumber"]    
    hfile.close()

def rslc_inconsistent_bands(file_in, file_out):

    shutil.copyfile(os.path.join(TEST_DIR_IN, file_in), os.path.join(TEST_DIR_OUT, file_out))

    hfile = h5py.File(os.path.join(TEST_DIR_OUT, file_out), "r+")

    hfile.move("/science/LSAR/RSLC/metadata", "/science/LSAR/SLC/metadata")
    hfile.close()
    
def rslc_lsar_vs_ssar(file_in, file_out):

    shutil.copyfile(os.path.join(TEST_DIR_IN, file_in), os.path.join(TEST_DIR_OUT, file_out))
    
    hfile = h5py.File(os.path.join(TEST_DIR_OUT, file_out), "r+")

    fix_rslc(hfile)
    hfile.copy("/science/LSAR/identification", "/science/SSAR/identification")
    hfile.copy("/science/LSAR/RSLC", "/science/SSAR/RSLC")
    orbit_lsar = hfile["/science/LSAR/identification/absoluteOrbitNumber"][...]
    hfile["/science/SSAR/identification/absoluteOrbitNumber"][...] = orbit_lsar + 1

    hfile.close()

def rslc_identification1(file_in, file_out):

    shutil.copyfile(os.path.join(TEST_DIR_IN, file_in), os.path.join(TEST_DIR_OUT, file_out))

    hfile = h5py.File(os.path.join(TEST_DIR_OUT, file_out), "r+")
    fix_rslc(hfile)
    
    time1 = numpy.array(["2018-07-30T16:15:47.000000"], dtype="S26")
    time2 = numpy.array(["2018-07-30T16:14:39.000000"], dtype="S26") 

    id = hfile["/science/LSAR/identification"]
    del id["zeroDopplerStartTime"]
    del id["zeroDopplerEndTime"]
    id.create_dataset("zeroDopplerStartTime", data=time1)
    id.create_dataset("zeroDopplerEndTime", data=time2)
    
    hfile.close()

def rslc_identification2(file_in, file_out):

    shutil.copyfile(os.path.join(TEST_DIR_IN, file_in), os.path.join(TEST_DIR_OUT, file_out))

    hfile = h5py.File(os.path.join(TEST_DIR_OUT, file_out), "r+")
    fix_rslc(hfile)
 
    id = hfile["/science/LSAR/identification"]
    id["absoluteOrbitNumber"][...] = -10
    hfile.close()

def rslc_identification3(file_in, file_out):

    shutil.copyfile(os.path.join(TEST_DIR_IN, file_in), os.path.join(TEST_DIR_OUT, file_out))

    hfile = h5py.File(os.path.join(TEST_DIR_OUT, file_out), "r+")
    fix_rslc(hfile)
 
    id = hfile["/science/LSAR/identification"]
    id["trackNumber"][...] = 900
    hfile.close()
    
def rslc_identification4(file_in, file_out):

    shutil.copyfile(os.path.join(TEST_DIR_IN, file_in), os.path.join(TEST_DIR_OUT, file_out))

    hfile = h5py.File(os.path.join(TEST_DIR_OUT, file_out), "r+")
    fix_rslc(hfile)
 
    id = hfile["/science/LSAR/identification"]
    id["frameNumber"][...] = -2
    hfile.close()

def rslc_identification5(file_in, file_out):

    shutil.copyfile(os.path.join(TEST_DIR_IN, file_in), os.path.join(TEST_DIR_OUT, file_out))

    hfile = h5py.File(os.path.join(TEST_DIR_OUT, file_out), "r+")
    fix_rslc(hfile)
 
    id = hfile["/science/LSAR/identification"]
    id["cycleNumber"][...] = -4
    hfile.close()

def rslc_identification6(file_in, file_out):

    shutil.copyfile(os.path.join(TEST_DIR_IN, file_in), os.path.join(TEST_DIR_OUT, file_out))

    hfile = h5py.File(os.path.join(TEST_DIR_OUT, file_out), "r+")
    fix_rslc(hfile)
    product = numpy.array(["ABCD"], dtype="S4")
    
    id = hfile["/science/LSAR/identification"]
    del id["productType"]
    id.create_dataset("productType", data=product)    

    hfile.close()

def rslc_identification7(file_in, file_out):

    shutil.copyfile(os.path.join(TEST_DIR_IN, file_in), os.path.join(TEST_DIR_OUT, file_out))

    hfile = h5py.File(os.path.join(TEST_DIR_OUT, file_out), "r+")
    fix_rslc(hfile)
    look = numpy.array(["leftr"], dtype="S5")
 
    id = hfile["/science/LSAR/identification"]
    del id["lookDirection"]
    id.create_dataset("lookDirection", data=look)
    hfile.close()

def rslc_identification8(file_in, file_out):

    shutil.copyfile(os.path.join(TEST_DIR_IN, file_in), os.path.join(TEST_DIR_OUT, file_out))

    hfile = h5py.File(os.path.join(TEST_DIR_OUT, file_out), "r+")
    fix_rslc(hfile)

    swaths = hfile["/science/LSAR/RSLC/swaths"]
    swaths.copy("frequencyA", "frequencyB")

    del hfile["/science/LSAR/identification/listOfFrequencies"]
    hfile.create_dataset("/science/LSAR/identification/listOfFrequencies", \
                         data=numpy.array(["A", "B"], dtype="S1"))
    
    freqA = hfile["/science/LSAR/RSLC/swaths/frequencyA"]
    freqB = hfile["/science/LSAR/RSLC/swaths/frequencyB"]
    freqA["acquiredCenterFrequency"][...] = 1243000000.0
    freqB["acquiredCenterFrequency"][...] = 1233000000.0
    freqB["processedCenterFrequency"][...] = freqA["processedCenterFrequency"][...] + 10.0

    hfile.close()

def rslc_identification9(file_in, file_out):

    shutil.copyfile(os.path.join(TEST_DIR_IN, file_in), os.path.join(TEST_DIR_OUT, file_out))

    hfile = h5py.File(os.path.join(TEST_DIR_OUT, file_out), "r+")
    fix_rslc(hfile)

    swaths = hfile["/science/LSAR/RSLC/swaths"]
    swaths.copy("frequencyA", "frequencyB")

    del hfile["/science/LSAR/identification/listOfFrequencies"]
    hfile.create_dataset("/science/LSAR/identification/listOfFrequencies", \
                         data=numpy.array(["A", "B"], dtype="S1"))
    
    freqA = hfile["/science/LSAR/RSLC/swaths/frequencyA"]
    freqB = hfile["/science/LSAR/RSLC/swaths/frequencyB"]
    freqA["processedCenterFrequency"][...] = 1243000000.0
    freqB["processedCenterFrequency"][...] = 1233000000.0
    freqB["acquiredCenterFrequency"][...] = freqA["acquiredCenterFrequency"][...] + 10.0

    hfile.close()

def rslc_identification2b(file_in, file_out):

    shutil.copyfile(os.path.join(TEST_DIR_IN, file_in), os.path.join(TEST_DIR_OUT, file_out))

    hfile = h5py.File(os.path.join(TEST_DIR_OUT, file_out), "r+")
    fix_rslc(hfile)

    id = hfile["/science/LSAR/identification"]
    id["absoluteOrbitNumber"][...] = -10
    id["trackNumber"][...] = -20

    hfile.close()
    
def rslc_nan1(file_in, file_out):

    shutil.copyfile(os.path.join(TEST_DIR_IN, file_in), os.path.join(TEST_DIR_OUT, file_out))

    hfile = h5py.File(os.path.join(TEST_DIR_OUT, file_out), "r+")
    fix_rslc(hfile)

    shape = hfile["/science/LSAR/RSLC/swaths/frequencyA/HH"].shape
    image = hfile["/science/LSAR/RSLC/swaths/frequencyA/HH"]

    with image.astype(numpy.complex64):
        data = image[...]

    data = remove_zeroes_complex(data)
    data.real[0:shape[0]//2, :] = numpy.nan    
    write_field(hfile, "/science/LSAR/RSLC/swaths/frequencyA/HH", data, dtype=complex32)
    
    hfile.close()
    
def rslc_nan2(file_in, file_out):

    shutil.copyfile(os.path.join(TEST_DIR_IN, file_in), os.path.join(TEST_DIR_OUT, file_out))

    hfile = h5py.File(os.path.join(TEST_DIR_OUT, file_out), "r+")
    fix_rslc(hfile)

    shape = hfile["/science/LSAR/RSLC/swaths/frequencyA/HH"].shape
    image = hfile["/science/LSAR/RSLC/swaths/frequencyA/HH"]

    with image.astype(numpy.complex64):
        data = image[...]

    data = remove_zeroes_complex(data)
    data.imag[:, :] = numpy.nan    
    write_field(hfile, "/science/LSAR/RSLC/swaths/frequencyA/HH", data, dtype=complex32)
    
    hfile.close()

def rslc_zero1(file_in, file_out):

    shutil.copyfile(os.path.join(TEST_DIR_IN, file_in), os.path.join(TEST_DIR_OUT, file_out))

    hfile = h5py.File(os.path.join(TEST_DIR_OUT, file_out), "r+")
    fix_rslc(hfile)

    shape = hfile["/science/LSAR/RSLC/swaths/frequencyA/HH"].shape
    image = hfile["/science/LSAR/RSLC/swaths/frequencyA/HH"]

    with image.astype(numpy.complex64):
        data = image[...]

    data = remove_zeroes_complex(data)
    data.real[0:shape[0]//4, :] = 0.0
    data.imag[0:shape[0]//4, :] = 0.0
    write_field(hfile, "/science/LSAR/RSLC/swaths/frequencyA/HH", data, dtype=complex32)
    
    hfile.close()

def rslc_zero2(file_in, file_out):

    shutil.copyfile(os.path.join(TEST_DIR_IN, file_in), os.path.join(TEST_DIR_OUT, file_out))

    hfile = h5py.File(os.path.join(TEST_DIR_OUT, file_out), "r+")
    fix_rslc(hfile)

    shape = hfile["/science/LSAR/RSLC/swaths/frequencyA/HH"].shape
    image = hfile["/science/LSAR/RSLC/swaths/frequencyA/HH"]

    with image.astype(numpy.complex64):
        data = image[...]


    data.real[:, :] = 0.0
    data.imag[:, :] = 0.0
    write_field(hfile, "/science/LSAR/RSLC/swaths/frequencyA/HH", data, dtype=complex32)
    
    hfile.close()

def rslc_nan_zero(file_in, file_out):

    shutil.copyfile(os.path.join(TEST_DIR_IN, file_in), os.path.join(TEST_DIR_OUT, file_out))

    hfile = h5py.File(os.path.join(TEST_DIR_OUT, file_out), "r+")
    fix_rslc(hfile)

    shape = hfile["/science/LSAR/RSLC/swaths/frequencyA/HH"].shape
    image = hfile["/science/LSAR/RSLC/swaths/frequencyA/HH"]

    with image.astype(numpy.complex64):
        data = image[...]

    data.real[0:100, :] = 0.0
    data.imag[0:100, :] = 0.0
    data.real[100:shape[0], :] = numpy.nan
    data.imag[100:shape[0], :] = numpy.nan
    write_field(hfile, "/science/LSAR/RSLC/swaths/frequencyA/HH", data, dtype=complex32)
    
    hfile.close()

def rslc_spacing1(file_in, file_out):

    shutil.copyfile(os.path.join(TEST_DIR_IN, file_in), os.path.join(TEST_DIR_OUT, file_out))

    hfile = h5py.File(os.path.join(TEST_DIR_OUT, file_out), "r+")
    fix_rslc(hfile)

    htime = hfile["/science/LSAR/RSLC/swaths/zeroDopplerTime"]
    hspacing = hfile["/science/LSAR/RSLC/swaths/zeroDopplerTimeSpacing"]
    
    time = htime[...]
    time[1] = time[0] - hspacing[...]/2.0
    htime[...] = time

    hfile.close()
    
def rslc_spacing2(file_in, file_out):

    shutil.copyfile(os.path.join(TEST_DIR_IN, file_in), os.path.join(TEST_DIR_OUT, file_out))

    hfile = h5py.File(os.path.join(TEST_DIR_OUT, file_out), "r+")
    fix_rslc(hfile)

    htime = hfile["/science/LSAR/RSLC/swaths/zeroDopplerTime"]
    hspacing = hfile["/science/LSAR/RSLC/swaths/zeroDopplerTimeSpacing"]
    
    time = htime[...]
    time[1] = time[0] + hspacing[...]/2.0
    htime[...] = time

    hfile.close()

def rslc_wrong_size(file_in, file_out):

    shutil.copyfile(os.path.join(TEST_DIR_IN, file_in), os.path.join(TEST_DIR_OUT, file_out))

    hfile = h5py.File(os.path.join(TEST_DIR_OUT, file_out), "r+")

    slant = hfile["/science/LSAR/RSLC/swaths/frequencyA/slantRange"][...]
    del hfile["/science/LSAR/RSLC/swaths/frequencyA/slantRange"]
    hfile.create_dataset("/science/LSAR/RSLC/swaths/frequencyA/slantRange", \
                         data=slant[0:slant.size-10])

    hfile.close()
    
def rslc_frequency1(file_in, file_out, dname):

    hfile_in = h5py.File(os.path.join(TEST_DIR_IN, file_in), "r")
    hfile_out = h5py.File(os.path.join(TEST_DIR_OUT, file_out), "w")
    repack_rslc(hfile_in, hfile_out, "A", "HH", dname)
    fix_rslc(hfile_out)
    
    hfile_in.close()
    hfile_out.close()

    hfile = h5py.File(os.path.join(TEST_DIR_OUT, file_out), "r+")
    
    rspacing = hfile["/science/LSAR/%s/swaths/frequencyA/slantRangeSpacing" % dname]
    tspacing = (constants.c/2.0)/rspacing[...]
    tinterval = 1.0/tspacing

    shape = (hfile["/science/LSAR/%s/swaths/zeroDopplerTime" % dname].shape[0], \
             hfile["/science/LSAR/%s/swaths/frequencyA/slantRange" % dname].shape[0])
    time = numpy.arange(0, 1.0*shape[1]*tinterval, tinterval)
    freq1 = 1.0*1.0E5
    freq2 = 5.0*1.0E5

    raw1 = 2*numpy.pi*freq1*time
    raw2 = 2*numpy.pi*freq2*time
    real1 = numpy.sin(raw1).astype(numpy.float16)
    real2 = numpy.sin(raw2).astype(numpy.float16)
    imag = numpy.zeros(real1.shape, dtype=numpy.float16)

    real = real1+real2
    xdata = real+imag*1j
    xdata = numpy.tile(xdata, (shape[0])).reshape(shape[0], xdata.size)

    write_field(hfile, "/science/LSAR/%s/swaths/frequencyA/HH" % dname, xdata, dtype=complex32)
    hfile.close()

def rslc_power_phase1(file_in, file_out):

    shutil.copyfile(os.path.join(TEST_DIR_IN, file_in), os.path.join(TEST_DIR_OUT, file_out))

    hfile = h5py.File(os.path.join(TEST_DIR_OUT, file_out), "r+")
    resize_rslc(hfile, "A", "HH", "RSLC", 128, 128)
    fix_rslc(hfile)

    dset = hfile["/science/LSAR/RSLC/swaths/frequencyA/HH"]
    shape = dset.shape
    nlines = shape[1]//4

    real = numpy.zeros(shape, dtype=numpy.float32)
    imag = numpy.zeros(shape, dtype=numpy.float32)
        
    real[:, 0:nlines] = -1.0
    real[:, nlines:2*nlines] = -2.0
    real[:, 2*nlines:3*nlines] = 3.0
    real[:, 3*nlines:4*nlines] = 4.0

    imag[:, 0:nlines] = -2.0
    imag[:, nlines:2*nlines] = -4.0
    imag[:, 2*nlines:3*nlines] = 1.5
    imag[:, 3*nlines:4*nlines] = 2.0

    xdata = real + 1.0j*imag
    dset[...] = xdata
    
    hfile.close()
    
def rslc_power_phase2(file_in, file_out):

    shutil.copyfile(os.path.join(TEST_DIR_IN, file_in), os.path.join(TEST_DIR_OUT, file_out))

    hfile = h5py.File(os.path.join(TEST_DIR_OUT, file_out), "r+")
    resize_rslc(hfile, "A", "HH", "RSLC", 128, 128)
    fix_rslc(hfile)

    dset = hfile["/science/LSAR/RSLC/swaths/frequencyA/HH"]
    shape = dset.shape
    nlines = shape[1]//4

    real = numpy.zeros(shape, dtype=numpy.float32)
    imag = numpy.zeros(shape, dtype=numpy.float32)
        
    real[:, 0:nlines] = -1.0
    real[:, nlines:2*nlines] = numpy.nan
    real[:, 2*nlines:3*nlines] = 3.0
    real[:, 3*nlines:4*nlines] = 0.0

    imag[:, 0:nlines] = -2.0
    imag[:, nlines:2*nlines] = -4.0
    imag[:, 2*nlines:3*nlines] = 1.5
    imag[:, 3*nlines:4*nlines] = 0.0

    xdata = real + 1.0j*imag
    dset[...] = xdata
    
    hfile.close()
    
def rslc_power_phase3(file_in, file_out):

    shutil.copyfile(os.path.join(TEST_DIR_IN, file_in), os.path.join(TEST_DIR_OUT, file_out))

    hfile = h5py.File(os.path.join(TEST_DIR_OUT, file_out), "r+")
    resize_rslc(hfile, "A", "HH", "RSLC", 128, 128)
    fix_rslc(hfile)

    hfile.copy("/science/LSAR/RSLC/swaths/frequencyA/HH", "/science/LSAR/RSLC/swaths/frequencyA/HV")
    del hfile["/science/LSAR/RSLC/swaths/frequencyA/listOfPolarizations"]
    hfile.create_dataset("/science/LSAR/RSLC/swaths/frequencyA/listOfPolarizations", \
                         data=numpy.array(["HH", "HV"], dtype="S2"))

    dset_hh = hfile["/science/LSAR/RSLC/swaths/frequencyA/HH"]
    dset_vv = hfile["/science/LSAR/RSLC/swaths/frequencyA/HV"]
    shape = dset_hh.shape
    nlines = shape[1]//4

    real1 = numpy.zeros(shape, dtype=numpy.float32) 
    imag1 = numpy.zeros(shape, dtype=numpy.float32)
    real2 = numpy.zeros(shape, dtype=numpy.float32) 
    imag2 = numpy.zeros(shape, dtype=numpy.float32)
        
    real1[:, 0:2*nlines] = 1.0
    real1[:, 2*nlines:4*nlines] = 0.0 
    real2[:, 0:2*nlines] = 10.0
    real2[:, 2*nlines:4*nlines] = numpy.nan

    imag1[:, 0:2*nlines] = 2.0
    imag1[:, 2*nlines:4*nlines] = 0.0
    imag2[:, 0:2*nlines] = 20.0
    imag2[:, 2*nlines:4*nlines] = 200.0

    xdata1 = real1 + 1.0j*imag1
    xdata2 = real2 + 1.0j*imag2
    dset_hh[...] = xdata1
    dset_vv[...] = xdata2
    
    hfile.close()

def rslc_subswath1(file_in, file_out):

    shutil.copyfile(os.path.join(TEST_DIR_IN, file_in), os.path.join(TEST_DIR_OUT, file_out))

    hfile = h5py.File(os.path.join(TEST_DIR_OUT, file_out), "r+")
    del hfile["/science/LSAR/RSLC/swaths/frequencyA/validSamplesSubSwath1"]

    hfile.close()
 
def rslc_subswath2(file_in, file_out):

    shutil.copyfile(os.path.join(TEST_DIR_IN, file_in), os.path.join(TEST_DIR_OUT, file_out))

    hfile = h5py.File(os.path.join(TEST_DIR_OUT, file_out), "r+")
    shape = hfile["/science/LSAR/RSLC/swaths/frequencyA/HH"].shape
    
    subswath = hfile["/science/LSAR/RSLC/swaths/frequencyA/validSamplesSubSwath1"]
    subswath[0, 1] = shape[1] + 10

    hfile.close()
    
def gcov_nan1(file_in, file_out):

    shutil.copyfile(os.path.join(TEST_DIR_IN, file_in), os.path.join(TEST_DIR_OUT, file_out))

    hfile = h5py.File(os.path.join(TEST_DIR_OUT, file_out), "r+")
    #fix_gcov(hfile)
    remove_nans_real(hfile)

    shape = hfile["/science/LSAR/GCOV/grids/frequencyA/HHHH"].shape
    image = hfile["/science/LSAR/GCOV/grids/frequencyA/HHHH"]
    data = image[...]
    
    data[0:shape[0]//2, :] = numpy.nan    
    write_field(hfile, "/science/LSAR/GCOV/grids/frequencyA/HHHH", data, dtype=numpy.float32)
    
    hfile.close()
    
def gcov_nan2(file_in, file_out):

    shutil.copyfile(os.path.join(TEST_DIR_IN, file_in), os.path.join(TEST_DIR_OUT, file_out))

    hfile = h5py.File(os.path.join(TEST_DIR_OUT, file_out), "r+")
    #fix_gcov(hfile)
    remove_nans_real(hfile)

    shape = hfile["/science/LSAR/GCOV/grids/frequencyA/HHHH"].shape
    image = hfile["/science/LSAR/GCOV/grids/frequencyA/HHHH"]
    data = image[...]

    data.real[:, :] = numpy.nan    
    write_field(hfile, "/science/LSAR/GCOV/grids/frequencyA/HHHH", data, dtype=numpy.float32)
    
    hfile.close()
    
def gcov_percentile(file_in, file_out):

    shutil.copyfile(os.path.join(TEST_DIR_IN, file_in), os.path.join(TEST_DIR_OUT, file_out))

    hfile = h5py.File(os.path.join(TEST_DIR_OUT, file_out), "r+")

    #fix_gcov(hfile)
    remove_nans_negatives_real(hfile)

    dset = hfile["/science/LSAR/GCOV/grids/frequencyA/HHHH"]
    xdata = dset[...]
    shape = dset.shape
    xline = numpy.arange(0, shape[1])
    for i in range(0, shape[0]):
        xdata[i, :] = xline[:]

    dset[...] = xdata

    hfile.close()

def gcov_missing_metadata(file_in, file_out):

    shutil.copyfile(os.path.join(TEST_DIR_IN, file_in), os.path.join(TEST_DIR_OUT, file_out))

    hfile = h5py.File(os.path.join(TEST_DIR_OUT, file_out), "r+")
    del hfile["/science/LSAR/GCOV/metadata"]
    hfile.close()

def gunw_nometadata(file_in, file_out):

    shutil.copyfile(os.path.join(TEST_DIR_IN, file_in), os.path.join(TEST_DIR_OUT, file_out))

    hfile = h5py.File(os.path.join(TEST_DIR_OUT, file_out), "r+")
    del hfile["/science/LSAR/GUNW/metadata"]
    hfile.close()

def gunw_nooffsets(file_in, file_out):

    shutil.copyfile(os.path.join(TEST_DIR_IN, file_in), os.path.join(TEST_DIR_OUT, file_out))

    hfile = h5py.File(os.path.join(TEST_DIR_OUT, file_out), "r+")
    del hfile["/science/LSAR/GUNW/grids/pixelOffsets/VV"]
    hfile.close()

def gunw_nophase(file_in, file_out):

    shutil.copyfile(os.path.join(TEST_DIR_IN, file_in), os.path.join(TEST_DIR_OUT, file_out))

    hfile = h5py.File(os.path.join(TEST_DIR_OUT, file_out), "r+")
    del hfile["/science/LSAR/GUNW/grids/frequencyA/HH/unwrappedPhase"]
    hfile.close()

def gunw_spacing_uneven(file_in, file_out):

    shutil.copyfile(os.path.join(TEST_DIR_IN, file_in), os.path.join(TEST_DIR_OUT, file_out))

    hfile = h5py.File(os.path.join(TEST_DIR_OUT, file_out), "r+")
    coordinates = hfile["/science/LSAR/GUNW/grids/frequencyA/xCoordinates"]
    spacing = hfile["/science/LSAR/GUNW/grids/frequencyA/xCoordinateSpacing"]

    coordinates[-1] += spacing[...]
    
    hfile.close()

def gunw_spacing_negative(file_in, file_out):

    shutil.copyfile(os.path.join(TEST_DIR_IN, file_in), os.path.join(TEST_DIR_OUT, file_out))

    hfile = h5py.File(os.path.join(TEST_DIR_OUT, file_out), "r+")
    coordinates = hfile["/science/LSAR/GUNW/grids/frequencyA/yCoordinates"]
    spacing = hfile["/science/LSAR/GUNW/grids/frequencyA/yCoordinateSpacing"]

    coordinates[1] -= 2*spacing[...]
    
    hfile.close()

def gunw_inconsistent_size(file_in, file_out):

    shutil.copyfile(os.path.join(TEST_DIR_IN, file_in), os.path.join(TEST_DIR_OUT, file_out))

    hfile = h5py.File(os.path.join(TEST_DIR_OUT, file_out), "r+")
    data = hfile["/science/LSAR/GUNW/grids/frequencyA/HH/unwrappedPhase"][0:10, 0:10]
    del hfile["/science/LSAR/GUNW/grids/frequencyA/HH/unwrappedPhase"]
    hfile.create_dataset("/science/LSAR/GUNW/grids/frequencyA/HH/unwrappedPhase", data=data)
    hfile.close()


def gunw_coord_size(file_in, file_out):

    shutil.copyfile(os.path.join(TEST_DIR_IN, file_in), os.path.join(TEST_DIR_OUT, file_out))

    hfile = h5py.File(os.path.join(TEST_DIR_OUT, file_out), "r+")
    xcoord = hfile["/science/LSAR/GUNW/grids/frequencyA/xCoordinates"][0:10]
    del hfile["/science/LSAR/GUNW/grids/frequencyA/xCoordinates"]
    hfile.create_dataset("/science/LSAR/GUNW/grids/frequencyA/xCoordinates", data=xcoord)
    hfile.close()

def gunw_coord_missing(file_in, file_out):

    shutil.copyfile(os.path.join(TEST_DIR_IN, file_in), os.path.join(TEST_DIR_OUT, file_out))

    hfile = h5py.File(os.path.join(TEST_DIR_OUT, file_out), "r+")
    del hfile["/science/LSAR/GUNW/grids/pixelOffsets/xCoordinates"]
    hfile.close()

def gunw_coord_shape(file_in, file_out):

    shutil.copyfile(os.path.join(TEST_DIR_IN, file_in), os.path.join(TEST_DIR_OUT, file_out))

    hfile = h5py.File(os.path.join(TEST_DIR_OUT, file_out), "r+")
    xcoord = numpy.arange(0, 16).reshape(4, 4)
    del hfile["/science/LSAR/GUNW/grids/pixelOffsets/xCoordinates"]
    hfile.create_dataset("/science/LSAR/GUNW/grids/pixelOffsets/xCoordinates", data=xcoord)
    hfile.close()

def gunw_connected(file_in, file_out):

    nline = 512
    nsmp = 512
    ngroup = nline*nsmp//(8*8)

    tile1 = numpy.arange(1, 5).reshape(2,2).astype(numpy.float32)
    tile2 = numpy.arange(5, 9).reshape(2,2).astype(numpy.float32)
    tile3 = 10*tile2

    tile1b = tile1.repeat(nline//8, axis=0).repeat(nsmp//4, axis=1)
    tile2b = tile2.repeat(nline//8, axis=0).repeat(nsmp//4, axis=1)
    tile1c = tile1.repeat(nline//8, axis=0).repeat(nsmp//2, axis=1)
    tile3b = tile3.repeat(nline//8, axis=0).repeat(nsmp//2, axis=1)

    connect1 = numpy.zeros((nline, nsmp), dtype=numpy.float32)
    connect1[0:nline//4, 0:nsmp//2] = tile1b[...]
    connect1[nline//4:nline//2, 0:nsmp//2] = tile2b[...]
    connect1[nline//2:3*nline//4, 0:nsmp] = tile1c[...]
    connect1[3*nline//4:nline, 0:nsmp] = tile3b[...]

    connect2 = numpy.zeros((nline, nsmp), dtype=numpy.float32)
    connect2[0:nline, 0:nsmp//2] = 1.0

    shutil.copyfile(os.path.join(TEST_DIR_IN, file_in), os.path.join(TEST_DIR_OUT, file_out))  
    hfile = h5py.File(os.path.join(TEST_DIR_OUT, file_out), "r+")
    hfile["/science/LSAR/GUNW/grids/frequencyA/HH/connectedComponents"][...] = connect1[...]
    hfile["/science/LSAR/GUNW/grids/frequencyA/VV/connectedComponents"][...] = connect2[...]
    
    hfile.close()

if __name__ == "__main__":

    gslc_wrong_size(GSLC_FILE, "gslc_arraysize.h5")
    gslc_missing_band(GSLC_FILE, "missing_band.h5")
    gslc_wrong_frequency1(GSLC_FILE, "wrong_frequencies1.h5")
    gslc_wrong_frequency2(GSLC_FILE, "wrong_frequencies2.h5")
    gslc_wrong_frequency3(GSLC_FILE, "wrong_frequencies3.h5")    
    gslc_wrong_polarizations1(GSLC_FILE, "wrong_polarizations1.h5")
    gslc_wrong_polarizations2(GSLC_FILE, "wrong_polarizations2.h5")
    gslc_wrong_polarizations3(GSLC_FILE, "wrong_polarizations3.h5")

    rslc_inconsistent_bands(RSLC_FILE, "inconsistent_bands.h5")
    rslc_missing_none(RSLC_FILE, "missing_none.h5")
    rslc_missing_one(RSLC_FILE, "missing_one.h5")
    rslc_lsar_vs_ssar(RSLC_FILE, "lsar_vs_ssar.h5")

    rslc_identification1(RSLC_FILE, "identification1.h5")
    rslc_identification2(RSLC_FILE, "identification2.h5")
    rslc_identification2b(RSLC_FILE, "identification2b.h5")
    rslc_identification3(RSLC_FILE, "identification3.h5")
    rslc_identification4(RSLC_FILE, "identification4.h5")
    rslc_identification5(RSLC_FILE, "identification5.h5")
    rslc_identification6(RSLC_FILE, "identification6.h5")
    rslc_identification7(RSLC_FILE, "identification7.h5")
    rslc_identification8(RSLC_FILE, "identification8.h5")
    rslc_identification9(RSLC_FILE, "identification9.h5")

    rslc_nan1(RSLC_FILE, "nan1.h5")
    rslc_nan2(RSLC_FILE, "nan2.h5")
    rslc_zero1(RSLC_FILE, "zeros1.h5")
    rslc_zero2(RSLC_FILE, "zeros2.h5")
    rslc_nan_zero(RSLC_FILE, "nan_zeros.h5")

    gcov_wrong_frequency1(GCOV_FILE, "gcov_wrong_frequencies1.h5")
    gcov_wrong_frequency2(GCOV_FILE, "gcov_wrong_frequencies2.h5")
    gcov_wrong_frequency3(GCOV_FILE, "gcov_wrong_frequencies3.h5")
    gcov_wrong_polarizations1(GCOV_FILE, "gcov_wrong_polarizations1.h5")
    gcov_wrong_polarizations2(GCOV_FILE, "gcov_wrong_polarizations2.h5")
    gcov_wrong_polarizations3(GCOV_FILE, "gcov_wrong_polarizations3.h5")
    gcov_wrong_polarizations4(GCOV_FILE, "gcov_wrong_polarizations4.h5")
    gcov_wrong_polarizations5(GCOV_FILE, "gcov_wrong_polarizations5.h5")
    gcov_wrong_polarizations6(GCOV_FILE, "gcov_wrong_polarizations6.h5")
    
    gcov_nan1(GCOV_FILE, "gcov_nan1.h5")
    gcov_nan2(GCOV_FILE, "gcov_nan2.h5")
    
    rslc_spacing1(RSLC_FILE, "time_spacing1.h5")
    rslc_spacing2(RSLC_FILE, "time_spacing2.h5")

    rslc_frequency1(RSLC_FILE2, "frequency1.h5", dname="SLC")
    rslc_power_phase1(RSLC_FILE, "power_phase1.h5")
    rslc_power_phase2(RSLC_FILE, "power_phase2.h5")
    rslc_power_phase3(RSLC_FILE, "power_phase3.h5")

    gcov_percentile(GCOV_FILE, "gcov_stats.h5")
    gcov_missing_metadata(GCOV_FILE, "gcov_missing_metadata.h5")

    rslc_wrong_size(RSLC_FILE, "slc_arraysize.h5")
    rslc_subswath1(RSLC_FILE, "missing_subswath.h5")
    rslc_subswath2(RSLC_FILE, "subswath_bounds.h5")

    gunw_nometadata(GUNW_FILE, "gunw_nometa.h5")
    gunw_nooffsets(GUNW_FILE, "gunw_nooffset.h5")
    gunw_nophase(GUNW_FILE, "gunw_nophase.h5")
    gunw_spacing_uneven(GUNW_FILE, "gunw_spacing_uneven.h5")
    gunw_spacing_negative(GUNW_FILE, "gunw_spacing_negative.h5")
    gunw_inconsistent_size(GUNW_FILE, "gunw_arraysize1.h5")
    gunw_coord_size(GUNW_FILE, "gunw_arraysize2.h5")
    gunw_coord_missing(GUNW_FILE, "gunw_nocoords1.h5")
    gunw_coord_shape(GUNW_FILE, "gunw_nocoords2.h5")
    gunw_connected(GUNW_FILE, "gunw_connected.h5")
