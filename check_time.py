# -*- coding: utf-8 -*-

import optparse
import os
import sys

import numpy as np
import h5py

EPS = 1.0E-05

def parse_args(parser, required=[], keep_none=[]):
    
    try:
        opts, args = parser.parse_args()
        kwds = eval(str(opts))
    except:
        print("Incorrect options.")
        sys.exit()

    for k in required:
        if (not kwds.has_key(k)):
            raise RuntimeError("Missing required keyword %s" % k)
    for k in list(kwds.keys()):
        if (kwds[k] is None) and (k not in keep_none):
            del kwds[k]

        
    return kwds, args

def validate_time(time, spacing):
    
    print("Verifying %i time samples with spacing %f" % (time.size, spacing))
    
    time_shift = np.roll(time, 1)
    deltat = time[1:] - time_shift[1:]
    assert(np.all(deltat > 0.0))

    diff = np.abs(deltat-spacing)
    assert(np.all(diff <= EPS))
        
    print("Found %i time elements with steps from %f to %f with expected spacing %f" \
           % (time.size, deltat.min(), deltat.max(), spacing))

    return deltat

if __name__ == "__main__":
    
    parser = optparse.OptionParser()
    parser.add_option("--fhdf", dest="fhdf", type="string", action="store")
    (kwds, args) = parse_args(parser)
    
    print("Commencing execution")

    if ("fhdf" in kwds.keys()):
        hfile_out = h5py.File(kwds["fhdf"], "w")
    
    for file in args:
        print("Checking file %s" % file)
        
        hfile = h5py.File(file, "r")
        time = hfile.get("/science/LSAR/SLC/swaths/zeroDopplerTime")
        spacing = hfile.get("/science/LSAR/SLC/swaths/zeroDopplerTimeSpacing")
        deltat = validate_time(time[...], spacing[...])
        hfile.close()

        if ("fhdf" in kwds.keys()):
            hfile_out.create_dataset("/Test/%s/TimeSpacing" % os.path.basename(file), deltat.shape, \
                                     dtype="f4", data=deltat)

    if ("fhdf" in kwds.keys()):
        hfile_out.close()

            
   
