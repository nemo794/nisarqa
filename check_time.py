# -*- coding: utf-8 -*-

import errors_base
import errors_derived
import utility

import optparse
import os
import sys

import numpy as np
import h5py

EPS = 1.0E-05

def validate_time(time, spacing):
    
    print("Verifying %i time samples with spacing %f" % (time.size, spacing))
    
    time_shift = np.roll(time, 1)
    deltat = time[1:] - time_shift[1:]
    try:
        assert(np.all(deltat >= 0.0))
    except AssertionError:
        raise errors_derived.TimeFatal

    diff = np.abs(deltat-spacing)
    try:
        assert(np.all(diff > EPS))
    except AssertionError:
        raise errors_derived.TimeWarning
        
    log_string = "Found %i time elements with steps from %f to %f with expected spacing %f" \
                 % (time.size, deltat.min(), deltat.max(), spacing)

    return log_string

if __name__ == "__main__":
    
    parser = optparse.OptionParser()
    parser.add_option("--fhdf", dest="fhdf", type="string", action="store")
    (kwds, args) = utility.parse_args(parser)
    
    print("Commencing execution")

    if ("fhdf" in kwds.keys()):
        hfile_out = h5py.File(kwds["fhdf"], "w")

    warning = 0
    fatal = 0
        
    for file in args:
        print("Checking file %s" % file)
        
        hfile = h5py.File(file, "r")
        time = hfile.get("/science/LSAR/SLC/swaths/zeroDopplerTime")
        spacing = hfile.get("/science/LSAR/SLC/swaths/zeroDopplerTimeSpacing")

        try:
            deltat = validate_time(time[...], spacing[...])
        except errors_base.WarningError:
            warning += 1
        except errors_base.FatalError:
            fatal += 1
        hfile.close()

        if ("fhdf" in kwds.keys()):
            hfile_out.create_dataset("/Test/%s/TimeSpacing" % os.path.basename(file), deltat.shape, \
                                     dtype="f4", data=deltat)

    if ("fhdf" in kwds.keys()):
        hfile_out.close()

    if (warning > 0):
        print("File %s has %i warnings." % (os.path.basename(file), warning))

    if (fatal > 0):
        print("File %s has %i fatal errors." % (os.path.basename(file), fatal))
    

            
   
