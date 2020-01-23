import params

import numpy as np

import datetime
import math
import tempfile
import traceback
import sys

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

def get_traceback(error, error_type):

    traceback_string = []
    
    fid = tempfile.NamedTemporaryFile(mode="w", delete=False)
    traceback.print_exception(error_type, "", error.__traceback__, file=fid)
    fid.close()
    
    fid = open(fid.name)
    for xline in fid:
        traceback_string.append(xline)
    fid.close()
    
    return traceback_string

def check_spacing(flname, start_time, data, spacing, dname, warning, fatal):
    
    data_shift = np.roll(data, 1)
    delta = data[1:] - data_shift[1:]
    diff = np.abs(delta-spacing)
    log_string = ""
    
    try:
        assert(np.all(delta > 0.0))
    except AssertionError as e:
        idx = np.where(delta <= 0.0)
        traceback_string = [get_traceback(e, AssertionError)]
        raise fatal(flname, start_time, traceback_string, \
                    ["%s: Found %i elements with negative spacing: %s at locations %s" \
                     % (dname, len(idx[0]), data[idx], idx)])
        
    try:
        assert(np.all(diff <= params.EPS))
    except AssertionError as e:
        idx = np.where(diff > params.EPS)
        traceback_string = [get_traceback(e, AssertionError)]
        raise warning(flname, start_time, traceback_string, \
                      ["%s: Found %i elements with unexpected steps: %s at locations %s" \
                       % (dname, len(idx[0]), diff[idx], idx)])

    return log_string

def round(num):

    #print("Rounding %s" % num)
    
    for i in range(0, 100):
        if (10.0**i > num):
            break

    if (num < 10**i/5.0):
        bounds = 10.0**i/5
    else:
        bounds = 10.0**i

    return bounds

def hist_bounds(counts, edges, thresh_of_max=0.10, thresh_increase=1.05):

    max_value = counts.max()
    max_idx = np.where(counts == max_value)[0][0]
    bin_left = -9999
    bin_right = -9999
    #print("Max value in array: %i at location %i" % (max_value, max_idx))
    
    for i in range(0, int(counts.size/2)):
        binl1 = max_idx - i
        binl2 = max_idx - i - 1
        binr1 = max_idx + i + 1
        binr2 = max_idx + i + 2

        if ( (binl1 < 0) and (bin_left == -9999) ) or \
           ( (binr2 >= counts.size) and (bin_right == -9999) ):
            return
        
        #print("Lbins %i %i, counts %i %i, increase %f, perc_of_max %f" \
        #      % (binl1, binl2, counts[binl1], counts[binl2], \
        #         1.0*counts[binl1]/counts[binl2], 1.0*counts[binl1]/max_value))
        #print("Rbins %i %i, counts %i %i, increase %f, perc_of_max %f\n" \
        #      % (binr1, binr2, counts[binr1], counts[binr2], \
        #         1.0*counts[binr1]/counts[binr2], 1.0*counts[binr1]/max_value))

        if (1.0*counts[binl1]/counts[binl2] > thresh_increase) and \
           (counts[binl1] <= thresh_of_max*max_value) and (bin_left == -9999):
            #print("Selecting left_bin %i" % edges[binl1])
            bin_left = edges[binl1]
        if (1.0*counts[binr1]/counts[binr2] > thresh_increase) and \
           (counts[binr1] <= thresh_of_max*max_value) and (bin_right == -9999):
            #print("Selecting right bin %i" % edges[binr1])
            bin_right = edges[binr1]
        if (bin_left != -9999) and (bin_right != -9999):
            bounds = round(max(math.fabs(bin_right), math.fabs(bin_left)))
            #print("min %f, max %f, bounds %f" % (bin_left, bin_right, bounds))
            return bounds
        
