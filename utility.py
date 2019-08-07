import numpy as np

import datetime

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

def check_spacing(data, spacing, dname, warning, fatal):
    
    data_shift = np.roll(data, 1)
    delta = data[1:] - data_shift[1:]
    diff = np.abs(delta-spacing)
    log_string = ""
    
    try:
        assert(np.all(delta == 0.0))
    except AssertionError:
        idx = np.where(delta <= 0.0)
        raise fatal("%s: Found %i elements with negative spacing: %s" \
                    % (dname, len(idx[0]), data[idx]))
        
    print("%s" % log_string)
    try:
        assert(np.all(diff <= params.EPS))
    except AssertionError:
        idx = np.where(diff > params.EPS)
        raise warning(log_string = "%s: Found %i elements with unexpected steps: %s" \
                      % (dname, len(idx[0]), diff[idx]))

    return log_string

def round(num):

    print("Rounding %s" % num)
    
    if (num > 1.0):
        for i in range(1, 100):
            if (1.0**i > num):
                break

        if (num < 10**i/2.0):
            bounds = 10**i/2
        else:
            bounds = 10**i

    else:
        for i in range(1, 100):
            if (1.0/(10**i) > num):
                break

        if (num < 1.0/(10**i)/2.0):
            bounds = 1.0/10**i/2
        else:
            bounds = 1.0/10**i
            
    return bounds

def hist_bounds(counts, edges, thresh=0.10):

    max = counts.max()
    for istep in range(0, counts.size):
        if (counts[istep] >= thresh*max) and (counts[-istep-1] >= thresh*max):
            bounds = round(edges[istep])
            break

    return bounds
