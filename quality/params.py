import itertools

NTRACKS = 173
ORBIT_OFFSET = 0
NSUBSWATHS = 4
FREQUENCIES = (["A"], ["B"], ["A", "B"])
PRODUCT_TYPE = ("b'RRST'", "b'RRSD'", "b'RSLC'", "b'RMLC'", "b'RCOV'", "b'RIFG'", "b'RUNW'", "b'GUNW'", \
                "b'CGOV'", "b'GSLC'")

GCOV_POLARIZATION_LIST = ("HHHH", "HVHV", "VHVH", "VVVV")
SLC_POLARIZATION_LIST = ("HH", "HV", "VH", "VV", "RH", "RV")
GCOV_POLARIZATION_GROUPS = (["VHVH", "VVVV"], ["HHHH", "HVHV", "VHVH", "VVVV"])
SLC_POLARIZATION_GROUPS = (["HH"], ["HH", "HV"], ["VV"], ["VV", "VH"], ["HH", "HV", "VH", "VV"], ["RH", "RV"])

GCOV_POLARIZATION_LIST = []
for p1 in ("HH", "HV", "VH", "VV"):
    for p2 in ("HH", "HV", "VH", "VV"):
        GCOV_POLARIZATION_LIST.append("%s%s" % (p1, p2))

        
GCOV_POLARIZATION_GROUPS = []
for nelement in range(0, 4):
    xlist = itertools.combinations(GCOV_POLARIZATION_LIST, nelement+1)
    GCOV_POLARIZATION_GROUPS += list(xlist)

#print("GCOV list %s, groups %s" % (GCOV_POLARIZATION_LIST, GCOV_POLARIZATION_GROUPS))
    
GCOV_FREQUENCY_NAMES = ["centerFrequency"]
SLC_FREQUENCY_NAMES = ("acquiredCenterFrequency", "processedCenterFrequency")

GCOV_ID_PARAMS = ("absoluteOrbitNumber", "trackNumber", "frameNumber", "lookDirection", \
                  "orbitPassDirection", "productType", "zeroDopplerStartTime", "zeroDopplerEndTime")
SLC_ID_PARAMS = ("absoluteOrbitNumber", "trackNumber", "frameNumber", "cycleNumber", "lookDirection", \
                 "orbitPassDirection", "productType", "zeroDopplerStartTime", "zeroDopplerEndTime")

PRODUCT_TYPES = ("RRST", "RRSD", "RSLC", "RMLC", "RCOV", "RIFG", "RUNW", "GUNW", "CGOV", "GSLC", "SLC")

EPS = 1.0E-05
TIMELEN = 26
