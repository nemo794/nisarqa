import errors_base

import os

class IdentificationFatal(errors_base.FatalError):
    file_list = []
    ninstance = 0
    name = "FatalIdentificationError"
    def __init__(self, flname, error_string):
        IdentificationFatal.ninstance += 1
        IdentificationFatal.file_list.append(os.path.basename(flname))
        raise errors_base.FatalError(flname, error_string)

class TimeSpacingWarning(errors_base.WarningError):
    file_list = []
    ninstance = 0
    name = "WarningTimeSpacing"
    def __init__(self, flname, error_string):
        TimeSpacingWarning.ninstance += 1
        TimeSpacingWarning.file_list.append(os.path.basename(flname))
        raise errors_base.WarningError(flname, error_string)

class TimeSpacingFatal(errors_base.FatalError):
    file_list = []
    ninstance = 0
    name = "FatalTimeSpacing"
    def __init__(self, flname, error_string):
        TimeSpacingFatal.ninstance += 1
        TimeSpacingFatal.file_list.append(os.path.basename(flname))
        raise errors_base.FatalError(flname, error_string)

class SlantSpacingWarning(errors_base.WarningError):
    file_list = []
    ninstance = 0
    name = "WarningSlantSpacing"
    def __init__(self, flname, error_string):
        SlantSpacingWarning.ninstance += 1
        SlantSpacingWarning.file_list.append(os.path.basename(flname))
        raise errors_base.WarningError(flname, error_string)

class SlantSpacingFatal(errors_base.FatalError):
    file_list = []
    ninstance = 0
    name = "SlantSpacingFatal"
    def __init__(self, flname, error_string):
        SlantSpacingFatal.ninstance += 1
        SlantSpacingFatal.file_list.append(os.path.basename(flname))
        name = "FatalSlantSpacing"
        raise errors_base.FatalError(flname, error_string)

class MissingSubswathFatal(errors_base.FatalError):
    file_list = []
    ninstance = 0
    name = "FatalMissingSubswath"
    def __init__(self, flname, error_string):
        MissingSubswathFatal.ninstance += 1
        MissingSubswathFatal.file_list.append(os.path.basename(flname))
        raise errors_base.FatalError(flname, error_string)

class NumSubswathFatal(errors_base.FatalError):
    file_list = []
    ninstance = 0
    name = "FatalNumSubswath"
    def __init__(self, flname, error_string):
        NumSubswathFatal.ninstance += 1
        NumSubswathFatal.file_list.append(os.path.basename(flname))
        raise errors_base.FatalError(flname, error_string)

class BoundsSubswathFatal(errors_base.FatalError):
    file_list = []
    ninstance = 0
    name = "FatalBoundsSubswath"
    def __init__(self, flname, error_string):
        BoundsSubswathFatal.ninstance += 1
        BoundsSubswathFatal.file_list.append(os.path.basename(flname))
        raise errors_base.FatalError(flname, error_string)

class FrequencyListFatal(errors_base.FatalError):
    file_list = []
    ninstance = 0
    name = "FatalFrequencyList"
    def __init__(self, flname, error_string):
        FrequencyListFatal.ninstance += 1
        FrequencyListFatal.file_list.append(os.path.basename(flname))
        raise errors_base.FatalError(flname, error_string)

class FrequencyOrderFatal(errors_base.FatalError):
    file_list = []
    ninstance = 0
    name = "FatalFrequencyOrder"
    def __init__(self, flname, error_string):
        FrequencyOrderFatal.ninstance += 1
        FrequencyOrderFatal.file_list.append(os.path.basename(flname))
        raise errors_base.FatalError(flname, error_string)

class PolarizationListFatal(errors_base.FatalError):
    file_list = []
    ninstance = 0
    name = "FatalPolarizationList"
    def __init__(self, flname, error_string):
        PolarizationListFatal.ninstance += 1
        PolarizationListFatal.file_list.append(os.path.basename(flname))
        raise errors_base.FatalError(flname, error_string)

class ArrayMissingFatal(errors_base.FatalError):
    file_list = []
    ninstance = 0
    name = "FatalArrayMissing"
    def __init__(self, flname, error_string):
        ArrayMissingFatal.ninstance += 1
        ArrayMissingFatal.file_list.append(os.path.basename(flname))
        raise errors_base.FatalError(flname, error_string)

class ArraySizeFatal(errors_base.FatalError):
    file_list = []
    ninstance = 0
    name = "FatalArraySize"
    def __init__(self, flname, error_string):
        ArraySizeFatal.ninstance += 1
        ArraySizeFatal.file_list.append(os.path.basename(flname))
        raise errors_base.FatalError(flname, error_string)

class NaNWarning(errors_base.WarningError):
    file_list = []
    ninstance = 0
    name = "WarningNaN"
    def __init__(self, flname, error_string):
        NanWarning.ninstance += 1
        NaNWarning.file_list.append(os.path.basename(flname))
        raise errors_base.WarningError(flname, error_string)

    

 
