import errors_base

import os

class IdentificationFatal(errors_base.FatalError):
    file_list = []
    ninstance = 0
    name = "FatalIdentificationError"
    def __init__(self, flname, start_time, traceback, description):
        IdentificationFatal.ninstance += 1
        IdentificationFatal.file_list.append(os.path.basename(flname))
        raise errors_base.FatalError(flname, start_time, self.name, traceback, description)

class TimeSpacingWarning(errors_base.WarningError):
    file_list = []
    ninstance = 0
    name = "WarningTimeSpacing"
    def __init__(self, flname, start_time, traceback, description):
        TimeSpacingWarning.ninstance += 1
        TimeSpacingWarning.file_list.append(os.path.basename(flname))
        raise errors_base.WarningError(flname, start_time, self.name, traceback, description)

class TimeSpacingFatal(errors_base.FatalError):
    file_list = []
    ninstance = 0
    name = "FatalTimeSpacing"
    def __init__(self, flname, start_time, traceback, description):
        TimeSpacingFatal.ninstance += 1
        TimeSpacingFatal.file_list.append(os.path.basename(flname))
        raise errors_base.FatalError(flname, start_time, self.name, traceback, description)

class SlantSpacingWarning(errors_base.WarningError):
    file_list = []
    ninstance = 0
    name = "WarningSlantSpacing"
    def __init__(self, flname, start_time, traceback, description):
        SlantSpacingWarning.ninstance += 1
        SlantSpacingWarning.file_list.append(os.path.basename(flname))
        raise errors_base.WarningError(flname, start_time, self.name, traceback, description)

class SlantSpacingFatal(errors_base.FatalError):
    file_list = []
    ninstance = 0
    name = "FatalSlantSpacing"
    def __init__(self, flname, start_time, traceback, description):
        SlantSpacingFatal.ninstance += 1
        SlantSpacingFatal.file_list.append(os.path.basename(flname))
        raise errors_base.FatalError(flname, start_time, self.name, traceback, description)

class MissingSubswathFatal(errors_base.FatalError):
    file_list = []
    ninstance = 0
    name = "FatalMissingSubswath"
    def __init__(self, flname, start_time, traceback, description):
        MissingSubswathFatal.ninstance += 1
        MissingSubswathFatal.file_list.append(os.path.basename(flname))
        raise errors_base.FatalError(flname, start_time, self.name, traceback, description)

class NumSubswathFatal(errors_base.FatalError):
    file_list = []
    ninstance = 0
    name = "FatalNumSubswath"
    def __init__(self, flname, start_time, traceback, description):
        NumSubswathFatal.ninstance += 1
        NumSubswathFatal.file_list.append(os.path.basename(flname))
        raise errors_base.FatalError(flname, start_time, self.name, traceback, description)

class BoundsSubswathFatal(errors_base.FatalError):
    file_list = []
    ninstance = 0
    name = "FatalBoundsSubswath"
    def __init__(self, flname, start_time, traceback, description):
        BoundsSubswathFatal.ninstance += 1
        BoundsSubswathFatal.file_list.append(os.path.basename(flname))
        raise errors_base.FatalError(flname, start_time, self.name, traceback, description)

class FrequencyListFatal(errors_base.FatalError):
    file_list = []
    ninstance = 0
    name = "FatalFrequencyList"
    def __init__(self, flname, start_time, traceback, description):
        FrequencyListFatal.ninstance += 1
        FrequencyListFatal.file_list.append(os.path.basename(flname))
        raise errors_base.FatalError(flname, start_time, self.name, traceback, description)

class FrequencyOrderFatal(errors_base.FatalError):
    file_list = []
    ninstance = 0
    name = "FatalFrequencyOrder"
    def __init__(self, flname, start_time, traceback, description):
        FrequencyOrderFatal.ninstance += 1
        FrequencyOrderFatal.file_list.append(os.path.basename(flname))
        raise errors_base.FatalError(flname, start_time, self.name, traceback, description)

class PolarizationListFatal(errors_base.FatalError):
    file_list = []
    ninstance = 0
    name = "FatalPolarizationList"
    def __init__(self, flname, start_time, traceback, description):
        PolarizationListFatal.ninstance += 1
        PolarizationListFatal.file_list.append(os.path.basename(flname))
        raise errors_base.FatalError(flname, start_time, self.name, traceback, description)

class ArrayMissingFatal(errors_base.FatalError):
    file_list = []
    ninstance = 0
    name = "FatalArrayMissing"
    def __init__(self, flname, start_time, traceback, description):
        ArrayMissingFatal.ninstance += 1
        ArrayMissingFatal.file_list.append(os.path.basename(flname))
        raise errors_base.FatalError(flname, start_time, self.name, traceback, description)

class ArraySizeFatal(errors_base.FatalError):
    file_list = []
    ninstance = 0
    name = "FatalArraySize"
    def __init__(self, flname, start_time, traceback, description):
        ArraySizeFatal.ninstance += 1
        ArraySizeFatal.file_list.append(os.path.basename(flname))
        raise errors_base.FatalError(flname, start_time, self.name, traceback, description)

class NaNWarning(errors_base.WarningError):
    file_list = []
    ninstance = 0
    name = "WarningNaN"
    def __init__(self, flname, start_time, traceback, description):
        NanWarning.ninstance += 1
        NaNWarning.file_list.append(os.path.basename(flname))
        raise errors_base.WarningError(flname, start_time, self.name, traceback, description)

    

 
