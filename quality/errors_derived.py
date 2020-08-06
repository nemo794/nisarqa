from quality import errors_base

import os

class IdentificationWarning(errors_base.WarningError):
    file_list = []
    ninstance = 0
    name = "WarningIdentificationError"
    def __init__(self, flname, start_time, traceback, description):
        IdentificationWarning.ninstance += 1
        IdentificationWarning.file_list.append(os.path.basename(flname))
        raise errors_base.WarningError(flname, start_time, self.name, traceback, description)

class IdentificationFatal(errors_base.FatalError):
    file_list = []
    ninstance = 0
    name = "FatalIdentificationError"
    def __init__(self, flname, start_time, traceback, description):
        IdentificationFatal.ninstance += 1
        IdentificationFatal.file_list.append(os.path.basename(flname))
        raise errors_base.FatalError(flname, start_time, self.name, traceback, description)

class FrequencyPolarizationFatal(errors_base.FatalError):
    file_list = []
    ninstance = 0
    name = "FatalFrequencyPolarizationError"
    def __init__(self, flname, start_time, traceback, description):
        FrequencyPolarizationFatal.ninstance += 1
        FrequencyPolarizationFatal.file_list.append(os.path.basename(flname))
        raise errors_base.FatalError(flname, start_time, self.name, traceback, description)

class MissingDatasetFatal(errors_base.FatalError):
    file_list = []
    ninstance = 0
    name = "FatalMissingDatasetError"
    def __init__(self, flname, start_time, traceback, description):
        MissingDatasetFatal.ninstance += 1
        MissingDatasetFatal.file_list.append(os.path.basename(flname))
        raise errors_base.FatalError(flname, start_time, self.name, traceback, description)

class MissingDatasetWarning(errors_base.WarningError):
    file_list = []
    ninstance = 0
    name = "WarningMissingDatasetError"
    def __init__(self, flname, start_time, traceback, description):
        MissingDatasetWarning.ninstance += 1
        MissingDatasetWarning.file_list.append(os.path.basename(flname))
        raise errors_base.WarningError(flname, start_time, self.name, traceback, description)

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

class CoordinateSpacingWarning(errors_base.WarningError):
    file_list = []
    ninstance = 0
    name = "WarningCoordinateSpacing"
    def __init__(self, flname, start_time, traceback, description):
        CoordinateSpacingWarning.ninstance += 1
        CoordinateSpacingWarning.file_list.append(os.path.basename(flname))
        raise errors_base.WarningError(flname, start_time, self.name, traceback, description)

class CoordinateSpacingFatal(errors_base.FatalError):
    file_list = []
    ninstance = 0
    name = "FatalCoordinateSpacing"
    def __init__(self, flname, start_time, traceback, description):
        CoordinateSpacingFatal.ninstance += 1
        CoordinateSpacingFatal.file_list.append(os.path.basename(flname))
        raise errors_base.FatalError(flname, start_time, self.name, traceback, description)
    
class MissingSubswathWarning(errors_base.FatalError):
    file_list = []
    ninstance = 0
    name = "WarningMissingSubswath"
    def __init__(self, flname, start_time, traceback, description):
        MissingSubswathWarning.ninstance += 1
        MissingSubswathWarning.file_list.append(os.path.basename(flname))
        raise errors_base.WarningError(flname, start_time, self.name, traceback, description)

class NumSubswathWarning(errors_base.WarningError):
    file_list = []
    ninstance = 0
    name = "WarningNumSubswath"
    def __init__(self, flname, start_time, traceback, description):
        NumSubswathWarning.ninstance += 1
        NumSubswathWarning.file_list.append(os.path.basename(flname))
        raise errors_base.WarningError(flname, start_time, self.name, traceback, description)

class BoundsSubswathWarning(errors_base.WarningError):
    file_list = []
    ninstance = 0
    name = "WarningBoundsSubswath"
    def __init__(self, flname, start_time, traceback, description):
        BoundsSubswathWarning.ninstance += 1
        BoundsSubswathWarning.file_list.append(os.path.basename(flname))
        raise errors_base.WarningError(flname, start_time, self.name, traceback, description)

class FrequencyListFatal(errors_base.FatalError):
    file_list = []
    ninstance = 0
    name = "FatalFrequencyList"
    def __init__(self, flname, start_time, traceback, description):
        FrequencyListFatal.ninstance += 1
        FrequencyListFatal.file_list.append(os.path.basename(flname))
        raise errors_base.FatalError(flname, start_time, self.name, traceback, description)

class FrequencyOrderWarning(errors_base.WarningError):
    file_list = []
    ninstance = 0
    name = "WarningFrequencyOrder"
    def __init__(self, flname, start_time, traceback, description):
        FrequencyOrderWarning.ninstance += 1
        FrequencyOrderWarning.file_list.append(os.path.basename(flname))
        raise errors_base.WarningError(flname, start_time, self.name, traceback, description)

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

class NaNFatal(errors_base.FatalError):
    file_list = []
    ninstance = 0
    name = "FatalNaN"
    def __init__(self, flname, start_time, traceback, description):
        NaNFatal.ninstance += 1
        NaNFatal.file_list.append(os.path.basename(flname))
        raise errors_base.FatalError(flname, start_time, self.name, traceback, description)

class NaNWarning(errors_base.WarningError):
    file_list = []
    ninstance = 0
    name = "WarningNaN"
    def __init__(self, flname, start_time, traceback, description):
        print("Raising NaN Warning")
        NaNWarning.ninstance += 1
        NaNWarning.file_list.append(os.path.basename(flname))
        raise errors_base.WarningError(flname, start_time, self.name, traceback, description)

class ZeroFatal(errors_base.FatalError):
    file_list = []
    ninstance = 0
    name = "FatalZero"
    def __init__(self, flname, start_time, traceback, description):
        ZeroFatal.ninstance += 1
        ZeroFatal.file_list.append(os.path.basename(flname))
        raise errors_base.FatalError(flname, start_time, self.name, traceback, description)

class ZeroWarning(errors_base.WarningError):
    file_list = []
    ninstance = 0
    name = "WarningZero"
    def __init__(self, flname, start_time, traceback, description):
        print("Raising Zero Warning")
        ZeroWarning.ninstance += 1
        ZeroWarning.file_list.append(os.path.basename(flname))
        raise errors_base.WarningError(flname, start_time, self.name, traceback, description)

class NegativeBackscatterWarning(errors_base.WarningError):
    file_list = []
    ninstance = 0
    name = "WarningNegativeBackscatter"
    def __init__(self, flname, start_time, traceback, description):
        print("Raising NegativeBackScatter Warning")
        NegativeBackscatterWarning.ninstance += 1
        NegativeBackscatterWarning.file_list.append(os.path.basename(flname))
        raise errors_base.WarningError(flname, start_time, self.name, traceback, description)

class RegionGrowingWarning(errors_base.FatalError):
    file_list = []
    ninstance = 0
    name = "WarningRegionGrowing"
    def __init__(self, flname, start_time, traceback, description):
        print("Raising RegionGrowing Warning")
        RegionGrowingWarning.ninstance += 1
        RegionGrowingWarning.file_list.append(os.path.basename(flname))
        raise errors_base.WarningError(flname, start_time, self.name, traceback, description)
    
    

 
