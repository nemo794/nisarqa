import errors_base

class IdentificationFatal(errors_base.FatalError):
    def __init__(self, error_string):
        raise errors_base.FatalError(error_string)

class TimeSpacingWarning(errors_base.WarningError):
    def __init__(self, error_string):
        raise errors_base.WarningError(error_string)

class TimeSpacingFatal(errors_base.FatalError):
    def __init__(self, error_string):
        raise errors_base.FatalError(error_string)

class SlantSpacingWarning(errors_base.WarningError):
    def __init__(self, error_string):
        raise errors_base.WarningError(error_string)

class SlantSpacingFatal(errors_base.FatalError):
    def __init__(self, error_string):
        raise errors_base.FatalError(error_string)

class ASlantWarning(SlantSpacingWarning):
    def __init__(self):
        SlantWarning.__init__(self, "A")

class BSlantWarning(SlantSpacingWarning):
    def __init__(self):
        SlantWarning.__init__(self, "B")

class ASlantFatal(SlantSpacingFatal):
    def __init__(self):
        SlantFatal.__init__(self, "A")

class BSlantFatal(SlantSpacingFatal):
    def __init__(self):
        SlantFatal.__init__(self, "B")

class MissingSubswathFatal(errors_base.FatalError):
    def __init__(self, error_string):
        raise errors_base.FatalError(error_string)

class NumSubswathFatal(errors_base.FatalError):
    def __init__(self, error_string):
        raise errors_base.FatalError(error_string)

class BoundsSubswathFatal(errors_base.FatalError):
    def __init__(self, error_string):
        raise errors_base.FatalError(error_string)

class FrequencyListFatal(errors_base.FatalError):
    def __init__(self, error_string):
        raise errors_base.FatalError(error_string)

class FrequencyOrderFatal(errors_base.FatalError):
    def __init__(self, error_string):
        raise errors_base.FatalError(error_string)

class PolarizationListFatal(errors_base.FatalError):
    def __init__(self, error_string):
        raise errors_base.FatalError(error_string)

class ArrayMissingFatal(errors_base.FatalError):
    def __init__(self, error_string):
        raise errors_base.FatalError(error_string)

class ArraySizeFatal(errors_base.FatalError):
    def __init__(self, error_string):
        raise errors_base.FatalError(error_string)

class NaNWarning(errors_base.WarningError):
    def __init__(self, error_string):
        raise errors_base.WarningError(error_string)

    

 
