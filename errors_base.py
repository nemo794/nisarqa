
class CustomError(Exception):
    log_string = {}
    spacing = "    "

    def __init__(self, flname, error_string):
        pass

    @classmethod
    def log_error(cls, flname, error_string):

        cls.ninstance_total += 1
        if (flname not in cls.ninstance_file.keys()):
            cls.ninstance_file[flname] = 1
        else:
            cls.ninstance_file[flname] += 1
            
        if (flname not in cls.log_string.keys()):
            cls.log_string[flname] = ""

        if (isinstance(error_string, str)):
            for xstring in error_string.split("\n"):
                cls.log_string[flname] += "%s%s\n" % (cls.spacing, xstring)
        elif (isinstance(error_string, list)):
              for s in error_string:
                  cls.log_string[flname] += "%s%s\n" % (cls.spacing, s)

class WarningError(CustomError):
    log_string = {}
    ninstance_file = {}
    ninstance_total = 0
    xtype = "WARNING"
    def __init__(self, flname, error_string):

        self.log_error(flname, error_string)
        CustomError.__init__(self, flname, error_string)

    def reset(self):
        WarningError.ninstance_file = 0 

class FatalError(CustomError):
    log_string = {}
    ninstance_file = {}
    ninstance_total = 0
    xtype = "FATAL"
    def __init__(self, flname, error_string):
        self.log_error(flname, error_string)
        CustomError.__init__(self, flname, error_string)
        
    def reset(self):
        FatalError.ninstance_file = 0 






   
