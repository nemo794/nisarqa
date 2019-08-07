
class CustomError(Exception):
    log_string = ""
    spacing = "    "
    def __init__(self, error_string):
        if (isinstance(error_string, str)):
            CustomError.log_string += "%s%s\n" % (CustomError.spacing, error_string)
        elif (isinstance(error_string, list)):
              for s in error_string:
                  CustomError.log_string += "%s%s\n" % (CustomError.spacing, s)

    def print_log(self, fname, flog):
        file_id = open(flog, "a+")
        file_id.write("File %s had %i %s errors.\n" % (fname, self.ninstance, self.xtype))
        if (self.ninstance > 0):
            file_id.write(self.log_string)
        file_id.close()
    
class WarningError(CustomError):
    ninstance = 0
    xtype = "WARNING"
    def __init__(self, error_string):
        WarningError.ninstance += 1
        CustomError.__init__(self, error_string)

class FatalError(CustomError):
    ninstance = 0
    xtype = "FATAL"
    def __init__(self, error_string):
        FatalError.ninstance += 1
        CustomError.__init__(self, error_string)


   
