
import os

class CustomError(Exception):
    #traceback_string = {}
    error_strings = {}
    spacing = "    "

    NFIELDS = 7
    DELIMITER = ";"
    PGE = "Q/A"
    MODULE = "Verify_slc"

    def __init__(self, flname, start_time, ename, error_traceback, error_description):

        #print("Logging error %s: %s" % (ename, error_description))
        self.retrieve_errors(flname, start_time, ename, error_traceback, error_description)

    @classmethod
    def retrieve_errors(cls, flname, start_time, ename, error_traceback, error_description):
        if (flname not in cls.error_strings.keys()):
            cls.error_strings[flname] = []

        estring = cls.retrieve_error_string(start_time, ename, error_traceback, error_description)
        cls.error_strings[flname] += estring
        
    @classmethod
    def log_error(cls, flname, error_traceback, error_description):

        cls.ninstance_total += 1
        if (flname not in cls.ninstance_file.keys()):
            cls.ninstance_file[flname] = 0
        cls.ninstance_file[flname] += 1

        assert( (flname in cls.error_description.keys()) == (flname in cls.traceback_string.keys()) )
        #if (flname not in cls.error_description.keys()):
        #    cls.error_description[flname] = []
        #    cls.traceback_string[flname] = []

        #cls.error_description[flname].append(error_description)
        #cls.traceback_string[flname].append(error_description)

    @classmethod
    def retrieve_error_string(cls, start_time, ename, traceback, description):

        estrings = []
        assert(len(traceback) == len(description))
        
        for (t, d) in zip(traceback, description):
        
            estring = "%s" % start_time

            if (ename.startswith("Fatal")):
                estring = "%s%s%s" % (estring, cls.DELIMITER, "Fatal")
            elif (ename.startswith("Warning")):
                estring = "%s%s%s" % (estring, cls.DELIMITER, "Warning")

            estring = "%s%s%s%s%s" % (estring, cls.DELIMITER, cls.PGE, cls.DELIMITER, cls.MODULE)
            estring = "%s%s%s" % (estring, cls.DELIMITER, ename)

            tstring = ""
            for t2 in t:
                if (t2.lstrip().startswith("File")) and ("line" in t2):
                    fields = t2.split(",")
                    fname = os.path.basename(fields[0].split('"')[1])
                    line = int(fields[1].replace("line", ""))
                    tstring = "%s Line %i" % (fname, line)
                    break

            assert(len(tstring) > 0)
            estring = "%s%s%s" % (estring, cls.DELIMITER, tstring)
            estring = "%s%s%s" % (estring, cls.DELIMITER, d)

            assert(len(estring.split(cls.DELIMITER)) == cls.NFIELDS)
            estrings.append(estring)
            
        return estrings

class WarningError(CustomError):
    error_string = {}
    ninstance_file = {}
    ninstance_total = 0
    xtype = "WARNING"
    def __init__(self, flname, start_time, ename, error_traceback, error_description):
        CustomError.__init__(self, flname, start_time, ename, error_traceback, error_description)

    def reset(self):
        WarningError.ninstance_file = 0 

class FatalError(CustomError):
    error_string = {}
    ninstance_file = {}
    ninstance_total = 0
    xtype = "FATAL"
    def __init__(self, flname, start_time, ename, error_traceback, error_description):
        CustomError.__init__(self, flname, start_time, ename, error_traceback, error_description)
        
    def reset(self):
        FatalError.ninstance_file = 0 






   
