import inspect
import logging
import traceback

class NISARLogger(object):

    def __init__(self, filename):
        self.filename = filename

        fid = logging.FileHandler(filename)
        formatter_str = '%(asctime)s, %(levelname)s, %(pge)s, %(module)s, %(error_code)i' \
                      + '%(source)s:%(line_number)i, "%(error_name)s: %(message)s"'
        formatter = logging.Formatter(formatter_str)
        fid.setFormatter(formatter)

        self.logger = logging.getLogger(name="NISAR")
        self.logger.setLevel(logging.DEBUG)
        self.logger.addHandler(fid)

    def log_message(self, class_filter, message):

        xfilter = class_filter()
        log_funct = getattr(self.logger, xfilter.level)
        
        self.logger.addFilter(xfilter)
        log_funct(message)

    def close(self):
        logging.shutdown()
        
class LogFilterAbstract(object):

    def __init__(self):
        self.extra = {"pge": "QA", "module": "misc"}
        self.call_string = "log_message"

        (source_file, line_no) = self.get_trace()
        self.extra["source"] = source_file
        self.extra["line_number"] = line_no

        #print("source has %i characters: %s" % (len(source_file), source_file))

    def get_trace(self):

        idx = None
        line_no = None
        source_file = None
        
        stack = [str(s) for s in inspect.stack()]
        for (i, s) in enumerate(stack):
            if (self.call_string in s):
                #print("Dissecting trace string %s" % s)
                idx = i
                break

        try:
            assert(idx is not None)
            xstack = stack[idx+1]
        except AssertionError:
            return ("no_such_file.py", -9999)
        except IndexError:
            xstack = stack[idx]
            
        for f in [f.strip() for f in xstack.split(",")]:
            if (f.startswith("line")):
                line_no = int(f.replace("line", ""))
            if (f.startswith("file")):
                source_file = f.replace("file", "").replace('"', "")
            if ( (line_no is not None) and (source_file is not None) ):
                break

        try:
            assert( (source_file is not None) and (line_no is not None) )
        except AssertionError:
            return ("no_such_file.py", -9999)
        else:
            return (source_file, line_no)        

    def filter(self, record):
        for key in self.extra.keys():
            setattr(record, key, self.extra[key])
        return True

class LogFilterDebug(LogFilterAbstract):

    def __init__(self):
        LogFilterAbstract.__init__(self)
        self.level = "debug"
        self.extra["error_code"] = 101000
        self.extra["error_name"] = "N/A"

class LogFilterInfo(LogFilterAbstract):

    def __init__(self):
        LogFilterAbstract.__init__(self)
        self.level = "info"
        self.extra["error_code"] = 100000
        self.extra["error_name"] = "N/A"

class LogFilterWarning(LogFilterAbstract):

    def __init__(self):
        LogFilterAbstract.__init__(self)
        self.level = "warning"
        self.extra["error_code"] = 102000
        self.extra["error_name"] = "Information"

class LogFilterError(LogFilterAbstract):

    def __init__(self):
        LogFilterAbstract.__init__(self)
        self.level = "error"
        self.extra["error_code"] = 103000
        self.extra["error_name"] = "Error"

        
