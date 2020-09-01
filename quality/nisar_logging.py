import logging

class NISARLogger(object):

    def __init__(self, filename):
        self.filename = filename

        fid = logging.FileHandler(filename)
        formatter_str = '%(asctime)s, %(levelname)s, %(pge)s, %(module)s, %(error_code)i, \
                         %(pathname)s:%(lineno)i, "%(message)s"'
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
        
class LogFilterAbstract(object):

    def __init__(self):
        self.extra = {"pge": "QA", "module": "misc"}

    def filter(self, record):
        for key in self.extra.keys():
            setattr(record, key, self.extra[key])
        return True

class LogFilterDebug(LogFilterAbstract):

    def __init__(self):
        self.level = "debug"
        self.extra["error_code"] = 101000

class LogFilterInfo(LogFilterAbstract):

    def __init__(self):
        self.level = "info"
        self.extra["error_code"] = 100000

class LogFilterWarning(LogFilterAbstract):

    def __init__(self):
        self.level = "warning"
        self.extra["error_code"] = 102000

class LogFilterError(LogFilterAbstract):

    def __init__(self):
        self.level = "error"
        self.extra["error_code"] = 103000

        
