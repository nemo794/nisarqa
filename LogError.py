import errors_base
import errors_derived

import os

class LogError(object):
    
    def __init__(self, flname):
        self.flname = flname
        self.errors_fatal = [e for e in dir(errors_derived) if (e.endswith("Fatal"))]
        self.errors_warning = [e for e in dir(errors_derived) if (e.endswith("Warning"))]
        self.errors_all = self.errors_fatal+self.errors_warning

        self.max_len_errors = 0
        
    def make_header(self, file_list):

        self.file_list = [os.path.basename(f) for f in file_list]
        self.max_len_files = max([len(os.path.basename(f)) for f in file_list])
        for e in self.errors_all:
            xerror = getattr(errors_derived, e)
            self.max_len_errors = max(self.max_len_errors, len(xerror.name))

        self.header = "FILE;".ljust(self.max_len_files)
        
        for e in self.errors_all:
            xerror = getattr(errors_derived, e)
            self.header = "%s %s;" % (self.header, xerror.name.rjust(self.max_len_errors))

        self.fid = open(self.flname, "w")
        self.fid.write("%s\n" % self.header)
        self.fid.close()
            
    def print_error_matrix(self, flname_slc):

        self.fid = open(self.flname, "a+")
        self.fid.write(os.path.basename(flname_slc).ljust(self.max_len_files))
        self.fid.write(";")
        for e in self.errors_all:
            xerror = getattr(errors_derived, e)
            if (os.path.basename(flname_slc) in xerror.file_list):
                self.fid.write("1;".rjust(self.max_len_errors))
            else:
                self.fid.write("0;".rjust(self.max_len_errors))
        self.fid.write("\n")
            
        self.fid.close()

    def print_file_logs(self, flname_slc):

        self.fid = open(self.flname, "w+")
        
        xerror = getattr(errors_base, "CustomError")
        #print("Errors: %s" % xerror.error_strings)
        if (flname_slc in xerror.error_strings.keys()):
            for xline in xerror.error_strings[flname_slc]:
                self.fid.write("%s\n" % xline)

        self.fid.close()
        
    def print_file_logs_bak(self):

        self.fid = open(self.flname, "a+")
        self.fid.write("\n\n")

        for e in self.errors_all:
            xerror = getattr(errors_derived, e)
            self.fid.write("%s: %i total errors\n" % (xerror.name, len(xerror.file_list)))

        for f in self.file_list:
            self.fid.write("\n")
            for e in ("FatalError", "WarningError"):
                xerror = getattr(errors_base, e)
                try:
                    nerrors = xerror.ninstance_file[f]
                    error_string = xerror.log_string[f]
                    self.fid.write("File %s had %i %s errors.\n" % (f, nerrors, xerror.xtype))
                    if (nerrors > 0):
                        self.fid.write(error_string)
                except KeyError:
                    self.fid.write("File %s had 0 %s errors.\n" % (f, xerror.xtype))
                
        self.fid.close()

        

        
