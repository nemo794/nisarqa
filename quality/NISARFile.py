from quality import errors_base
from quality import errors_derived
from quality.HdfGroup import HdfGroup
from quality import logging_base
from quality import params
from quality.SLCImage import SLCImage
from quality import utility

import h5py
from matplotlib import cm, pyplot, ticker
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.font_manager import FontProperties
import numpy as np

import copy
import datetime
import logging
import os
import sys
import tempfile
import traceback

class NISARFile(h5py.File):

    def __init__(self, flname, logger, xml_tree=None, mode="r"):
        self.flname = os.path.basename(flname)
        self.logger = logger
        self.is_open = False
        try:
            h5py.File.__init__(self, flname, mode)
        except:
            self.logger.log_message(logging_base.LogFilterError, "Could not open file %s" % flname)
            return
        else:
            self.logger.log_message(logging_base.LogFilterInfo, "Opening file %s" % flname)
            self.is_open = True

        self.xml_tree = xml_tree
        
        self.bands = []
        self.SWATHS = {}
        self.METADATA = {}
        self.IDENTIFICATION = {}

        self.images = {}
        self.start_time = {}

    def get_start_time(self):
        
        for b in ("LSAR", "SSAR"):
            try:
                xtime = self["/science/%s/identification/zeroDopplerStartTime" % b][...]
                xtime = xtime.item().replace(b"\x00", b"")
                try:
                    self.start_time[b] = xtime.decode("utf-8")
                except (AttributeError, UnicodeDecodeError):
                    self.start_time[b] = xtime
            except KeyError:
                self.start_time[b] = "9999-99-99T99:99:99"

        for b in ("LSAR", "SSAR"):
            self.logger.log_message(logging_base.LogFilterInfo, \
                               "%s Start time %s" % (b, self.start_time[b])) 

    def get_freq_pol(self):

        self.FREQUENCIES = {}
        self.polarizations = {}
        
        for b in self.bands:

            # Find list of frequencies by directly querying dataset

            self.FREQUENCIES[b] = {}
            self.polarizations[b] = {}
            for f in ("A", "B"):
                try:
                    f2 = self.SWATHS[b].get("frequency%s" % (f))
                except KeyError:
                    pass
                else:
                    self.logger.log_message(logging_base.LogFilterInfo, "Found %s Frequency%s" % (b, f))
                    self.FREQUENCIES[b][f] = f2
                    self.polarizations[b][f] = []
                    for p in self.polarization_list:
                        try:
                            p2 = self.FREQUENCIES[b][f][p]
                        except KeyError:
                            pass
                        else:
                            self.polarizations[b][f].append(p)
                            

    def check_freq_pol(self, band, fgroups1, fgroups2, fnames2):

        # Check for correct frequencies and polarizations

        error_string = []
        traceback_string = []
        if (fgroups1 is None):
            fgroups1 = [self.SWATHS]
        if (fgroups2 is None):
            fgroups2 = [self.FREQUENCIES]
            fnames2 = [""]
         
        try:
            frequencies = [f.decode() for f in self.IDENTIFICATION[band].get("listOfFrequencies")[...]]
            assert(frequencies in params.FREQUENCIES)
        except KeyError as e:
            traceback_string += [utility.get_traceback(e, KeyError)]
            self.logger.log_message(logging_base.LogFilterError, "%s is missing frequency list" % band)
        except (AssertionError, TypeError, UnicodeDecodeError) as e:
            traceback_string += [utility.get_traceback(e, AssertionError)]
            self.logger.log_message(logging_base.LogFilterError, "%s has invalid frequency list" % band)
        else:
            for f in frequencies:
                for (flist, fname) in zip(fgroups1, fnames2):
                    try:
                        assert("frequency%s" % f in flist[band].keys())
                    except AssertionError as e:
                        error_string = "%s missing Frequency%s" % (band, f)
                        if (len(fname) > 0):
                            error_string = error_string.replace("Frequency", "%s Frequency" % fname)
                        self.logger.log_message(logging_base.LogFilterError, error_string)

            for flist in fgroups2:
                for f in flist[band].keys():
                    try:
                        assert(f in frequencies)
                    except AssertionError as e:
                        error_string = "%s frequency list missing %s" % (band, f)
                        if (len(fname) > 0):
                            error_string = error_string.replace("Frequency", "%s Frequency" % fname)
                        self.logger.log_message(logging_base.LogFilterError, error_string)

        for (flist, fname) in zip(fgroups2, fnames2):
            for f in flist[band].keys():
                try:
                    polarizations_found = [p.decode().upper() for p in flist[band][f].get("listOfPolarizations")[...]]
                except KeyError as e:
                    error_string = "%s %s is missing polarization list" % (band, f)
                    if (len(fname) > 0):
                        error_string = error_string.replace("Frequency", "%s Frequency" % fname)
                    self.logger.log_message(logging_base.LogFilterError, error_string)
                    continue
                except UnicodeDecodeError as e:
                    error_string = "%s Frequency%s has invalid polarization list: %s"
                    if (len(fname) > 0):
                        error_string = error_string.replace("Frequency", "%s Frequency" % fname)
                    self.logger.log_message(logging_base.LogFilterError, error_string)
                        
                try:
                    polarization_ok = False
                    for plist in self.polarization_groups:
                        if (set(polarizations_found) == set(plist)):
                            polarization_ok = True
                            break
                    assert(polarization_ok)
                except (AssertionError) as e:
                    error_string = "%s Frequency%s has invalid polarization list: %s" \
                                   % (band, f, polarizations_found)
                    if (len(fname) > 0):
                        error_string = error_string.replace("Frequency", "%s Frequency" % fname)
                    self.logger.log_message(logging_base.LogFilterError, error_string)

                else:
                    if ("interferogram" in flist[band][f].keys()):
                        polarization_groups = ["interferogram", "pixelOffsets"]
                        polarization_keys = [flist[band][f]["interferogram"], flist[band][f]["pixelOffsets"]]
                    else:
                        polarization_groups = [""]
                        polarization_keys = [flist[band][f]]

                    for (pgroup, pkey) in zip(polarization_groups, polarization_keys):
                        self.logger.log_message(logging_base.LogFilterInfo, \
                                                "Checking polarization %s in %s" % (pgroup, type(pkey)))
                        for p in polarizations_found:
                            try:
                                assert(p in pkey.keys())
                            except AssertionError as e:
                                error_string = "%s Frequency%s missing polarization %s" % (band, f, p)
                                if (len(fname) > 0):
                                    error_string = error_string.replace("Frequency", "%s Frequency" % fname)
                                if (len(pgroup) > 0):
                                    error_string = error_string.replace("missing", "%s missing" % pgroup)
                                self.logger.log_message(logging_base.LogFilterError, error_string)

                    for (pgroup, pkey) in zip(polarization_groups, polarization_keys):
                        plist = [p for p in pkey.keys() if p in self.polarizations_possible] # ("HH", "HV", "VH", "VV")]
                        for p in plist:
                            try:
                                assert(p in polarizations_found)
                            except AssertionError as e:
                                error_string = "%s Frequency%s has extra polarization %s" % (band, f, p)
                                if (len(fname) > 0):
                                   error_string = error_string.replace("Frequency", "%s Frequency" % fname)
                                if (len(pgroup) > 0):
                                   error_string = error_string.replace("missing", "%s missing" % pgroup) 
                                self.logger.log_message(logging_base.LogFilterError, error_string) 

        return error_string

    def create_images(self, time_step=1, range_step=1):

        missing_images = []
        missing_params = []
        traceback_string = []
        error_string = []

        freq = {}
        srange = {}
        
        for b in self.bands:
            for f in self.FREQUENCIES[b].keys():
                if (not isinstance(self, SLCFile)):
                    continue

                try:
                    freq["%s %s" % (b, f)] = self.FREQUENCIES[b][f].get("processedCenterFrequency")
                    srange["%s %s" % (b, f)] = self.FREQUENCIES[b][f].get("slantRangeSpacing")
                except KeyError:
                    missing_params += ["%s %s" % (b, f)]
                    self.logger.log_message(logging_base.LogError, \
                                            "%s %s cannot initialize SLCImage due to missing " % (b, f) \
                                            + "frequency or spacing")

        for b in self.bands:
            for f in self.FREQUENCIES[b].keys():
                if ("%s %s" % (b, f) in missing_params):
                    break

                for p in self.polarizations[b][f]:
                    key = "%s %s %s" % (b, f, p)
                    try:
                        if (isinstance(self, SLCFile)) or (isinstance(self, GSLCFile)):
                            self.images[key] = SLCImage(b, f, p, freq["%s %s" % (b, f)][...], \
                                                        srange["%s %s" % (b, f)][...])
                        elif (isinstance(self.GCOVFile)):
                            self.images[key] = GCOVImage(b, f, p)
                        self.images[key].read(self.FREQUENCIES[b], time_step=time_step, range_step=range_step)
                    except KeyError:
                        missing_images.append(key)
                        
        try:
            assert(len(missing_images) == 0)
        except AssertionError as e:
            self.logger.log_message(logging_base.LogFilterError, \
                                    "Missing %i images: %s" % (len(missing_images), missing_images))

        return error_string
            #raise errors_derived.ArrayMissingFatal(self.flname, self.start_time[b], traceback_string, \
            #                                       [error_string])
                

    def find_missing_datasets(self, swaths, frequencies):

        if (swaths is None):
            swaths = [self.SWATHS]
        if (frequencies is None):
            frequencies = [self.FREQUENCIES]

        missing_groups = []
        for b in self.bands:
            for gname in ("SWATHS", "METADATA", "IDENTIFICATION"):
                try:
                    group = getattr(self, gname)
                    group[b].get_dataset_list(self.xml_tree, b)
                except KeyError:
                    missing_groups.append(gname)
            #self.SWATHS[b].get_dataset_list(self.xml_tree, b)
            #self.METADATA[b].get_dataset_list(self.xml_tree, b)
            #self.IDENTIFICATION[b].get_dataset_list(self.xml_tree, b)

        error_string = []
        traceback_string = []
        
        for b in self.bands:
            no_look = []
            for f in ("A", "B"):
                if (f not in self.FREQUENCIES[b].keys()):
                    no_look.append("frequency%s" % f)
                    continue

                try:
                    nsubswaths = self.FREQUENCIES[b][f].get("numberOfSubSwaths")[...]
                except (KeyError, ValueError):
                    nsubswaths = 0
                    pass
                else:
                    self.logger.log_message(logging_base.LogFilterDebug, \
                                            "nsubswaths %s = %s" % (type(nsubswaths), nsubswaths))
                    for isub in range(nsubswaths+1, params.NSUBSWATHS+1):
                        no_look.append("frequency%s/validSamplesSubSwath%i" % (f, isub))

                try:
                    found_polarizations = self.polarizations[b][f] + self.component_plist[b][f]
                except AttributeError:
                    found_polarizations = self.polarizations[b][f]

                for p in list(params.GCOV_POLARIZATION_LIST) + list(params.SLC_POLARIZATION_LIST):
                    if (f in self.FREQUENCIES[b].keys()) and (p not in found_polarizations):
                        no_look.append("frequency%s/%s" % (f, p))

            self.logger.log_message(logging_base.LogFilterInfo, "%s: no_look=%s" % (b, no_look))

            for gname in ("SWATHS", "METADATA", "IDENTIFICATION"):
                if (gname in missing_groups):
                    continue
                group = getattr(self, gname)
                group = group[b]
                group.verify_dataset_list(no_look=no_look)
            #self.SWATHS[b].verify_dataset_list(no_look=no_look)
            #self.METADATA[b].verify_dataset_list(no_look=no_look)
            #self.IDENTIFICATION[b].verify_dataset_list(no_look=no_look)

                try:
                    assert(len(group.missing) == 0)
                except AssertionError as e:
                    self.logger.log_message(logging_base.LogFilterWarning, "%s missing %i fields: %s" \
                                            % (group.name, len(group.missing), ":".join(group.missing)))

    def check_identification(self):

        error_string = []
        traceback_string = []

        identifications = {}
        for dname in self.identification_list:
            is_present = True
            identifications[dname] = []
            for b in self.bands:
                if (dname in self.IDENTIFICATION[b].missing):
                    is_present = False
                    continue

                try:
                    xid = np.ndarray.item(self.IDENTIFICATION[b].get(dname)[...])
                    identifications[dname].append(xid)
                except KeyError as e:
                    if (dname != "cycleNumber"):
                        self.logger.log_message(logging_base.LogFilterWarning, \
                                                "%s missing dataset %s" % (b, dname))
                except TypeError as e:
                    if (dname == "cycleNumber"):
                        identifications[dname].append(-9999)

            if (not is_present):
                continue
                        
            try:
                assert( (len(self.bands) == 1) or (identifications[dname][0] == identifications[dname][1]) )
            except IndexError:
                continue
            except AssertionError as e:
                self.logger.log_message(logging_base.LogFilterWarning, \
                                        "Values of %s differ between bands" % dname)
                
        # Verify that all identification parameters are correct
            
        try:
            start_time = str(identifications["zeroDopplerStartTime"][0])
            end_time = str(identifications["zeroDopplerEndTime"][0])
            assert(end_time > start_time)
        except IndexError as e:
            pass
        except AssertionError as e:
            self.logger.log_message(logging_base.LogFilterWarning, \
                                    "Start Time %s not less than End Time %s" % (start_time, end_time))
            traceback_string.append(utility.get_traceback(e, AssertionError))
            
        try:
            orbit = int(identifications["absoluteOrbitNumber"][0])
            assert(orbit > 0)
        except IndexError as e:
            pass
        except (AssertionError, ValueError) as e:
            self.logger.log_message(logging_base.LogFilterWarning, "Invalid Orbit Number: %i" % orbit)
            traceback_string.append(utility.get_traceback(e, AssertionError))
            
        try:
            track = int(identifications["trackNumber"][0])
            assert( (track > 0) and (track <= params.NTRACKS) )
        except (IndexError, TypeError, ValueError) as e:
            pass
        except AssertionError as e:
            self.logger.log_message(logging_base.LogFilterWarning, "Invalid Track Number: %i" % track)
            traceback_string.append(utility.get_traceback(e, AssertionError))
            
        try:
            frame = int(identifications["frameNumber"][0])
            assert(frame > 0)
        except (IndexError, TypeError, ValueError) as e:
            pass
        except AssertionError as e:
            self.logger.log_message(logging_base.LogFilterWarning, "Invalid Frame Number: %i" % frame)
            traceback_string.append(utility.get_traceback(e, AssertionError))
            
        try:
            cycle = int(identifications["cycleNumber"][0])
            assert(cycle > 0)
        except (IndexError, TypeError, ValueError, KeyError) as e:
            pass
        except AssertionError as e:
            self.logger.log_message(logging_base.LogFilterWarning, "Invalid Cycle Number: %i" % cycle)
            traceback_string.append(utility.get_traceback(e, AssertionError))

        try:
            ptype = identifications["productType"][0]
            if (isinstance(ptype, bytes)):
                ptype = ptype.decode("utf-8").replace("\x00", "")
            assert(ptype == self.product_type)
        except IndexError as e:
            pass
        except (UnicodeDecodeError, AssertionError) as e:
            self.logger.log_message(logging_base.LogFilterWarning, "Invalid Product Type: %s" % ptype)
            traceback_string.append(utility.get_traceback(e, AssertionError))

        try:
            lookdir = identifications["lookDirection"][0]
            if (isinstance(lookdir, bytes)):
                lookdir = lookdir.decode("utf-8").replace("\x00", "")
            assert(lookdir.lower() in ("left", "right"))
        except IndexError as e:
            pass
        except (UnicodeDecodeError, AssertionError) as e:
            self.logger.log_message(logging_base.LogFilterWarning, "Invalid Look Direction: %s" % lookdir)
            traceback_string.append(utility.get_traceback(e, AssertionError))

    def check_nans(self):

        self.empty_images = []
        
        for key in self.images.keys():
            ximg = self.images[key]
            ximg.check_for_nan()
            band = ximg.band
            self.logger.log_message(logging_base.LogFilterInfo, "Checking image %s for zeros and NaNs" % key)

            if (not(hasattr(ximg, "num_zero"))):
                ximg.num_zero = 0
            try:
                assert(ximg.num_nan == 0)
                assert(ximg.num_zero == 0)
            except AssertionError as e:
                if (ximg.empty):
                    self.empty_images.append(key)
                    self.logger.log_message(logging_base.LogFilterError, ximg.empty_string)                                
                else:
                    if (ximg.empty):
                        self.logger.log_message(logging_base.LogFilterError, ximg.empty_string)
                    elif (ximg.num_nan > 0) and (ximg.num_zero == 0):
                        self.logger.log_message(logging_base.LogFilterWarning, ximg.nan_string)
                    elif (ximg.num_zero > 0) and (ximg.num_nan == 0):
                        self.logger.log_message(logging_base.LogFilterWarning, ximg.zero_string)
                    elif (ximg.num_zero > 0) and (ximg.num_nan > 0):
                        self.logger.log_message(logging_base.LogFilterWarning, \
                                                "%s:%s" % (ximg.nan_string, ximg.zero_string))

        for key in self.empty_images:
            del self.images[key]

    def check_time(self):

        error_string = []
        
        for b in self.bands:

            try:
                time = self.SWATHS[b].get("zeroDopplerTime")[...]
                spacing = self.SWATHS[b].get("zeroDopplerTimeSpacing")[...]
                start_time = self.IDENTIFICATION[b].get("zeroDopplerStartTime")[...]
                end_time = self.IDENTIFICATION[b].get("zeroDopplerEndTime")[...]
            except KeyError:
                pass
            else:
                self.logger.log_message(logging_base.LogFilterInfo, \
                                   "Start time %s, End time %s" % (start_time, end_time))

                try:
                    start_time = bytes(start_time).split(b".")[0].decode("utf-8").replace("\x00", "")
                    end_time = bytes(end_time).split(b".")[0].decode("utf-8").replace("\x00", "")
                except UnicodeDecodeError as e:
                    traceback_string = [utility.get_traceback(e, UnicodeDecodeError)]
                    self.logger.log_message(logging_base.LogFilterError, \
                                            "%s Start/End Times could not be read." % b)
                else:

                    tformats = ("%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M:%S:%f")
                    for (i, tform) in enumerate(tformats):
                        try:
                            time1 = datetime.datetime.strptime(start_time, tform)
                            time2 = datetime.datetime.strptime(end_time, tform)
                        except ValueError as e:
                            self.logger.log_message(logging_base.LogFilterDebug, \
                                               "Time format conversion failed for iteration %i" % i)
                            if ((i+1) == len(tformats)):
                                traceback_string = [utility.get_traceback(e, ValueError)]
                                self.logger.log_message(logging_base.LogFilterError, \
                                                        "%s Invalid Start and End Times" % b)
                        else:
                            break
                        
                    try:
                        assert( (time1.year >= 2000) and (time1.year < 2100) )
                        assert( (time2.year >= 2000) and (time2.year < 2100) )
                    except AssertionError as e:
                        traceback_string = [utility.get_traceback(e, AssertionError)]
                        self.logger.log_message(logging_base.LogFilterError, \
                                                "%s End Time < Start Time" % b)
                    
                    error_string = utility.check_spacing(self.flname, self.start_time[b], time, spacing, \
                                                         "%s zeroDopplerTime" % b)
                    if (len(error_string) > 0):
                        for e in error_string:
                            self.logger.log_message(logging_base.LogFilterWarning, e)
                    #try:
                    #    assert(len(error_string) == 0)
                    #except AssertionError:
                    #    logging.log_message(logging_base.LogFilterWarning, "%s" % error_string)

            try:
                time = self.METADATA[b].get("orbit/time")
            except KeyError:
                continue
            else:
                error_string = utility.check_spacing(self.flname, time[0], time, time[1] - time[0], \
                                                     "%s orbitTime" % b)
                if (len(error_string) > 0):
                    for e in error_string:
                        self.logger.log_message(logging_base.LogFilterWarning, e)
                
        
    def check_frequencies(self, band, flist):

        self.logger.log_message(logging_base.LogFilterInfo, \
                                "Checking frequencies for flist: %s" % flist)

        nfrequencies = len(flist)
        if (nfrequencies == 1):
            return

        for freq_name in self.frequency_checks:
            freq = {}
            try:
                for f in list(flist.keys()):
                    xfreq = flist[f].get(freq_name, default=None)
                    assert(xfreq is not None)
                    freq[f] = xfreq[...]
            except AssertionError as e:
                self.logger.log_message(logging_base.LogFilterWarning, \
                                        "%s Frequency%s missing dataset %s" % (band, f, freq_name))
                    
            try:
                assert(freq["A"] < freq["B"])
            except AssertionError as e:
                self.logger.log_message(logging_base.LogFilterWarning, \
                                        "%s A=%f not less than B=%f" % (freq_name, freq["A"], freq["B"]))
                
    def check_num_subswaths(self, b, f):

        try:
            nsubswath = self.FREQUENCIES[b][f].get("numberOfSubSwaths")[...]
            assert( (nsubswath >= 0) and (nsubswath <= params.NSUBSWATHS) )
        except AssertionError as e:
            traceback_string = [utility.get_traceback(e, AssertionError)]
            self.logger.log_message(logging_base.LogFilterError, 
                                    "%s Frequency%s had invalid number of subswaths: %i" \
                                    % (b, f, nsubswath))

        return nsubswath
        


               

