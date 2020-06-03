from quality import errors_base
from quality import errors_derived
from quality.HdfGroup import HdfGroup
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

    def __init__(self, flname, xml_tree=None, mode="r"):
        self.flname = os.path.basename(flname)
        h5py.File.__init__(self, flname, mode)
        print("Opening file %s" % flname)

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
                self.start_time[b] = self["/science/%s/identification/zeroDopplerStartTime" % b][...]
            except KeyError:
                self.start_time[b] = "9999-99-99T99:99:99"
            else:
                break
        
    def junk_get_bands(self):
        
        for band in ("LSAR", "SSAR"):
            try:
                xband = self["/science/%s" % band]
            except KeyError:
                print("%s not present" % band)
                pass
            else:
                print("Found band %s" % band)
                self.bands.append(band)
                traceback_string = []
                error_string = []

                # Determine if data is SLC or RSLC and create HdfGroups.

                name_swath = None
                name_metadata = None
                for (gname, name_found, xdict, dict_name) in zip( (self.swath_path, self.metadata_path), \
                                                                  (name_swath, name_metadata), \
                                                                  (self.SWATHS, self.METADATA), \
                                                                  ("%s Swath" % band, "%s Metadata" % band) ):
                    for dtype in self.ftype_list:
                        try:
                            group = self[gname % (band, dtype)]
                        except KeyError:
                            continue
                        else:
                            if (dict_name.endswith("Swath")):
                                name_swath = dtype
                            elif (dict_name.endswith("Metadata")):
                                name_metadata = dtype
                            name_found = dtype
                            xdict[band] = HdfGroup(self, dict_name, gname % (band, dtype))
                            break

                        
                    try:
                        assert(name_found is not None)
                    except AssertionError as e:
                        gname2 = gname % (band, dtype)
                        if (isinstance(self, SLCFile)):
                            gname2 = gname2.replace("RSLC", "[SLC,RSLC]")
                        elif (isinstance(self, GCOVFile)):
                            gname2 = gname
                        traceback_string = [utility.get_traceback(e, AssertionError)]
                        error_string += ["%s does not exist" % gname2]

                # Check for missing Swath and/or Metadata groups
                        
                assert(len(traceback_string) == len(error_string))
                if (len(error_string) > 0):
                    raise errors_derived.IdentificationFatal(self.flname, self.start_time, traceback_string, \
                                                             error_string)
                        
                # Check that both Swath and Metadata groups are either SLC or RLSC
                    
                try:
                    assert(name_swath == name_metadata)
                    self.ftype = name_swath
                except AssertionError as e:
                    traceback_string = [utility.get_traceback(e, AssertionError)]
                    raise errors_derived.IdentificationFatal(self.flname, self.start_time, traceback_string, \
                                                             ["Metadata and swath have inconsistent naming"])

                # Check that Identification group exists
                
                try:
                    self.IDENTIFICATION[band] = HdfGroup(self, "%s Identification" % band, \
                                                         self.identification_path % band)
                except KeyError as e:
                    traceback_string = [utility.get_traceback(e, KeyError)]
                    raise errors_derived.IdentificationFatal(self.flname, self.start_time, traceback_string, \
                                                             ["File missing identification data for %s." % band])

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
                    print("Found %s Frequency%s" % (b, f))
                    self.FREQUENCIES[b][f] = f2
                    self.polarizations[b][f] = []
                    for p in self.polarization_list:
                        try:
                            p2 = self.FREQUENCIES[b][f][p]
                        except KeyError:
                            pass
                        else:
                            self.polarizations[b][f].append(p)
                            

    def check_freq_pol(self):

        # Check for correct frequencies and polarizations

        error_string = []
        traceback_string = []
         
        for b in self.bands:

            try:
                frequencies = [f.decode() for f in self.IDENTIFICATION[b].get("listOfFrequencies")[...]]
                assert(frequencies in params.FREQUENCIES)
            except KeyError as e:
                traceback_string += [utility.get_traceback(e, KeyError)]
                error_string += ["%s Band is missing frequency list" % b]
            except (AssertionError, TypeError, UnicodeDecodeError) as e:
                traceback_string += [utility.get_traceback(e, AssertionError)]
                error_string += ["%s Band has invalid frequency list" % b]
            else:
                for f in frequencies:
                    try:
                        assert("frequency%s" % f in self.SWATHS[b].keys())
                    except AssertionError as e:
                        traceback_string += [utility.get_traceback(e, AssertionError)]
                        error_string += ["%s Band missing Frequency%s" % (b, f)]

                for f in self.FREQUENCIES[b].keys():
                    try:
                        assert(f in frequencies)
                    except AssertionError as e:
                        traceback_string += [utility.get_traceback(e, AssertionError)]
                        error_string += ["%s Band frequency list missing %s" % (b, f)]


            for f in self.FREQUENCIES[b].keys():
                 try:
                    polarization_ok = False
                    try:
                        polarizations_found = [p.decode() for p in self.FREQUENCIES[b][f].get("listOfPolarizations")[...]]
                    except KeyError as e:
                        traceback_string += [utility.get_traceback(e, KeyError)]
                        error_string += ["%s %s is missing polarization list" % (b, f)]
                        continue

                    for plist in self.polarization_groups:
                        #print("Comparing %s against %s" % (polarizations_found, plist))
                        if (set(polarizations_found) == set(plist)):
                            polarization_ok = True
                            break
                    assert(polarization_ok)
                 except (AssertionError, UnicodeDecodeError) as e:
                    traceback_string += [utility.get_traceback(e, AssertionError)]
                    error_string += ["%s Frequency%s has invalid polarization list: %s" % (b, f, polarizations_found)]

                 else:
                    for p in polarizations_found:
                        try:
                            assert(p in self.FREQUENCIES[b][f].keys())
                        except AssertionError as e:
                            traceback_string += [utility.get_traceback(e, AssertionError)]
                            error_string += ["%s Frequency%s missing polarization %s" % (b, f, p)]

                    plist = [p for p in self.FREQUENCIES[b][f].keys() if p in ("HH", "HV", "VH", "VV")]
                    for p in plist:
                        try:
                            assert(p in polarizations_found)
                        except AssertionError as e:
                            traceback_string += [utility.get_traceback(e, AssertionError)]
                            error_string += ["%s Frequency%s has extra polarization %s" % (b, f, p)]
                            

        assert(len(traceback_string) == len(error_string))
        try:
            assert(len(error_string) == 0)
        except AssertionError as e:
            #print("error_string: %s" % error_string)
            #print("traceback_string: %s" % traceback_string)
            raise errors_derived.IdentificationFatal(self.flname, self.start_time[b], traceback_string, error_string)

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
                    error_string += ["%s %s cannot initialize SLCImage due to missing frequency or spacing" % (b, f)]

        for b in self.bands:
            for f in self.FREQUENCIES[b].keys():
                if ("%s %s" % (b, f) in missing_params):
                    break

                for p in self.polarizations[b][f]:
                    key = "%s %s %s" % (b, f, p)
                    try:
                        if (isinstance(self, SLCFile)):
                            
                            self.images[key] = SLCImage(b, f, p, freq["%s %s" % (b, f)][...], \
                                                        srange["%s %s" % (b, f)][...])
                        elif (isinstance(self.GCOVFile)):
                            self.images[key] = GCOVImage(b, f, p)
                        self.images[key].read(self.FREQUENCIES[b], time_step=time_step, range_step=range_step)
                    except KeyError:
                        missing_images.append(key)
                        
        try:
            assert(len(missing_images) == 0)
            assert(len(missing_params) == 0)
        except AssertionError as e:
            traceback_string = [utility.get_traceback(e, AssertionError)]
            error_string = ""
            if (len(missing_images) > 0):
                error_string += "Missing %i images: %s" % (len(missing_images), missing_images)
            if (len(missing_params) > 0):
                error_string += "Could not initialize %i images" % (len(missing_params), missing_params)
            raise errors_derived.ArrayMissingFatal(self.flname, self.start_time[b], traceback_string, \
                                                   [error_string])
                

    def find_missing_datasets(self):

        for b in self.bands:
            self.SWATHS[b].get_dataset_list(self.xml_tree, b)
            self.METADATA[b].get_dataset_list(self.xml_tree, b)
            self.IDENTIFICATION[b].get_dataset_list(self.xml_tree, b)

        error_string = []
        traceback_string = []
        
        for b in self.bands:
            no_look = []
            for f in ("A", "B"):
                if (f not in self.FREQUENCIES[b].keys()):
                    no_look.append("frequency%s" % f)
                    continue

                try:
                    nsubswaths = self.FREQUENCIES[b][f]["numberOfSubSwaths"][...]
                except KeyError:
                    pass
                else:
                    for isub in range(nsubswaths+1, params.NSUBSWATHS+1):
                        no_look.append("frequency%s/validSamplesSubSwath%i" % (f, isub))
                
                for p in self.polarization_list:
                    if (f in self.FREQUENCIES[b].keys()) and (p not in self.polarizations[b][f]):
                        no_look.append("frequency%s/%s" % (f, p))


            self.SWATHS[b].verify_dataset_list(no_look=no_look)
            self.METADATA[b].verify_dataset_list(no_look=no_look)
            self.IDENTIFICATION[b].verify_dataset_list(no_look=no_look)

            for xdict in (self.SWATHS[b], self.METADATA[b], self.IDENTIFICATION[b]):
                try:
                    assert(len(xdict.missing) == 0)
                except AssertionError as e:
                    traceback_string += [utility.get_traceback(e, AssertionError)]
                    error_string += ["%s missing %i fields: %s" % (xdict.name, len(xdict.missing), \
                                                                   ":".join(xdict.missing))]

            assert(len(error_string) == len(traceback_string))
            try:
                assert(len(error_string) == 0)
            except AssertionError as e:
                #print("Missing %i datasets: %s" % (len(error_string), error_string))
                raise errors_derived.MissingDatasetFatal(self.flname, self.start_time, \
                                                         traceback_string, error_string)
            
                
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
                    identifications[dname].append(self.IDENTIFICATION[b].get(dname)[...])
                except KeyError as e:
                    if (dname != "cycleNumber"):
                        traceback_string.append(utility.get_traceback(e, KeyError))
                        error_string += ["%s missing dataset %s" % (b, dname)]
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
                traceback_string.append(utility.get_traceback(e, AssertionError))
                error_string += ["Values of %s differ between bands" % dname]
                
        # Verify that all identification parameters are correct
            
        try:
            start_time = str(identifications["zeroDopplerStartTime"][0])
            end_time = str(identifications["zeroDopplerEndTime"][0])
            assert(end_time > start_time)
        except IndexError as e:
            pass
        except AssertionError as e:
            error_string += ["Start Time %s not less than End Time %s" % (start_time, end_time)]
            traceback_string.append(utility.get_traceback(e, AssertionError))
            
        try:
            orbit = int(identifications["absoluteOrbitNumber"][0])
            assert(orbit > 0)
        except IndexError as e:
            pass
        except (AssertionError, ValueError) as e:
            error_string += ["Invalid Orbit Number: %i" % orbit]
            traceback_string.append(utility.get_traceback(e, AssertionError))
            
        try:
            track = int(identifications["trackNumber"][0])
            assert( (track > 0) and (track <= params.NTRACKS) )
        except (IndexError, TypeError, ValueError) as e:
            pass
        except AssertionError as e:
            error_string += ["Invalid Track Number: %i" % track]
            traceback_string.append(utility.get_traceback(e, AssertionError))
            
        try:
            frame = int(identifications["frameNumber"][0])
            assert(frame > 0)
        except (IndexError, TypeError, ValueError) as e:
            pass
        except AssertionError as e:
            error_string += ["Invalid Frame Number: %i" % frame]
            traceback_string.append(utility.get_traceback(e, AssertionError))
            
        try:
            cycle = int(identifications["cycleNumber"][0])
            assert(cycle > 0)
        except (IndexError, TypeError, ValueError, KeyError) as e:
            pass
        except AssertionError as e:
            error_string += ["Invalid Cycle Number: %i" % cycle]
            traceback_string.append(utility.get_traceback(e, AssertionError))

        try:
            ptype = str(identifications["productType"][0]).lstrip("b").replace("'", "")
            assert(ptype == self.product_type)
        except IndexError as e:
            pass
        except AssertionError as e:
            error_string += ["Invalid Product Type: %s" % ptype]
            traceback_string.append(utility.get_traceback(e, AssertionError))

        try:
            lookdir = str(identifications["lookDirection"][0])
            assert(str(lookdir).lower() in ("b'left'", "b'right'"))
        except IndexError as e:
            pass
        except AssertionError as e:
            error_string += ["Invalid Look Direction: %s" % lookdir]
            traceback_string.append(utility.get_traceback(e, AssertionError))

        # raise errors if needed

        assert(len(error_string) == len(traceback_string))
            
        try:
            assert(len(error_string) == 0)
        except AssertionError as e:
            raise errors_derived.IdentificationFatal(self.flname, self.start_time[self.bands[0]], \
                                                     traceback_string, error_string)

    def check_nans(self):

        nan_warning = []
        nan_fatal = []
        
        for key in self.images.keys():
            ximg = self.images[key]
            ximg.check_for_nan()
            if (ximg.num_nan > 0):
                if (not ximg.empty):
                    nan_warning.append(key)
                else:
                    nan_fatal.append(key)

        # Raise Warning if any NaNs are found and Fatal if an image is entirely NaN
            
        try:
            assert(len(nan_warning) == 0)
        except AssertionError as e:
            traceback_string = [utility.get_traceback(e, AssertionError)]
            error_string = ["%i images had NaN's:" % len(nan_warning)]
            for key in nan_warning:
                band = self.images[key].band
                error_string[0] += " (%s, %i=%.0f%%)" % (key, self.images[key].num_nan, \
                                                         self.images[key].perc_nan)
            raise errors_derived.NaNWarning(self.flname, self.start_time[band], \
                                            traceback_string, error_string)

        try:
            assert(len(nan_fatal) == 0)
        except AssertionError as e:
            traceback_string = [utility.get_traceback(e, AssertionError)]
            error_string = ["%i images were empty:" % (len(nan_fatal))]
            for key in nan_fatal:
                band = self.images[key].band
                error_string[0] += " (%s)" % key
            raise errors_derived.NaNFatal(self.flname, self.start_time[band], \
                                          traceback_string, error_string)          

    def check_time(self):

        for b in self.bands:

            try:
                time = self.SWATHS[b].get("zeroDopplerTime")[...]
                spacing = self.SWATHS[b].get("zeroDopplerTimeSpacing")[...]
                start_time = self.IDENTIFICATION[b].get("zeroDopplerStartTime")[...]
                end_time = self.IDENTIFICATION[b].get("zeroDopplerEndTime")[...]
            except KeyError:
                pass
            else:

                try:
                    start_time = bytes(start_time).split(b".")[0].decode("utf-8")
                    end_time = bytes(end_time).split(b".")[0].decode("utf-8")
                except UnicodeDecodeError as e:
                    traceback_string = [utility.get_traceback(e, UnicodeDecodeError)]
                    raise errors_derived.IdentificationFatal(self.flname, self.start_time[b], traceback_string, \
                                                         ["%s Start/End Times could not be read." % b])
                else:
                
                    try:
                        time1 = datetime.datetime.strptime(start_time, "%Y-%m-%dT%H:%M:%S")
                        time2 = datetime.datetime.strptime(end_time, "%Y-%m-%dT%H:%M:%S")
                        assert(time2 > time1)
                        assert( (time1.year >= 2000) and (time1.year < 2100) )
                        assert( (time2.year >= 2000) and (time2.year < 2100) )
                    except (AssertionError, ValueError) as e:
                        traceback_string = [utility.get_traceback(e, AssertionError)]
                        raise errors_derived.IdentificationFatal(self.flname, self.start_time[b], traceback_string, \
                                                                 ["%s Invalid Start and End Times" % b])
                    
                    try:
                        utility.check_spacing(self.flname, self.start_time[b], time, spacing, "%s zeroDopplerTime" % b, \
                                              errors_derived.TimeSpacingWarning, errors_derived.TimeSpacingFatal)
                    except (KeyError, errors_base.WarningError, errors_base.FatalError):
                        pass


        try:
            time = self.METADATA[b].get("orbit/time")
        except KeyError:
            contine
        else:
            try:
                utility.check_spacing(self.flname, time[0], time, time[1] - time[0], "%s orbitTime" % b, \
                                      errors_derived.TimeSpacingWarning, errors_derived.TimeSpacingFatal)
            except (errors_base.WarningError, errors_base.FatalError):
                pass
        
    def check_frequencies(self):

        for b in self.bands:
            nfrequencies = len(self.FREQUENCIES[b])
            if (nfrequencies == 2):
                for freq_name in self.frequency_checks:
                    freq = {}
                    for f in list(self.FREQUENCIES[b].keys()):
                        freq[f] = self.FREQUENCIES[b][f].get(freq_name)[...]

                    try:
                        assert(freq["A"] < freq["B"])
                    except AssertionError as e:
                        traceback_string = [utility.get_traceback(e, AssertionError)]
                        raise errors_derived.FrequencyOrderFatal(self.flname, self.start_time[b], traceback_string, \
                                                                 ["%s A=%f not less than B=%f" \
                                                                  % (freq_name, freq["A"], freq["B"])])

    def check_num_subswaths(self, b, f):

        try:
            nsubswath = self.FREQUENCIES[b][f].get("numberOfSubSwaths")[...]
            assert( (nsubswath >= 0) and (nsubswath <= params.NSUBSWATHS) )
        except AssertionError as e:
            traceback_string = [utility.get_traceback(e, AssertionError)]
            raise errors_derived.NumSubswathFatal(self.flname, self.start_time[b], traceback_string, \
                                                  ["%s Frequency%s had invalid number of subswaths: %i" \
                                                   % (b, f, nsubswath)])

        return nsubswath
        


               

