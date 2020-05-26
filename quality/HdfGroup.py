import h5py

import os
import sys

class HdfGroup(object):

    def __init__(self, hfile, name, group_name):

        self.name = name
        self.group_name = group_name
        self.hfile = hfile
        self.hdf_group = hfile[group_name]

        self.file_list = []
        self.missing = []

        self.default_band = "LSAR"

        print("Initializing %s with %s" % (name, group_name))

    def get(self, name):

        return self.hdf_group[os.path.join(self.group_name, name)]

    def keys(self):

        return self.hdf_group.keys()
    
    def get_dataset_list(self, xml_tree, band):

        self.dset_list = [d.get("name") for d in xml_tree.iter() if ("name" in d.keys())]
        self.dset_list = [d.replace("%s/" % self.group_name, "") for d in self.dset_list if (d.startswith(self.group_name))]

        if (band != self.default_band):
            self.dset_list = [d.replace(self.default_band, band) for d in self.dset_list]

    def verify_dataset_list(self, no_look=[]):

        skip = []
        for s in no_look:
            skip += [d for d in self.dset_list if (s in d)]
        data_check = [d for d in self.dset_list if (d not in skip)]

        for dset in data_check:
            if (dset not in self.hdf_group.keys()):    
                self.missing.append(dset)


        
