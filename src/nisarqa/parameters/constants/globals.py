import numpy as np
from cycler import cycler

# List of first 6 Seaborn colorblind colors:
# Hardcode these so that we do not add another dependency of `seaborn`
# Source for values: output from seaborn.color_palette(palette='colorblind', n_colors=6)
seaborn_colorblind = [(0.00392156862745098, 0.45098039215686275, 0.6980392156862745), 
                      (0.8705882352941177, 0.5607843137254902, 0.0196078431372549),
                      (0.00784313725490196, 0.6196078431372549, 0.45098039215686275),
                      (0.8352941176470589, 0.3686274509803922, 0.0),
                      (0.8, 0.47058823529411764, 0.7372549019607844),
                      (0.792156862745098, 0.5686274509803921, 0.3803921568627451)]
CUSTOM_CYCLER = (cycler(color=seaborn_colorblind) +
                 cycler(linestyle=['-', '-.', '--', (0, (3, 1, 1, 1)), (0, (3, 5, 1, 5, 1, 5)), ':']) +
                 cycler(lw=np.linspace(3,1,6)))

LIST_OF_NISAR_PRODUCTS = ['rslc', 'gslc', 'gcov', 'rifg',
                          'runw', 'gunw', 'roff', 'goff']

NISAR_BANDS = ('LSAR', 'SSAR')

NISAR_FREQS = ('A', 'B')


def get_possible_pols(product_type):
    '''
    Return the possible polarizations for the requested product type.

    Parameters
    ----------
    product_type : str
        One of: 'rslc', 'gslc', 'gcov', 'rifg',
                'runw', 'gunw', 'roff', 'goff'
    Returns
    -------
    polarization_list : tuple of str
        Tuple of all possible polarizations for the given product type.
        Ex: ('HH', 'VV', 'HV', 'VH')

    '''
    if product_type not in LIST_OF_NISAR_PRODUCTS:
        raise ValueError(f'input product type is {product_type}, but '
                         f'must be one of: {LIST_OF_NISAR_PRODUCTS}')
    
    if product_type == 'rslc':
        return ('HH', 'VV', 'HV', 'VH', 'RH', 'RV', 'LH', 'LV')
    else:
        raise NotImplementedError

# Directory Structure and Paths in QA STATS.h5 file
STATS_H5_BASE_GROUP = '/science/%s'
STATS_H5_IDENTIFICATION_GROUP = STATS_H5_BASE_GROUP + '/identification'
processing_group = '/processing'
data_group = '/data'

# QA Directory Structure and Paths in STATS.h5 file
STATS_H5_QA_STATS_H5_BASE_GROUP = STATS_H5_BASE_GROUP + '/QA'
STATS_H5_QA_PROCESSING_GROUP = STATS_H5_QA_STATS_H5_BASE_GROUP + processing_group
STATS_H5_QA_DATA_GROUP = STATS_H5_QA_STATS_H5_BASE_GROUP + data_group
# Frequency group. Note: There are two '%s' here. The first is for the band,
# the second for the frequency.
# Example end result: '/science/%s/QA/data/frequency%s'
STATS_H5_QA_FREQ_GROUP = STATS_H5_QA_DATA_GROUP + '/frequency%s'  # Two '%s' here!

# CalTools
STATS_H5_ABSCAL_STATS_H5_BASE_GROUP = STATS_H5_BASE_GROUP + '/absoluteCalibrationFactor'
STATS_H5_ABSCAL_PROCESSING_GROUP = STATS_H5_ABSCAL_STATS_H5_BASE_GROUP + processing_group
STATS_H5_ABSCAL_DATA_GROUP = STATS_H5_ABSCAL_STATS_H5_BASE_GROUP + data_group

STATS_H5_PTA_STATS_H5_BASE_GROUP = STATS_H5_BASE_GROUP + '/pointTargetAnalyzer'
STATS_H5_PTA_PROCESSING_GROUP = STATS_H5_PTA_STATS_H5_BASE_GROUP + processing_group
STATS_H5_PTA_DATA_GROUP = STATS_H5_PTA_STATS_H5_BASE_GROUP + data_group

STATS_H5_NOISE_EST_STATS_H5_BASE_GROUP = STATS_H5_BASE_GROUP + '/noiseEstimation'
STATS_H5_NOISE_EST_PROCESSING_GROUP = STATS_H5_NOISE_EST_STATS_H5_BASE_GROUP + processing_group
STATS_H5_NOISE_EST_DATA_GROUP = STATS_H5_NOISE_EST_STATS_H5_BASE_GROUP + data_group

# The are global constants and not functions nor classes,
# so manually create the __all__ attribute.
__all__ = [
     'CUSTOM_CYCLER',
     'LIST_OF_NISAR_PRODUCTS',
     'NISAR_BANDS',
     'NISAR_FREQS',
     'get_possible_pols',

     'STATS_H5_BASE_GROUP',
     'STATS_H5_IDENTIFICATION_GROUP',

     'STATS_H5_QA_STATS_H5_BASE_GROUP',
     'STATS_H5_QA_PROCESSING_GROUP',
     'STATS_H5_QA_DATA_GROUP',
     'STATS_H5_QA_FREQ_GROUP',

     'STATS_H5_ABSCAL_STATS_H5_BASE_GROUP',
     'STATS_H5_ABSCAL_PROCESSING_GROUP',
     'STATS_H5_ABSCAL_DATA_GROUP',
     'STATS_H5_PTA_STATS_H5_BASE_GROUP',
     'STATS_H5_PTA_PROCESSING_GROUP',
     'STATS_H5_PTA_DATA_GROUP',
     'STATS_H5_NOISE_EST_STATS_H5_BASE_GROUP',
     'STATS_H5_NOISE_EST_PROCESSING_GROUP',
     'STATS_H5_NOISE_EST_DATA_GROUP'
]
