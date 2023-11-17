from itertools import product

import numpy as np
from cycler import cycler

# List of first 6 Seaborn colorblind colors:
# Hardcode these so that we do not add another dependency of `seaborn`
# Source for values: output from seaborn.color_palette(palette='colorblind', n_colors=6)
seaborn_colorblind = [
    (0.00392156862745098, 0.45098039215686275, 0.6980392156862745),
    (0.8705882352941177, 0.5607843137254902, 0.0196078431372549),
    (0.00784313725490196, 0.6196078431372549, 0.45098039215686275),
    (0.8352941176470589, 0.3686274509803922, 0.0),
    (0.8, 0.47058823529411764, 0.7372549019607844),
    (0.792156862745098, 0.5686274509803921, 0.3803921568627451),
]
CUSTOM_CYCLER = (
    cycler(color=seaborn_colorblind)
    + cycler(
        linestyle=[
            "-",
            "-.",
            "--",
            (0, (3, 1, 1, 1)),
            (0, (3, 5, 1, 5, 1, 5)),
            ":",
        ]
    )
    + cycler(lw=np.linspace(3, 1, 6))
)

FIG_SIZE_ONE_PLOT_PER_PAGE = (6.4, 4.8)
FIG_SIZE_TWO_PLOTS_PER_PAGE = (10.0, 4.8)

LIST_OF_NISAR_PRODUCTS = [
    "rslc",
    "gslc",
    "gcov",
    "rifg",
    "runw",
    "gunw",
    "roff",
    "goff",
]

NISAR_BANDS = ("L", "S")
NISAR_FREQS = ("A", "B")

# As of June 2023, baseline for GCOV NISAR products is to only include
# on-diagonal terms. However, the ISCE3 GCOV processor is capable of processing
# both on-diagonal and off-diagonal terms. There is an ongoing push from
# some NISAR Science Team members to include the off-diagonal terms
# in the baseline GCOV product, so P. Rosen requested that QA handle both
# on-diag and off-diag terms. But, they need to be handled differently
# in the code, so have these lists be independent.
GCOV_DIAG_POLS = []
GCOV_OFF_DIAG_POLS = []


def _append_gcov_terms(txrx_iterable):
    # For global GCOV*_POLS variables, we should include all possible
    # polarizations.
    # ISCE3 GCOV processor can output e.g. either HHVV or VVHH, depending
    # on the order the user typed the polarizations into the runconfig.
    # So, use `product()` to list the full matrix, and not only the upper
    # triangle or the lower triangle of the matrix.
    for term in product(txrx_iterable, repeat=2):
        if term[0] == term[1]:
            # On-diag term
            GCOV_DIAG_POLS.append(term[0] + term[1])
        else:
            GCOV_OFF_DIAG_POLS.append(term[0] + term[1])


_append_gcov_terms(("HH", "HV", "VH", "VV"))
_append_gcov_terms(("RH", "RV"))
_append_gcov_terms(("LH", "LV"))


def get_possible_pols(product_type):
    """
    Return all possible polarizations for the requested product type.

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

    """
    if product_type not in LIST_OF_NISAR_PRODUCTS:
        raise ValueError(
            f"input product type is {product_type}, but "
            f"must be one of: {LIST_OF_NISAR_PRODUCTS}"
        )

    if product_type.endswith("slc"):
        return ("HH", "VV", "HV", "VH", "RH", "RV", "LH", "LV")
    elif product_type == "gcov":
        return GCOV_DIAG_POLS + GCOV_OFF_DIAG_POLS
    elif product_type in ("rifg", "runw", "gunw", "roff", "goff"):
        # As of 6/21/2023, baseline for NISAR InSAR mission processing
        # is to produce only HH and/or VV.
        # ISCE3 can also handle HV and/or VH. Compact Pol not supported.
        return ("HH", "VV", "HV", "VH")
    else:
        raise NotImplementedError


# Directory Structure and Paths in QA STATS.h5 file
STATS_H5_BASE_GROUP = "/science/%sSAR"
STATS_H5_IDENTIFICATION_GROUP = STATS_H5_BASE_GROUP + "/identification"
processing_group = "/processing"
data_group = "/data"

# QA Directory Structure and Paths in STATS.h5 file
STATS_H5_QA_STATS_H5_BASE_GROUP = STATS_H5_BASE_GROUP + "/QA"
STATS_H5_QA_PROCESSING_GROUP = (
    STATS_H5_QA_STATS_H5_BASE_GROUP + processing_group
)
STATS_H5_QA_DATA_GROUP = STATS_H5_QA_STATS_H5_BASE_GROUP + data_group
# Frequency group. Note: There are two '%s' here. The first is for the band,
# the second for the frequency.
# Example end result: '/science/%s/QA/data/frequency%s'
STATS_H5_QA_FREQ_GROUP = (
    STATS_H5_QA_DATA_GROUP + "/frequency%s"
)  # Two '%s' here!

# Pol group. Note: There are three '%s' here. The first is for the band,
# the second for the frequency, third for polarization.
# Example end result: '/science/%s/QA/data/frequency%s/%s'
STATS_H5_QA_POL_GROUP = STATS_H5_QA_FREQ_GROUP + "/%s"


# InSAR Products
# Add a place to mirror the groups in INSAR_GROUPS
# Example end result: '/science/%s/QA/data/frequency%s/%s'
STATS_H5_QA_INSAR_GROUP_GROUP = (
    STATS_H5_QA_FREQ_GROUP + "/%s"
)  # Three '%s' here!

# Add a place to mirror the polarizations
# Example end result: '/science/%s/QA/data/frequency%s/%s/%s'
STATS_H5_QA_INSAR_POL_GROUP = (
    STATS_H5_QA_INSAR_GROUP_GROUP + "/%s"
)  # Four '%s' here!

# CalTools
STATS_H5_ABSCAL_STATS_H5_BASE_GROUP = (
    STATS_H5_BASE_GROUP + "/absoluteRadiometricCalibration"
)
STATS_H5_ABSCAL_PROCESSING_GROUP = (
    STATS_H5_ABSCAL_STATS_H5_BASE_GROUP + processing_group
)
STATS_H5_ABSCAL_DATA_GROUP = STATS_H5_ABSCAL_STATS_H5_BASE_GROUP + data_group

STATS_H5_PTA_STATS_H5_BASE_GROUP = STATS_H5_BASE_GROUP + "/pointTargetAnalyzer"
STATS_H5_PTA_PROCESSING_GROUP = (
    STATS_H5_PTA_STATS_H5_BASE_GROUP + processing_group
)
STATS_H5_PTA_DATA_GROUP = STATS_H5_PTA_STATS_H5_BASE_GROUP + data_group

STATS_H5_NOISE_EST_STATS_H5_BASE_GROUP = (
    STATS_H5_BASE_GROUP + "/noiseEstimation"
)
STATS_H5_NOISE_EST_PROCESSING_GROUP = (
    STATS_H5_NOISE_EST_STATS_H5_BASE_GROUP + processing_group
)
STATS_H5_NOISE_EST_DATA_GROUP = (
    STATS_H5_NOISE_EST_STATS_H5_BASE_GROUP + data_group
)

# The are global constants and not functions nor classes,
# so manually create the __all__ attribute.
__all__ = [
    "CUSTOM_CYCLER",
    "LIST_OF_NISAR_PRODUCTS",
    "NISAR_BANDS",
    "NISAR_FREQS",
    "FIG_SIZE_ONE_PLOT_PER_PAGE",
    "FIG_SIZE_TWO_PLOTS_PER_PAGE",
    "get_possible_pols",
    "GCOV_DIAG_POLS",
    "GCOV_OFF_DIAG_POLS",
    "STATS_H5_BASE_GROUP",
    "STATS_H5_IDENTIFICATION_GROUP",
    "STATS_H5_QA_STATS_H5_BASE_GROUP",
    "STATS_H5_QA_PROCESSING_GROUP",
    "STATS_H5_QA_DATA_GROUP",
    "STATS_H5_QA_FREQ_GROUP",
    "STATS_H5_QA_POL_GROUP",
    "STATS_H5_QA_INSAR_GROUP_GROUP",
    "STATS_H5_QA_INSAR_POL_GROUP",
    "STATS_H5_ABSCAL_STATS_H5_BASE_GROUP",
    "STATS_H5_ABSCAL_PROCESSING_GROUP",
    "STATS_H5_ABSCAL_DATA_GROUP",
    "STATS_H5_PTA_STATS_H5_BASE_GROUP",
    "STATS_H5_PTA_PROCESSING_GROUP",
    "STATS_H5_PTA_DATA_GROUP",
    "STATS_H5_NOISE_EST_STATS_H5_BASE_GROUP",
    "STATS_H5_NOISE_EST_PROCESSING_GROUP",
    "STATS_H5_NOISE_EST_DATA_GROUP",
]
