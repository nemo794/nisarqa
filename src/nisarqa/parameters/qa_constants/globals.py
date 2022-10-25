BANDS = ('LSAR', 'SSAR')

RSLC_FREQS = ('A', 'B')
RSLC_POLS = ('HH', 'VV', 'HV', 'VH')

# The are global constants and not functions nor classes,
# so manually create the __all__ attribute.
__all__ = [
     'BANDS',
     'RSLC_FREQS',
     'RSLC_POLS'
]
