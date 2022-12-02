
from dataclasses import dataclass, field, fields
from typing import Union, Iterable, Tuple, Optional, Any


@dataclass
class Param:
    name: str
    value: Any
    description: str = ''''''
    units: Optional[Any] = None
    stats_h5_path: Optional[str] = None # where to store this parameter in the stats.h5 file. If None, will not be stored.


@dataclass
class QACalValParams:
    '''
    Data structure to hold the core parameters for the
    QA code's output plots and statistics files that are
    common to all NISAR products.

    Parameters
    ----------
    plots_pdf : PdfPages
        The output file to append the power image plot to
    stats_h5 : h5py.File
        The output file to save QA metrics, etc. to
    '''

    # Attributes that are common to all NISAR QA Products
    stats_file: str = './stats.h5'
    plots_file: str = './plots.pdf'
    summary_file: str = './summary.csv'

    @classmethod
    def from_parent(cls, parent, **kwargs):
        '''
        Construct a child parameters class from an existing
        instance of its immediate parent class.

        Parameters
        ----------
        parent : parent class
            Instance of the immediate parent class whose attributes will
            populate the new child class instance.
            Note that a only shallow copy is performed when populating
            the new instance; for parent attributes that contain
            references, the child object will reference the same
            same object.
        **kwargs : optional
            Attributes specific to the child class.

        Example
        -------
        >>> parent = CoreQAParams()
        >>> @dataclass
        ... class ChildParams(CoreQAParams):
        ...     x: int
        ... 
        >>> y = ChildParams.from_parent(parent=parent, x=2)
        >>> print(y)
        ChildParams(stats_file='./stats.h5', plots_file='./plots.pdf',
        summary_file='./summary.csv', x=2)        
        '''

        # Get the immediate parent class of this dataclass
        immediate_parent = self.__bases__

        # Only support one base class
        assert len(immediate_parent) == 1, f'{self.__class__.__name__} can only inherit from one base class.'

        if not isinstance(parent, immediate_parent[0]):
            raise ValueError(f'`parent` input must be of type {immediate_parent[0].__name__}.')

        # Create shallow copy of the dataclass into a dict.
        # (Using the asdict() method to create a deep copy throws a
        # "TypeError: cannot serialize '_io.BufferedWriter' object" 
        # exception when copying the field with the PdfPages object.)
        parent_dict = dict((field.name, getattr(parent, field.name)) 
                                            for field in fields(parent))

        return cls(**parent_dict, **kwargs)


@dataclass
class QAParams(QACalValParams):
    '''
    Data structure to hold the core parameters for the
    QA code's output plots and statistics files that are
    common to all NISAR products.

    Parameters
    ----------
    **core : QACalValParams
        All fields from the parent class QACalValParams.
    bands : sequence of str
        A sequence of the bands in the input file,
        Examples: ['LSAR','SSAR'] or ('LSAR')
    browse_image_file : str, optional
        QA Browse image file name.
        To account for when multiple browse images are generated, such as
        when a NISAR product contains multiple polarizations, the QA code
        will temporarily remove the `.png` ending and update browse_image_file
        per this naming convention:
            <prefix>_<product name>_BAND_F_PP_qqq.png
                <prefix>        : `browse_image_prefix`, supplied from SDS
                <product name>  : RSLC, GLSC, etc.
                BAND            : LSAR or SSAR
                F               : frequency A or B 
                PP              : polarization
                qqq             : pow (RSLC's browse images are of power images)
        Defaults to './<product name>_BAND_F_PP_qqq.png' (current working directory)
    tile_shape : tuple of ints, optional
        Shape of each tile to be processed. If `tile_shape` is
        larger than the shape of `arr`, or if the dimensions of `arr`
        are not integer multiples of the dimensions of `tile_shape`,
        then smaller tiles may be used.
        -1 to use all rows / all columns (respectively).
        Format: (num_rows, num_cols) 
        Defaults to (1024, -1) to use all columns (i.e. full rows of data)
        and leverage Python's row-major ordering.
    '''

    # Attributes that are common to all NISAR QA Products
    bands: Iterable[str]
    browse_image_file: str = '.'
    tile_shape: tuple = (1024,-1)

