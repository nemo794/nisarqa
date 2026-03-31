from __future__ import annotations

from dataclasses import dataclass

import h5py

import nisarqa

from .non_insar_geo_product import NonInsarGeoProduct
from .slc_product import SLCProduct

objects_to_skip = nisarqa.get_all(name=__name__)


@dataclass
class GSLC(SLCProduct, NonInsarGeoProduct):
    @property
    def product_type(self) -> str:
        return "GSLC"

    def center_freq(self, freq: str) -> float:
        """
        The processed center frequency for input product's Frequency `freq`.

        For GSLC products, this is read from the `centerFrequency` dataset
        in the grids frequency group.

        Parameters
        ----------
        freq : str
            Must be either "A" or "B".

        Returns
        -------
        center_freq : float
            The processed center frequency for input product's Frequency `freq`,
            in hertz.
        """

    def center_freq(self, freq: str) -> float:
        """
        The processed center frequency for input product's Frequency `freq`.

        For GCOV products, this is read from the `centerFrequency` dataset
        in the metadata sourceData swaths group.

        Parameters
        ----------
        freq : str
            Must be either "A" or "B".

        Returns
        -------
        center_freq : float
            The processed center frequency for input product's Frequency `freq`,
            in hertz.
        """
        freq_group = self.get_freq_path(freq=freq)

        with h5py.File(self.filepath) as f:
            try:
                center_freq = f[freq_group]["centerFrequency"][()]
            except KeyError as e:
                raise nisarqa.DatasetNotFoundError from e

        return center_freq


__all__ = nisarqa.get_all(__name__, objects_to_skip)
