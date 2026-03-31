from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property

import nisarqa

from .geo_product import NisarGeoProduct
from .non_insar_product import NonInsarProduct

objects_to_skip = nisarqa.get_all(name=__name__)


@dataclass
class NonInsarGeoProduct(NonInsarProduct, NisarGeoProduct):
    @cached_property
    def browse_x_range(self) -> tuple[float, float]:
        # All rasters used for the browse should have the same grid specs
        # So, WLOG parse the specs from the first one of them.
        layers = self.get_layers_for_browse()
        freq = next(iter(layers.keys()))
        pol = layers[freq][0]

        with self.get_raster(freq=freq, pol=pol) as img:
            x_start = img.x_start
            x_stop = img.x_stop

        return (x_start, x_stop)

    @cached_property
    def browse_y_range(self) -> tuple[float, float]:
        # All rasters used for the browse should have the same grid specs
        # So, WLOG parse the specs from the first one of them.
        layers = self.get_layers_for_browse()
        freq = next(iter(layers.keys()))
        pol = layers[freq][0]

        with self.get_raster(freq=freq, pol=pol) as img:
            y_start = img.y_start
            y_stop = img.y_stop

        return (y_start, y_stop)

    def _source_data_path(self) -> str:
        """
        Get the path to the sourceData group.

        Returns
        -------
        source_data_path : str
            Path to the sourceData group.
                Standard Format:
                    "/science/<band>/<product_type>/metadata/sourceData"
                Example:
                    "/science/LSAR/GCOV/metadata/sourceData"
        """

        return f"{self._metadata_group_path}/sourceData"

    def _source_data_swaths_path(self) -> str:
        """
        Get the path to the sourceData/swaths group.

        Returns
        -------
        source_data_path : str
            Path to the sourceData group.
                Standard Format:
                    "/science/<band>/<product_type>/metadata/sourceData/swaths"
                Example:
                    "/science/LSAR/GCOV/metadata/sourceData/swaths"
        """

        return f"{self._source_data_path}/swaths"

    def _source_data_swaths_freq_path(self, freq: str) -> str:
        """
        Get the path to the sourceData/swaths/frequencyX group.

        Parameters
        ----------
        freq : str
            Must be either "A" or "B".

        Returns
        -------
        source_data_path : str
            Path to the sourceData group.
                Standard Format:
                    "/science/<band>/<product_type>/metadata/sourceData/swaths"
                Example:
                    "/science/LSAR/GCOV/metadata/sourceData/swaths"
        """
        if freq not in ("A", "B"):
            raise ValueError(f"{freq=}, must be either 'A' or 'B'")

        return f"{self._source_data_swaths_path}/frequency{freq}"


__all__ = nisarqa.get_all(__name__, objects_to_skip)
