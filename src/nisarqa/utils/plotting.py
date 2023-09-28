from __future__ import annotations

import os
from collections.abc import Iterable, Sequence
from dataclasses import fields, replace
from fractions import Fraction
from typing import Optional, overload

import numpy as np
import numpy.typing as npt
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import hsv_to_rgb
from matplotlib.ticker import FuncFormatter

import nisarqa

# List of objects from the import statements that
# should not be included when importing this module
objects_to_skip = nisarqa.get_all(name=__name__)


def make_hsi_browse_wrapped(
    product: nisarqa.WrappedGroup,
    params: nisarqa.HSIImageParamGroup,
    browse_png: str | os.PathLike,
) -> None:
    """
    Create and save HSI wrapped interferogram browse PNG for input product.

    Parameters
    ----------
    product : nisarqa.WrappedGroup
        Input NISAR product.
    params : nisarqa.HSIImageParamGroup
        A structure containing the parameters for creating the HSI image.
    browse_png : path-like
        Filename (with path) for the browse image PNG.
    """
    freq, pol = product.get_browse_freq_pol()

    with product.get_wrapped_igram(
        freq=freq, pol=pol
    ) as igram_r, product.get_wrapped_coh_mag(freq=freq, pol=pol) as coh_r:
        rgb_img, _ = make_hsi_raster(
            phs_or_complex_raster=igram_r,
            coh_raster=coh_r,
            equalize=params.equalize_browse,
            rewrap=None,
            longest_side_max=params.longest_side_max,
        )

    nisarqa.rslc.plot_to_rgb_png(
        red=rgb_img.data[:, :, 0],
        green=rgb_img.data[:, :, 1],
        blue=rgb_img.data[:, :, 2],
        filepath=browse_png,
    )


def make_hsi_browse_unwrapped(
    product: nisarqa.UnwrappedGroup,
    params: nisarqa.UNWHSIImageParamGroup,
    browse_png: str | os.PathLike,
) -> None:
    """
    Create and save HSI unwrapped phase image browse png for input product.

    Parameters
    ----------
    product : nisarqa.UnwrappedGroup
        Input NISAR product.
    params : nisarqa.UNWHSIImageParamGroup
        A structure containing the parameters for creating the HSI image.
    browse_png : path-like
        Filename (with path) for the browse image PNG.
    """
    freq, pol = product.get_browse_freq_pol()

    with product.get_unwrapped_phase(
        freq=freq, pol=pol
    ) as phs_r, product.get_unwrapped_coh_mag(freq=freq, pol=pol) as coh_r:
        rgb_img, _ = make_hsi_raster(
            phs_or_complex_raster=phs_r,
            coh_raster=coh_r,
            equalize=params.equalize_browse,
            rewrap=params.rewrap,
            longest_side_max=params.longest_side_max,
        )
    nisarqa.rslc.plot_to_rgb_png(
        red=rgb_img.data[:, :, 0],
        green=rgb_img.data[:, :, 1],
        blue=rgb_img.data[:, :, 2],
        filepath=browse_png,
    )


def hsi_images_to_pdf_wrapped(
    product: nisarqa.WrappedGroup,
    report_pdf: PdfPages,
) -> None:
    """
    Create HSI wrapped interferogram images and save to PDF.

    Parameters
    ----------
    product : nisarqa.WrappedGroup
        Input NISAR product.
    report_pdf : PdfPages
        The output PDF file to append the HSI image plot to.
    """
    for freq in product.freqs:
        for pol in product.get_pols(freq=freq):
            with product.get_wrapped_igram(
                freq=freq, pol=pol
            ) as igram_r, product.get_wrapped_coh_mag(
                freq=freq, pol=pol
            ) as coh_r:
                # Create a *Raster with the HSI image

                # The HSI colorbar generated in the HSI PDF function (below)
                # has a linear scale from [0, 1] for the intensity channel
                # (coherence magnitude layer). If we equalize that channel,
                # then the colorbar scale would be inaccurate.
                # So, ensure equalize=False when creating the HSI Raster.
                rgb_img, cbar_min_max = make_hsi_raster(
                    phs_or_complex_raster=igram_r,
                    coh_raster=coh_r,
                    equalize=False,  # Do not equalize the PDF HSI images
                    rewrap=None,
                    longest_side_max=None,  # Unnecessary for the PDF
                )

            product.save_hsi_img_to_pdf(
                img=rgb_img,
                report_pdf=report_pdf,
                cbar_min_max=cbar_min_max,
                plot_title_prefix="Wrapped Phase Image and Coherence Magnitude",
            )


def hsi_images_to_pdf_unwrapped(
    product: nisarqa.UnwrappedGroup,
    report_pdf: PdfPages,
    rewrap: Optional[float | int] = None,
) -> None:
    """
    Create HSI unwrapped phase images and save to PDF.

    Parameters
    ----------
    product : nisarqa.UnwrappedGroup
        Input NISAR product.
    report_pdf : PdfPages
        The output PDF file to append the HSI image plot to.
    rewrap : float or int or None, optional
        The multiple of pi to rewrap the unwrapped phase image when generating
        the HSI image(s). If None, no rewrapping will occur.
        Ex: If 3 is provided, the image is rewrapped to the interval [0, 3pi).
    """
    for freq in product.freqs:
        for pol in product.get_pols(freq=freq):
            with product.get_unwrapped_phase(
                freq=freq, pol=pol
            ) as phs_r, product.get_unwrapped_coh_mag(
                freq=freq, pol=pol
            ) as coh_r:
                # The HSI colorbar generated in the HSI PDF function (below)
                # has a linear scale from [0, 1] for the intensity channel
                # (coherence magnitude layer). If we equalize that channel,
                # then the colorbar scale would be inaccurate.
                # So, ensure equalize=False when creating the HSI Raster.
                rgb_img, cbar_min_max = make_hsi_raster(
                    phs_or_complex_raster=phs_r,
                    coh_raster=coh_r,
                    equalize=False,  # Do not equalize the PDF HSI images
                    rewrap=rewrap,
                    longest_side_max=None,  # Unnecessary for the PDF
                )

            product.save_hsi_img_to_pdf(
                img=rgb_img,
                report_pdf=report_pdf,
                cbar_min_max=cbar_min_max,
                plot_title_prefix=(
                    "Unwrapped Phase Image and Coherence Magnitude"
                ),
            )


def make_hsi_as_rgb_img(
    phase_img: np.ndarray,
    coh_mag: np.ndarray,
    phs_img_range: Optional[Sequence[float | int]] = None,
    equalize: bool = False,
) -> np.ndarray:
    """
    Create HSI interferogram image array, returned in RGB colorspace.

    The phase image and coherence magnitude rasters are first processed into
    the HSI (Hue, Saturation, Intensity) colorspace, which is then converted
    to RGB values in normalized range [0, 1].

    If any input layer was NaN-valued for a given pixel, then the output
    value for that pixel will be NaN (for all channels).

    TODO: This algorithm currently uses the built-in matplotlib.hsv_to_rgb()
    to convert to RGB due to delivery schedule constraints.
    But, there is a subtle mathematical difference between the HSI and HSV
    colorspaces, and we should definitely use HSI (not HSV).
    This will require a custom implementation of HSI->RGB.

    Parameters
    ----------
    phase_img : numpy.ndarray
        2D array of a phase image (e.g. `unwrappedInterferogram`).
        This should contain real-valued pixels; if your raster is a complex
        valued interferogram, please compute the phase (e.g. use np.angle())
        and then pass in the resultant raster for `phase_img`.
    coh_mag : numpy.ndarray
        2D raster of `coherenceMagnitude` layer corresponding to `phase_img`.
        This should already be normalized to range [0, 1]. (Otherwise
        something must have gone rather wrong in the InSAR processing!)
    phs_img_range : pair of float or None, optional
        The expected range for the phase image raster. `None` to use the
        min and max of the image data. This will be used for normalization
        of the data to range [0, 1].
        Note: If `phs_img`'s data is within a known range, it is strongly
        suggested to set this parameter, otherwise unintended image correction
        will occur.
        For example, if an image was generated via np.angle(), the known range
        is [-pi, pi]. But if the actual image data only contains values from
        [-pi/2, pi/3] and `None` was provided, then the phase image will appear
        "stretched", because during normalization -pi/2 -> 0 and pi/3 -> 1.
        Defaults to None.
    equalize : bool, optional
        True to perform histogram equalization on the Intensity channel
        (the coherence magnitude layer) for the HSI image.
        See: https://scikit-image.org/docs/stable/auto_examples/color_exposure/plot_equalize.html
        Default is False.

    Returns
    -------
    rgb : numpy.ndarray
        3D array of HSI image converted to the RGB (Red Green Blue) colorspace.
        This RGB array is ready for plotting in e.g. matplotlib.
    """
    if not np.issubdtype(phase_img.dtype, np.floating):
        raise TypeError(
            f"`phase_img` has type {type(phase_img)}, must be float. "
            "Hint: If complex, perhaps compute numpy.angle(phase_img) "
            "before passing in the array."
        )

    if phase_img.ndim != 2:
        raise ValueError(
            f"`phase_img` has shape {np.shape(phase_img)}"
            "but must be a 2D array."
        )

    if coh_mag.ndim != 2:
        raise ValueError(
            f"`coh_mag` has shape {np.shape(coh_mag)} but must be a 2D array."
        )

    if np.shape(phase_img) != np.shape(coh_mag):
        raise ValueError(
            f"`phase_img` (shape: {np.shape(phase_img)}) and `coh_mag` "
            f"(shape: {np.shape(coh_mag)}) must have the same shape."
        )

    # coh mag should already be within range [0, 1].
    if np.any(coh_mag < 0.0) or np.any(coh_mag > 1.0):
        raise ValueError("`coh_mag` contains values outside of range [0, 1].")

    # Initialize HSI array
    # Note: per hsv_to_rgb(): "All values assumed to be in range [0, 1]".
    # So, we'll need to normalize all rasters added to this array.
    hsi = np.ones((phase_img.shape[0], phase_img.shape[1], 3), dtype=np.float32)

    # First channel is hue. The phase image should be represented by hue.
    # Note: If available, make sure to specify the known min/max range for
    # the image, such as [-pi, pi] for images created with np.angle().
    # Otherwise, `nisarqa.normalize()` will take the min and max of the actual
    # data, which could have the undesirable effect of applying image
    # correction to e.g. phase images.
    hsi[:, :, 0] = nisarqa.normalize(arr=phase_img, min_max=phs_img_range)

    # Second channel is saturation. We set it to 1 always.
    # Note: Nothing more to do --> It was previously initialized to 1.

    # Third channel is intensity scaled between 0, 1.
    # If the user desires, equalize histogram
    if equalize:
        # image_histogram_equalization handles normalization
        hsi[:, :, 2] = image_histogram_equalization(image=coh_mag)
    else:
        # coh mag is already within range [0, 1].
        hsi[:, :, 2] = coh_mag

    # The input arrays may contain some nan values.
    # But, we need to set them to zero for generating the HSI array.
    # So, create a temporary mask of all NaN pixels, do the computation,
    # and then set those pixels back to NaN for the final image.

    # nan_mask: True where pixels are non-finite, False where pixels are finite
    nan_mask = ~np.isfinite(phase_img) | ~np.isfinite(coh_mag)

    # For the HSI-> RGB conversion, replace all non-finite values with 0
    hsi[nan_mask] = 0.0

    # For plotting the image, convert to RGB color space
    # TODO - We need to use HSI not HSV. There is a subtle mathematical
    # difference between the two colorspaces.
    # However, this will require a custom implementation of HSI->RGB.
    rgb = hsv_to_rgb(hsi)

    # Restore the NaN values
    rgb[nan_mask] = np.nan

    return rgb


@overload
def make_hsi_raster(
    phs_or_complex_raster: nisarqa.RadarRaster,
    coh_raster: nisarqa.RadarRaster,
    equalize: bool,
    rewrap: Optional[float] = None,
    longest_side_max: Optional[int] = None,
) -> tuple[nisarqa.RadarRaster, list[float, float]]:
    ...


@overload
def make_hsi_raster(
    phs_or_complex_raster: nisarqa.GeoRaster,
    coh_raster: nisarqa.GeoRaster,
    equalize: bool,
    rewrap: Optional[float] = None,
    longest_side_max: Optional[int] = None,
) -> tuple[nisarqa.GeoRaster, list[float, float]]:
    ...


def make_hsi_raster(
    phs_or_complex_raster,
    coh_raster,
    equalize,
    rewrap=None,
    longest_side_max=None,
):
    """
    Create HSI interferogram *Raster with square pixels, and colorbar range.

    Parameters
    ----------
    phs_or_complex_raster : nisarqa.GeoRaster | nisarqa.RadarRaster
        *Raster of complex interferogram or unwrapped phase data to use to
        construct the Hue layer for the HSI *Raster.
        If *Raster is complex valued, numpy.angle() will be used to compute
        the phase image (float-valued).
        This should correspond to `coh_raster`.
    coh_raster : nisarqa.GeoRaster | nisarqa.RadarRaster
        *Raster for the coherence magnitude raster to use to construct
        the intesity layer for the HSI *Raster.
        This should correspond to `phs_or_complex_raster`.
    equalize : bool, optional
        True to perform histogram equalization on the Intensity channel
        (the coherence magnitude layer) for the HSI image.
        See: https://scikit-image.org/docs/stable/auto_examples/color_exposure/plot_equalize.html
        Default is False.
    rewrap : float or int or None, optional
        The multiple of pi to rewrap the unwrapped phase image when generating
        the HSI image(s). If None, no rewrapping will occur.
        Ex: If 3 is provided, the image is rewrapped to the interval [0, 3pi).
    longest_side_max : int or None, optional
        Decimate the generated HSI raster so that the max length of
        axis 0 and axis 1 in `hsi_raster` is `longest_side_max`.

    Returns
    -------
    hsi_raster : nisarqa.GeoRaster | nisarqa.RadarRaster
        A *Raster of the HSI image converted to RGB color space. This raster
        will have square pixels. (If the rasters in `phs_or_complex_raster` and
        `coh_raster` do not have square pixels, then decimation will be applied
        to achieve square pixels.)
        The type of `hsi_raster` will be the same as the type of
        `phs_or_complex_raster` and `coh_raster`.
    cbar_min_max : pair of float
        The suggested range to use for the Hue axis of the
        HSI colorbar for `hsi_raster`.
        If `phs_or_complex_raster` has complex valued data, then `cbar_min_max`
        will be the range [-pi, +pi].
    """
    # Input validation:
    # Check that phs and coh *Raster instances have consistent metadata
    for phs, coh in zip(fields(phs_or_complex_raster), fields(coh_raster)):
        phs_val = getattr(phs_or_complex_raster, phs.name)
        coh_val = getattr(coh_raster, coh.name)

        if phs.name == "data":
            # raster data layers should have the same shape
            assert np.shape(phs_val) == np.shape(coh_val)
        elif phs.name == "name":
            # "name" dataclass attributes should be the same
            # except for the final layer name
            assert phs_val.split("_")[:-1] == coh_val.split("_")[:-1]
        elif isinstance(phs_val, str):
            assert phs_val == coh_val
        else:
            assert np.abs(phs_val - coh_val) < 1e-6
    # END validation check

    phs_img = phs_or_complex_raster.data[...]

    if np.issubdtype(phs_img.dtype, np.complexfloating):
        # complex data; take the phase angle.
        phs_img = np.angle(phs_img.data)

        # np.angle() returns output in range [-pi, pi]
        # So, set the colobar's min and max to be the range [-pi, +pi].
        cbar_min_max = [-np.pi, np.pi]

        # Helpful hint for user!
        if rewrap is not None:
            raise RuntimeWarning(
                "Input raster has a complex dtype (implying a wrapped"
                f" interferogram), but input parameter {rewrap=}. `rewrap` is"
                " only used in the case of real-valued data (implying an"
                " unwrapped phase image). Please check that this is intended."
            )

    else:
        # unwrapped phase image
        if rewrap is None:
            # TODO - look into adding a "percentile_for_clipping" option
            cbar_min_max = [
                np.nanmin(phs_img),
                np.nanmax(phs_img),
            ]
        else:
            # `rewrap` is a multiple of pi. Convert to the full float value.
            rewrap_final = rewrap * np.pi

            # The sign of the output of the modulo operator
            # is the same as the sign of `rewrap_final`.
            # This means that it will always put the output
            # into range [0, <rewrap_final>]
            phs_img %= rewrap_final
            cbar_min_max = [0, rewrap_final]

    coh_img = coh_raster.data[...]

    rgb = nisarqa.make_hsi_as_rgb_img(
        phase_img=phs_img,
        coh_mag=coh_img,
        phs_img_range=cbar_min_max,
        equalize=equalize,
    )

    # Square the pixels. Decimate if requested.
    y_axis_spacing = phs_or_complex_raster.y_axis_spacing
    x_axis_spacing = phs_or_complex_raster.x_axis_spacing

    if longest_side_max is None:
        # Update to be the longest side of the array. This way no downsizing
        # of the image will occur, but we can still output square pixels.
        longest_side_max = max(np.shape(rgb)[:2])

    ky, kx = nisarqa.compute_square_pixel_nlooks(
        img_shape=np.shape(rgb)[:2],  # only need the x and y dimensions
        sample_spacing=[y_axis_spacing, x_axis_spacing],
        longest_side_max=longest_side_max,
    )
    rgb = rgb[::ky, ::kx]

    # Update the ground spacing so that the new *Raster we are building will
    # have correct metadata.
    y_axis_spacing = y_axis_spacing * ky
    x_axis_spacing = x_axis_spacing * kx

    # Construct the name
    # (Remove the layer name from the `name`)
    name = "_".join(phs_or_complex_raster.name.split("_")[:-1])
    if rewrap:
        pi_unicode = "\u03c0"
        name += f" - rewrapped to [0, {rewrap}{pi_unicode})"

    # Construct the HSI *Raster object
    if isinstance(phs_or_complex_raster, nisarqa.RadarRaster):
        hsi_raster = replace(
            phs_or_complex_raster,
            data=rgb,
            name=name,
            ground_az_spacing=y_axis_spacing,
            ground_range_spacing=x_axis_spacing,
        )
    elif isinstance(phs_or_complex_raster, nisarqa.GeoRaster):
        hsi_raster = replace(
            phs_or_complex_raster,
            data=rgb,
            name=name,
            y_spacing=y_axis_spacing,
            x_spacing=x_axis_spacing,
        )
    else:
        raise TypeError(
            f"Input rasters have type {type(phs_img)}, but must be either"
            " nisarqa.RadarRaster or nisarqa.GeoRaster."
        )

    return hsi_raster, cbar_min_max


def img2pdf_hsi(
    img_arr: npt.ArrayLike,
    plots_pdf: PdfPages,
    cbar_min_max: Optional[Sequence[float]] = None,
    title: Optional[str] = None,
    xlim: Optional[Sequence[float | int]] = None,
    ylim: Optional[Sequence[float | int]] = None,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
) -> None:
    """
    Plot image array with a linear HSI "colorbar", then append to PDF.

    Parameters
    ----------
    img_arr : array_like
        Image to plot; image should represent an HSI (Hue,
        Saturation, Intensity) image that has been already converted to
        RGB colorspace (such as via matplotlib.colors.hsv_to_rgb()).
    plots_pdf : PdfPages
        The output PDF file to append the HSI image plot to.
    cbar_min_max : pair of float or None, optional
        The range for the Hue axis of the HSI colorbar for the image raster.
        `None` to use the min and max of the image data for the colorbar range.
        Defaults to None.
    title : str, optional
        The full title for the plot.
    xlim, ylim : sequence of numeric, optional
        Lower and upper limits for the axes ticks for the plot.
        Format: xlim=[<x-axis lower limit>, <x-axis upper limit>]
                ylim=[<y-axis lower limit>, <y-axis upper limit>]
    xlabel, ylabel : str, optional
        Axes labels for the x-axis and y-axis (respectively).
    """
    fig = plt.figure(constrained_layout=True)

    if title is not None:
        fig.suptitle(title)

    # Create two subplots; one to hold the actual raster image, the second
    # (smaller one) for the HSI "colorbar".
    # (Use add_subplot() to force the colorbar to be tall and skinny.)
    ax = fig.add_gridspec(5, 6)
    ax1 = fig.add_subplot(ax[:, :-2])
    ax2 = fig.add_subplot(ax[1:4, -1])

    # Set all NaN pixels to 1 in each of the red-green-blue layers.
    # This way, the NaN pixels will appear white in the PDF.
    img_to_plot = img_arr.copy()
    img_to_plot[~np.isfinite(img_arr)] = 1

    # Decimate image to a size that fits on the axes without interpolation
    # and without making the size (in MB) of the PDF explode.

    # Get size of ax1 window in inches
    bbox = ax1.get_window_extent().transformed(fig.dpi_scale_trans.inverted())

    height, width = np.shape(img_to_plot)[:2]
    if height >= width:
        # In this conditional, the image is taller than it is wide.
        # So, "shrink" the image to the height of the axis.
        # (Multiply by fig.dpi to convert from inches to pixels.)
        desired_longest = bbox.height * fig.dpi
        stride = int(height / desired_longest)
    else:
        # In this conditional, the image is shorter than it is tall.
        # So, "shrink" the image to the width of the axis.
        # (Multiply by fig.dpi to convert from inches to pixels.)
        desired_longest = bbox.width * fig.dpi
        stride = int(width / desired_longest)

    img_to_plot = img_to_plot[::stride, ::stride, :]

    # Plot the raster image and label it
    ax1.imshow(img_to_plot, aspect="equal", cmap="hsv", interpolation="none")

    # There are two subplots, and we want the main plot title to appear
    # over both subplots (aka centered in the figure). So, do not supply
    # the title here, otherwise the main plot title will only be centered
    # over `ax1``. (The main plot title was set above, via `fig.suptitle()`.)
    nisarqa.rslc.format_axes_ticks_and_labels(
        ax=ax1,
        img_arr_shape=np.shape(img_to_plot),
        xlim=xlim,
        ylim=ylim,
        xlabel=xlabel,
        ylabel=ylabel,
    )

    # Create the HSI "colorbar" (which is actually a subplot)
    v, h = np.mgrid[0:1:100j, 0:1:300j]
    s = np.ones_like(v)
    hsv = np.dstack((h, s, v))
    rgb = hsv_to_rgb(hsv)
    rgb = np.rot90(rgb, k=3)
    rgb = np.fliplr(rgb)

    if cbar_min_max is None:
        cbar_max = np.nanmax(img_arr)
        cbar_min = np.nanmin(img_arr)

    else:
        if (len(cbar_min_max) != 2) or (cbar_min_max[0] >= cbar_min_max[1]):
            raise ValueError(
                f"{cbar_min_max=}, must be a pair of increasing values."
            )
        cbar_max = cbar_min_max[1]
        cbar_min = cbar_min_max[0]

    ax2.imshow(
        rgb,
        origin="lower",
        extent=[0, 1, cbar_min, cbar_max],
    )
    ax2.set_xlabel("InSAR\nCoherence\nMagnitude", fontsize=8.5)
    ax2.set_ylabel("InSAR Phase", fontsize=8.5, rotation=270, labelpad=10)
    ax2.yaxis.set_label_position("right")

    # If the colorbar range covers an even multiple of pi, then re-format
    # the ticks marks to look nice.
    if (np.abs(cbar_max - cbar_min) % np.pi) < 1e-6:
        # Compute number of ticks
        tick_vals = np.arange(cbar_min, cbar_max + np.pi, np.pi)

        # Only pretty-format if there are a small-ish number of ticks
        # If support for a higher number is desired, then add'l code will
        # need to be written to decimate `tick_vals` appropriately.
        if len(tick_vals) < 9:
            ax2.set_yticks(tick_vals)
            ax2.yaxis.set_major_formatter(
                FuncFormatter(
                    lambda val, pos: (
                        f"{Fraction(f'{val/np.pi:.2f}')}$\pi$"
                        if val != 0
                        else "0"
                    )
                )
            )

    ax2.yaxis.tick_right()

    ax2.set_title("HSI Color Space\nSaturation=1", fontsize=8.5)

    # Append figure to the output PDF
    plots_pdf.savefig(fig)

    # Close the plot
    plt.close(fig)


def image_histogram_equalization(
    image: np.ndarray, nbins: int = 256
) -> np.ndarray:
    """
    Perform histogram equalization of a grayscale image.

    Parameters
    ----------
    image : numpy.ndarray
        N-dimensional image array. All dimensions will be combined when
        computing the histogram.
    nbins : int, optional
        Number of bins for computing the histogram.
        Defaults to 256.

    Returns
    -------
    equalized_img : numpy.ndarray
        The image with histogram equalization applied.
        This image will be in range [0, 1].

    References
    ----------
    Adapted from: skimage.exposure.equalize_hist
    Description of histogram equalization:
    https://scikit-image.org/docs/stable/auto_examples/color_exposure/plot_equalize.html
    """
    # Do not include NaN values
    img = image[np.isfinite(image)]

    hist, bin_edges = np.histogram(img.flatten(), bins=nbins, range=None)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0

    cdf = hist.cumsum()
    cdf = cdf / float(cdf[-1])

    out = np.interp(image.flatten(), bin_centers, cdf)
    out = out.reshape(image.shape)

    # Sanity Check. Mathematically, the output should be within range [0, 1].
    assert np.all(
        (0.0 <= out[np.isfinite(out)]) & (out[np.isfinite(out)] <= 1.0)
    ), "`out` contains values outside of range [0, 1]."

    # Unfortunately, np.interp currently always promotes to float64, so we
    # have to cast back to single precision when float32 output is desired
    return out.astype(image.dtype, copy=False)


__all__ = nisarqa.get_all(__name__, objects_to_skip)