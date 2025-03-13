from typing import Optional
from PIL import Image


def reduce_size(
    uncompressed_image,
    output_compressed_image,
    max_size: Optional[tuple[float, float]] = None,
    optimize=False,
    quality=100.0,
):
    """
    Reduce the shape size and compress a PNG into a JPG.

    Suggest: Do not compress ROFF and GOFF images; the quiver arrows do not
    appear crisp in the jpg.
    """

    # Open the image
    img = Image.open(uncompressed_image)

    # Resize
    if max_size is not None:
        img.thumbnail(size=max_size, resample=Image.Resampling.BILINEAR)

    # Ensure that the transparent pixels appear white by removing the alpha channel.
    if img.mode == "RGBA":
        # color image
        bkgd = Image.new("RGB", (img.width, img.height), (255, 255, 255))
        bkgd.paste(img, mask=img.split()[3])

    elif img.mode == "L":
        # grayscale image
        img = img.convert("LA")
        bkgd = Image.new("L", (img.width, img.height), 255)
        bkgd.paste(img, mask=img.split()[1])
    else:
        raise Exception("Nope. Input file: ", uncompressed_image)

    bkgd.save(output_compressed_image, optimize=optimize, quality=quality)


if __name__ == "__main__":
    basepath = (
        "/Users/niemoell/Desktop/qa_public/nisarqa/doc/product_specs/images"
    )
    # for i in ("RSLC", "GSLC", "GCOV", "RIFG", "RUNW", "GUNW", "ROFF", "GOFF"):
    # for i in ("RIFG", "RUNW", "GUNW", "ROFF", "GOFF"):

    # for i in ("RIFG", "RUNW", "GUNW"):
    #     if i == "RIFG":
    #         pix = 450
    #     elif i == "RUNW":
    #         pix = 448
    #     elif i == "GUNW":
    #         pix = 500
    #     reduce_size(
    #         f"{basepath}/full_size_already_converted/browse/BROWSE_{i}_{pix}.png",
    #         f"{basepath}/browse_{i}_compressed.jpg",
    #         # max_size=(500, 500),
    #         optimize=True,
    #         quality=85,
    #     )

    # for i in ("RSLC", "GSLC", "GCOV"):
    #     reduce_size(
    #         f"{basepath}/full_size_already_converted/browse/BROWSE_{i}_400.png",
    #         f"{basepath}/browse_{i}_compressed.jpg",
    #         # max_size=(500, 500),
    #         optimize=True,
    #         quality=85,
    #     )

    # for i in ("rslc", "gslc", "gcov"):
    #     reduce_size(
    #         f"{basepath}/full_size_already_converted/report_backscatter_{i}.png",
    #         f"{basepath}/report_backscatter_{i}.jpg",
    #         max_size=(500, 500),
    #         always_optimize=True,
    #     )

    # reduce_size(
    #     f"{basepath}/full_size_already_converted/report_cover_page.png",
    #     f"{basepath}/report_cover_page.jpg",
    #     max_size=(2000, 1000),
    #     always_optimize=True,
    # )

    for i in (
        "pta_impulse_response",
        "phase_histogram",
        "power_histogram",
        "pta_cr_offsets",
        "range_spectra",
    ):
        name = f"report_{i}_rslc"
        reduce_size(
            f"{basepath}/full_size_already_converted/{name}.png",
            f"{basepath}/{name}.jpg",
            max_size=(600, 288),
            optimize=True,
            quality=80,
        )

    name = "report_az_spectra_rslc"
    reduce_size(
        f"{basepath}/full_size_already_converted/{name}.png",
        f"{basepath}/{name}.jpg",
        max_size=(600, 600),
        optimize=True,
        quality=80,
    )

    # reduce_size(
    #     "{basepath}/full_size_already_converted/product_dependency.png",
    #     "{basepath}/product_dependency.jpg",
    #     max_size=(700, 700),
    #     always_optimize=True,
    # )

    # reduce_size(
    #     "{basepath}/full_size_already_converted/qa_processing_pipeline.png",
    #     "{basepath}/qa_processing_pipeline.jpg",
    #     max_size=(700, 700),
    #     always_optimize=True,
    # )

    # reduce_size(
    #     "{basepath}/full_size_already_converted/geocoding.png",
    #     "{basepath}/geocoding.jpg",
    #     max_size=(700, 700),
    #     always_optimize=True,
    # )

    for i in (
        "report_az_rng_offsets.png",
        "report_az_rng_offsets_hist.png",
        "report_offsets_quiver.png",
        "report_offsets_variance.png",
        "report_offsets_variance_hist.png",
        "report_offsets_cov_and_surf_peak.png",
        "report_offsets_cov_and_surf_peak_hist.png",
        "report_hsi.png",
        "report_iono_phs_screen_hist.png",
        "report_iono_phs_screen.png",
        "report_wrapped_igram_coh.png",
        "report_wrapped_igram_coh_hist.png",
        "report_unw_igram.png",
        "report_unw_igram_hist.png",
        "report_unw_coh.png",
        "report_unw_coh_hist.png",
        "report_cc.png",
    ):
        reduce_size(
            f"{basepath}/{i}",
            f"{basepath}/{i.replace('png', 'jpg')}",
            max_size=(600, 288),
            optimize=True,
            quality=80,
        )
