from PIL import Image


def reduce_size(
    uncompressed_image,
    output_compressed_image,
    max_size=(500, 500),
    always_optimize=False,
):
    """Reduce the shape size and compress a PNG into a JPG."""

    # Open the image
    img = Image.open(uncompressed_image)

    # Resize
    img.thumbnail(size=max_size, resample=Image.Resampling.BILINEAR)

    # Ensure that the transparent pixels appear white by removing the alpha channel.
    if img.mode == "RGBA":
        # color image
        bkgd = Image.new("RGB", (img.width, img.height), (255, 255, 255))
        bkgd.paste(img, mask=img.split()[3])
        optimize = False
        quality = 100

    elif img.mode == "L":
        # grayscale image
        img = img.convert("LA")
        bkgd = Image.new("L", (img.width, img.height), 255)
        bkgd.paste(img, mask=img.split()[1])
        optimize = True
        quality = 75
    else:
        raise Exception("Nope. Input file: ", uncompressed_image)

    if always_optimize:
        optimize = True
        quality = 85

    bkgd.save(output_compressed_image, optimize=optimize, quality=quality)


if __name__ == "__main__":
    # for i in ("RSLC", "GSLC", "GCOV", "RIFG", "RUNW", "GUNW", "ROFF", "GOFF"):
    #     reduce_size(
    #         f"full_size_already_converted/browse_{i}_reduced_size.png",
    #         f"browse_{i}_reduced_size.jpg",
    #         max_size=(250, 250),
    #     )

    # for i in ("rslc", "gslc", "gcov"):
    #     reduce_size(
    #         f"full_size_already_converted/report_backscatter_{i}.png",
    #         f"report_backscatter_{i}.jpg",
    #         max_size=(500, 500),
    #         always_optimize=True,
    #     )

    # reduce_size(
    #     f"full_size_already_converted/report_cover_page.png",
    #     f"report_cover_page.jpg",
    #     max_size=(2000, 1000),
    #     always_optimize=True,
    # )

    # for i in (
    #     "pta_impulse_response",
    #     "az_spectra",
    #     "phase_histogram",
    #     "power_histogram",
    #     "pta_cr_offsets",
    #     "range_spectra",
    # ):
    #     name = f"report_{i}_rslc"
    #     reduce_size(
    #         f"full_size_already_converted/{name}.png",
    #         f"{name}.jpg",
    #         max_size=(600, 600),
    #         always_optimize=True,
    #     )

    # reduce_size(
    #     "full_size_already_converted/product_dependency.png",
    #     "product_dependency.jpg",
    #     max_size=(700, 700),
    #     always_optimize=True,
    # )

    # reduce_size(
    #     "full_size_already_converted/qa_processing_pipeline.png",
    #     "qa_processing_pipeline.jpg",
    #     max_size=(700, 700),
    #     always_optimize=True,
    # )

    # reduce_size(
    #     "full_size_already_converted/geocoding.png",
    #     "geocoding.jpg",
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
            i,
            i.replace("png", "jpg"),
            max_size=(600, 600),
            always_optimize=True,
        )
