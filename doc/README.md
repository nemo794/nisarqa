# Instructions to Build the QA Product Specification PDF

## Install pandoc
Official [instructions to install `pandoc` are here](https://pandoc.org/installing.html).

Instructions on a Mac Silicon running Sonoma 14.7.4:
```bash
# Keep separate from the `nisarqa` conda environment
conda create -n pandoc
conda activate pandoc

# install
conda install -c conda-forge pandoc
```

## Install LaTeX
By default, `pandoc` creates an intermediary LaTeX file when converting from markdown to PDF. This requires LaTeX to be installed. Suggestions for installing LaTeX on your platform are included in the [`pandoc` installation instructions](https://pandoc.org/installing.html).

Instructions on a Mac Silicon running Sonoma 14.7.4:

1. Download MacTeX and follow .pkg installation instructions: https://tug.org/mactex/mactex-download.html
2. Verify that `pandoc` can generate a PDF:

    ```bash
    # cd into the `product_specs` directory. If starting from top level `nisarqa`:
    cd ./doc/product_specs

    # Test pandoc
    pandoc 01_cover_page.md -o test.pdf
    ```
    If this command succeeds, great! Open the new test.pdf file you've created.

    But if the command fails with this error:
    ```bash
    pdflatex not found. Please select a different --pdf-engine or install pdflatex
    ```
    Then try following the guidance here: https://stackoverflow.com/questions/22081991/rmarkdown-pandoc-pdflatex-not-found

    This command succeeded for me:
    ```bash
    pandoc 01_cover_page.md -o test.pdf --pdf-engine=/Library/TeX/Distributions/.DefaultTeX/Contents/Programs/texbin/pdflatex
    ```

[!Note]
In lieu of installing LaTeX and Eisvogel independently, consider using the [pre-built Docker container](https://hub.docker.com/r/pandoc/extra).


## Download Eisvogel Template
To make the final PDF look a bit nicer, we'll use [the Eisvogel template](https://github.com/Wandmalfarbe/pandoc-latex-template).

Instructions on a Mac Silicon running Sonoma 14.7.4:
1. Download and unzip Eisvogel package. The most recent release as of this writing was [Release 3.1.0](https://github.com/Wandmalfarbe/pandoc-latex-template/releases/tag/v3.1.0), which is what we tested with and used.
2. Per the Eisvogel instructions, manually move the pondoc LaTeX template file `eisvogel.latex` to your pandoc templates folder.

    a. As noted in [this SO post](https://stackoverflow.com/a/39710146), there was not an existing pandoc templates directory, so first I needed to manually create it:
    ```bash
    mkdir -p ~/.pandoc/templates

    # (Optional) Add a default 
    pandoc -D latex > ~/.pandoc/templates/default.latex
    ```
    b. Copy (or move) the template file from wherever it was downloaded, e.g.:
    ```bash
    cp ~/Downloads/Eisvogel-3.1.0/eisvogel.latex ~/.pandoc/templates 
    ```
3. Check that this succeeds, and that the PDF generated looks a bit nicer:
```bash
pandoc 01_cover_page.md -o test.pdf \
    --pdf-engine=/Library/TeX/Distributions/.DefaultTeX/Contents/Programs/texbin/pdflatex \
    --template eisvogel
```

## Concatonate the `.md` files and generate the intermediary DOCX
```bash
pandoc --from=markdown_github+multiline_tables+table_captions *.md  -o nisarqa_product_specs_tmp.docx -V linkcolor=blue
```

## Compress all image files before pushing
To keep the size of the git repo manageable, users are requested to compress all images prior to submitting a pull request.


Here is a suggested method to reduce the file size:

```python
from PIL import Image

uncompressed_image = "path/to/uncompressed/image.png"
resize_factor = 3
# Use .jpg extension to enable optimization and lossy compression
output_compressed_image = "path/for/compressed/image.jpg"

img = Image.open(uncompressed_image)
(width, height) = (img.width // resize_factor, img.height // resize_factor)
img_resized = img.resize((width, height), resample=Image.Resampling.BILINEAR)

img_resized.save(output_compressed_image, optimize=True, quality=50.0)

img.close()
img_resized.close()
```