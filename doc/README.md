# Table of 


# Instructions to Update the Product Spec Markdown Files

The markdown files have two primary venues for viewing: the public `nisarqa` Github repository, and the NISAR Product Specification document which is signed and released publicly.


## Update markdown text to match new QA PRs

As new QA PRs are drafted, the best practise is for the product specs to be updated in those PRs to keep them in sync. But, when preparing to cut a QA releases, the developer should do a final pass updating the product specs in advance of cutting the release.

When updating the product specs, the author needs to be mindful that the output looks "nice" in both Github and in .docx output. To view correctly on the Github repository, the markdown must be written in "Github-Flavored Markdown". However, when reformatting the markdown files into .docx files via `pandoc`, not all formatting will translate correctly. Here is guidance on how to correctly format the raw markdown for both venues:
* 



## Auto-Generate Updated QA HDF5 Product Specification tables

First, locate a sample QA HDF5 file with a representative internal file structure for that QA HDF5 file type. Tips:
* If updating the QA HDF5 product specs for e.g. all eight L1/L2 QA HDF5 products, locate a unique QA HDF5 file for each L1/L2 type.
* When selecting these files, the contents and size of the e.g. imagery Datasets do not matter; this script only looks at the internal structure, descriptions, etc. However, it is useful to consider the chosen sample files should be for single-pol observations, dual-pol, quad-pol, multi-frequency, etc. 
* When selecting RSLC and GSLC QA HDF5 files, please chose samples with outputs from the AbsCal and PTA Caltools.

Then, `cd` into to `nisarqa/docs/product_specs` directory. Use the `generate_stats_markdown.py` script to auto-generate updated markdown product specs using the input QA HDF5 file:

```
python ./generate_stats_markdown.py /path/to/QA/STATS.h5
```

The `generate_stats_markdown.py` script automatically adds header information into each output markdown file, iterates through the input product to parse the structure, formats the tables, and saves it with the correct file name in the `product_specs` directory. It also handles a few known edge cases. **To make changes to the QA HDF5 product spec markdown documents, please update this `generate_stats_markdown.py` script and re-run the script. _Do not edit the markdown files directly._**

## Compress all image files before pushing
To keep the size of the git repo manageable, users are requested to compress all images prior to submitting a pull request. The script `nisarqa/docs/product_specs/images/resize_png_to_jpg.py` is provided to quickly downsample and compress images for use in the documentation.

After compressing images, remember to double-check that the images' links in the markdown files are all still correct.


# Instructions to Build the QA Product Specification .docx

## Update all markdown files
Per the instructions (above).

## Create new environment with dependencies
Official [instructions to install `pandoc` are here](https://pandoc.org/installing.html).

Instructions on a Mac Silicon running Sonoma 14.7.4:
```bash
# Keep separate from the `nisarqa` conda environment
conda create -n qa_docs
conda activate qa_docs

# install
conda install -c conda-forge pandoc pillow h5py numpy
```

## Concatonate the `.md` files and generate the intermediary DOCX

`cd` into to `nisarqa/docs/product_specs` directory. 

Then, build a temporary, intermediary docx. This will
contain all of the "content" of the final product specs, but lack the front matter and the correct formatting.

```bash
pandoc --from=markdown_github+multiline_tables+table_captions *.md  -o nisarqa_product_specs_tmp.docx -V linkcolor=blue
```

## Prepare the final .docx product spec

Open the intermediary `nisarqa_product_specs_tmp.docx` file in Word.
 * Select all (command-A), and copy to the clipboard.

Open the final "front matter" .docx file, which contains the cover page, signature pages, NISAR style formats, etc.
* Scroll to the first page after the table of contents, which should be the beginning of Section 1.
* Delete everything from Section 1 to the end.
* Paste the clipboard contents verbatim into the body. (In effect, we're wholesale replacing to old body contents with the updated contents.)
* Scroll up to the table of contents. Right-click, and "update field". This should rebuild the TOC with the new material
* In the Word document, under the "View" section, in the Macros dropdown select View Macros. Run the `TableStyleAndCenterImages` macro. (Will take ~4-8 minutes.)
    - If this macro does not exist, then copy-paste the code (below) into the Visual Basic Editor (VBA) in Word, and that Macro should appear.
* Update the front matter with dates, revision history, etc.


Macro for formatting the tables and images in Word:
```
Sub TableStyleAndCenterImages()
    
    ' Apply the "NISAR table" style format to all tables
    Dim tbl As Table
    For Each tbl In ActiveDocument.Tables
        tbl.Style = "NISAR table"
    tbl.AutoFitBehavior wdAutoFitWindow
    tbl.Range.Font.Size = 9
    Next

  ' Center all images horizontally
  Dim objInLineShape As InlineShape
  Dim objDoc As Document
  Set objDoc = ActiveDocument
  For Each objInLineShape In objDoc.InlineShapes
    objInLineShape.Select
    Selection.ParagraphFormat.Alignment = wdAlignParagraphCenter
  Next objInLineShape
  
  ' For the product spec tables, set the "Path" rows to light blue color
  Dim tb As Table
    Dim rw As Row
    Dim cl As Cell
    Dim tx As String
    ' Loop through the tables in the document
    For Each tb In ActiveDocument.Tables
        Dim isSpecTable As Boolean
        isSpecTable = False
        ' Loop through the rows of the table
        For Each rw In tb.Rows
            ' Loop through the cells of the row
            For Each cl In rw.Cells
                ' Get the text of the cell
                tx = cl.Range.Text
                ' Remove the paragraph mark and cell marker
                tx = Left(tx, Len(tx) - 2)
                If tx = "Path:" Then
                    isSpecTable = True
                    ' If so, color the row pale blue
                    rw.Shading.ForegroundPatternColor = wdColorPaleBlue
                    ' And exit the cell loop
                    Exit For
                End If
            Next cl
        Next rw
        If isSpecTable Then
            tb.Rows(3).Shading.ForegroundPatternColor = wdColorAutomatic
                    'tb.PreferredWidthType = wdPreferredWidthPoints
                    'tb.Columns(1).Width = InchesToPoints(1.2)

        End If
    Next tb
    
End Sub
```

