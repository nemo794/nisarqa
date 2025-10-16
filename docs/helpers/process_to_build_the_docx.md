# Instructions to Build the QA Product Specification DOCX

## Update all Markdown files
Please follow the [Process to update the Markdown files](process_to_update_the_markdown_files.md).


## Install `pandoc`

[Official instructions to install `pandoc` are here](https://pandoc.org/installing.html).

```bash
conda install -c conda-forge pandoc
```

## Concatenate the `.md` files and generate the intermediary DOCX

`cd` into to `nisarqa/docs/product_specs` directory. 

Then, build a temporary, intermediary docx. This will contain all of the 
"content" of the final product specs, but lack the front matter and 
the correct formatting.

```bash
pandoc --from=markdown_github+multiline_tables+table_captions *.md  -o nisarqa_product_specs_tmp.docx -V linkcolor=blue
```

> [!CAUTION]
> `pandoc` has deprecated using `--from=markdown_github` in favor of the 
newer `--from=gfm`. However, `gfm` does not allow the `+multiline_tables` 
extension, which is required for the long descriptions in the tables to 
wrap to multiple lines within a cell. For now, keep using 
`--from=markdown_github`; it still worked with `pandoc` v3.4.


## Prepare the DOCX version of the QA product spec

Open the "front matter" DOCX file, which contains the cover page, 
signature pages, NISAR style formats, etc. This is DOCX located:

```
nisarqa/docs/helpers/nisarqa_product_specs_front_matter_template.docx
```

* Scroll to the first page after the table of contents, which should be 
the beginning of Section 1.
* If needed, delete everything from Section 1 to the end (it should 
already be deleted).
* Update the front matter with dates, revision history, names, etc.
    - Check with JPL/NISAR Mission management for the most-recent released
    version of the QA specs. When updating the front matter, ensure that
    the new version is incremented correctly from the most-recent previous
    version.
* Save, and `git commit` this updated front matter to the `nisarqa` repo.
    - **Commit this front matter prior to copy-pasting the full spec!**
    - To keep the GitHub repo small, and to avoid having a DOCX fall out of
    sync with the markdown files within the same `nisarqa` repo,
    let's only store the front matter portion the repo.
    - Each final DOCX will need to go through JPL and the NISAR Mission's 
    official processes for documentation, and should be tracked exclusively
    within those processes.

Next, open the intermediary `nisarqa_product_specs_tmp.docx` file in Word.

* Select all (command-A on Mac), and copy the entire contents of the 
intermediary file to the clipboard.
* Click over to the front matter DOCX. Paste the clipboard contents 
verbatim into the body of the front matter DOCX. 
* Scroll up to the table of contents. Right-click, and do "Update Field". 
This should rebuild the table of contents with the new material.
* In the Word document, under the "View" section, in the Macros dropdown 
select View Macros. Run the `TableStyleAndCenterImages` macro. (Will take 
~4-8 minutes.)
    - If this macro does not exist, then copy-paste the code (below) into 
    the Visual Basic Editor (VBA) in Word, and that Macro should appear.
* This is the final QA product specs DOCX! Save with the appropriate filename,
and follow the required JPL / NISAR mission procedures for this spec.

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
                If tx = "Path" Then
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
            tb.PreferredWidthType = wdPreferredWidthPoints
                    
            ' Set first column narrow (0.8")
            tb.Columns(1).Width = InchesToPoints(0.8)

            'Set second column to fill remaining width
            '(6.5 inches total, to allow for 1" page margins)
            tb.Columns(2).Width = InchesToPoints(5.7)
        End If
    Next tb
    
End Sub
```
