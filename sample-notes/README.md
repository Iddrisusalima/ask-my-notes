# Sample notes

This folder holds the note corpus the pipeline runs against. It is the default value of
`ASKMYDOCS_NOTES_FOLDER`.

## Expected contents

Put **5 to 10 note files** here. Fewer than 5 or more than 10 supported files is not an error —
the tool emits a warning stating the discovered count and this recommended range, and then
processes every discovered file anyway.

## Supported file extensions

| Extension | Loader |
|---|---|
| `.pdf` | PDF page text extraction |
| `.md` | markdown, full text preserved |
| `.markdown` | markdown, full text preserved |

Extensions are matched without regard to letter case, so `.PDF` and `.Md` are discovered too.
Files with any other extension are skipped and listed with their extension.

## This README is not a note file

`README.md` in this folder is excluded from discovery and **is not counted** toward the 5 to 10
note files. The count covers only the supported note files beside it.

Also excluded from discovery:

- every entry whose name begins with a period
- every symbolic link, which is listed among the skipped entries with its reason

Subdirectories **are** scanned: a supported file nested at any depth below this folder is
discovered.

## Version control

Personal note files in this folder are git-ignored. Only this README is tracked, so cloning the
repository gives you the folder and these instructions but none of anyone else's notes. Keep
anything committed here small and non-sensitive.
