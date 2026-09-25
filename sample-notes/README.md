# sample-notes

The corpus this repository ships with. Six short markdown notes, deliberately
non-sensitive, so anyone can clone the repository and run the tool immediately.

**These files are committed.** Do not put personal notes here. Keep your own notes in
`my-notes/`, which `.gitignore` excludes, and point the tool at it:

```powershell
$env:ASKMYDOCS_NOTES_FOLDER = "my-notes"
```

## What belongs here

- 5 to 10 note files. This README is not counted as one.
- Supported extensions: `.pdf`, `.md`, `.markdown`, matched case-insensitively.
- Files are discovered recursively. Entries beginning with a period, symbolic links, and
  this README are skipped.

At the default chunk size of 500 with 50 characters of overlap, the six notes here
produce 20 chunks.
