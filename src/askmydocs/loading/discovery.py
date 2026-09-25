"""Note discovery: recursive, case-insensitive, and deterministically ordered.

Ordering is by code-point comparison of the forward-slash relative path, which
makes the order identical on every operating system (Requirement 6.8). Relying
on filesystem iteration order would make chunk ids differ between machines, and
chunk ids are what the vector store is keyed on.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from askmydocs.errors import NotesFolderError
from askmydocs.loading.base import SUPPORTED_EXTENSIONS


@dataclass(frozen=True)
class DiscoveredFile:
    """One file that will be loaded."""

    absolute_path: Path
    relative_path: str  # forward-slash, relative to the notes folder
    extension: str  # lower-cased, no leading period


@dataclass(frozen=True)
class SkippedEntry:
    """One entry that will not be loaded, and why."""

    relative_path: str
    reason: str


@dataclass(frozen=True)
class DiscoveryResult:
    files: tuple[DiscoveredFile, ...]
    skipped: tuple[SkippedEntry, ...]
    counts_by_extension: dict[str, int]


def discover_notes(notes_folder: Path) -> DiscoveryResult:
    """Find every supported note file under ``notes_folder``.

    Walks subdirectories. Matches extensions case-insensitively. Excludes the
    folder's own README, every entry whose name begins with a period, and every
    symbolic link, each recorded in ``skipped`` with a reason (Req 6.7, 6.9).

    Raises:
        NotesFolderError: the path is missing, or is not a directory (Req 6.5).
    """
    resolved = notes_folder.resolve()
    if not resolved.exists():
        raise NotesFolderError(f"The notes folder does not exist: {resolved}")
    if not resolved.is_dir():
        raise NotesFolderError(f"The notes folder is not a directory: {resolved}")

    files: list[DiscoveredFile] = []
    skipped: list[SkippedEntry] = []

    for path in resolved.rglob("*"):
        relative = path.relative_to(resolved).as_posix()

        if any(part.startswith(".") for part in path.relative_to(resolved).parts):
            if path.is_file():
                skipped.append(SkippedEntry(relative, "name begins with a period"))
            continue
        if path.is_symlink():
            skipped.append(SkippedEntry(relative, "symbolic link"))
            continue
        if not path.is_file():
            continue
        if relative.lower() == "readme.md":
            skipped.append(SkippedEntry(relative, "the notes folder README"))
            continue

        extension = path.suffix.lower().lstrip(".")
        if extension not in SUPPORTED_EXTENSIONS:
            skipped.append(
                SkippedEntry(relative, f"unsupported extension {path.suffix!r}")
            )
            continue

        files.append(DiscoveredFile(path, relative, extension))

    # Deterministic, OS-independent order (Requirement 6.8).
    files.sort(key=lambda f: f.relative_path)
    skipped.sort(key=lambda s: s.relative_path)

    counts: dict[str, int] = {}
    for discovered in files:
        counts[discovered.extension] = counts.get(discovered.extension, 0) + 1

    return DiscoveryResult(tuple(files), tuple(skipped), counts)
