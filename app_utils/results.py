from __future__ import annotations
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any
import pandas as pd
import streamlit as st
import mimetypes

_TEXT_SUFFIXES = {".txt", ".log", ".md"}
_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg"}


def _snapshot(folders: Iterable[Path]) -> dict[Path, float]:
    """
    Map every file under the given folders to its modification time.
    """
    seen: dict[Path, float] = {}
    for folder in folders:
        folder = Path(folder)
        if not folder.exists():
            continue
        for path in folder.rglob("*"):
            if path.is_file():
                seen[path] = path.stat().st_mtime   # .stat() returns all file metadata; st_mtime is last-modified time (Unix seconds)
    return seen


def _changed_files(before: dict[Path, float], after: dict[Path, float]) -> list[Path]:
    """
    Return files that were added or modified between two snapshots,
    sorted by path. Returns an empty list if nothing changed.

    A file counts as modified if its mtime differs in either direction.
    Deleted files are not reported.
    """
    changed = [
        path for path, mtime in after.items() 
        if path not in before or before[path] != mtime
    ]
    return sorted(changed)


def _render_file(path: Path, key_prefix: str, index: int) -> None:
    """
    Render an output file: a preview when the type supports one,
    then a download button for the original, unmodified file.

    CSVs show as a table, images inline, and text files as a code
    block. Other types get a download button only. If a preview fails,
    a warning is shown and the download is still offered.

    Args:
        path: File to render.
        key_prefix: Separates this group's widget keys from other
            sections of the page.
        index: The file's position in its list; keeps keys unique when
            two files share a name.
    """
    suffix = path.suffix.lower()
    key = f"dl_{key_prefix}_{index}_{path.name}"
    # MIME type is a 2-part label identifying type and subtype (ie. text/html)
    # guess_type maps the extension to a mime type; None if unknown
    mime = mimetypes.guess_type(path.name)[0] or "application/octet-stream" # standard MIME type for arbitrary binary data (unknown files)
    label = f"Download {path.name}"

    st.markdown(f"**{path.name}**")

    try:
        if suffix == ".csv":
            label = "Download CSV"
            st.dataframe(pd.read_csv(path), use_container_width=True)
        elif suffix in _IMAGE_SUFFIXES:
            label = "Download image"
            st.image(str(path), use_container_width=True)
        elif suffix in _TEXT_SUFFIXES:
            label = "Download text"
            text = path.read_text(encoding="utf-8", errors="replace")
            st.code(text, language="text")
    except Exception as e:
        # A failed preview shouldn't block the download or crash the page
        st.warning(f"Could not preview {path.name}: {e}")

    # Raw bytes so the download is identical to the file on disk
    st.download_button(
        label,
        data=path.read_bytes(),
        file_name=path.name,
        mime=mime,
        key=key,
    )


def run_and_render(
    label: str,
    fn: Callable[..., Any],
    watch_folders: Iterable[Path],
    *args,
    **kwargs,
    ) -> Any:
    """
    Run an analysis function and display everything it produced.

    The watch folders are snapshotted before and after calling fn, so
    any file fn adds or modifies there is shown with a preview and a
    download button. fn's return value is also displayed if it is:
      - a non-empty DataFrame: shown as a table with a CSV download
      - a non-empty string: shown as a warning message
    Any other return value is not displayed.

    If fn raises, the error and its traceback are shown in the app
    instead of crashing the page, and files fn wrote before failing
    are not shown.

    Args:
        label: Display name for the analysis; also used in headings,
            download file names, and widget keys.
        fn: The analysis function to run.
        watch_folders: Folders where fn writes its output files.
        *args, **kwargs: Passed through to fn unchanged.

    Returns:
        Whatever fn returned, or None if fn raised. A None return can
        also mean fn succeeded and returned None.
    """
    # Baseline: every file in the watch folders with its mtime
    before = _snapshot(watch_folders)

    # Run fn; on failure show a short message plus a collapsible traceback, 
    # then stop without rendering results
    with st.spinner(f"Running {label}…"):
        try:
            result = fn(*args, **kwargs)
        except Exception as e:
            st.error(f"{label} failed: {e}")
            with st.expander("Traceback"):
                st.exception(e)
            return None

    # Snapshot again and diff to find files fn added or modified
    after = _snapshot(watch_folders)
    produced = _changed_files(before, after)

    st.subheader(f"{label} - results")

    # Tracks whether anything at all (table, message, or file) was shown
    rendered_anything = False

    # Display the return value. Only non-empty DataFrames and strings are handled; 
    # any other type is silently skipped.
    if isinstance(result, pd.DataFrame) and not result.empty:
        st.dataframe(result, use_container_width=True)
        # Keep the index when it carries labels (describe(), coefficient
        # tables, groupby results); drop it when it's just 0 -> n-1
        keep_index = not isinstance(result.index, pd.RangeIndex)
        st.download_button(
            "Download summary CSV",
            data=result.to_csv(index=keep_index).encode("utf-8"),
            file_name=f"{label.lower().replace(' ', '_')}_summary.csv",
            mime="text/csv",
            key=f"dl_summary_{label}",
        )
        rendered_anything = True
    elif isinstance(result, str) and result.strip():
        # a returned string here is a problem warrenting a notice message
        st.warning(result)
        rendered_anything = True

    # Preview each produced file and give it a download button;
    # index keeps widget keys unique when filenames repeat
    for index, path in enumerate(produced):
        _render_file(path, key_prefix=label.replace(" ", "_"), index=index)
        rendered_anything = True

    # Distinguish "ran and showed results" from "ran but nothing to show"
    if not rendered_anything:
        st.info(f"{label} ran but produced no displayable output.")
    else:
        st.success(f"{label} complete.")

    return result

