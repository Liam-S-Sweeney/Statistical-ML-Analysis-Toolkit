"""
Render pipeline results in the browser.

Analysis modules write their output to disk under outputs/. On a hosted
deployment that filesystem is ephemeral and invisible to the user, so a run is
wrapped: snapshot the output tree, call the pipeline, diff the tree, render
whatever appeared.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from pathlib import Path

import pandas as pd
import streamlit as st

_TEXT_SUFFIXES = {".txt", ".log", "md"}
_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg"}


def _snapshot(folders: Iterable[Path]) -> dict[Path, float]:
    """Map every file under the given folders to its modification time."""
    seen: dict[Path, float] = {}
    for folder in folders:
        folder = Path(folder)
        if not folder.exists():
            continue
        for path in folder.rglob("*"):
            if path.is_file():
                seen[path] = path.stat().st_mtime
    return seen


def _new_files(before: dict[Path, float], after: dict[Path, float]) -> list[Path]:
    changed = [
        path
        for path, mtime in after.items()
        if path not in before or before[path] != mtime
    ]
    return sorted(changed)

### Review ###
def _render_file(path: Path, key_prefix: str, index: int) -> None:
    suffix = path.suffix.lower()
    key = f"{key_prefix}_{index}_{path.name}"

    st.markdown(f"**{path.name}**")

    if suffix == ".csv":
        try:
            frame = pd.read_csv(path)
            st.dataframe(frame, use_container_width=True)
        except Exception as exc:
            st.warning(f"Could not preview {path.name}: {exc}")
        st.download_button(
            "Download CSV",
            data=path.read_bytes(),
            file_name=path.name,
            mime="text/csv",
            key=f"dl_{key}",
        )

    elif suffix in _IMAGE_SUFFIXES:
        st.image(str(path), use_container_width=True)
        st.download_button(
            "Download image",
            data=path.read_bytes(),
            file_name=path.name,
            mime="image/png",
            key=f"dl_{key}",
        )

    elif suffix in _TEXT_SUFFIXES:
        text = path.read_text(errors="replace")
        st.code(text, language="text")
        st.download_button(
            "Download text",
            data=text,
            file_name=path.name,
            mime="text/plain",
            key=f"dl_{key}",
        )

    else:
        st.download_button(
            f"Download {path.name}",
            data=path.read_bytes(),
            file_name=path.name,
            key=f"dl_{key}",
        )


def run_and_render(
    label: str,
    fn: Callable,
    watch_folders: Iterable[Path],
    *args,
    **kwargs,
):
    """
    Run a pipeline function and display everything it produced.
 
    Parameters
    ----------
    label : str
        Human-readable analysis name, used in headings and error messages.
    fn : Callable
        The pipeline function to call.
    watch_folders : Iterable[Path]
        Output folders to diff for newly written artifacts.
    """
    before = _snapshot(watch_folders)

    with st.spinner(f"Running {label}…"):
        try:
            result = fn(*args, **kwargs)
        except Exception as exc:
            st.error(f"{label} failed: {exc}")
            with st.expander("Traceback"):
                st.exception(exc)
            return None

    after = _snapshot(watch_folders)
    produced = _new_files(before, after)

    st.subheader(f"{label} — results")

    rendered_anything = False

    if isinstance(result, pd.DataFrame) and not result.empty:
        st.dataframe(result, use_container_width=True)
        st.download_button(
            "Download summary CSV",
            data=result.to_csv(index=False).encode("utf-8"),
            file_name=f"{label.lower().replace(' ', '_')}_summary.csv",
            mime="text/csv",
            key=f"dl_summary_{label}",
        )
        rendered_anything = True

    elif isinstance(result, str) and result.strip():
        st.warning(result)
        rendered_anything = True

    for index, path in enumerate(produced):
        _render_file(path, key_prefix=label.replace(" ", "_"), index=index)
        rendered_anything = True

    if not rendered_anything:
        st.info(f"{label} ran but produced no displayable output.")
    else:
        st.success(f"{label} complete.")

    return result

