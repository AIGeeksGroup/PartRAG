import os
from typing import Iterable, Optional, Sequence

from huggingface_hub import snapshot_download


def _unique_paths(paths: Iterable[str]) -> list[str]:
    seen = set()
    ordered = []
    for path in paths:
        if not path:
            continue
        if path in seen:
            continue
        seen.add(path)
        ordered.append(path)
    return ordered


def _is_valid_dir(
    path: str,
    required_subdirs: Sequence[str],
    required_files: Sequence[str],
) -> bool:
    if not os.path.isdir(path):
        return False
    for subdir in required_subdirs:
        if not os.path.isdir(os.path.join(path, subdir)):
            return False
    for file_name in required_files:
        if not os.path.isfile(os.path.join(path, file_name)):
            return False
    return True


def resolve_existing_weight_dir(
    candidates: Iterable[str],
    required_subdirs: Sequence[str] = (),
    required_files: Sequence[str] = (),
) -> Optional[str]:
    for path in _unique_paths(candidates):
        if _is_valid_dir(path, required_subdirs=required_subdirs, required_files=required_files):
            return path
    return None


def resolve_or_download_weights(
    preferred_dir: str,
    repo_id: str,
    legacy_dirs: Iterable[str] = (),
    required_subdirs: Sequence[str] = (),
    required_files: Sequence[str] = (),
    local_files_only: bool = False,
) -> str:
    candidates = _unique_paths([preferred_dir, *legacy_dirs])
    resolved = resolve_existing_weight_dir(
        candidates,
        required_subdirs=required_subdirs,
        required_files=required_files,
    )
    if resolved is not None:
        return resolved

    if local_files_only:
        joined = ", ".join(candidates)
        raise FileNotFoundError(
            f"Could not find local weights for {repo_id}. Checked: {joined}"
        )

    os.makedirs(preferred_dir, exist_ok=True)
    snapshot_download(repo_id=repo_id, local_dir=preferred_dir)
    return preferred_dir

