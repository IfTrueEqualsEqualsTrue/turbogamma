"""
Rename EPI files in a directory.

Each file is named:  <dicom_id>_<num>_<type>.epi.content
    e.g.  2.16.840.1.114337.1.1.1617962306.0_1_acquired.epi.content

This script:
  - replaces each unique dicom_id with an incremental index (0, 1, 2, ...)
  - renames the type:  acquired -> ref,  computed -> eval

Result:  0_1_ref.epi.content,  1_21_eval.epi.content, ...
"""
import os
from pathlib import Path

from dotenv import load_dotenv

# --- constants (edit here, not via terminal args) ---------------------------
load_dotenv()
DIRECTORY = Path(os.getenv("FILES"))
TYPE_MAP = {
    "acquired": "ref",
    "computed": "eval",
}
SUFFIX = ".epi.content"
# ---------------------------------------------------------------------------


def main():
    files = sorted(p for p in DIRECTORY.iterdir() if p.name.endswith(SUFFIX))
    if not files:
        print(f"No {SUFFIX} files found in {DIRECTORY}")
        return

    dicom_index = {}  # dicom_id -> incremental index
    planned = []  # (old_path, new_name)

    for path in files:
        stem = path.name[: -len(SUFFIX)]  # strip ".epi.content"
        dicom_id, num, kind = stem.rsplit("_", 2)

        if dicom_id not in dicom_index:
            dicom_index[dicom_id] = len(dicom_index)

        new_kind = TYPE_MAP.get(kind, kind)
        new_name = f"{dicom_index[dicom_id]}_{num}_{new_kind}{SUFFIX}"
        planned.append((path, new_name))

    for old_path, new_name in planned:
        new_path = old_path.with_name(new_name)
        if new_path == old_path:
            continue
        if new_path.exists():
            print(f"SKIP (target exists): {old_path.name} -> {new_name}")
            continue
        print(f"{old_path.name} -> {new_name}")
        old_path.rename(new_path)


if __name__ == "__main__":
    main()
