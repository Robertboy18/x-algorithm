# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 X.AI Corp.
from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)

ORBAX_TMP_DIR_SUFFIX = ".orbax-checkpoint-tmp-"


def ensure_orbax_dir_finalized(local_ckpt_dir: str, tag: str) -> None:
    orbax_dir = os.path.join(local_ckpt_dir, tag)
    if os.path.isdir(orbax_dir):
        return
    for entry in os.listdir(local_ckpt_dir):
        if entry.startswith(f"{tag}{ORBAX_TMP_DIR_SUFFIX}"):
            tmp_dir = os.path.join(local_ckpt_dir, entry)
            try:
                os.rename(tmp_dir, orbax_dir)
                logger.info("Renamed %s -> %s", entry, tag)
            except FileNotFoundError:
                pass
            return


def collect_cloud_upload_files(
    local_ckpt_dir: str,
    tag: str,
    proc_idx: int,
) -> list[tuple[str, int]]:
    orbax_dir = os.path.join(local_ckpt_dir, tag)
    proc_subdir = os.path.join(tag, f"ocdbt.process_{proc_idx}")
    my_subdir = os.path.join(local_ckpt_dir, proc_subdir)

    result: list[tuple[str, int]] = []

    if os.path.isdir(my_subdir):
        for root, _, files in os.walk(my_subdir):
            for fname in files:
                fpath = os.path.join(root, fname)
                result.append((os.path.relpath(fpath, local_ckpt_dir), os.path.getsize(fpath)))

    if proc_idx != 0:
        return result

    for entry in os.listdir(orbax_dir):
        entry_path = os.path.join(orbax_dir, entry)
        if entry.startswith("ocdbt.process_"):
            continue
        if os.path.isfile(entry_path):
            result.append((f"{tag}/{entry}", os.path.getsize(entry_path)))
        elif os.path.isdir(entry_path):
            for root, _, files in os.walk(entry_path):
                for fname in files:
                    fpath = os.path.join(root, fname)
                    result.append((os.path.relpath(fpath, local_ckpt_dir), os.path.getsize(fpath)))

    for entry in os.listdir(local_ckpt_dir):
        entry_path = os.path.join(local_ckpt_dir, entry)
        if entry == tag:
            continue
        if os.path.isfile(entry_path):
            result.append((entry, os.path.getsize(entry_path)))

    commit_file = os.path.join(local_ckpt_dir, "commit_success.txt")
    if not os.path.exists(commit_file):
        with open(commit_file, "w") as f:
            f.write(f"Checkpoint commit was successful to {local_ckpt_dir}\n")
        result.append(("commit_success.txt", os.path.getsize(commit_file)))

    return result
