#!/usr/bin/env python3
"""
Download FSDP shards from yiheng0824/smdm (folder latest_new.pth),
merge with merge.py into latest.pth, then upload latest.pth back to the repo.

Requires: huggingface_hub, and HF token for upload (huggingface-cli login or HF_TOKEN).
"""
import os
import sys
import tempfile
import argparse

REPO_ID = "yiheng0824/smdm"
SHARD_FOLDER = "latest_new.pth"
OUTPUT_FILENAME = "latest.pth"


def main():
    parser = argparse.ArgumentParser(description="Download latest_new.pth from HF, merge to latest.pth, upload.")
    parser.add_argument("--no-upload", action="store_true", help="Only download and merge; do not upload.")
    parser.add_argument("--work-dir", default=None, help="Work directory for download/merge (default: temp dir).")
    parser.add_argument("--output", "-o", default=None, help="Copy merged latest.pth to this path (e.g. ./latest.pth).")
    parser.add_argument("--keep", action="store_true", help="Keep work directory after run (for debugging).")
    args = parser.parse_args()

    work_dir = args.work_dir or tempfile.mkdtemp(prefix="smdm_merge_")
    os.makedirs(work_dir, exist_ok=True)
    shard_dir = os.path.join(work_dir, SHARD_FOLDER)
    output_path = os.path.join(work_dir, OUTPUT_FILENAME)

    try:
        # 1) Download folder latest_new.pth from yiheng0824/smdm
        print(f"Downloading {REPO_ID}/{SHARD_FOLDER} ...")
        from huggingface_hub import snapshot_download
        snapshot_download(
            repo_id=REPO_ID,
            local_dir=work_dir,
            allow_patterns=[f"{SHARD_FOLDER}/*", f"{SHARD_FOLDER}/.*"],
            local_dir_use_symlinks=False,
        )
        if not os.path.isdir(shard_dir):
            print(f"错误: 下载后未找到目录 {shard_dir}")
            sys.exit(1)
        print(f"Downloaded to {shard_dir}")

        # 2) Merge shards -> latest.pth (reuse merge.merge_fsdp_shards)
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from merge import merge_fsdp_shards
        merge_fsdp_shards(shard_dir, output_path)
        if not os.path.isfile(output_path):
            print(f"错误: 合并后未生成 {output_path}")
            sys.exit(1)

        # 3) Upload latest.pth to repo
        if args.output:
            import shutil
            shutil.copy2(output_path, args.output)
            print(f"✅ Copied merged file to: {args.output}")

        if not args.no_upload:
            path_to_upload = args.output if args.output and os.path.isfile(args.output) else output_path
            print(f"Uploading {OUTPUT_FILENAME} to {REPO_ID} ...")
            from huggingface_hub import HfApi
            api = HfApi()
            api.upload_file(
                path_or_fileobj=path_to_upload,
                path_in_repo=OUTPUT_FILENAME,
                repo_id=REPO_ID,
                repo_type="model",
            )
            print(f"✅ Uploaded {REPO_ID}/{OUTPUT_FILENAME}")
        else:
            print(f"✅ Merged file at: {output_path} (--no-upload, skip upload)")
    finally:
        if not args.keep and (args.work_dir is None):
            import shutil
            if os.path.exists(work_dir):
                shutil.rmtree(work_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
