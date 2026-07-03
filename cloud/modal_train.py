"""Run Stage-2 VLA training on Modal (single H200).

Reproduces the cluster's qwen3vl env (torch 2.6.0+cu124, transformers 4.57.6, flash-attn
2.7.4.post1) in a Modal image, mounts the repo, and runs training with checkpoints persisted to a
Volume. Qwen2.5-VL-3B weights + the FAST tokenizer download from HF on first run (cached on a
Volume so later runs skip it).

One-time setup
--------------
    pip install modal && modal setup                       # auth
    modal secret create wandb WANDB_API_KEY=<key>          # optional (else metrics.jsonl only)

    # upload the dataset (1.8 GB) into the 'qwen-data' volume once:
    modal run cloud/modal_train.py::upload_dataset \
        --local-dir /iris/projects/humanoid/trossen_data/0528_merge_block_mem

Train (run3 config)
-------------------
    modal run --detach cloud/modal_train.py::train --args \
      "--out-dir /vol/ckpt/run3 --steps 15000 --resolution 224 --fps 2 --num-pairs 8 \
       --ckpt-stride 2 --compile --micro-batch 16 --grad-accum 2 --num-workers 8"

`--detach` keeps it running if your laptop disconnects. Checkpoints land in the 'qwen-ckpt'
volume; download with:  modal volume get qwen-ckpt run3/step_015000 ./run3_step15k
"""

import os

import modal

REPO_URL = "https://github.com/ZJU-Walker/qwen.git"
GIT_REF = "main"  # pin to a commit sha for reproducibility if desired
DATA_DIR = "/vol/data/0528_merge_block_mem"   # dataset volume mount + dataset subdir
CKPT_DIR = "/vol/ckpt"                          # checkpoint volume mount
HF_DIR = "/vol/hf"                              # HF cache volume mount

# Image: CUDA 12.4 base matching torch 2.6.0+cu124; flash-attn from the prebuilt wheel (compiling
# it from source in-image takes ~30 min, the wheel is instant).
image = (
    modal.Image.from_registry("nvidia/cuda:12.4.1-devel-ubuntu22.04", add_python="3.11")
    .apt_install("git")
    .pip_install(
        "torch==2.6.0",
        "torchvision==0.21.0",
        index_url="https://download.pytorch.org/whl/cu124",
    )
    .pip_install(
        "transformers==4.57.6",
        "accelerate==1.7.0",
        "qwen-vl-utils==0.0.14",
        "av==17.0.0",
        "numpy>=2.0",
        "pyarrow>=24.0",
        "scipy",
        "wandb",
        "protobuf",
        "sentencepiece",
        "safetensors",
        "einops",
    )
    # flash-attn prebuilt wheel for torch 2.6 / cu12 / py311 / cxx11abiFALSE
    .pip_install(
        "https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/"
        "flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp311-cp311-linux_x86_64.whl"
    )
    .env({"HF_HOME": HF_DIR, "TOKENIZERS_PARALLELISM": "false"})
)

app = modal.App("qwen-vla-train", image=image)

data_vol = modal.Volume.from_name("qwen-data", create_if_missing=True)
ckpt_vol = modal.Volume.from_name("qwen-ckpt", create_if_missing=True)
hf_vol = modal.Volume.from_name("qwen-hf", create_if_missing=True)
VOLUMES = {"/vol/data": data_vol, CKPT_DIR: ckpt_vol, HF_DIR: hf_vol}


@app.function(volumes={"/vol/data": data_vol}, timeout=10 * 60)
def check_dataset():
    """Confirm the dataset uploaded correctly. Upload itself is a CLI step (see cloud/README.md):
        modal volume put qwen-data <local_dataset_dir> /0528_merge_block_mem
    """
    ok = True
    for sub in ("data", "frames/cam_high", "meta"):
        p = os.path.join(DATA_DIR, sub)
        n = len(os.listdir(p)) if os.path.isdir(p) else -1
        print(f"{p}: {'MISSING' if n < 0 else f'{n} entries'}")
        ok = ok and n > 0
    print("dataset OK" if ok else "DATASET INCOMPLETE — re-run `modal volume put`")


@app.function(
    gpu="H200",
    volumes=VOLUMES,
    timeout=24 * 60 * 60,           # up to 24 h; use --detach so it survives disconnects
    secrets=[modal.Secret.from_name("wandb")] if os.environ.get("MODAL_HAVE_WANDB") else [],
)
def train(args: str = ""):
    """Clone the repo at GIT_REF and run training/train.py with `args` (a single string)."""
    import shlex
    import subprocess

    subprocess.run(["git", "clone", "--depth", "1", "-b", GIT_REF, REPO_URL, "/root/qwen"],
                   check=True)
    workdir = "/root/qwen"

    argv = shlex.split(args)
    # Inject cloud defaults if the caller didn't override them.
    if "--data-root" not in args:
        argv += ["--data-root", DATA_DIR]
    if "--out-dir" not in args:
        argv += ["--out-dir", os.path.join(CKPT_DIR, "run")]
    if "--norm-stats" not in args:
        argv += ["--norm-stats", os.path.join(CKPT_DIR, "norm_stats.json")]

    if "--fast-revision" not in args:  # pin the tokenizer to the sha verified on the cluster
        argv += ["--fast-revision", "ec4d7aa71691cac0b8bed6942be45684db2110f4"]

    env = dict(os.environ, PYTHONPATH="src:.")
    print("launching: python -m streaming_qwen_vlm.training.train", " ".join(argv), flush=True)
    subprocess.run(
        ["python", "-m", "streaming_qwen_vlm.training.train", *argv],
        cwd=workdir, env=env, check=True,
    )
    ckpt_vol.commit()  # flush final checkpoints to persistent storage


SMOKE_ARGS = ("--overfit-episode 0 --steps 60 --warmup-steps 10 --eval-every 30 "
              "--save-every 30 --keep-every 60 --resolution 224 --fps 2 --num-pairs 8 "
              "--ckpt-stride 2 --compile --micro-batch 16 --grad-accum 2 --num-workers 8 "
              f"--out-dir {CKPT_DIR}/smoke3")


@app.local_entrypoint()
def smoke():
    """Short overfit-one-episode smoke on the H200 to validate the image before a long run."""
    train.remote(SMOKE_ARGS)
