# Training on Modal (single H200)

Runs the same Stage-2 VLA training as the cluster, on a Modal-provisioned H200, with checkpoints
persisted to a Modal Volume. See `modal_train.py` for the image/entrypoints.

## 0. One-time local setup

```bash
pip install modal
modal setup                                            # browser auth
# optional: wandb logging (else it falls back to metrics.jsonl on the volume)
modal secret create wandb WANDB_API_KEY=<your-key>
export MODAL_HAVE_WANDB=1                               # so train() attaches the secret
```

Make sure the repo is pushed — Modal clones `github.com/ZJU-Walker/qwen.git` at `main`:
```bash
git push origin main
```

## 1. Upload the dataset once (1.8 GB)

Run this FROM THE CLUSTER (where the dataset lives), streaming straight into the volume:
```bash
modal volume put qwen-data \
  /iris/projects/humanoid/trossen_data/0528_merge_block_mem /0528_merge_block_mem
```
Confirm the layout (should show data/ frames/ meta/ ...):
```bash
modal volume ls qwen-data /0528_merge_block_mem
modal run cloud/modal_train.py::check_dataset      # prints entry counts, flags missing dirs
```
Only `frames/cam_high/`, `data/`, and `meta/` are actually read; uploading the whole dir is
simplest and the wrist-cam `videos/` (unused) add little.

## 2. Smoke test (~10 min, validates image + GPU + HF downloads)

```bash
modal run cloud/modal_train.py::smoke
```
First run downloads Qwen2.5-VL-3B (~7 GB) and the FAST tokenizer into the `qwen-hf` volume, so it
takes longer; later runs reuse the cache. Watch for `compiled N/... decoder layers`, then step
40/60 lines with a `samp/s` figure and `cuda_gb` well under 141.

## 3. Full training (run3 config)

```bash
modal run --detach cloud/modal_train.py::train --args \
  "--out-dir /vol/ckpt/run3 --steps 15000 --resolution 224 --fps 2 --num-pairs 8 \
   --ckpt-stride 2 --compile --micro-batch 16 --grad-accum 2 --num-workers 8"
```
`--detach` keeps it running after you close your laptop. The entrypoint injects `--data-root`,
`--norm-stats`, and a pinned `--fast-revision` automatically if you don't pass them.

For the 336/30f/3fps config instead, drop `--resolution/--fps/--num-pairs` (defaults) and use
`--micro-batch 8 --grad-accum 4` (needs ~96 GB — fits the H200 141 GB).

## 4. Retrieve checkpoints

```bash
modal volume ls qwen-ckpt run3
modal volume get qwen-ckpt run3/step_015000 ./run3_step15k     # deployable snapshot (~few GB)
```
Then serve locally/anywhere with `realtime/server.py --checkpoint ./run3_step15k` (the policy
rebuilds the grid from the checkpoint's meta.json).

## Notes / gotchas
- **GPU memory**: 224/2fps config is light; 336 config needs ~96 GB → keep it on the H200 141 GB,
  not an 80 GB card.
- **Cost**: ~13 h for run3 at ~10 samp/s; H200 on Modal is ~$4-5/h → roughly $55-70.
- **flash-attn** is installed from the prebuilt wheel (torch2.6/cu12/py311/cxx11abiFALSE) — no
  30-min source build. If Modal's base CUDA drifts, rebuild the image.
- **Resuming**: add `--resume`; the fp32 resume bundle (`latest/`, ~55 GB) lives on the volume.
  Volumes persist across runs, so a `--detach` job that gets preempted can be resumed.
- The dataset/checkpoint/HF paths are the only cloud-specific bits; everything else is the same
  code the cluster runs.
