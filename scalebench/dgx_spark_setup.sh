#!/usr/bin/env bash
# ============================================================================
# scalebench — DGX Spark bootstrap (run from the repo root on the Spark)
#   bash scalebench/dgx_spark_setup.sh
# Installs the GPU stack, fetches CelebA, verifies the identity file, and
# launches the full sweep. DGX OS / Ubuntu aarch64 + CUDA assumed present.
# ============================================================================
set -euo pipefail

echo "== [1/5] Python environment =="
python3 -m venv .venv-spark
# shellcheck disable=SC1091
source .venv-spark/bin/activate
pip install --upgrade pip -q
pip install -q insightface opencv-python-headless numpy pyyaml pandas \
    matplotlib psutil tqdm kagglehub markdown onnxruntime-gpu
# cuVS-accelerated FAISS (GPU flat/IVF + CAGRA). Falls back to faiss-cpu if
# no matching wheel exists for this CUDA/aarch64 combo.
pip install -q faiss-gpu-cuvs || pip install -q faiss-cpu

echo "== [2/5] Sanity: GPU visibility =="
python - <<'PY'
import onnxruntime as ort
print("onnxruntime providers:", ort.get_available_providers())
import faiss
print("faiss GPU:", hasattr(faiss, "StandardGpuResources"),
      "| CAGRA:", hasattr(faiss, "GpuIndexCagra"))
PY

echo "== [3/5] CelebA images (kagglehub, ~1.3 GB) =="
python - <<'PY'
import os, kagglehub
root = kagglehub.dataset_download("jessicali9530/celeba-dataset")
src = os.path.join(root, "img_align_celeba", "img_align_celeba")
os.makedirs("data/celeba", exist_ok=True)
dst = "data/celeba/img_align_celeba"
if not os.path.exists(dst):
    os.symlink(src, dst, target_is_directory=True)
n = len(os.listdir(dst))
print("images:", n)
assert n == 202599, "expected 202,599 aligned images"
PY

echo "== [4/5] identity_CelebA.txt (MD5-verified against torchvision reference) =="
python - <<'PY'
import hashlib, os, urllib.request
p = "data/celeba/identity_CelebA.txt"
if not os.path.exists(p):
    url = ("https://raw.githubusercontent.com/Golbstein/"
           "keras-face-recognition/master/identity_CelebA.txt")
    urllib.request.urlretrieve(url, p)
h = hashlib.md5(open(p, "rb").read()).hexdigest()
assert h == "32bd1bd63d3c78cd57e08160ec5ed1e2", f"identity file MD5 mismatch: {h}"
print("identity file verified:", h)
PY

echo "== [5/5] Full sweep (embeddings cache makes reruns fast) =="
python -m scalebench.run --config scalebench/config.dgx-spark.yaml
echo "Done. Report: scalebench/results-dgx-spark/report.html (+ results.csv)"
