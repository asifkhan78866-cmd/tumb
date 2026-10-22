#!/usr/bin/env bash
# Put the datasets in place on a machine that has only the clone.
#
#   scripts/unpack_data.sh            # unpack what the repository ships
#   scripts/unpack_data.sh --msd      # also fetch the 7.1 GB Decathlon volumes
#
# Two of the three datasets travel with the repository as archives; the third
# is far too large for that and is downloaded instead, from a public bucket
# that needs no account.
set -euo pipefail
cd "$(dirname "$0")/.."

WANT_MSD=0
[ "${1:-}" = "--msd" ] && WANT_MSD=1

have() { command -v "$1" >/dev/null 2>&1; }
have unzip || { echo "ERROR: 'unzip' is not installed." >&2; exit 1; }

# --- 1. The 4-class MRI dataset (7,023 images) ----------------------------- #
# Shipped as two parts because GitHub refuses a single file over 100 MB.
if [ -d data/bri/archive/Training ] && \
   [ "$(find data/bri/archive -name '*.jpg' | wc -l | tr -d ' ')" -gt 5000 ]; then
  echo "[data] MRI dataset already unpacked"
elif ls data/bri.zip.part-* >/dev/null 2>&1; then
  echo "[data] reassembling the MRI dataset from its parts"
  cat data/bri.zip.part-* > data/bri.zip
  # -o: the repository already ships a few of these images as samples, and
  # unzip would otherwise stop to ask whether to replace each one.
  unzip -qo data/bri.zip -d data/
  rm -f data/bri.zip
  echo "[data] MRI dataset: $(find data/bri/archive -name '*.jpg' | wc -l | tr -d ' ') images"
else
  echo "[data] MRI dataset parts not found — run scripts/download_dataset.sh instead"
fi

# --- 2. The paired MRI-SPECT slices (357 pairs) ---------------------------- #
if [ "$(ls data/spect_mri/MRI/*.png 2>/dev/null | wc -l | tr -d ' ')" -gt 300 ]; then
  echo "[data] MRI-SPECT pairs already unpacked"
elif [ -f data/spect_mri.zip ]; then
  echo "[data] unpacking the MRI-SPECT pairs"
  unzip -qo data/spect_mri.zip -d data/
  echo "[data] MRI-SPECT: $(ls data/spect_mri/MRI/*.png | wc -l | tr -d ' ') pairs"
else
  echo "[data] data/spect_mri.zip not found"
fi

# --- 3. The Decathlon (BraTS) volumes -------------------------------------- #
# 7.1 GB unpacked. Far past what a git repository should carry, so it is a
# download — but a public one: no Kaggle account, no login, no token.
if [ -d data/msd_brain/Task01_BrainTumour/imagesTr ]; then
  echo "[data] Decathlon volumes already present"
elif [ "$WANT_MSD" = "1" ]; then
  have curl || { echo "ERROR: 'curl' is not installed." >&2; exit 1; }
  echo "[data] downloading Task01_BrainTumour (7.6 GB — this takes a while)"
  mkdir -p data/msd_brain
  curl -L -C - --retry 5 --retry-delay 10 \
    -o data/msd_brain/Task01_BrainTumour.tar \
    https://msd-for-monai.s3-us-west-2.amazonaws.com/Task01_BrainTumour.tar
  echo "[data] extracting"
  tar -xf data/msd_brain/Task01_BrainTumour.tar -C data/msd_brain
  rm -f data/msd_brain/Task01_BrainTumour.tar
  echo "[data] Decathlon: $(find data/msd_brain -name '*.nii.gz' | wc -l | tr -d ' ') volumes"
else
  cat <<'MSG'
[data] Decathlon (BraTS) volumes: not present, and not shipped — 7.1 GB is past
       what a git repository should carry. They are only needed to *retrain*
       segmentation or the 3D U-Net; every trained model already works without
       them. To fetch them (public, no account):

         scripts/unpack_data.sh --msd
MSG
fi

echo
echo "[data] done. What this copy can run:"
echo "         python -m backend.cli.doctor"
