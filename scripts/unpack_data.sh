#!/usr/bin/env bash
# Put the datasets in place on a machine that has only the clone.
#
#   scripts/unpack_data.sh            # unpack what the repository ships
#   scripts/unpack_data.sh --msd      # also fetch the 7.1 GB Decathlon volumes
#
# No dataset travels with the repository any more: between them they were
# 160 MB of archives that every clone had to carry whether or not it ever
# trained anything. Each is a download now, and this script knows where from.
set -euo pipefail
cd "$(dirname "$0")/.."

WANT_MSD=0
[ "${1:-}" = "--msd" ] && WANT_MSD=1

have() { command -v "$1" >/dev/null 2>&1; }
have unzip || { echo "ERROR: 'unzip' is not installed." >&2; exit 1; }

# --- 1. The 4-class MRI dataset (7,023 images) ----------------------------- #
if [ -d data/bri/archive/Training ] && \
   [ "$(find data/bri/archive -name '*.jpg' | wc -l | tr -d ' ')" -gt 5000 ]; then
  echo "[data] MRI dataset already unpacked"
elif ls data/bri.zip.part-* >/dev/null 2>&1; then
  # Kept for anyone who still has the split archive from a shared folder.
  echo "[data] reassembling the MRI dataset from its parts"
  cat data/bri.zip.part-* > data/bri.zip
  # -o: the repository ships a few of these images as samples, and unzip would
  # otherwise stop to ask whether to replace each one.
  unzip -qo data/bri.zip -d data/
  rm -f data/bri.zip
  echo "[data] MRI dataset: $(find data/bri/archive -name '*.jpg' | wc -l | tr -d ' ') images"
elif [ -f data/bri.zip ]; then
  echo "[data] unpacking data/bri.zip"
  unzip -qo data/bri.zip -d data/
  echo "[data] MRI dataset: $(find data/bri/archive -name '*.jpg' | wc -l | tr -d ' ') images"
else
  cat <<'MSG'
[data] MRI dataset: not here. Nine sample images ship with the repository, which
       is enough to run every method; the full 7,023 are needed only to train.
       Get them either way:

         scripts/download_dataset.sh        # from Kaggle, needs a token
         # or drop the shared data.zip / bri.zip into data/ and re-run this
MSG
fi

# --- 2. The paired MRI-SPECT slices (357 pairs) ---------------------------- #
if [ "$(ls data/spect_mri/MRI/*.png 2>/dev/null | wc -l | tr -d ' ')" -gt 300 ]; then
  echo "[data] MRI-SPECT pairs already unpacked"
elif [ -f data/spect_mri.zip ]; then
  echo "[data] unpacking the MRI-SPECT pairs"
  unzip -qo data/spect_mri.zip -d data/
  echo "[data] MRI-SPECT: $(ls data/spect_mri/MRI/*.png | wc -l | tr -d ' ') pairs"
else
  echo "[data] MRI-SPECT pairs: not here (one sample pair ships with the repository)."
  echo "       Drop the shared spect_mri.zip into data/ and re-run this script."
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
