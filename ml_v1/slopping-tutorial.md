You don’t need to move the whole 17 GB folder. Training only needs two files from it, about 200 MB:

derived/ml/features.parquet, the cell-month feature table. The training tensors are built from this on the other machine.
derived/app/cell_country.parquet, the country of each cell, which the evaluation uses.

The code is already on GitHub, and docs/validation/model.json (the outage-month list) is in the repo. I’ll bundle the two files:

The bundle is at /home/nisbenz/firms/brasa-ml-data.tar (195 MB, SHA-256 b4251739…0f69f0).

1. Copy it to the GPU machine

Pick whichever transfer suits you:

SSH: run this from this laptop, replacing user@gpu-box with your login on the other machine:
rsync -avP ~/firms/brasa-ml-data.tar user@gpu-box:~/
No SSH: upload it to Google Drive (as you did with the FIRMS zips) or a USB stick, and download it there.
GitHub release: attach it to a release like your data-2000-2025 one. This needs gh, which isn’t installed on this laptop, or you can upload through the GitHub web page.
2. On the GPU machine
git clone https://github.com/nisbenz/brasa && cd brasa
mkdir -p ~/firms && tar -xf ~/brasa-ml-data.tar -C ~/firms && sha256sum ~/brasa-ml-data.tar

The checksum should end in 0f69f0.

python3.12 -m venv .venv && .venv/bin/pip install -r requirements.txt

Install the PyTorch build that matches the machine’s CUDA version. Check it with nvidia-smi; this example is for CUDA 12.8:

.venv/bin/pip install torch==2.15.0 --index-url https://download.pytorch.org/whl/cu128
.venv/bin/python -c "import torch; print(torch.cuda.is_available())"

That should print True.

3. Build the tensors, train, evaluate

This writes about 3.7 GB of tensors to ~/firms/derived/ml/overlap:

.venv/bin/python -I ml_v1/build_tensor.py ~/firms/derived --years 2012-2025

Training can run for a while, so it’s wrapped in nohup to survive a dropped SSH session:

nohup .venv/bin/python -I ml_v1/train.py ~/firms/derived/ml/overlap ~/firms/derived/ml/unet.pt > train.log 2>&1 &
tail -f train.log

When training finishes:

.venv/bin/python -I ml_v1/evaluate.py ~/firms/derived ~/firms/derived/ml/overlap ~/firms/derived/ml/unet.pt --importance

Results land in ~/firms/derived/ml/unet/comparison.csv and importance.json. Send those back and I’ll go through them.

Before training, you could run the 2-second smoke test from ml/README.md on the new machine to confirm everything works there.

If the GPU has less than 16 GB of memory, add --batch 8 or --patch 96 to the training command. The defaults are untested on a real GPU, and I haven’t measured training time.