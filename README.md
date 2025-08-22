# sam2-mri-segmentation
(USC SPAN) Segmenting mri video avi files using Meta's SAM2 

# MRI Tongue Segmentation with SAM 2

This repo runs [Meta’s SAM 2](https://github.com/facebookresearch/sam2) on MRI `.avi` videos to segment tongue motion.

## Outputs
- `workdir_mri/frames/` — extracted JPEG frames (optional to keep)
- `workdir_mri/masks/*.png` — one **binary** mask per frame (white = tongue)

---

## Quickstart

git clone <YOUR_REPO_URL> facebook_sam2
cd facebook_sam2

python3 -m venv sam2env
source sam2env/bin/activate
pip install -r requirements.txt

