# sam2-mri-segmentation
(USC SPAN) Segmenting mri video avi files using Meta's SAM2
**MAIN CODE FILE IS LOCATED runs/segment_mri_sam2.py
**
# MRI Tongue Segmentation with SAM 2

This repo runs [Meta’s SAM 2](https://github.com/facebookresearch/sam2) on MRI `.avi` videos to segment tongue motion.

## Outputs
- `workdir_mri/frames/` — extracted JPEG frames (optional to keep)
- `workdir_mri/masks/*.png` — one **binary** mask per frame (white = tongue)

---

## Quickstart

git clone https://github.com/anura111/sam2-mri-segmentation facebook_sam2

cd facebook_sam2


python3 -m venv sam2env

source sam2env/bin/activate

pip install -r requirements.txt

---

## Usage
- change paths to your own!
  
PYTORCH_ENABLE_MPS_FALLBACK=1 \
~/Desktop/facebook_sam2/sam2env/bin/python segment_mri_sam2.py 
  --video YOUR/PATH\
  --workdir ~/YOUR/PATH \
  --out ~/YOUR/PATH \
  --vos_optimized \
  --device mps
