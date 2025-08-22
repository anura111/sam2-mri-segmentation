#!/usr/bin/env python3
"""
Segment a moving mouth/tongue region in an MRI .avi using SAM 2.
Outputs:
  - masks/<frame>.png  (binary masks)
  - overlay.mp4        (original video + colored overlay)
"""

import os, argparse, shutil, glob
import numpy as np
import cv2
from contextlib import nullcontext
import torch

# Option A: load via Hugging Face (easiest)
from sam2.sam2_video_predictor import SAM2VideoPredictor

# Option B: local checkpoint (uncomment if you prefer local files)
# from sam2.build_sam import build_sam2_video_predictor


def extract_frames(video_path, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 25.0  # fallback

    idx = 0
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        # MRI is often grayscale; ensure 3-channel BGR JPEGs for SAM2 helpers.
        if len(frame.shape) == 2 or frame.shape[2] == 1:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        out_path = os.path.join(out_dir, f"{idx:06d}.jpg")
        cv2.imwrite(out_path, frame, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        frames.append(out_path)
        idx += 1
    cap.release()

    if idx == 0:
        raise RuntimeError("No frames extracted; unsupported video?")
    return frames, fps


def auto_motion_point(frames, sample=20, threshold_ratio=0.10):
    """
    Heuristic: find the centroid of the largest moving blob across the first
    `sample` frames (mouth area should move during speech).
    """
    sample = min(sample, len(frames))
    first = cv2.imread(frames[0], cv2.IMREAD_GRAYSCALE)
    h, w = first.shape[:2]
    acc = np.zeros((h, w), np.float32)
    prev = first.astype(np.float32)

    for i in range(1, sample):
        img = cv2.imread(frames[i], cv2.IMREAD_GRAYSCALE).astype(np.float32)
        diff = cv2.absdiff(img, prev)
        acc += diff
        prev = img

    norm = cv2.normalize(acc, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    # threshold by a fraction of max motion
    _, bw = cv2.threshold(norm, int(threshold_ratio * 255), 255, cv2.THRESH_BINARY)
    bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8), iterations=1)

    cnts, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        # fallback: center of the frame
        return (w // 2, h // 2), 0
    biggest = max(cnts, key=cv2.contourArea)
    M = cv2.moments(biggest)
    if M["m00"] == 0:
        x, y, ww, hh = cv2.boundingRect(biggest)
        cx, cy = x + ww // 2, y + hh // 2
    else:
        cx, cy = int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])
    return (int(cx), int(cy)), 0  # (x, y), frame_idx


def ensure_dtype():
    if torch.cuda.is_available():
        # BF16 is recommended by the repo examples; fallback to FP16 if BF16 unsupported.
        try:
            # Rough check; most recent GPUs support bf16; we’ll still guard with autocast.
            return torch.bfloat16
        except Exception:
            return torch.float16
    return torch.float32


def union_from_outputs(masks_or_logits, target_hw=None):
    """
    Normalize SAM2 outputs to a single 2D uint8 mask (H, W) with values {0,255}.
    Handles tensors, lists, dicts, extra singleton dims, and multi-object unions.
    """
    import numpy as np
    import torch

    # 1) Get to a raw array/tensor we can work with
    m = masks_or_logits
    if isinstance(m, dict):
        # Try common keys; fall back to first value
        for k in ["masks", "mask_logits", "logits", "low_res_masks"]:
            if k in m:
                m = m[k]
                break
        else:
            # just take the first item
            m = next(iter(m.values()))

    if torch.is_tensor(m):
        m = m.detach().to("cpu")
        arr = m
    else:
        arr = np.array(m, dtype=np.float32)

    # 2) Convert torch -> numpy
    if 'torch' in str(type(arr)):
        arr = arr.cpu().numpy()

    # 3) Squeeze singleton dims
    arr = np.squeeze(arr)

    # 4) Union across any leading dims so we end up with (H, W)
    #    Cases seen in the wild: (H,W), (1,H,W), (N,H,W), (N,1,H,W), (H,W,1)
    if arr.ndim == 2:
        mask2d = arr
    elif arr.ndim == 3:
        # union across the first axis (objects/channels)
        mask2d = (arr > 0).any(axis=0)
    elif arr.ndim == 4:
        mask2d = (arr > 0).any(axis=(0,1))
    else:
        # If shape is weird, create an empty mask with target size as a fallback
        if target_hw is None:
            raise RuntimeError(f"Unexpected mask shape: {arr.shape}")
        h, w = target_hw
        mask2d = np.zeros((h, w), dtype=bool)

    # 5) Ensure boolean -> uint8 {0,255}
    if mask2d.dtype != bool:
        mask2d = mask2d > 0
    mask2d = (mask2d.astype(np.uint8)) * 255

    # 6) Make sure it’s C-contiguous for OpenCV
    return np.ascontiguousarray(mask2d)



def overlay_mask(frame_bgr, mask_uint8, alpha=0.7, thickness=8):
    """Overlay segmentation mask in bright red with a thick outline."""
    frame_out = frame_bgr.copy()

    # Fill region in red (semi-transparent)
    color = np.zeros_like(frame_bgr)
    color[:, :] = (0, 0, 255)  # BGR red
    m = (mask_uint8 > 0)[:, :, None]
    frame_out = np.where(m, (alpha * color + (1 - alpha) * frame_out).astype(np.uint8), frame_out)

    # Add bold outline on top
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(frame_out, contours, -1, (0, 0, 255), thickness=thickness)

    return frame_out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True, help="Path to input .avi MRI")
    ap.add_argument("--workdir", default="workdir_mri", help="Temp workspace (frames/masks)")
    ap.add_argument("--out", default="overlay.mp4", help="Output overlay video")
    ap.add_argument("--model", default="facebook/sam2-hiera-small",
                    help="HF model id (e.g., facebook/sam2-hiera-small or -large)")
    ap.add_argument("--init_mode", choices=["auto_point", "box"], default="auto_point",
                    help="How to initialize SAM2")
    ap.add_argument("--box", type=int, nargs=4, metavar=("X1","Y1","X2","Y2"),
                    help="Manual box if --init_mode box (pixels)")
    ap.add_argument("--init_frame", type=int, default=0, help="Annotation frame index")
    ap.add_argument("--keep_frames", action="store_true", help="Do not delete extracted frames")
    ap.add_argument("--vos_optimized", action="store_true",
                    help="Enable torch.compile speedup when supported")
    ap.add_argument("--device", default=None, choices=[None, "cpu", "cuda", "mps"],
                help="Compute device. Default: auto (mps on Apple, else cuda if available, else cpu)")

    args = ap.parse_args()
    # Decide device
    if args.device is not None:
        device = args.device
    else:
        if torch.backends.mps.is_available():
            device = "mps"
        elif torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"
    print(f"Using device: {device}")


    frames_dir = os.path.join(args.workdir, "frames")
    masks_dir = os.path.join(args.workdir, "masks")
    os.makedirs(masks_dir, exist_ok=True)
    if os.path.isdir(frames_dir):
        shutil.rmtree(frames_dir)

    print("Extracting frames...")
    frames, fps = extract_frames(args.video, frames_dir)
    h, w = cv2.imread(frames[0]).shape[:2]

    print("Loading SAM 2 model...")
    # Option A: from_pretrained (no local files needed)
    predictor = SAM2VideoPredictor.from_pretrained(args.model, device=device)

    # Option B: local ckpt (uncomment & adjust if you downloaded checkpoints)
    # cfg = "configs/sam2.1/sam2.1_hiera_s.yaml"
    # ckpt = "checkpoints/sam2.1_hiera_small.pt"
    # predictor = build_sam2_video_predictor(cfg, ckpt, vos_optimized=args.vos_optimized)

    dtype = ensure_dtype()
    # Autocast only on CUDA; MPS/CPU will use a no-op context
    if device == "cuda":
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        device_ctx = torch.autocast("cuda", dtype=dtype)
    else:
        dtype = torch.float32
        device_ctx = nullcontext()


    print("Initializing video state...")
    # SAM2 expects a directory of JPEG frames
    with torch.inference_mode(), device_ctx:
        state = predictor.init_state(
            video_path=frames_dir,
            offload_video_to_cpu=True,   # keep full frame stack on CPU
            offload_state_to_cpu=True    # keep large cached tensors on CPU
        )

        if args.init_mode == "auto_point":
            (cx, cy), ann_f = auto_motion_point(frames)
            ann_f = args.init_frame if args.init_frame is not None else ann_f
            print(f"Auto init at frame {ann_f}, point=({cx},{cy})")
            points = np.array([[cx, cy]], dtype=np.float32)
            labels = np.array([1], dtype=np.int32)  # 1 = positive click
            _ = predictor.add_new_points_or_box(
                state, frame_idx=ann_f, obj_id=1, points=points, labels=labels
            )
        else:
            if args.box is None:
                raise ValueError("Provide --box X1 Y1 X2 Y2 with --init_mode box")
            (x1, y1, x2, y2) = args.box
            box = np.array([x1, y1, x2, y2], dtype=np.float32)
            print(f"Init with box {box.tolist()} at frame {args.init_frame}")
            _ = predictor.add_new_points_or_box(
                state, frame_idx=args.init_frame, obj_id=1, box=box
            )

        print("Propagating through video...")
        # Collect masks per frame and write PNGs
        mask_paths = {}
        for f_idx, obj_ids, out_masks in predictor.propagate_in_video(state):
            union_mask = union_from_outputs(out_masks, target_hw=(h, w))
            # If mask accidentally has a third dim of size 1, squeeze it:
            if union_mask.ndim == 3 and union_mask.shape[2] == 1:
                union_mask = union_mask[:, :, 0]
            mask_path = os.path.join(masks_dir, f"{f_idx:06d}.png")
            cv2.imwrite(mask_path, union_mask)

    print("Writing overlay video...")
    # Reassemble overlay video
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(args.out, fourcc, fps, (w, h))
    for i, f in enumerate(frames):
        frame = cv2.imread(f)
        mask_path = mask_paths.get(i, None)
        if mask_path and os.path.exists(mask_path):
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            frame = overlay_mask(frame, mask, alpha=0.5)
        writer.write(frame)
    writer.release()

    if not args.keep_frames:
        shutil.rmtree(frames_dir)

    print(f"Done.\n- Masks: {masks_dir}\n- Overlay video: {args.out}")


if __name__ == "__main__":
    main()
