import json
import shutil
import time
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import torch
from diff_gaussian_rasterization import GaussianRasterizer as Renderer
from PIL import Image

from helpers import setup_camera

# image size & camera clipping planes
# w, h = 640, 360
# w, h = 1280, 720
w, h = 625, 625
near, far = 0.01, 100.0

# output method name
# METHOD = "ours"
METHOD = "ours-625x625"


def load_checkpoint_data(seq: str, exp: str, out_dir: Path, iteration: int = None) -> list[dict]:
    """
    Load checkpoint data from checkpoints folder.
    If iteration is None, loads all checkpoints found.
    Returns list of dicts with checkpoint info.
    """
    checkpoint_dir = out_dir / exp / seq / "checkpoints"
    
    if not checkpoint_dir.exists():
        return None
    
    if iteration is not None:
        checkpoint_files = list(checkpoint_dir.glob(f"*_iter_{iteration}.npz"))
    else:
        checkpoint_files = sorted(checkpoint_dir.glob("*.npz"))
    
    checkpoints = []
    for cp_file in checkpoint_files:
        raw = dict(np.load(cp_file))
        params = {k: torch.tensor(v).cuda().float() for k, v in raw.items()}
        
        # Extract timestep and iteration from filename
        filename = cp_file.stem
        parts = filename.split('_')
        t = int(parts[1]) if 'timestep' in filename else 0
        iter_num = int(parts[-1].replace('iter_', '')) if 'iter_' in filename else 0
        
        scene_entry = {
            "means3D": params["means3D"],
            "colors_precomp": params["rgb_colors"],
            "rotations": torch.nn.functional.normalize(params["unnorm_rotations"]),
            "opacities": torch.sigmoid(params.get("logit_opacities", torch.zeros(1))),
            "scales": torch.exp(params.get("log_scales", torch.zeros(1))),
            "means2D": torch.zeros_like(params["means3D"], device="cuda"),
            "timestep": t,
            "iteration": iter_num,
            "filename": str(cp_file.name)
        }
        checkpoints.append(scene_entry)
    
    return sorted(checkpoints, key=lambda x: (x['timestep'], x['iteration']))


def load_scene_data(seq: str, exp: str, out_dir: Path) -> list[dict]:
    """
    Load per-timestep Gaussian params from params.npz,
    return a list of length T where each entry is the dict
    the renderer expects for that timestep.
    """
    npz_path = out_dir / exp / seq / "params.npz"
    raw = dict(np.load(npz_path))
    params = {k: torch.tensor(v).cuda().float() for k, v in raw.items()}

    T = params["means3D"].shape[0]
    scene = []
    for t in range(T):
        scene.append(
            {
                "means3D": params["means3D"][t],
                "colors_precomp": params["rgb_colors"][t],
                "rotations": torch.nn.functional.normalize(
                    params["unnorm_rotations"][t]
                ),
                "opacities": torch.sigmoid(params["logit_opacities"]),
                "scales": torch.exp(params["log_scales"]),
                "means2D": torch.zeros_like(params["means3D"][0], device="cuda"),
            }
        )
    return scene


def tensor_to_pil(im: torch.Tensor) -> Image.Image:
    """
    Convert a [C,H,W] float32 tensor with values in [0,1]
    into a PIL Image (uint8 RGB).
    """
    arr = (
        (torch.permute(im, (1, 2, 0)).cpu().numpy() * 255.0)
        .clip(0, 255)
        .astype(np.uint8)
    )
    return Image.fromarray(arr)


def render_checkpoints(seq: str, exp: str, out_dir: Path, data_dir: Path, iteration: int = None):
    """
    Render checkpoints from training.
    If iteration is specified, renders that specific checkpoint.
    Otherwise renders all checkpoints found.
    """
    checkpoints = load_checkpoint_data(seq, exp, out_dir, iteration)
    
    if checkpoints is None:
        print(f"No checkpoints found in {out_dir / exp / seq / 'checkpoints'}")
        return
    
    meta_path = data_dir / seq / "train_meta.json"
    with open(meta_path, "r") as f:
        meta = json.load(f)
    
    base_dir = out_dir / exp / seq / "checkpoints"
    renders_base = base_dir / "renders"
    gt_base = base_dir / "gt"
    
    for cp in checkpoints:
        t = cp['timestep']
        iter_num = cp['iteration']
        
        scene_data = {
            "means3D": cp["means3D"],
            "colors_precomp": cp["colors_precomp"],
            "rotations": cp["rotations"],
            "opacities": cp["opacities"],
            "scales": cp["scales"],
            "means2D": cp["means2D"],
        }
        
        for c, (fn, ks, w2cs) in enumerate(zip(meta["fn"][t], meta["k"][t], meta["w2c"][t])):
            cam = setup_camera(w, h, np.array(ks), np.array(w2cs), near=near, far=far)
            with torch.no_grad():
                im, _, _ = Renderer(raster_settings=cam)(**scene_data)
            
            timestep_dir = renders_base / f"t{t:04d}" / f"cam{c:04d}"
            timestep_dir.mkdir(parents=True, exist_ok=True)
            
            gt_dir = gt_base / f"t{t:04d}" / f"cam{c:04d}"
            gt_dir.mkdir(parents=True, exist_ok=True)
            
            name = f"iter{iter_num:06d}.png"
            img = tensor_to_pil(im)
            img.save(timestep_dir / name)
            
            src = data_dir / seq / "ims" / fn
            dst = gt_dir / name
            shutil.copy(src, dst)
            
            print(f"Saved: t{t:04d}/cam{c:04d}/{name}")
    
    print(f"\nRendered {len(checkpoints)} checkpoint(s) to {base_dir}")


def render_and_save(seq: str, exp: str, out_dir: Path, data_dir: Path):
    """
    For each (timestep, view) in train_meta.json:
      1. grab scene[t]
      2. render with that view's (k, w2c)
      3. save to .../test/METHOD/renders/<t>_<c>.png
      4. copy GT from data_dir/.../ims/<fn> → .../test/METHOD/gt/<t>_<c>.png
    """
    # 1) load scene data (length = # timesteps)
    scene = load_scene_data(seq, exp, out_dir)

    # 2) load metadata & build flat list of views
    meta_path = data_dir / seq / "train_meta.json"
    with open(meta_path, "r") as f:
        meta = json.load(f)

    views = []
    for t, (fns, ks, w2cs) in enumerate(zip(meta["fn"], meta["k"], meta["w2c"])):
        for c, fn in enumerate(fns):
            views.append(
                {
                    "t": t,
                    "c": c,
                    "fn": fn,
                    "k": np.array(ks[c]),
                    "w2c": np.array(w2cs[c]),
                }
            )

    # 3) prepare output folders
    base = out_dir / exp / seq / "test" / METHOD
    renders_dir = base / "renders"
    gt_dir = base / "gt"
    fps_path = base / "fps.txt"
    renders_dir.mkdir(parents=True, exist_ok=True)
    gt_dir.mkdir(parents=True, exist_ok=True)

    timings = []
    # 4) render + copy GT for each view
    for view in views:
        ts = time.time()
        t, c, fn = view["t"], view["c"], view["fn"]
        data_vars = scene[t]  # pick the right timestep’s dict

        # build camera & render
        cam = setup_camera(w, h, view["k"], view["w2c"], near=near, far=far)
        with torch.no_grad():
            im, _, _ = Renderer(raster_settings=cam)(**data_vars)

        timings.append(time.time() - ts)
        # convert and save render
        img = tensor_to_pil(im)
        name = f"{t:04d}_{c:04d}.png"
        img.save(renders_dir / name)

        # copy the matching ground-truth image
        src = data_dir / seq / "ims" / fn
        dst = gt_dir / name
        shutil.copy(src, dst)

        print(f"Saved render → {renders_dir/name}    GT → {gt_dir/name}")

    # 5) save average FPS
    with open(fps_path, 'w') as f:
        total_time = sum(timings)
        f.write("0") if total_time < 1e-5 else f.write(str(len(views) / total_time))
    
    print("", flush=True)


if __name__ == "__main__":
    parser = ArgumentParser(
        description="Render every (t, view) and pair with GT for evaluation"
    )
    parser.add_argument("--exp-name", type=str, default="exp1")
    parser.add_argument("--output-dir", type=Path, default=Path("./output"))
    parser.add_argument("--data-dir", type=Path, default=Path("./data"))
    parser.add_argument(
        "--dataset",
        type=str,
        default="sphere-bounce-5",
        choices=["sphere-bounce", "sphere-bounce-5", "basketball", "boxes", "football", "juggle", "softball", "tennis"],
        help="Name of the dataset to use for training (e.g., basketball, boxes, etc.)",
    )
    parser.add_argument(
        "--checkpoint",
        type=int,
        default=None,
        help="Render specific checkpoint iteration. If not specified, renders all checkpoints.",
    )
    parser.add_argument(
        "--render-checkpoints-only",
        action="store_true",
        help="Only render checkpoints, skip final params rendering.",
    )
    args = parser.parse_args()

    print(f"\n=== Sequence: {args.dataset} ===", flush=True)

    if args.render_checkpoints_only or args.checkpoint is not None:
        render_checkpoints(args.dataset, args.exp_name, args.output_dir, args.data_dir, args.checkpoint)
    else:
        render_and_save(args.dataset, args.exp_name, args.output_dir, args.data_dir)
