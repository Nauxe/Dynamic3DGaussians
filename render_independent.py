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

w, h = 625, 625
near, far = 0.01, 100.0

METHOD = "ours-independent-625x625"


def tensor_to_pil(im: torch.Tensor) -> Image.Image:
    arr = (
        (torch.permute(im, (1, 2, 0)).cpu().numpy() * 255.0)
        .clip(0, 255)
        .astype(np.uint8)
    )
    return Image.fromarray(arr)


def load_scene_data(seq: str, exp: str, out_dir: Path):
    """
    Load per-timestep scene parameters from independent training output.
    Each timestep is saved in its own folder: timestep_0/params.npz, timestep_1/params.npz, etc.
    """
    base_dir = out_dir / exp / seq
    scene = []
    t = 0
    
    while True:
        timestep_dir = base_dir / f"timestep_{t}"
        npz_path = timestep_dir / "params.npz"
        
        if not npz_path.exists():
            break
        
        raw = dict(np.load(npz_path, allow_pickle=True))
        params = {k: torch.tensor(v).cuda().float() for k, v in raw.items()}
        
        scene.append({
            "means3D": params["means3D"],
            "colors_precomp": params["rgb_colors"],
            "rotations": torch.nn.functional.normalize(params["unnorm_rotations"]),
            "opacities": torch.sigmoid(params["logit_opacities"]),
            "scales": torch.exp(params["log_scales"]),
            "means2D": torch.zeros_like(params["means3D"], device="cuda"),
        })
        t += 1
    
    return scene


def render_and_save(seq: str, exp: str, out_dir: Path, data_dir: Path):
    scene = load_scene_data(seq, exp, out_dir)

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

    base = out_dir / exp / seq / "test" / METHOD
    renders_dir = base / "renders"
    gt_dir = base / "gt"
    fps_path = base / "fps.txt"
    renders_dir.mkdir(parents=True, exist_ok=True)
    gt_dir.mkdir(parents=True, exist_ok=True)

    timings = []
    for view in views:
        ts = time.time()
        t, c, fn = view["t"], view["c"], view["fn"]
        data_vars = scene[t]

        cam = setup_camera(w, h, view["k"], view["w2c"], near=near, far=far)
        with torch.no_grad():
            im, _, _ = Renderer(raster_settings=cam)(**data_vars)

        timings.append(time.time() - ts)
        img = tensor_to_pil(im)
        name = f"{t:04d}_{c:04d}.png"
        img.save(renders_dir / name)

        src = data_dir / seq / "ims" / fn
        dst = gt_dir / name
        shutil.copy(src, dst)

        print(f"Saved render -> {renders_dir/name}    GT -> {gt_dir/name}")

    with open(fps_path, 'w') as f:
        total_time = sum(timings)
        f.write("0") if total_time < 1e-5 else f.write(str(len(views) / total_time))
    
    print("", flush=True)


if __name__ == "__main__":
    parser = ArgumentParser(
        description="Render independent training outputs"
    )
    parser.add_argument("--exp-name", type=str, default="sphere-bounce-5")
    parser.add_argument("--output-dir", type=Path, default=Path("./output-indep"))
    parser.add_argument("--data-dir", type=Path, default=Path("./data"))
    parser.add_argument(
        "--dataset",
        type=str,
        default="sphere-bounce-5",
        choices=["sphere-bounce-5", "basketball", "boxes", "football", "juggle", "softball", "tennis"],
        help="Name of the dataset",
    )
    args = parser.parse_args()

    print(f"\n=== Sequence: {args.dataset} ===", flush=True)
    render_and_save(args.dataset, args.exp_name, args.output_dir, args.data_dir)
