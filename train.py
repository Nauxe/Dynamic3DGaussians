from argparse import ArgumentParser
import torch
import os
import json
import copy
import numpy as np
from PIL import Image
from random import randint
from tqdm import tqdm
from diff_gaussian_rasterization import GaussianRasterizer as Renderer
from helpers import setup_camera, l1_loss_v1, l1_loss_v2, weighted_l2_loss_v1, weighted_l2_loss_v2, quat_mult, \
    o3d_knn, params2rendervar, params2cpu, save_params
from external import calc_ssim, calc_psnr, build_rotation, densify, update_params_and_optimizer


def get_dataset(t, md, seq, data_dir):
    dataset = []
    for c in range(len(md['fn'][t])):
        w, h, k, w2c = md['w'], md['h'], md['k'][t][c], md['w2c'][t][c]
        cam = setup_camera(w, h, k, w2c, near=1.0, far=100)
        fn = md['fn'][t][c]
        im = np.array(copy.deepcopy(Image.open(f"{data_dir}/{seq}/ims/{fn}")))[:,:,:3]
        im = torch.tensor(im).float().cuda().permute(2, 0, 1) / 255
        seg = np.array(copy.deepcopy(Image.open(f"{data_dir}/{seq}/seg/{fn.replace('.jpg', '.png')}"))).astype(np.float32)
        seg = torch.tensor(seg).float().cuda()
        seg_col = torch.stack((seg, torch.zeros_like(seg), 1 - seg))
        dataset.append({'cam': cam, 'im': im, 'seg': seg_col, 'id': c})
    return dataset


def get_batch(todo_dataset, dataset):
    if not todo_dataset:
        todo_dataset = dataset.copy()
    curr_data = todo_dataset.pop(randint(0, len(todo_dataset) - 1))
    return curr_data


def initialize_params(seq, md, data_dir):
    init_pt_cld = np.load(f"{data_dir}/{seq}/init_pt_cld.npz")["data"]
    seg = init_pt_cld[:, 6]
    max_cams = 50
    sq_dist, _ = o3d_knn(init_pt_cld[:, :3], 3)
    mean3_sq_dist = sq_dist.mean(-1).clip(min=0.0000001)
    params = {
        'means3D': init_pt_cld[:, :3],
        'rgb_colors': init_pt_cld[:, 3:6],
        'seg_colors': np.stack((seg, np.zeros_like(seg), 1 - seg), -1),
        'unnorm_rotations': np.tile([1, 0, 0, 0], (seg.shape[0], 1)),
        'logit_opacities': np.zeros((seg.shape[0], 1)),
        'log_scales': np.tile(np.log(np.sqrt(mean3_sq_dist))[..., None], (1, 3)),
        'cam_m': np.zeros((max_cams, 3)),
        'cam_c': np.zeros((max_cams, 3)),
    }
    params = {k: torch.nn.Parameter(torch.tensor(v).cuda().float().contiguous().requires_grad_(True)) for k, v in
              params.items()}
    cam_centers = np.linalg.inv(md['w2c'][0])[:, :3, 3]  # Get scene radius
    scene_radius = 1.1 * np.max(np.linalg.norm(cam_centers - np.mean(cam_centers, 0)[None], axis=-1))
    variables = {'max_2D_radius': torch.zeros(params['means3D'].shape[0]).cuda().float(),
                 'scene_radius': scene_radius,
                 'means2D_gradient_accum': torch.zeros(params['means3D'].shape[0]).cuda().float(),
                 'denom': torch.zeros(params['means3D'].shape[0]).cuda().float()}
    return params, variables


def initialize_optimizer(params, variables):
    lrs = {
        'means3D': 0.00016 * variables['scene_radius'],
        'rgb_colors': 0.0025,
        'seg_colors': 0.0,
        'unnorm_rotations': 0.001,
        'logit_opacities': 0.05,
        'log_scales': 0.001,
        'cam_m': 1e-4,
        'cam_c': 1e-4,
    }
    param_groups = [{'params': [v], 'name': k, 'lr': lrs[k]} for k, v in params.items()]
    return torch.optim.Adam(param_groups, lr=0.0, eps=1e-15)


def get_loss(params, curr_data, variables, is_initial_timestep):
    losses = {}

    rendervar = params2rendervar(params)
    rendervar['means2D'].retain_grad()
    im, radius, _, = Renderer(raster_settings=curr_data['cam'])(**rendervar)
    curr_id = curr_data['id']
    im = torch.exp(params['cam_m'][curr_id])[:, None, None] * im + params['cam_c'][curr_id][:, None, None]
    losses['im'] = 0.8 * l1_loss_v1(im, curr_data['im']) + 0.2 * (1.0 - calc_ssim(im, curr_data['im']))
    variables['means2D'] = rendervar['means2D']  # Gradient only accum from colour render for densification

    # segrendervar = params2rendervar(params)
    # segrendervar['colors_precomp'] = params['seg_colors']
    # seg, _, _, = Renderer(raster_settings=curr_data['cam'])(**segrendervar)
    # losses['seg'] = 0.8 * l1_loss_v1(seg, curr_data['seg']) + 0.2 * (1.0 - calc_ssim(seg, curr_data['seg']))

    if not is_initial_timestep:
        is_fg = (params['seg_colors'][:, 0] > 0.5).detach()
        fg_pts = rendervar['means3D'][is_fg]
        fg_rot = rendervar['rotations'][is_fg]

        rel_rot = quat_mult(fg_rot, variables["prev_inv_rot_fg"])
        rot = build_rotation(rel_rot)
        neighbor_pts = fg_pts[variables["neighbor_indices"]]
        curr_offset = neighbor_pts - fg_pts[:, None]
        curr_offset_in_prev_coord = (rot.transpose(2, 1)[:, None] @ curr_offset[:, :, :, None]).squeeze(-1)
        losses['rigid'] = weighted_l2_loss_v2(curr_offset_in_prev_coord, variables["prev_offset"],
                                              variables["neighbor_weight"])

        losses['rot'] = weighted_l2_loss_v2(rel_rot[variables["neighbor_indices"]], rel_rot[:, None],
                                            variables["neighbor_weight"])

        curr_offset_mag = torch.sqrt((curr_offset ** 2).sum(-1) + 1e-20)
        # losses['iso'] = weighted_l2_loss_v1(curr_offset_mag, variables["neighbor_dist"], variables["neighbor_weight"])

        # losses['floor'] = torch.clamp(fg_pts[:, 1], min=0).mean()
        #
        # bg_pts = rendervar['means3D'][~is_fg]
        # bg_rot = rendervar['rotations'][~is_fg]
        # losses['bg'] = l1_loss_v2(bg_pts, variables["init_bg_pts"]) + l1_loss_v2(bg_rot, variables["init_bg_rot"])
        
        losses['soft_col_cons'] = l1_loss_v2(params['rgb_colors'], variables["prev_col"])

    loss_weights = {'im': 1.0, 'seg': 3.0, 'rigid': 4.0, 'rot': 4.0, 'iso': 2.0, 'floor': 2.0, 'bg': 20.0,
                    'soft_col_cons': 0.01}
    loss = sum([loss_weights[k] * v for k, v in losses.items()])
    seen = radius > 0
    variables['max_2D_radius'][seen] = torch.max(radius[seen], variables['max_2D_radius'][seen])
    variables['seen'] = seen
    return loss, variables


def initialize_per_timestep(params, variables, optimizer):
    pts = params['means3D']
    rot = torch.nn.functional.normalize(params['unnorm_rotations'])
    new_pts = pts + (pts - variables["prev_pts"])
    new_rot = torch.nn.functional.normalize(rot + (rot - variables["prev_rot"]))

    is_fg = params['seg_colors'][:, 0] > 0.5
    prev_inv_rot_fg = rot[is_fg]
    prev_inv_rot_fg[:, 1:] = -1 * prev_inv_rot_fg[:, 1:]
    fg_pts = pts[is_fg]
    prev_offset = fg_pts[variables["neighbor_indices"]] - fg_pts[:, None]
    variables['prev_inv_rot_fg'] = prev_inv_rot_fg.detach()
    variables['prev_offset'] = prev_offset.detach()
    variables["prev_col"] = params['rgb_colors'].detach()
    variables["prev_pts"] = pts.detach()
    variables["prev_rot"] = rot.detach()

    new_params = {'means3D': new_pts, 'unnorm_rotations': new_rot}
    params = update_params_and_optimizer(new_params, params, optimizer)

    return params, variables


def initialize_post_first_timestep(params, variables, optimizer, num_knn=20):
    is_fg = params['seg_colors'][:, 0] > 0.5
    init_fg_pts = params['means3D'][is_fg]
    init_bg_pts = params['means3D'][~is_fg]
    init_bg_rot = torch.nn.functional.normalize(params['unnorm_rotations'][~is_fg])
    neighbor_sq_dist, neighbor_indices = o3d_knn(init_fg_pts.detach().cpu().numpy(), num_knn)
    neighbor_weight = np.exp(-2000 * neighbor_sq_dist)
    neighbor_dist = np.sqrt(neighbor_sq_dist)
    variables["neighbor_indices"] = torch.tensor(neighbor_indices).cuda().long().contiguous()
    variables["neighbor_weight"] = torch.tensor(neighbor_weight).cuda().float().contiguous()
    variables["neighbor_dist"] = torch.tensor(neighbor_dist).cuda().float().contiguous()

    variables["init_bg_pts"] = init_bg_pts.detach()
    variables["init_bg_rot"] = init_bg_rot.detach()
    variables["prev_pts"] = params['means3D'].detach()
    variables["prev_rot"] = torch.nn.functional.normalize(params['unnorm_rotations']).detach()
    params_to_fix = ['logit_opacities', 'log_scales', 'cam_m', 'cam_c']
    for param_group in optimizer.param_groups:
        if param_group["name"] in params_to_fix:
            param_group['lr'] = 0.0
    return variables


def report_progress(params, data, i, current_loss, best_loss, window_avg, loss_history, window_size, progress_bar, every_i=100):
    if i % every_i == 0:
        im, _, _, = Renderer(raster_settings=data['cam'])(**params2rendervar(params))
        curr_id = data['id']
        im = torch.exp(params['cam_m'][curr_id])[:, None, None] * im + params['cam_c'][curr_id][:, None, None]
        psnr = calc_psnr(im, data['im']).mean()
        progress_bar.set_postfix({
            "train img 0 PSNR": f"{psnr:.{7}f}",  
            'loss': f'{current_loss:.4f}',
            'best': f'{best_loss:.4f}',
            'window': f'{window_avg if len(loss_history) >= window_size else 0:.4f}'
})
        progress_bar.update(every_i)


def train(seq, exp, data_dir, output_dir):
    if os.path.exists(f"{output_dir}/{exp}/{seq}"):
        print(f"Experiment '{exp}' for sequence '{seq}' already exists. Exiting.")
        return

    # LX: Create log file for loss
    log_file = f"{output_dir}/{exp}/{seq}/training_log.txt"
    os.makedirs(f"{output_dir}/{exp}/{seq}", exist_ok=True)

    md = json.load(open(f"{data_dir}/{seq}/train_meta.json", 'r'))  # metadata
    num_timesteps = len(md['fn'])
    params, variables = initialize_params(seq, md, data_dir)
    optimizer = initialize_optimizer(params, variables)
    output_params = []

    # Open log file for writing
    with open(log_file, 'w') as f:
        f.write(f"Training log for sequence: {seq}, experiment: {exp}\n")
        f.write("="*50 + "\n")
        f.write(f"Total timesteps: {num_timesteps}\n\n")

    for t in range(num_timesteps):
        dataset = get_dataset(t, md, seq, data_dir)
        todo_dataset = []
        is_initial_timestep = (t == 0)
        if not is_initial_timestep:
            params, variables = initialize_per_timestep(params, variables, optimizer)
        
        # LX Changes: Train till convergence
        # num_iter_per_timestep = 10000 if is_initial_timestep else 2000
        # num_iter_per_timestep = 30000 if is_initial_timestep else 2000
        i = 0
        max_iter = 100000 if is_initial_timestep else 30000 # Initial: 100000
        min_iter = 20000 if is_initial_timestep else 3000 # Initial: 20000
        window_size = 100  # Window for moving average
        patience = 5  # Number of windows to wait for improvement
        percent_tol = 1e-7 # Convergence tolerance as a percentage
        # rel_tol = 1e-7  # Relative tolerance for loss improvement
        # abs_tol = 1e-8  # Absolute tolerance for loss improvement
        
        loss_history = []
        window_losses = []
        best_loss = float('inf')
        best_iter = 0
        no_improve_count = 0
        has_converged = False
        psnr_value = 0
        
        # Create timestep-specific log file
        timestep_log = f"{output_dir}/{exp}/{seq}/timestep_{t}_loss.txt"
        with open(timestep_log, 'w') as f:
            f.write(f"Iteration,Loss,Im,Seg,Rigid,Rot,Iso,Floor,Bg,Soft_Col_Cons,PSNR,Window_Avg,Best_Loss\n")
        
        # progress_bar = tqdm(range(num_iter_per_timestep), desc=f"timestep {t}")
        # for i in range(num_iter_per_timestep):
        progress_bar = tqdm(range(max_iter), desc=f"timestep {t}")
        while not has_converged and i < max_iter:
            curr_data = get_batch(todo_dataset, dataset)
            loss, variables = get_loss(params, curr_data, variables, is_initial_timestep)
            current_loss = loss.item()
            
            # Calculate PSNR for logging
            rendervar = params2rendervar(params)
            im_rendered, _, _, = Renderer(raster_settings=curr_data['cam'])(**rendervar)
            curr_id = curr_data['id']
            im_rendered = torch.exp(params['cam_m'][curr_id])[:, None, None] * im_rendered + params['cam_c'][curr_id][:, None, None]
            psnr_value = calc_psnr(im_rendered, curr_data['im']).mean().item()

            # Track best loss
            if current_loss < best_loss:
                best_loss = current_loss
                best_iter = i

            # Store loss history
            loss_history.append(current_loss)
            
            # Compute moving average over window
            window_avg = -1
            if len(loss_history) >= window_size:
                window_avg = sum(loss_history[-window_size:]) / window_size
                window_losses.append(window_avg)
                
                # Check convergence criteria after minimum iterations
                if i >= min_iter and len(window_losses) >= patience:
                    # Method 1: Check if loss has stabilized (small relative change)
                    recent_avg = window_losses[-1]
                    older_avg = window_losses[-patience]
                    
                    rel_change = abs(recent_avg - older_avg) / (abs(older_avg) + 1e-8)
                    abs_change = abs(recent_avg - older_avg)
                    
                    # Percentage‑based absolute change (relative to the older average)
                    perc_abs_change = abs_change / (abs(older_avg) + 1e-8)
                    
                    # Method 2: Check if no improvement in best loss for patience windows
                    if best_iter <= i - window_size * patience:
                        no_improve_count += 1
                    else:
                        no_improve_count = 0
                    
                    # Convergence conditions
                    stable_loss = rel_change < percent_tol or perc_abs_change < percent_tol 
                    no_improvement = no_improve_count >= 4
                    
                    # Additional condition: gradient norm (if available)
                    # grad_norm = compute_gradient_norm(params)
                    
                    if stable_loss or no_improvement:
                        has_converged = True
                        convergence_reason = "Stabilized loss" if stable_loss else "No improvement"
                        
                        # Log detailed convergence info
                        with open(timestep_log, 'a') as f:
                            f.write(f"{i},{current_loss:.6f},"
                                   f"{psnr_value:.6f},"
                                   f"{window_avg:.6f},{best_loss:.6f}\n")
                        
                        # Early stop message
                        im, _, _, = Renderer(raster_settings=dataset[0]['cam'])(**params2rendervar(params))
                        curr_id = dataset[0]['id']
                        im = torch.exp(params['cam_m'][curr_id])[:, None, None] * im + params['cam_c'][curr_id][:, None, None]
                        psnr = calc_psnr(im, dataset[0]['im']).mean()
                        progress_bar.set_postfix({
                            "train img 0 PSNR": f"{psnr:.{7}f}",  
                            'loss': f'{current_loss:.4f}',
                            'best': f'{best_loss:.4f}',
                            'window': f'{window_avg if len(loss_history) >= window_size else 0:.4f}',
                            'status': f'CONVERGED ({convergence_reason})'
                        }) 
                        break
            
            # Log every iteration
            with open(timestep_log, 'a') as f:
                f.write(f"{i},{current_loss:.6f},"
                       f"{psnr_value:.6f},"
                       f"{window_avg if window_avg != -1 else 0:.6f},{best_loss:.6f}\n")

            loss.backward()
            with torch.no_grad():
                report_progress(params, dataset[0], i, current_loss, best_loss, window_avg, loss_history, window_size, progress_bar)
                if is_initial_timestep:
                    # first timestep: densify for first 500-5000 iter avery 100 iter 
                    params, variables = densify(params, variables, optimizer, i)
                    # if not is_initial_timestep:
                    #     variables = recompute_neighbor_struct(params, variables)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                
            i += 1


        progress_bar.close()

        # Log final status
        if not has_converged and i >= max_iter:
            convergence_status = f"Reached max iterations ({max_iter})"
        else:
            convergence_status = f"Converged after {i} iterations"
        
        # Log convergence info to main log
        with open(log_file, 'a') as f:
            f.write(f"Timestep {t}: {convergence_status}\n")
            f.write(f"  - Final loss: {loss_history[-1]:.6f}\n")
            f.write(f"  - Best loss: {best_loss:.6f} at iteration {best_iter}\n")
            f.write(f"  - Total iterations: {i}\n")
            f.write(f"  - Final PSNR: {psnr_value}\n\n")
        
        # Save loss history plot (optional - requires matplotlib)
        try:
            import matplotlib.pyplot as plt
            plt.figure(figsize=(10, 6))
            plt.plot(loss_history, label='Loss')
            plt.axhline(y=best_loss, color='r', linestyle='--', label=f'Best loss: {best_loss:.6f}')
            plt.xlabel('Iteration')
            plt.ylabel('Loss')
            plt.title(f'Timestep {t} - Loss History')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.savefig(f"{output_dir}/{exp}/{seq}/timestep_{t}_loss_plot.png")
            plt.close()
        except ImportError:
            pass  # matplotlib not available
        
        output_params.append(params2cpu(params, is_initial_timestep))

        if is_initial_timestep:
            variables = initialize_post_first_timestep(params, variables, optimizer)

    save_params(output_params, seq, exp, output_dir)



if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--data-dir", type=str, default="./data", help="Path to the data directory"
    )
    parser.add_argument("--exp-name", type=str, default="sphere-bounce-5", help="Experiment name")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./output",
        help="Path to the output directory",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="sphere-bounce-5",
        choices=["sphere-bounce-5", "basketball", "boxes", "football", "juggle", "softball", "tennis"],
        help="Name of the dataset to use for training (e.g., basketball, boxes, etc.)",
    )
    args = parser.parse_args()
    train(args.dataset, args.exp_name, args.data_dir, args.output_dir)
