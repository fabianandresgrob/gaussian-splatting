import os
import torch
from random import randint
from utils.loss_utils import l1_loss, ssim
from lpipsPyTorch import LPIPS
from gaussian_renderer import render
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
from PIL import Image
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import json
from view_selection import build_selector

TENSORBOARD_FOUND = True

def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from, view_selection_strategy, view_selection_config, use_gui=False):
    print(f"positions: init={opt.position_lr_init} final={opt.position_lr_final} delay_mult={opt.position_lr_delay_mult} max_steps={opt.position_lr_max_steps}")
    print(f"feature={opt.feature_lr} opacity={opt.opacity_lr} scaling={opt.scaling_lr} rotation={opt.rotation_lr}")
    print(f"densification: interval={opt.densification_interval} from={opt.densify_from_iter} until={opt.densify_until_iter} grad_threshold={opt.densify_grad_threshold}")
    first_iter = 0
    tb_writer = SummaryWriter(log_dir=dataset.model_path)
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)

    # Initialize View Selector
    strategy = view_selection_strategy
    try:
        config = json.loads(view_selection_config)
    except:
        config = {}

    selector = build_selector(strategy, config=config)
    selector.initialize(scene.getTrainCameras())

    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    
    # List to store metrics for retrospective analysis
    metrics_history = []

    first_iter += 1
    for iteration in range(first_iter, opt.iterations + 1):
        iter_start.record()

        gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        for _ in range(opt.optimizer_step_interval):
            # Pick a Camera using the strategy
            viewpoint_cam = selector.select_view(gaussians, iteration)

            # Render
            if (iteration - 1) == debug_from:
                pipe.debug = True

            bg = torch.rand((3), device="cuda") if opt.random_background else background

            render_pkg = render(viewpoint_cam, gaussians, pipe, bg)
            image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

            # Loss
            gt_image = viewpoint_cam.original_image.cuda()
            Ll1 = l1_loss(image, gt_image)
            loss = Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))
            loss /= opt.optimizer_step_interval  # Gradient accumulation
            loss.backward()

        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            # Modified to return metrics dict
            current_metrics = training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background))
            
            if current_metrics:
                metrics_history.append(current_metrics)
                # Save history to JSON incrementally
                with open(os.path.join(dataset.model_path, "metrics_history.json"), "w") as f:
                    json.dump(metrics_history, f, indent=4)

            if iteration in saving_iterations:
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            # Densification
            if iteration < opt.densify_until_iter:
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(opt.densify_grad_threshold, opt.prune_alpha_threshold, scene.cameras_extent, size_threshold)

                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()

            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none=True)

            if (iteration in checkpoint_iterations) or iteration == opt.iterations:
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")
    
    # Final evaluation
    final_results = eval_and_save(dataset.model_path, scene, render, (pipe, background))
    
    # Save final summary
    with open(os.path.join(dataset.model_path, "final_results.json"), "w") as f:
        json.dump(final_results, f, indent=4)


def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene: Scene, renderFunc, renderArgs):
    if tb_writer:
        tb_writer.add_scalar("train_loss_patches/l1_loss", Ll1.item(), iteration)
        tb_writer.add_scalar("train_loss_patches/total_loss", loss.item(), iteration)
        tb_writer.add_scalar("iter_time", elapsed, iteration)

    metrics_data = None

    # Report test and samples of training set
    if (iteration + 1) % testing_iterations[0] == 0:
        lpips = LPIPS(net_type="vgg").to("cuda")
        torch.cuda.empty_cache()
        validation_configs = (
            {"name": "test", "cameras": scene.getTestCameras()},
            {"name": "train", "cameras": scene.getTrainCameras()[::10]},
        )
        
        metrics_data = {"iteration": iteration, "elapsed_time": elapsed}

        for config in validation_configs:
            if config["cameras"] and len(config["cameras"]) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                ssim_test = 0.0
                lpips_test = 0.0
                for idx, viewpoint in enumerate(config["cameras"]):
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    
                    if tb_writer and (idx < 5):
                        image = image.unsqueeze(0)
                        # gt_image = gt_image.unsqueeze(0)
                        viz_image = torch.nn.functional.interpolate(image, scale_factor=0.25, mode="bilinear", align_corners=False)
                        tb_writer.add_images(config["name"] + "_view_{}/render".format(viewpoint.image_name), viz_image, global_step=iteration)

                    psnr_val = psnr(image, gt_image).mean()
                    ssim_val = ssim(image, gt_image)
                    lpips_val = lpips(image.unsqueeze(0), gt_image.unsqueeze(0))

                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr_val.item()
                    ssim_test += ssim_val.item()
                    lpips_test += lpips_val.item()
                
                count = len(config["cameras"])
                psnr_test /= count
                ssim_test /= count
                lpips_test /= count
                l1_test /= count
                
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config["name"], l1_test, psnr_test))
                
                # Store in dict
                metrics_data[config["name"]] = {
                    "L1": float(l1_test),
                    "PSNR": float(psnr_test),
                    "SSIM": float(ssim_test),
                    "LPIPS": float(lpips_test)
                }

                if tb_writer:
                    tb_writer.add_scalar(config["name"] + "/loss_viewpoint - l1_loss", l1_test, iteration)
                    tb_writer.add_scalar(config["name"] + "/loss_viewpoint - psnr", psnr_test, iteration)
                    tb_writer.add_scalar(config["name"] + "/loss_viewpoint - ssim", ssim_test, iteration)
                    tb_writer.add_scalar(config["name"] + "/loss_viewpoint - lpips", lpips_test, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar("total_points", scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()
    
    return metrics_data


@torch.no_grad()
def eval_and_save(model_path, scene: Scene, renderFunc, renderArgs):
    cameras = scene.getTestCameras()
    assert len(cameras) > 0, "No test cameras found"
    lpips = LPIPS(net_type="vgg").to("cuda")

    all_psnr = []
    all_ssim = []
    all_lpips = []
    render_out_dir = os.path.join(model_path, "eval")
    print("Output folder: {}".format(render_out_dir))
    combine_out_dir = os.path.join(model_path, "combine")
    os.makedirs(render_out_dir, exist_ok=True)
    os.makedirs(combine_out_dir, exist_ok=True)

    for idx, viewpoint in enumerate(cameras):
        image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"], 0.0, 1.0)
        gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
        save_path = os.path.join(render_out_dir, viewpoint.image_name + ".png")
        image_np = (image * 255.0).permute(1, 2, 0).detach().cpu().byte().numpy()
        gt_image_np = (gt_image * 255.0).permute(1, 2, 0).detach().cpu().byte().numpy()
        Image.fromarray(image_np).save(save_path)
        save_path = os.path.join(combine_out_dir, viewpoint.image_name + ".png")
        Image.fromarray(np.concatenate((image_np, gt_image_np), axis=1)).save(save_path)

        image = image.unsqueeze(0)
        gt_image = gt_image.unsqueeze(0)
        psnr_val = psnr(image, gt_image)
        ssim_val = ssim(image, gt_image)
        lpips_val = lpips(image, gt_image)

        all_psnr.append(psnr_val.item())
        all_ssim.append(ssim_val.item())
        all_lpips.append(lpips_val.item())
    
    results = {
        "mean_psnr": float(np.mean(all_psnr)),
        "mean_ssim": float(np.mean(all_ssim)),
        "mean_lpips": float(np.mean(all_lpips))
    }
    
    print("Evaluation results:")
    print("PSNR: {}".format(results["mean_psnr"]))
    print("SSIM: {}".format(results["mean_ssim"]))
    print("LPIPS: {}".format(results["mean_lpips"]))
    return results


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument("--debug_from", type=int, default=-1)
    parser.add_argument("--detect_anomaly", action="store_true", default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[1000, 7000, 30000]) # Added 1000 for early curve
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default=None)
    parser.add_argument("--view_selection_strategy", type=str, default="random", 
                        choices=["random", "fixed_prob", "epoch_based", "clustering", "no_replace"])
    parser.add_argument("--view_selection_config", type=str, default="{}")

    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)

    print("Optimizing " + args.model_path)

    safe_state(args.quiet)

    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from, args.view_selection_strategy, args.view_selection_config, use_gui=False)

    print("\nTraining complete.")