import os
import torch
from utils.loss_utils import l1_loss, ssim
from lpipsPyTorch import LPIPS
from gaussian_renderer import render
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, OptimizationParams
from PIL import Image
import numpy as np
import json
from view_selection import build_selector, configure_logging

# Optional logging backends
TENSORBOARD_FOUND = False
WANDB_FOUND = False

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    pass

try:
    import wandb
    WANDB_FOUND = True
except ImportError:
    pass


class Logger:
    """Unified logger supporting TensorBoard and Weights & Biases."""
    
    def __init__(self, log_dir, backend="tensorboard", project_name="3dgs-view-selection", run_name=None, config=None, entity=None):
        self.backend = backend
        self.log_dir = log_dir
        self.writer = None
        self.wandb_run = None
        
        if backend == "tensorboard" and TENSORBOARD_FOUND:
            self.writer = SummaryWriter(log_dir=log_dir)
            print(f"[Logger] TensorBoard initialized at {log_dir}")
        elif backend == "wandb" and WANDB_FOUND:
            self.wandb_run = wandb.init(
                project=project_name,
                entity=entity,  # Team/organization name, None = personal account
                name=run_name or os.path.basename(log_dir),
                config=config or {},
                dir=log_dir,
                reinit=True
            )
            print(f"[Logger] W&B initialized: {self.wandb_run.url}")
        elif backend == "none":
            print("[Logger] Logging disabled")
        else:
            print(f"[Logger] Backend '{backend}' not available. Logging disabled.")
    
    def add_scalar(self, tag, value, step):
        if self.writer:
            self.writer.add_scalar(tag, value, step)
        if self.wandb_run:
            wandb.log({tag: value}, step=step)
    
    def add_scalars(self, main_tag, tag_scalar_dict, step):
        """Log multiple scalars at once."""
        for tag, value in tag_scalar_dict.items():
            self.add_scalar(f"{main_tag}/{tag}", value, step)
    
    def add_image(self, tag, img_tensor, step):
        if self.writer:
            self.writer.add_images(tag, img_tensor, global_step=step)
        if self.wandb_run:
            # Convert tensor to wandb Image
            if img_tensor.dim() == 4:
                img_tensor = img_tensor[0]  # Take first image if batched
            img_np = (img_tensor.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            wandb.log({tag: wandb.Image(img_np)}, step=step)
    
    def add_histogram(self, tag, values, step):
        if self.writer:
            self.writer.add_histogram(tag, values, step)
        if self.wandb_run:
            wandb.log({tag: wandb.Histogram(values.cpu().numpy())}, step=step)
    
    def log_metrics(self, metrics_dict, step):
        """Log a dictionary of metrics."""
        if self.writer:
            for key, value in metrics_dict.items():
                self.writer.add_scalar(key, value, step)
        if self.wandb_run:
            wandb.log(metrics_dict, step=step)
    
    def finish(self):
        if self.writer:
            self.writer.close()
        if self.wandb_run:
            wandb.finish()

def training(
    dataset,
    opt,
    pipe,
    testing_iterations,
    saving_iterations,
    checkpoint_iterations,
    checkpoint,
    debug_from,
    view_selection_strategy,
    view_selection_config,
    seed,
    logger_backend="tensorboard",
    wandb_project="3dgs-view-selection",
    wandb_entity=None,
    use_gui: bool = False,
    no_checkpoints: bool = False,
    checkpoint_on_interrupt: bool = False,
    disable_view_selection_logs: bool = False,
    view_selection_verbose: bool = True,
    skip_final_eval: bool = False,
    log_distribution_snapshots: bool = False,
):
    print(f"positions: init={opt.position_lr_init} final={opt.position_lr_final} delay_mult={opt.position_lr_delay_mult} max_steps={opt.position_lr_max_steps}")
    print(f"feature={opt.feature_lr} opacity={opt.opacity_lr} scaling={opt.scaling_lr} rotation={opt.rotation_lr}")
    print(f"densification: interval={opt.densification_interval} from={opt.densify_from_iter} until={opt.densify_until_iter} grad_threshold={opt.densify_grad_threshold}")
    first_iter = 0
    
    # Initialize logger (TensorBoard, W&B, or none)
    run_name = f"{view_selection_strategy}_seed{seed}"
    logger_config = {
        "strategy": view_selection_strategy,
        "seed": seed,
        "iterations": opt.iterations,
        "source_path": dataset.source_path,
    }
    logger = Logger(
        log_dir=dataset.model_path,
        backend=logger_backend,
        project_name=wandb_project,
        run_name=run_name,
        config=logger_config,
        entity=wandb_entity
    )
    
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians)

    # ---------------------------------------------------------------------
    # Train/Test split logging
    # ---------------------------------------------------------------------
    def _safe_pct(n: int, d: int) -> float:
        return float(n) / float(d) if d > 0 else 0.0

    def _infer_dataset_type(source_path: str) -> str:
        if os.path.isdir(os.path.join(source_path, "sparse")):
            return "colmap"
        if os.path.exists(os.path.join(source_path, "nerfstudio", "transforms_undistorted.json")) or os.path.exists(
            os.path.join(source_path, "nerfstudio", "transforms.json")
        ):
            return "scannetpp"
        if os.path.exists(os.path.join(source_path, "transforms_train.json")):
            return "blender"
        return "unknown"

    def _infer_scannetpp_split_method(source_path: str) -> str:
        transforms_path = os.path.join(source_path, "nerfstudio", "transforms_undistorted.json")
        if not os.path.exists(transforms_path):
            transforms_path = os.path.join(source_path, "nerfstudio", "transforms.json")

        try:
            with open(transforms_path, "r") as f:
                transforms = json.load(f)
        except Exception:
            return "unknown"

        test_frames = transforms.get("test_frames", None)
        if not test_frames:
            return "fallback_seeded"

        # Check whether test_frames contain any valid entries after filtering.
        images_dir = os.path.join(source_path, "resized_undistorted_images")
        extrinsics_path = os.path.join(source_path, "colmap", "images.txt")
        try:
            from scene.colmap_loader import read_extrinsics_text

            extr = read_extrinsics_text(extrinsics_path)
            names = set()
            for _, image in extr.items():
                fn = os.path.basename(image.name)
                names.add(fn)
                names.add(fn.lower())
        except Exception:
            names = set()

        def _valid_frame(fr: dict) -> bool:
            fp = fr.get("file_path", "")
            if not fp:
                return False
            if not os.path.exists(os.path.join(images_dir, fp)):
                return False
            if names:
                return (fp in names) or (fp.lower() in names)
            return True

        valid_cnt = sum(1 for fr in test_frames if _valid_frame(fr))
        return "nerfstudio_test_frames" if valid_cnt > 0 else "fallback_seeded"

    try:
        train_cams = scene.getTrainCameras()
        test_cams = scene.getTestCameras()
        n_train = len(train_cams)
        n_test = len(test_cams)
        n_total = n_train + n_test

        dataset_type = _infer_dataset_type(dataset.source_path)
        split_method = "unknown"
        split_params = {}

        if dataset_type == "colmap":
            if getattr(dataset, "eval", False):
                llffhold = 8
                split_method = "llff_hold"
                split_params = {"llffhold": llffhold}
            else:
                split_method = "none"
        elif dataset_type == "scannetpp":
            split_method = _infer_scannetpp_split_method(dataset.source_path)
            if split_method == "fallback_seeded":
                split_params = {"seed": 0, "max_test": 10}
        elif dataset_type == "blender":
            split_method = "transforms_json"

        split_summary = {
            "source_path": dataset.source_path,
            "dataset_type": dataset_type,
            "eval": bool(getattr(dataset, "eval", False)),
            "train_test_exp": bool(getattr(dataset, "train_test_exp", False)),
            "num_total": n_total,
            "num_train": n_train,
            "num_test": n_test,
            "pct_test": _safe_pct(n_test, n_total),
            "split_method": split_method,
            "split_params": split_params,
            "test_image_names": sorted([c.image_name for c in test_cams]),
        }

        out_path = os.path.join(dataset.model_path, "split_summary.json")
        with open(out_path, "w") as f:
            json.dump(split_summary, f, indent=2)

        print(
            f"[Split] dataset={dataset_type} method={split_method} "
            f"train={n_train} test={n_test} total={n_total} pct_test={split_summary['pct_test']:.4f} "
            f"(logged to {out_path})"
        )
    except Exception as e:
        print(f"[Split] Warning: failed to write split summary: {e}")

    gaussians.training_setup(opt)
    if checkpoint:
        try:
            (model_params, first_iter) = torch.load(checkpoint)
            gaussians.restore(model_params, opt)
            print(f"[Checkpoint] Resumed from {checkpoint} at iter {first_iter}")
        except Exception as e:
            print(f"[Checkpoint] Failed to load '{checkpoint}', starting from scratch: {e}")
            first_iter = 0

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)

    # Initialize View Selector
    strategy = view_selection_strategy
    try:
        config = json.loads(view_selection_config)
    except Exception:
        config = {}

    # Provide dataset root to selectors for robust path resolution (e.g., DINO embeddings)
    if isinstance(config, dict) and getattr(dataset, "source_path", None):
        config.setdefault("source_path", dataset.source_path)

    # Configure logging for view selection module
    # When running large ablations, writing per-iteration selection logs can be unnecessary
    selection_log_dir = None if disable_view_selection_logs else dataset.model_path
    configure_logging(
        log_dir=selection_log_dir,
        level="DEBUG" if view_selection_verbose else "INFO",
        use_tqdm_handler=True,
    )

    selector = build_selector(
        strategy,
        config=config,
        log_dir=selection_log_dir,
        verbose=view_selection_verbose,
        seed=seed,
    )
    selector.initialize(scene.getTrainCameras())

    # Set up render context for deterministic_max_loss selector
    if hasattr(selector, 'set_render_context'):
        def combined_loss_fn(rendered, gt):
            """Combined L1 + D-SSIM loss for max-loss selector."""
            return l1_loss(rendered, gt) + opt.lambda_dssim * (1.0 - ssim(rendered, gt))
        selector.set_render_context(render, pipe, background, combined_loss_fn)

    ema_loss_for_log = 0.0
    start_iter = first_iter + 1
    progress_bar = tqdm(range(start_iter, opt.iterations + 1), desc="Training progress")
    
    # List to store metrics for retrospective analysis
    metrics_history = []

    def _save_checkpoint(iteration: int, suffix: str = "") -> None:
        filename = f"chkpnt{iteration}{suffix}.pth"
        path = os.path.join(scene.model_path, filename)
        torch.save((gaussians.capture(), iteration), path)

    last_completed_iteration = first_iter

    try:
        for iteration in range(start_iter, opt.iterations + 1):
            last_completed_iteration = iteration
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
                image = render_pkg["render"]
                viewspace_point_tensor = render_pkg["viewspace_points"]
                visibility_filter = render_pkg["visibility_filter"]
                radii = render_pkg["radii"]

                # Loss
                gt_image = viewpoint_cam.original_image.cuda()
                Ll1 = l1_loss(image, gt_image)
                loss = Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))
                loss /= opt.optimizer_step_interval  # Gradient accumulation
                loss.backward()

                # Update loss-based selector with this view's loss
                if hasattr(selector, "update_loss"):
                    selector.update_loss(viewpoint_cam, loss.item() * opt.optimizer_step_interval)

            iter_end.record()

            with torch.no_grad():
                # Progress bar
                ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
                if iteration % 10 == 0:
                    progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})

                # Log and save
                current_metrics = training_report(
                    logger,
                    iteration,
                    Ll1,
                    loss,
                    l1_loss,
                    iter_start.elapsed_time(iter_end),
                    testing_iterations,
                    scene,
                    render,
                    (pipe, background),
                )

                if current_metrics:
                    metrics_history.append(current_metrics)
                    # Save history to JSON incrementally
                    with open(os.path.join(dataset.model_path, "metrics_history.json"), "w") as f:
                        json.dump(metrics_history, f, indent=4)
                    
                    # Capture distribution snapshot at test iterations if enabled
                    if log_distribution_snapshots:
                        selector.snapshot_distribution(gaussians, iteration)
                        # Save snapshots incrementally
                        snapshots_path = os.path.join(dataset.model_path, "distribution_snapshots.json")
                        selector.save_distribution_snapshots(snapshots_path)

                if iteration in saving_iterations:
                    print("\n[ITER {}] Saving Gaussians".format(iteration))
                    scene.save(iteration)

                # Densification
                if iteration < opt.densify_until_iter:
                    gaussians.max_radii2D[visibility_filter] = torch.max(
                        gaussians.max_radii2D[visibility_filter], radii[visibility_filter]
                    )
                    gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                    if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                        size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                        gaussians.densify_and_prune(
                            opt.densify_grad_threshold,
                            opt.prune_alpha_threshold,
                            scene.cameras_extent,
                            size_threshold,
                            radii,
                        )

                    if iteration % opt.opacity_reset_interval == 0 or (
                        dataset.white_background and iteration == opt.densify_from_iter
                    ):
                        gaussians.reset_opacity()

                # Optimizer step
                if iteration < opt.iterations:
                    gaussians.optimizer.step()
                    gaussians.optimizer.zero_grad(set_to_none=True)

                if not no_checkpoints and ((iteration in checkpoint_iterations) or iteration == opt.iterations):
                    print("\n[ITER {}] Saving Checkpoint".format(iteration))
                    _save_checkpoint(iteration)

            progress_bar.update(1)

        if not skip_final_eval:
            # Final evaluation
            final_results = eval_and_save(dataset.model_path, scene, render, (pipe, background))

            # Save final summary
            with open(os.path.join(dataset.model_path, "final_results.json"), "w") as f:
                json.dump(final_results, f, indent=4)

    except KeyboardInterrupt:
        if checkpoint_on_interrupt:
            try:
                print(f"\n[INTERRUPT] Saving checkpoint at iter {last_completed_iteration}")
                _save_checkpoint(last_completed_iteration, suffix="_interrupt")
            except Exception as e:
                print(f"[INTERRUPT] Failed to save checkpoint: {e}")
        raise
    except Exception:
        if checkpoint_on_interrupt:
            try:
                print(f"\n[ERROR] Saving checkpoint at iter {last_completed_iteration}")
                _save_checkpoint(last_completed_iteration, suffix="_error")
            except Exception as e:
                print(f"[ERROR] Failed to save checkpoint: {e}")
        raise
    finally:
        # Close logger even on error/interrupt
        logger.finish()


def training_report(logger, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene: Scene, renderFunc, renderArgs):
    if logger:
        logger.add_scalar("train_loss_patches/l1_loss", Ll1.item(), iteration)
        logger.add_scalar("train_loss_patches/total_loss", loss.item(), iteration)
        logger.add_scalar("iter_time", elapsed, iteration)

    metrics_data = None

    # Report test and samples of training set
    should_test = False
    if isinstance(testing_iterations, int):
        should_test = (iteration % testing_iterations) == 0
    elif testing_iterations:
        try:
            should_test = iteration in set(testing_iterations)
        except TypeError:
            should_test = False

    if should_test:
        lpips_fn = LPIPS(net_type="vgg").to("cuda")
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
                    
                    if logger and (idx < 5):
                        viz_image = torch.nn.functional.interpolate(image.unsqueeze(0), scale_factor=0.25, mode="bilinear", align_corners=False)
                        logger.add_image(config["name"] + "_view_{}/render".format(viewpoint.image_name), viz_image, iteration)

                    psnr_val = psnr(image, gt_image).mean()
                    ssim_val = ssim(image, gt_image)
                    lpips_val = lpips_fn(image.unsqueeze(0), gt_image.unsqueeze(0))

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

                if logger:
                    logger.add_scalar(config["name"] + "/loss_viewpoint - l1_loss", l1_test, iteration)
                    logger.add_scalar(config["name"] + "/loss_viewpoint - psnr", psnr_test, iteration)
                    logger.add_scalar(config["name"] + "/loss_viewpoint - ssim", ssim_test, iteration)
                    logger.add_scalar(config["name"] + "/loss_viewpoint - lpips", lpips_test, iteration)

        if logger:
            logger.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            logger.add_scalar("total_points", scene.gaussians.get_xyz.shape[0], iteration)
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
    parser.add_argument(
        "--view_selection_strategy",
        type=str,
        default="stack",
        choices=[
            "stack",
            "uniform_random",
            "geometric",
            "clustering",
            "loss_based",
            "gaussian_aware",
            "scheduled_hybrid",
            "dino",
            "sequential",
            "deterministic_max_loss",
        ],
    )
    parser.add_argument("--view_selection_config", type=str, default="{}")
    parser.add_argument(
        "--view_selection_verbose",
        action="store_true",
        default=True,
        help="Enable verbose logging inside view selectors (default: enabled)",
    )
    parser.add_argument(
        "--no_view_selection_verbose",
        action="store_false",
        dest="view_selection_verbose",
        help="Disable verbose logging inside view selectors",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--logger", type=str, default="tensorboard",
                        choices=["tensorboard", "wandb", "none"],
                        help="Logging backend: tensorboard (default), wandb, or none")
    parser.add_argument("--wandb_project", type=str, default="3dgs-view-selection",
                        help="W&B project name (only used if --logger=wandb)")
    parser.add_argument("--wandb_entity", type=str, default=None,
                        help="W&B team/organization name (only used if --logger=wandb). None = personal account")
    parser.add_argument("--no_save", action="store_true",
                        help="Skip saving model checkpoints entirely (useful for hyperparameter sweeps)")
    parser.add_argument(
        "--no_checkpoints",
        action="store_true",
        help="Do not save any .pth checkpoints during training or at the end",
    )
    parser.add_argument(
        "--checkpoint_on_interrupt",
        action="store_true",
        help="If interrupted (SIGINT) or crashes, save a checkpoint at the last completed iteration",
    )
    parser.add_argument(
        "--disable_view_selection_logs",
        action="store_true",
        help="Disable view-selection file logs (selection_history.jsonl, view_selection_*.log)",
    )
    parser.add_argument("--skip_final_eval", action="store_true",
                        help="Skip final eval_and_save() (useful for quick smoke tests / avoiding LPIPS OOM)")
    parser.add_argument(
        "--log_distribution_snapshots",
        action="store_true",
        help="Log full probability distribution at each test iteration (saved to distribution_snapshots.json)"
    )

    args = parser.parse_args(sys.argv[1:])
    if args.no_save:
        # "Minimal disk" mode for training script: no gaussians snapshots, no checkpoints
        args.save_iterations = []
        args.checkpoint_iterations = []
        args.no_checkpoints = True
    else:
        if args.iterations not in args.save_iterations:
            args.save_iterations.append(args.iterations)

    print("Optimizing " + args.model_path)

    safe_state(args.quiet)

    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(
        lp.extract(args), op.extract(args), pp.extract(args), 
        args.test_iterations, args.save_iterations, args.checkpoint_iterations, 
        args.start_checkpoint, args.debug_from, 
        args.view_selection_strategy, args.view_selection_config, args.seed,
        logger_backend=args.logger, wandb_project=args.wandb_project, 
        wandb_entity=args.wandb_entity, use_gui=False, skip_final_eval=args.skip_final_eval,
        no_checkpoints=args.no_checkpoints,
        checkpoint_on_interrupt=args.checkpoint_on_interrupt,
        disable_view_selection_logs=args.disable_view_selection_logs,
        view_selection_verbose=args.view_selection_verbose,
        log_distribution_snapshots=args.log_distribution_snapshots,
    )

    print("\nTraining complete.")