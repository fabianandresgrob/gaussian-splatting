import os
import subprocess
import sys
import json
import numpy as np
import glob
import pandas as pd
import matplotlib.pyplot as plt

def run_training(repo_path, data_path, output_dir, strategy, seed, exp_name,
                 iterations=30000, test_iterations=None, save_iterations=None,
                 data_device="cuda", view_selection_config=None, extra_args=None,
                 verbose=True, resolution=-1, logger="tensorboard", 
                 wandb_project="3dgs-view-selection", wandb_entity=None):
    """
    Executes the modified train_gsplat.py script.
    
    Args:
        repo_path: Path to the gaussian-splatting repository
        data_path: Path to the scene data
        output_dir: Root directory for outputs
        strategy: View selection strategy (random, fixed_prob, epoch_based, clustering, no_replace)
        seed: Random seed for reproducibility
        exp_name: Experiment name (used for output folder)
        iterations: Total training iterations (default: 30000)
        test_iterations: List of iterations to run evaluation (default: [1000, 3000, 7000, 15000, 30000])
        save_iterations: List of iterations to save model (default: [30000])
        data_device: Device for data loading, "cuda" or "cpu" (default: "cuda")
        view_selection_config: JSON string for view selection configuration (default: "{}")
        extra_args: List of additional command line arguments (default: None)
        verbose: If True, stream output to notebook in real-time. If False, only log to file. (default: True)
        resolution: Image resolution factor. -1 or 1 = original, 2 = half, 4 = quarter. (default: -1)
        logger: Logging backend - "tensorboard", "wandb", or "none" (default: "tensorboard")
        wandb_project: W&B project name when logger="wandb" (default: "3dgs-view-selection")
        wandb_entity: W&B team/organization name when logger="wandb" (default: None = personal account)
    
    Returns:
        run_dir path on success, None on failure
    """
    os.chdir(repo_path)
    
    run_dir = os.path.join(output_dir, exp_name, f"seed_{seed}")
    os.makedirs(run_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"[START] {exp_name} | Strategy: {strategy} | Seed: {seed}")
    print(f"{'='*60}")
    print(f"Output: {run_dir}")

    # Set defaults
    if test_iterations is None:
        test_iterations = [1000, 3000, 7000, 15000, 30000]
    if save_iterations is None:
        save_iterations = [iterations]  # Save at final iteration by default
    if view_selection_config is None:
        view_selection_config = "{}"

    cmd = [
        "python", "-u", "train_gsplat.py",  # -u for unbuffered output
        "--source_path", data_path,
        "--model_path", run_dir,
        "--seed", str(seed),
        "--view_selection_strategy", strategy,
        "--view_selection_config", view_selection_config,
        "--iterations", str(iterations),
        "--data_device", data_device,
        "--resolution", str(resolution),
        "--logger", logger,
    ]
    
    # Add wandb project if using wandb
    if logger == "wandb":
        cmd.extend(["--wandb_project", wandb_project])
        if wandb_entity:
            cmd.extend(["--wandb_entity", wandb_entity])
    
    # Add test iterations
    cmd.extend(["--test_iterations"] + [str(i) for i in test_iterations])
    
    # Add save iterations
    cmd.extend(["--save_iterations"] + [str(i) for i in save_iterations])
    
    # Add any extra arguments
    if extra_args:
        cmd.extend(extra_args)
    
    log_path = os.path.join(run_dir, "console_log.txt")
    
    try:
        if verbose:
            # Stream output to both console and log file
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                bufsize=1  # Line buffered
            )
            
            with open(log_path, "w") as log_file:
                for line in process.stdout:
                    # Write to log file
                    log_file.write(line)
                    log_file.flush()
                    # Print to notebook (handle \r for progress bar updates)
                    if '\r' in line or 'Training progress' in line:
                        print(line, end='', flush=True)
                    else:
                        print(line, end='', flush=True)
                
            process.wait()
            if process.returncode != 0:
                raise subprocess.CalledProcessError(process.returncode, cmd)
        else:
            # Silent mode - just log to file
            with open(log_path, "w") as f:
                subprocess.run(cmd, check=True, stdout=f, stderr=subprocess.STDOUT)
        
        print(f"\n[SUCCESS] {exp_name} seed {seed} completed!")
        return run_dir
        
    except subprocess.CalledProcessError as e:
        print(f"\n[FAILED] {exp_name} seed {seed} crashed (exit code {e.returncode})")
        print(f"   Check log: {log_path}")
        return None
    except KeyboardInterrupt:
        print(f"\n[INTERRUPTED] {exp_name} seed {seed}")
        if 'process' in locals():
            process.terminate()
        return None

def analyze_retrospective(history_path, threshold_percent=0.01):
    """
    Finds the iteration where the metric first entered and stayed within 
    the threshold interval of the final metric value.
    """
    if not os.path.exists(history_path):
        return None, None

    with open(history_path, 'r') as f:
        history = json.load(f)
    
    if not history:
        return None, None

    # Sort by iteration just in case
    history.sort(key=lambda x: x['iteration'])
    
    # Get final PSNR (from the last entry)
    final_entry = history[-1]
    if 'test' not in final_entry: return None, None # Safety check
    
    final_psnr = final_entry['test']['PSNR']
    threshold_value = final_psnr * (1.0 - threshold_percent)

    # Find retrospective stop point
    # We look for the first iteration 'i' where ALL iterations 'j >= i' have PSNR > threshold
    early_stop_iter = final_entry['iteration']
    
    for i in range(len(history)):
        # Check if this point and all subsequent points meet the criteria
        subsequent_metrics = [h['test']['PSNR'] for h in history[i:]]
        if all(m >= threshold_value for m in subsequent_metrics):
            early_stop_iter = history[i]['iteration']
            break
            
    return final_psnr, early_stop_iter

def aggregate_results(output_root_path, scene, strategy):
    """
    Aggregates metrics across 5 seeds.
    """
    exp_folder = os.path.join(output_root_path, f"{scene}_{strategy}")
    seeds = glob.glob(os.path.join(exp_folder, "seed_*"))
    
    psnrs = []
    ssims = []
    lpips = []
    early_stops = []
    
    for seed_path in seeds:
        history_file = os.path.join(seed_path, "metrics_history.json")
        final_file = os.path.join(seed_path, "final_results.json")
        
        # Get Retrospective Stop
        _, stop_iter = analyze_retrospective(history_file)
        if stop_iter:
            early_stops.append(stop_iter)
            
        # Get Final Metrics
        if os.path.exists(final_file):
            with open(final_file, 'r') as f:
                res = json.load(f)
                psnrs.append(res['mean_psnr'])
                ssims.append(res['mean_ssim'])
                lpips.append(res['mean_lpips'])

    if not psnrs:
        return None

    report = {
        "Scene": scene,
        "Strategy": strategy,
        "Avg PSNR": np.mean(psnrs),
        "Std PSNR": np.std(psnrs),
        "Avg SSIM": np.mean(ssims),
        "Avg LPIPS": np.mean(lpips),
        "Avg Retro Stop Iter": np.mean(early_stops) if early_stops else "N/A"
    }
    return report