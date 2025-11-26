import os
import subprocess
import json
import numpy as np
import glob
import pandas as pd
import matplotlib.pyplot as plt

def run_training(repo_path, data_path, output_dir, strategy, seed, exp_name):
    """
    Executes the modified train_gsplat.py script.
    """
    os.chdir(repo_path)
    
    run_dir = os.path.join(output_dir, exp_name, f"seed_{seed}")
    os.makedirs(run_dir, exist_ok=True)

    print(f"--- Starting: {exp_name} | Seed: {seed} ---")

    cmd = [
        "python", "train_gsplat.py",
        "--source_path", data_path,
        "--model_path", run_dir,
        "--seed", str(seed),
        "--view_selection_strategy", strategy,
        "--test_iterations", "1000", "3000", "7000", "15000", "30000", # Granular testing for curves
        "--save_iterations", "30000",
        "--iterations", "30000",
        "--data_device", "cuda" # Ensure GPU usage
    ]
    
    try:
        with open(os.path.join(run_dir, "console_log.txt"), "w") as f:
            subprocess.run(cmd, check=True, stdout=f, stderr=subprocess.STDOUT)
        print(f"SUCCESS: Seed {seed} completed.")
        return run_dir
    except subprocess.CalledProcessError:
        print(f"FAILURE: Seed {seed} crashed. Check console_log.txt.")
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