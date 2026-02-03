#!/usr/bin/env python3
"""
Ablation Study Runner for 3D Gaussian Splatting View Selection Strategies.

This script orchestrates ablation experiments comparing different view selection
strategies for 3D Gaussian Splatting training.

================================================================================
                              EXPERIMENT CONFIGURATIONS
================================================================================

AVAILABLE CONFIGS:
------------------
    ID   | Strategy          | Description
    -----|-------------------|--------------------------------------------------
    B1   | stack             | Baseline: epoch-based shuffle (original 3DGS)
    B2   | uniform_random    | Baseline: true uniform random sampling
    S1   | geometric         | Standalone: geometric heuristics (pose diversity)
    S2   | loss_based        | Standalone: loss-weighted sampling (hard mining)
    S3   | dino              | Standalone: DINO feature diversity
    CL   | clustering        | Clustering: DBSCAN pose clustering (static)

================================================================================

Usage:
    # Run experiments on specific scenes and configs
    python run_ablation.py --data_root /path/to/scenes --output_root /path/to/results \
        --scenes scene1 scene2 --configs B1 S1 CL --seeds 0 1 2

    # Dry run (print what would be executed)
    python run_ablation.py --dry_run --data_root ... --output_root ...

    # Resume interrupted experiments
    python run_ablation.py --data_root ... --output_root ...

    # List all experiment configurations
    python run_ablation.py --list_configs

Author: ADL4CV Team
Date: 2024
"""

import os
import sys
import json
import argparse
import subprocess
import time
import hashlib
import glob
import logging
from datetime import datetime
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Any

# ============================================================================
#                           CONFIGURATION CLASSES
# ============================================================================

@dataclass
class ExperimentConfig:
    """Configuration for a single experiment type."""
    id: str                          # Unique identifier (e.g., "B1", "S1", "CL")
    name: str                        # Human-readable name
    strategy: str                    # View selection strategy name
    config: Dict[str, Any]           # Strategy-specific configuration
    description: str = ""            # Brief description for logging
    requires_dino: bool = False      # Whether this config needs DINO features

    def get_config_json(self) -> str:
        """Return config as JSON string for CLI."""
        return json.dumps(self.config)

    def get_config_hash(self) -> str:
        """Return short hash of config for deduplication."""
        config_str = f"{self.strategy}:{json.dumps(self.config, sort_keys=True)}"
        return hashlib.md5(config_str.encode()).hexdigest()[:8]


@dataclass
class RunConfig:
    """Configuration for a single training run."""
    experiment: ExperimentConfig
    scene: str
    seed: int
    output_dir: str
    data_path: str

    @property
    def run_id(self) -> str:
        """Unique identifier for this run."""
        return f"{self.experiment.id}_{self.scene}_seed{self.seed}"

    @property
    def run_dir(self) -> str:
        """Output directory for this run."""
        return os.path.join(self.output_dir, self.experiment.id, self.scene, f"seed_{self.seed}")

    def is_completed(self) -> bool:
        """Check if this run has already completed successfully."""
        final_results = os.path.join(self.run_dir, "final_results.json")
        return os.path.exists(final_results)

    def is_partially_completed(self) -> bool:
        """Check if this run has any checkpoints."""
        checkpoint_pattern = os.path.join(self.run_dir, "chkpnt*.pth")
        return len(glob.glob(checkpoint_pattern)) > 0


@dataclass
class AblationState:
    """Persistent state for resumable ablation runs."""
    started_at: str = ""
    last_updated: str = ""
    completed_runs: List[str] = field(default_factory=list)
    failed_runs: List[str] = field(default_factory=list)
    skipped_runs: List[str] = field(default_factory=list)
    total_runs: int = 0

    def save(self, path: str) -> None:
        """Save state to JSON file."""
        self.last_updated = datetime.now().isoformat()
        with open(path, 'w') as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def load(cls, path: str) -> 'AblationState':
        """Load state from JSON file."""
        if os.path.exists(path):
            with open(path, 'r') as f:
                data = json.load(f)
            return cls(**data)
        return cls(started_at=datetime.now().isoformat())


# ============================================================================
#                         EXPERIMENT DEFINITIONS
# ============================================================================

def get_all_experiment_configs() -> Dict[str, ExperimentConfig]:
    """
    Define all experiment configurations.

    Returns:
        Dictionary mapping config ID to ExperimentConfig
    """
    configs = {}

    # --- Baselines ---

    configs["B1"] = ExperimentConfig(
        id="B1",
        name="Baseline: Stack (No Replacement)",
        strategy="stack",
        config={},
        description="Original 3DGS epoch-based shuffle (stack-based selection)"
    )

    configs["B2"] = ExperimentConfig(
        id="B2",
        name="Baseline: Uniform Random",
        strategy="uniform_random",
        config={},
        description="Uniform random sampling with replacement"
    )

    # --- Standalone Strategies ---

    configs["S1"] = ExperimentConfig(
        id="S1",
        name="Geometric",
        strategy="geometric",
        config={
            "mode": "distance_to_selected",
            "temperature": 0.3,
            "recency_window": 50,
            "use_cumulative_penalty": False,
        },
        description="Pose-based diversity with dynamic selection tracking"
    )

    configs["S2"] = ExperimentConfig(
        id="S2",
        name="Loss-Based",
        strategy="loss_based",
        config={
            "temperature": 0.3,
            "min_samples_before_bias": 0,
        },
        description="Raw loss tracking, prioritize high-loss views"
    )

    configs["S3"] = ExperimentConfig(
        id="S3",
        name="DINO Features",
        strategy="dino",
        config={
            "embeddings_path": "auto",
            "require_embeddings": True,
            "temperature": 0.3,
            "diversity_mode": "distance_to_selected",
            "recency_window": 50,
            "normalize_embeddings": True,
            "use_cumulative_penalty": False,
        },
        description="DINO feature-based diversity sampling",
        requires_dino=True
    )

    # --- Clustering ---

    configs["CL"] = ExperimentConfig(
        id="CL",
        name="Clustering (DBSCAN)",
        strategy="clustering",
        config={
            "clustering_method": "dbscan",
            "eps": 0.85,
            "min_samples": 2,
            "temperature": 1.0,
            "use_orientation": True,
        },
        description="DBSCAN clustering on camera poses, inverse cluster-size weighting"
    )

    # --- Additional Strategies ---

    configs["SEQ"] = ExperimentConfig(
        id="SEQ",
        name="Sequential",
        strategy="sequential",
        config={},
        description="Deterministic sequential iteration through cameras (no randomness)"
    )

    configs["DML"] = ExperimentConfig(
        id="DML",
        name="Deterministic Max-Loss",
        strategy="deterministic_max_loss",
        config={},
        description="Always select the camera with highest current loss (expensive)"
    )

    return configs


# ============================================================================
#                              SCENE DEFINITIONS
# ============================================================================

# Default scenes for experiments (ScanNet++ scene IDs)
DEFAULT_SCENES = [
    "0c5385e84b",
    "5371eff4f9",
    "56a0ec536c",
    "5a269ba6fe",
    "ab046f8faf",
    "c0f5742640",
    "c173f62b15",
    "dc263dfbf0",
    "dffce1cf9a",
    "f248c2bcdc",
]

# Default seeds for reproducibility
DEFAULT_SEEDS = [0, 1, 2, 3, 4]


# ============================================================================
#                            TRAINING PARAMETERS
# ============================================================================

TRAINING_DEFAULTS = {
    "iterations": 30000,
    "test_iterations": [1000, 2500, 5000, 7500, 10000, 12500, 15000, 17500, 20000, 25000, 30000],
    # NOTE: By default we do not save point clouds or periodic checkpoints to reduce disk usage.
    # Final evaluation images and JSON summaries are still saved.
    "save_iterations": [],
    "checkpoint_iterations": [],
    "data_device": "cpu",
    "resolution": 2,  # Half resolution for speed (adjust as needed)
    # Forwarded to train_gsplat.py as --images. Useful for COLMAP datasets like mip-nerf-360
    # that ship pre-downscaled folders (images_2/images_4/images_8)
    "images": None,
    "logger": "wandb",
    # Random point cloud initialization (instead of COLMAP/dataset points)
    "random_pcd": False,
    "random_pcd_num_points": 100000,
    # Log full probability distribution at test iterations
    "log_distribution_snapshots": False,
    # Optimizer type: "default" (Adam), "sparse_adam", "sgd", "sgd_no_momentum"
    "optimizer_type": "default",
    # Densification control for ablation studies
    "disable_densification": False,
    "densification_multiplier": 1.0,
}


# ============================================================================
#                              RUNNER CLASS
# ============================================================================

class AblationRunner:
    """
    Orchestrates ablation study experiments.

    Features:
    - Resume capability: tracks completed runs in state file
    - Dry run mode: preview what would be executed
    - Flexible filtering: run specific config IDs
    """

    def __init__(
        self,
        repo_path: str,
        data_root: str,
        output_root: str,
        scenes: Optional[List[str]] = None,
        seeds: Optional[List[int]] = None,
        logger_backend: str = "wandb",
        wandb_project: str = "3DGS",
        wandb_entity: str = "fabian-grob-technical-university-of-munich",
        verbose: bool = True,
        minimal_disk: bool = True,
        keep_wandb_local: bool = True,
        disable_selection_logs: bool = False,
        view_selection_verbose: bool = True,
        training_params: Optional[Dict[str, Any]] = None,
        no_save: bool = False,
        skip_final_eval: bool = False,
        eval_test_only: bool = False,
    ):
        """
        Initialize the ablation runner.

        Args:
            repo_path: Path to gaussian-splatting repository
            data_root: Root directory containing scene data
            output_root: Root directory for experiment outputs
            scenes: List of scene names for experiments
            seeds: List of seeds for experiments
            logger_backend: Logging backend ("tensorboard", "wandb", "none")
            wandb_project: W&B project name if using wandb
            wandb_entity: W&B entity/team name if using wandb
            verbose: Whether to print verbose output
        """
        self.repo_path = os.path.abspath(repo_path)
        self.data_root = os.path.abspath(data_root)
        self.output_root = os.path.abspath(output_root)

        self.scenes = scenes or DEFAULT_SCENES
        self.seeds = seeds or DEFAULT_SEEDS

        self.logger_backend = logger_backend
        self.wandb_project = wandb_project
        self.wandb_entity = wandb_entity
        self.verbose = verbose
        self.minimal_disk = minimal_disk
        self.keep_wandb_local = keep_wandb_local
        self.disable_selection_logs = disable_selection_logs
        self.view_selection_verbose = view_selection_verbose

        self.training_params = TRAINING_DEFAULTS.copy()
        if training_params:
            self.training_params.update(training_params)
        self.no_save = no_save
        self.skip_final_eval = skip_final_eval
        self.eval_test_only = eval_test_only

        # Load experiment configs
        self.all_configs = get_all_experiment_configs()

        # State file for resume capability
        self.state_file = os.path.join(output_root, "ablation_state.json")
        self.state = AblationState.load(self.state_file)

        # Setup logging
        self._setup_logging()

    def _setup_logging(self) -> None:
        """Configure logging."""
        log_dir = os.path.join(self.output_root, "logs")
        os.makedirs(log_dir, exist_ok=True)

        log_file = os.path.join(log_dir, f"ablation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s | %(levelname)s | %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Ablation runner initialized. Log file: {log_file}")

    def _get_scene_path(self, scene: str) -> str:
        """Get the data path for a scene."""
        # Adjust this based on your dataset structure
        # Example for ScanNet++: data_root/scene_id/dslr
        scene_path = os.path.join(self.data_root, scene, "dslr")
        if not os.path.exists(scene_path):
            # Try alternative structure: data_root/scene_id
            scene_path = os.path.join(self.data_root, scene)
        return scene_path

    @staticmethod
    def _is_colmap_scene(scene_path: str) -> bool:
        """Heuristic: returns True if the scene looks like a COLMAP dataset."""
        sparse0 = os.path.join(scene_path, "sparse", "0")
        if not os.path.isdir(sparse0):
            return False
        # Require cameras + images to exist (bin or txt)
        has_cams = os.path.exists(os.path.join(sparse0, "cameras.bin")) or os.path.exists(os.path.join(sparse0, "cameras.txt"))
        has_imgs = os.path.exists(os.path.join(sparse0, "images.bin")) or os.path.exists(os.path.join(sparse0, "images.txt"))
        return has_cams and has_imgs

    @staticmethod
    def _find_latest_checkpoint(run_dir: str) -> Optional[str]:
        """Return the newest checkpoint path in run_dir, or None.

        Prefers *_interrupt/_error checkpoints if present, otherwise falls back to
        any chkpnt*.pth. Sorting is by (iteration, suffix priority).
        """
        candidates = glob.glob(os.path.join(run_dir, "chkpnt*.pth"))
        if not candidates:
            return None

        def _score(path: str) -> tuple:
            base = os.path.basename(path)
            # chkpnt15000_interrupt.pth / chkpnt15000.pth
            it = 0
            try:
                rest = base[len("chkpnt"):]
                rest = rest.split(".pth")[0]
                num = ""
                for ch in rest:
                    if ch.isdigit():
                        num += ch
                    else:
                        break
                it = int(num) if num else 0
            except Exception:
                it = 0

            # Prefer interrupt/error checkpoints at same iteration
            if "_interrupt" in base:
                suffix_prio = 2
            elif "_error" in base:
                suffix_prio = 1
            else:
                suffix_prio = 0
            return (it, suffix_prio)

        return max(candidates, key=_score)

    def generate_run_configs(
        self,
        config_ids: Optional[List[str]] = None,
    ) -> List[RunConfig]:
        """
        Generate all run configurations based on filters.

        Args:
            config_ids: Only include these specific config IDs

        Returns:
            List of RunConfig objects
        """
        runs = []

        for config_id, exp_config in self.all_configs.items():
            # Filter by specific config IDs
            if config_ids is not None and config_id not in config_ids:
                continue

            # Generate runs for each scene x seed combination
            for scene in self.scenes:
                scene_path = self._get_scene_path(scene)

                if not os.path.exists(scene_path):
                    self.logger.warning(
                        f"Skipping scene '{scene}': path not found under data_root: {scene_path}"
                    )
                    continue

                for seed in self.seeds:
                    run = RunConfig(
                        experiment=exp_config,
                        scene=scene,
                        seed=seed,
                        output_dir=self.output_root,
                        data_path=scene_path,
                    )
                    runs.append(run)

        return runs

    def run_single_experiment(self, run_config: RunConfig, resume: bool = True) -> bool:
        """
        Execute a single training run.

        Args:
            run_config: Configuration for this run

        Returns:
            True if successful, False otherwise
        """
        run_id = run_config.run_id
        run_dir = run_config.run_dir

        self.logger.info(f"Starting: {run_id}")
        self.logger.info(f"  Strategy: {run_config.experiment.strategy}")
        self.logger.info(f"  Scene: {run_config.scene}")
        self.logger.info(f"  Seed: {run_config.seed}")
        self.logger.info(f"  Output: {run_dir}")

        # Create output directory
        os.makedirs(run_dir, exist_ok=True)

        # If resuming and a checkpoint exists, continue from it automatically
        start_checkpoint = None
        if resume and (not run_config.is_completed()):
            start_checkpoint = self._find_latest_checkpoint(run_dir)

        # Build command
        cmd = [
            "python", "-u", "train_gsplat.py",
            "--source_path", run_config.data_path,
            "--model_path", run_dir,
            "--seed", str(run_config.seed),
            "--view_selection_strategy", run_config.experiment.strategy,
            "--view_selection_config", run_config.experiment.get_config_json(),
            "--iterations", str(self.training_params["iterations"]),
            "--data_device", self.training_params["data_device"],
            "--logger", self.logger_backend,
        ]

        # Optional: pick an images folder (e.g., images_2/images_4/images_8) for COLMAP datasets
        if self.training_params.get("images"):
            cmd.extend(["--images", str(self.training_params["images"])])
            # set resolution to 1 when using pre-downscaled images
            cmd.extend(["--resolution", "1"])

        # For COLMAP datasets (e.g., mip-nerf-360), enable eval mode so a test split exists
        # Without this, train_gsplat.py's final evaluation will assert on empty test cameras
        if self._is_colmap_scene(run_config.data_path):
            cmd.append("--eval")

        if not self.view_selection_verbose:
            cmd.append("--no_view_selection_verbose")

        if start_checkpoint:
            cmd.extend(["--start_checkpoint", start_checkpoint])

        if self.no_save:
            cmd.append("--no_save")
        if self.skip_final_eval:
            cmd.append("--skip_final_eval")
        if self.eval_test_only:
            cmd.append("--eval_test_only")

        # Add test iterations
        cmd.extend(["--test_iterations"] + [str(i) for i in self.training_params["test_iterations"]])

        # If resolution hasn't been set via --images, set it now
        if not any(arg == "--resolution" for arg in cmd):
            cmd.extend(["--resolution", str(self.training_params["resolution"])])

        # Random point cloud initialization
        if self.training_params.get("random_pcd"):
            cmd.append("--random_pcd")
            num_points = self.training_params.get("random_pcd_num_points", 100000)
            cmd.extend(["--random_pcd_num_points", str(num_points)])

        # Log distribution snapshots at test iterations
        if self.training_params.get("log_distribution_snapshots"):
            cmd.append("--log_distribution_snapshots")

        # Optimizer type
        optimizer_type = self.training_params.get("optimizer_type", "default")
        if optimizer_type != "default":
            cmd.extend(["--optimizer_type", optimizer_type])

        # Densification control
        if self.training_params.get("disable_densification"):
            cmd.append("--disable_densification")
        densification_multiplier = self.training_params.get("densification_multiplier", 1.0)
        if densification_multiplier != 1.0:
            cmd.extend(["--densification_multiplier", str(densification_multiplier)])

        # Disk-minimal defaults:
        # - Do not write point_cloud/*.ply snapshots
        # - Do not write checkpoints on success
        # - Still write a checkpoint on interrupt/crash for resume/debug
        # - Keep view-selection logs by default (selection_history.jsonl, view_selection_*.log)
        #   so we can analyze what was selected; disable explicitly via --disable_selection_logs.
        if self.minimal_disk:
            if "--no_save" not in cmd:
                cmd.append("--no_save")
            if "--no_checkpoints" not in cmd:
                cmd.append("--no_checkpoints")
            if "--checkpoint_on_interrupt" not in cmd:
                cmd.append("--checkpoint_on_interrupt")
            if self.disable_selection_logs and "--disable_view_selection_logs" not in cmd:
                cmd.append("--disable_view_selection_logs")
        else:
            # Optional: allow disabling view-selection logs even in full-disk mode
            if self.disable_selection_logs:
                cmd.append("--disable_view_selection_logs")

            # Add save iterations
            if self.training_params.get("save_iterations"):
                cmd.extend(["--save_iterations"] + [str(i) for i in self.training_params["save_iterations"]])
            # Add checkpoint iterations
            if self.training_params.get("checkpoint_iterations"):
                cmd.extend(["--checkpoint_iterations"] + [str(i) for i in self.training_params["checkpoint_iterations"]])

        # Add wandb config if using wandb
        if self.logger_backend == "wandb":
            cmd.extend(["--wandb_project", self.wandb_project])
            if self.wandb_entity:
                cmd.extend(["--wandb_entity", self.wandb_entity])

        # Save run metadata
        metadata = {
            "run_id": run_id,
            "experiment_id": run_config.experiment.id,
            "experiment_name": run_config.experiment.name,
            "strategy": run_config.experiment.strategy,
            "config": run_config.experiment.config,
            "scene": run_config.scene,
            "seed": run_config.seed,
            "start_checkpoint": start_checkpoint,
            "started_at": datetime.now().isoformat(),
            "command": " ".join(cmd),
            "view_selection_verbose": self.view_selection_verbose,
        }
        with open(os.path.join(run_dir, "run_metadata.json"), 'w') as f:
            json.dump(metadata, f, indent=2)

        # Execute training
        log_path = os.path.join(run_dir, "console_log.txt")

        try:
            start_time = time.time()

            with open(log_path, 'w') as log_file:
                process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    universal_newlines=True,
                    bufsize=1,
                    cwd=self.repo_path,
                )

                for line in process.stdout:
                    log_file.write(line)
                    log_file.flush()
                    if self.verbose:
                        print(line, end='', flush=True)

                process.wait()

            elapsed = time.time() - start_time

            if process.returncode != 0:
                raise subprocess.CalledProcessError(process.returncode, cmd)

            # Best-effort cleanup after successful run when minimal_disk is enabled
            # This is defensive in case older settings produced large artifacts
            if self.minimal_disk:
                try:
                    import shutil

                    pc_dir = os.path.join(run_dir, "point_cloud")
                    if os.path.isdir(pc_dir):
                        shutil.rmtree(pc_dir, ignore_errors=True)

                    for pth in glob.glob(os.path.join(run_dir, "chkpnt*.pth")):
                        try:
                            os.remove(pth)
                        except OSError:
                            pass

                    if not self.keep_wandb_local:
                        wb_dir = os.path.join(run_dir, "wandb")
                        if os.path.isdir(wb_dir):
                            shutil.rmtree(wb_dir, ignore_errors=True)
                except Exception as cleanup_err:
                    self.logger.warning(f"Cleanup skipped/failed for {run_id}: {cleanup_err}")

            self.logger.info(f"Completed: {run_id} in {elapsed/60:.1f} minutes")
            return True

        except subprocess.CalledProcessError as e:
            self.logger.error(f"Failed: {run_id} (exit code {e.returncode})")
            self.logger.error(f"  See log: {log_path}")
            return False
        except KeyboardInterrupt:
            self.logger.warning(f"Interrupted: {run_id}")
            if 'process' in locals():
                process.terminate()
            raise
        except Exception as e:
            self.logger.error(f"Error in {run_id}: {str(e)}")
            return False

    def run(
        self,
        config_ids: Optional[List[str]] = None,
        dry_run: bool = False,
        resume: bool = True,
    ) -> Dict[str, Any]:
        """
        Run the ablation study.

        Args:
            config_ids: Only run these specific config IDs
            dry_run: If True, just print what would be run
            resume: If True, skip already completed runs

        Returns:
            Summary dictionary with results
        """
        # Generate all run configurations
        all_runs = self.generate_run_configs(config_ids=config_ids)

        self.logger.info("=" * 70)
        self.logger.info("ABLATION STUDY")
        self.logger.info("=" * 70)
        self.logger.info(f"Total runs to process: {len(all_runs)}")
        self.logger.info(f"Output directory: {self.output_root}")
        self.logger.info(f"Data root: {self.data_root}")
        self.logger.info(f"Logger: {self.logger_backend}")
        if config_ids:
            self.logger.info(f"Config filter: {config_ids}")
        self.logger.info("=" * 70)

        # Filter out completed runs if resuming
        pending_runs = []
        skipped_count = 0

        for run in all_runs:
            if resume and run.is_completed():
                self.logger.info(f"Skipping (completed): {run.run_id}")
                skipped_count += 1
                if run.run_id not in self.state.completed_runs:
                    self.state.completed_runs.append(run.run_id)
            else:
                pending_runs.append(run)

        self.logger.info(f"Skipped (already completed): {skipped_count}")
        self.logger.info(f"Pending runs: {len(pending_runs)}")

        if dry_run:
            self.logger.info("\n[DRY RUN] Would execute the following runs:")
            for i, run in enumerate(pending_runs, 1):
                self.logger.info(f"  {i:3d}. {run.run_id}")
                self.logger.info(f"       Strategy: {run.experiment.strategy}")
                self.logger.info(f"       Config: {run.experiment.get_config_json()[:80]}...")
            return {"dry_run": True, "pending_runs": len(pending_runs)}

        # Execute runs
        self.state.total_runs = len(all_runs)
        self.state.save(self.state_file)

        successful = 0
        failed = 0

        for i, run in enumerate(pending_runs, 1):
            self.logger.info(f"\n{'='*70}")
            self.logger.info(f"RUN {i}/{len(pending_runs)}: {run.run_id}")
            self.logger.info(f"{'='*70}")

            try:
                success = self.run_single_experiment(run, resume=resume)

                if success:
                    successful += 1
                    self.state.completed_runs.append(run.run_id)
                else:
                    failed += 1
                    self.state.failed_runs.append(run.run_id)

                self.state.save(self.state_file)

            except KeyboardInterrupt:
                self.logger.warning("\nAblation interrupted by user")
                self.state.save(self.state_file)
                break

        # Summary
        self.logger.info(f"\n{'='*70}")
        self.logger.info("ABLATION SUMMARY")
        self.logger.info(f"{'='*70}")
        self.logger.info(f"Total runs: {len(all_runs)}")
        self.logger.info(f"Completed (this session): {successful}")
        self.logger.info(f"Failed (this session): {failed}")
        self.logger.info(f"Skipped (previously completed): {skipped_count}")
        self.logger.info(f"Total completed: {len(self.state.completed_runs)}")
        self.logger.info(f"Total failed: {len(self.state.failed_runs)}")
        self.logger.info(f"State saved to: {self.state_file}")

        return {
            "total_runs": len(all_runs),
            "successful": successful,
            "failed": failed,
            "skipped": skipped_count,
            "completed_total": len(self.state.completed_runs),
            "failed_total": len(self.state.failed_runs),
        }

    def print_experiment_summary(self) -> None:
        """Print a summary of all experiment configurations."""
        print("\n" + "=" * 80)
        print("EXPERIMENT CONFIGURATIONS")
        print("=" * 80)

        for config_id, config in self.all_configs.items():
            print(f"\n  {config_id}: {config.name}")
            print(f"      Strategy: {config.strategy}")
            print(f"      Description: {config.description}")

            # Print key config params
            if config.config:
                key_params = []
                for key, val in config.config.items():
                    if not key.endswith("_config"):
                        key_params.append(f"{key}={val}")
                if key_params:
                    print(f"      Params: {', '.join(key_params[:3])}")

        print("\n" + "=" * 80)

        # Count runs
        num_configs = len(self.all_configs)
        total_runs = num_configs * len(self.scenes) * len(self.seeds)

        print("\nRUN COUNTS:")
        print(f"  {num_configs} configs x {len(self.scenes)} scenes x {len(self.seeds)} seeds = {total_runs} runs")
        print("=" * 80)


# ============================================================================
#                              CLI INTERFACE
# ============================================================================

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run ablation study for 3DGS view selection strategies",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run all experiments with default settings
    python run_ablation.py --data_root /path/to/scenes --output_root /path/to/results

    # Run specific configs only
    python run_ablation.py --configs B1 S1 CL --data_root ... --output_root ...

    # Run on specific scenes and seeds
    python run_ablation.py --scenes scene1 scene2 --seeds 0 1 2 --data_root ... --output_root ...

    # Dry run (preview what would be executed)
    python run_ablation.py --dry_run --data_root ... --output_root ...

    # List all experiment configurations
    python run_ablation.py --list_configs

    # Use Weights & Biases logging
    python run_ablation.py --logger wandb --wandb_project my-project --data_root ... --output_root ...
        """
    )

    # Required arguments
    parser.add_argument(
        "--data_root",
        type=str,
        help="Root directory containing scene data"
    )
    parser.add_argument(
        "--output_root",
        type=str,
        help="Root directory for experiment outputs"
    )

    # Optional arguments
    parser.add_argument(
        "--repo_path",
        type=str,
        default=None,
        help="Path to gaussian-splatting repository (default: current directory)"
    )
    parser.add_argument(
        "--configs",
        nargs="+",
        type=str,
        help="Only run these specific config IDs (e.g., B1 S1 CL)"
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Print what would be run without executing"
    )
    parser.add_argument(
        "--no_resume",
        action="store_true",
        help="Don't skip already completed runs"
    )
    parser.add_argument(
        "--list_configs",
        action="store_true",
        help="List all experiment configurations and exit"
    )
    parser.add_argument(
        "--full_disk",
        action="store_true",
        help="Disable minimal-disk mode (save point clouds, checkpoints, and selection logs as configured)",
    )
    parser.add_argument(
        "--delete_wandb_local",
        action="store_true",
        help="After successful runs, delete local 'wandb/' artifacts inside each run directory",
    )

    parser.add_argument(
        "--disable_selection_logs",
        action="store_true",
        help="Disable view-selection logs (selection_history.jsonl and view_selection_*.log)",
    )

    # Logging options
    parser.add_argument(
        "--logger",
        type=str,
        choices=["tensorboard", "wandb", "none"],
        default="wandb",
        help="Logging backend (default: wandb)"
    )
    parser.add_argument(
        "--wandb_project",
        type=str,
        default="3dgs-ablation",
        help="W&B project name (only used with --logger wandb)"
    )
    parser.add_argument(
        "--wandb_entity",
        type=str,
        default=None,
        help="W&B entity/team name (only used with --logger wandb)"
    )

    # Scene/seed overrides
    parser.add_argument(
        "--scenes",
        nargs="+",
        type=str,
        help="Override default scenes"
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        help="Override default seeds"
    )

    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Reduce output verbosity"
    )

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

    # Training overrides (useful for smoke tests)
    parser.add_argument(
        "--iterations",
        type=int,
        default=TRAINING_DEFAULTS["iterations"],
        help="Override training iterations for all runs"
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=TRAINING_DEFAULTS["resolution"],
        help="Override training resolution for all runs"
    )
    parser.add_argument(
        "--images",
        type=str,
        default=TRAINING_DEFAULTS["images"],
        help="Forwarded to train_gsplat.py as --images (e.g., images_2/images_4/images_8 for COLMAP/mip-nerf-360)"
    )
    parser.add_argument(
        "--data_device",
        type=str,
        default=TRAINING_DEFAULTS["data_device"],
        help="Override data_device for all runs (e.g., cpu)"
    )
    parser.add_argument(
        "--test_iterations",
        nargs="+",
        type=int,
        default=TRAINING_DEFAULTS["test_iterations"],
        help="Override --test_iterations passed to train_gsplat.py"
    )
    parser.add_argument(
        "--save_iterations",
        nargs="+",
        type=int,
        default=TRAINING_DEFAULTS["save_iterations"],
        help="Override --save_iterations passed to train_gsplat.py"
    )
    parser.add_argument(
        "--checkpoint_iterations",
        nargs="+",
        type=int,
        default=TRAINING_DEFAULTS["checkpoint_iterations"],
        help="Override --checkpoint_iterations passed to train_gsplat.py"
    )
    parser.add_argument(
        "--no_save",
        action="store_true",
        help="Pass --no_save to train_gsplat.py (skip saving gaussians/checkpoints as configured there)"
    )
    parser.add_argument(
        "--skip_final_eval",
        action="store_true",
        help="Pass --skip_final_eval to train_gsplat.py (avoid final LPIPS eval; useful for smoke tests)"
    )
    parser.add_argument(
        "--eval_test_only",
        action="store_true",
        help="Only evaluate on test set during training (skip train set eval, saves time for large/imbalanced datasets)"
    )
    parser.add_argument(
        "--random_pcd",
        action="store_true",
        help="Use random point cloud initialization instead of COLMAP/dataset points"
    )
    parser.add_argument(
        "--random_pcd_num_points",
        type=int,
        default=TRAINING_DEFAULTS["random_pcd_num_points"],
        help="Number of random points to initialize (only used with --random_pcd)"
    )
    parser.add_argument(
        "--log_distribution_snapshots",
        action="store_true",
        help="Log full probability distribution at each test iteration (saved to distribution_snapshots.json)"
    )

    # Optimizer and densification ablation options
    parser.add_argument(
        "--optimizer_type",
        type=str,
        choices=["default", "sparse_adam", "sgd", "sgd_no_momentum"],
        default=TRAINING_DEFAULTS["optimizer_type"],
        help="Optimizer type: 'default' (Adam), 'sparse_adam', 'sgd', 'sgd_no_momentum' (default: default)"
    )
    parser.add_argument(
        "--disable_densification",
        action="store_true",
        help="Completely disable densification and pruning during training"
    )
    parser.add_argument(
        "--densification_multiplier",
        type=float,
        default=TRAINING_DEFAULTS["densification_multiplier"],
        help="Multiply densification thresholds (>1 = less aggressive densification, default: 1.0)"
    )

    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()

    # Determine repo path
    repo_path = args.repo_path or os.path.dirname(os.path.abspath(__file__))

    # Handle list_configs without requiring other args
    if args.list_configs:
        runner = AblationRunner(
            repo_path=repo_path,
            data_root=args.data_root or "/tmp",
            output_root=args.output_root or "/tmp",
        )
        runner.print_experiment_summary()
        return

    # Validate required arguments
    if not args.data_root or not args.output_root:
        print("Error: --data_root and --output_root are required")
        print("Use --help for usage information")
        sys.exit(1)

    # Create output directory
    os.makedirs(args.output_root, exist_ok=True)

    # Initialize runner
    runner = AblationRunner(
        repo_path=repo_path,
        data_root=args.data_root,
        output_root=args.output_root,
        scenes=args.scenes,
        seeds=args.seeds,
        logger_backend=args.logger,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
        verbose=not args.quiet,
        minimal_disk=not args.full_disk,
        keep_wandb_local=not args.delete_wandb_local,
        disable_selection_logs=args.disable_selection_logs,
        view_selection_verbose=args.view_selection_verbose,
        training_params={
            "iterations": args.iterations,
            "resolution": args.resolution,
            "data_device": args.data_device,
            "images": args.images,
            "test_iterations": args.test_iterations,
            "save_iterations": args.save_iterations,
            "checkpoint_iterations": args.checkpoint_iterations,
            "random_pcd": args.random_pcd,
            "random_pcd_num_points": args.random_pcd_num_points,
            "log_distribution_snapshots": args.log_distribution_snapshots,
            "optimizer_type": args.optimizer_type,
            "disable_densification": args.disable_densification,
            "densification_multiplier": args.densification_multiplier,
        },
        no_save=args.no_save,
        skip_final_eval=args.skip_final_eval,
        eval_test_only=args.eval_test_only,
    )

    # Print experiment summary
    runner.print_experiment_summary()

    # Run ablation
    results = runner.run(
        config_ids=args.configs,
        dry_run=args.dry_run,
        resume=not args.no_resume,
    )

    print("\nDone!")
    return results


if __name__ == "__main__":
    main()
