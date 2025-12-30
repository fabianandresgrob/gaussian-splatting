import os
import sys
import subprocess

def setup_environment(repo_path):
    """
    Installs dependencies and compiles the CUDA extensions 
    (simple-knn, diff-gaussian-rasterization) for CUDA 11.8.
    """
    print("="*50)
    print("Setting up Environment (Dependencies & Compilation)")
    print("="*50)

    # 1. Install Python Dependencies
    print("\n--- Installing Python dependencies ---")
    subprocess.run([sys.executable, "-m", "pip", "install", "tqdm", "plyfile", "opencv-python<4.10.0", "joblib"], check=True)

    # 2. Set CUDA environment variables to use conda environment
    print("\n--- Setting CUDA environment variables ---")
    conda_prefix = os.environ.get('CONDA_PREFIX')
    if conda_prefix:
        os.environ['CUDA_HOME'] = conda_prefix
        os.environ['PATH'] = f"{conda_prefix}/bin:" + os.environ.get('PATH', '')
        os.environ['LD_LIBRARY_PATH'] = f"{conda_prefix}/lib:" + os.environ.get('LD_LIBRARY_PATH', '')
        print(f"CUDA_HOME set to: {conda_prefix}")
    else:
        print("Warning: CONDA_PREFIX not found. Make sure conda environment is activated.")
        return

    # 3. Compile CUDA Extensions
    print("\n--- Compiling CUDA Extensions ---")
    
    # Compile diff-gaussian-rasterization
    diff_gaussian_path = os.path.join(repo_path, "submodules/diff-gaussian-rasterization")
    if os.path.exists(diff_gaussian_path):
        print("Compiling diff-gaussian-rasterization...")
        subprocess.run([
            sys.executable, "-m", "pip", "install", 
            "--no-build-isolation", diff_gaussian_path
        ], check=True)
    else:
        print(f"Warning: {diff_gaussian_path} not found.")

    # Compile simple-knn
    simple_knn_path = os.path.join(repo_path, "submodules/simple-knn")
    if os.path.exists(simple_knn_path):
        print("Compiling simple-knn...")
        subprocess.run([
            sys.executable, "-m", "pip", "install", 
            "--no-build-isolation", simple_knn_path
        ], check=True)
    else:
        print(f"Warning: {simple_knn_path} not found.")

    print("\nEnvironment setup complete!")

if __name__ == "__main__":
    repo_path = os.path.dirname(os.path.abspath(__file__))
    setup_environment(repo_path)