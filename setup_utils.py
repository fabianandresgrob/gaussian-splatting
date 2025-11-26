import os
import sys
import subprocess

def setup_environment(repo_path):
    """
    Installs dependencies, patches code for CUDA 12 compatibility,
    and compiles the CUDA extensions (simple-knn, diff-gaussian-rasterization).
    """
    print("="*50)
    print("Setting up Environment (Dependencies & Compilation)")
    print("="*50)

    # 1. Install Python Dependencies (No requirements.txt needed)
    print("\n--- Installing tqdm and plyfile ---")
    subprocess.run([sys.executable, "-m", "pip", "install", "tqdm", "plyfile"], check=True)

    # 2. Apply CUDA 12 Compatibility Patches
    print("\n--- Applying Compatibility Patches ---")
    
    # Fix simple-knn: Add missing float.h include
    simple_knn_src = os.path.join(repo_path, "submodules/simple-knn/simple_knn.cu")
    if os.path.exists(simple_knn_src):
        print(f"Patching {simple_knn_src}...")
        # Using sed to insert the include at the top
        subprocess.run(["sed", "-i", '1i #include <float.h>', simple_knn_src], check=True)
    
    # Fix diff-gaussian-rasterization: Add missing cstdint include
    rasterizer_header = os.path.join(repo_path, "submodules/diff-gaussian-rasterization/cuda_rasterizer/rasterizer_impl.h")
    if os.path.exists(rasterizer_header):
        print(f"Patching {rasterizer_header}...")
        subprocess.run(["sed", "-i", '1i #include <cstdint>', rasterizer_header], check=True)

    # 3. Compile CUDA Extensions
    print("\n--- Compiling CUDA Extensions ---")
    
    # Set CUDA environment variables
    os.environ['CUDA_HOME'] = '/usr/local/cuda'
    if '/usr/local/cuda/bin' not in os.environ['PATH']:
        os.environ['PATH'] = '/usr/local/cuda/bin:' + os.environ['PATH']

    # Compile diff-gaussian-rasterization
    diff_gaussian_path = os.path.join(repo_path, "submodules/diff-gaussian-rasterization")
    if os.path.exists(diff_gaussian_path):
        print("Compiling diff-gaussian-rasterization...")
        os.chdir(diff_gaussian_path)
        subprocess.run([sys.executable, "setup.py", "install"], check=True)
        os.chdir(repo_path) # Go back
    else:
        print(f"Warning: {diff_gaussian_path} not found.")

    # Compile simple-knn
    simple_knn_path = os.path.join(repo_path, "submodules/simple-knn")
    if os.path.exists(simple_knn_path):
        print("Compiling simple-knn...")
        os.chdir(simple_knn_path)
        subprocess.run([sys.executable, "setup.py", "install"], check=True)
        os.chdir(repo_path) # Go back
    else:
        print(f"Warning: {simple_knn_path} not found.")

    print("\nEnvironment setup complete!")