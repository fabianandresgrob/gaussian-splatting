import os
import sys
import subprocess
from google.colab import drive

def init_colab_env(repo_url, branch='main', working_dir='/content/gsplat_workspace'):
    if not os.path.exists('/content/drive'):
        drive.mount('/content/drive')

    os.makedirs(working_dir, exist_ok=True)
    os.chdir(working_dir)

    repo_name = repo_url.split("/")[-1].replace(".git", "")
    repo_path = os.path.join(working_dir, repo_name)

    if not os.path.exists(repo_path):
        subprocess.run(["git", "clone", repo_url], check=True)
        os.chdir(repo_path)
        subprocess.run(["git", "checkout", branch], check=True)
        # Initialize submodules (essential for 3DGS)
        subprocess.run(["git", "submodule", "update", "--init", "--recursive"], check=True)
    else:
        os.chdir(repo_path)
        subprocess.run(["git", "pull"], check=True)
        subprocess.run(["git", "checkout", branch], check=True)

    print("Installing dependencies...")
    subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], check=True)
    # Install submodules explicitly if requirements.txt misses them
    subprocess.run([sys.executable, "-m", "pip", "install", "submodules/diff-gaussian-rasterization"], check=True)
    subprocess.run([sys.executable, "-m", "pip", "install", "submodules/simple-knn"], check=True)
    
    return repo_path