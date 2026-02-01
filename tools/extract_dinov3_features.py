#!/usr/bin/env python3
"""
Extract DINOv3 features for ScanNet++ scenes using HuggingFace Transformers.

This script extracts global CLS token features from DINOv3 models for all images
in ScanNet++ scenes, storing them for use with DINOSelector.

DINOv3 offers significant improvements over DINOv2:
- Better feature quality across classification, segmentation, and depth tasks
- Improved geographical fairness and robustness
- Uses patch size 16 (vs 14 in DINOv2)

Usage:
    # Extract features for all scenes (default: ViT-L/16)
    python extract_dinov3_features.py --data_root ~/data/scenes/data
    
    # Extract for specific scene
    python extract_dinov3_features.py --data_root ~/data/scenes/data --scene_id 5a269ba6fe
    
    # Use different model size
    python extract_dinov3_features.py --data_root ~/data/scenes/data --model dinov3-vitb16

Output:
    Creates dino_features/features.pt under each scene containing:
    {
        'DSC00001': tensor([...]),  # shape: (embedding_dim,)
        'DSC00002': tensor([...]),
        ...
        '_model': 'facebook/dinov3-vitl16-pretrain-lvd1689m',
        '_embedding_dim': 1024,
    }

Requirements:
    pip install transformers torch torchvision tqdm pillow

Note on image sizes:
    DINOv3 uses patch_size=16. The model accepts images of any size that is
    a multiple of 16. If not, it will crop to the closest smaller multiple.
    Common choices: 256x256, 512x512. The AutoImageProcessor defaults to 256.
"""

import argparse
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import warnings

import torch
from PIL import Image
from tqdm import tqdm

# HuggingFace imports
from transformers import AutoImageProcessor, AutoModel


# Model configurations for DINOv3
# See: https://huggingface.co/collections/facebook/dinov3-68924841bd6b561778e31009
MODEL_CONFIGS = {
    # ViT models pretrained on LVD-1689M (web images)
    'dinov3-vits16': {
        'hf_name': 'facebook/dinov3-vits16-pretrain-lvd1689m',
        'embedding_dim': 384,
        'params': '21M',
    },
    'dinov3-vits16plus': {
        'hf_name': 'facebook/dinov3-vits16plus-pretrain-lvd1689m',
        'embedding_dim': 384,  # Same dim, but SwiGLU FFN
        'params': '29M',
    },
    'dinov3-vitb16': {
        'hf_name': 'facebook/dinov3-vitb16-pretrain-lvd1689m',
        'embedding_dim': 768,
        'params': '86M',
    },
    'dinov3-vitl16': {
        'hf_name': 'facebook/dinov3-vitl16-pretrain-lvd1689m',
        'embedding_dim': 1024,
        'params': '300M',
    },
    'dinov3-vith16plus': {
        'hf_name': 'facebook/dinov3-vith16plus-pretrain-lvd1689m',
        'embedding_dim': 1280,
        'params': '840M',
    },
    # ConvNeXt models (alternative architecture)
    'dinov3-convnext-tiny': {
        'hf_name': 'facebook/dinov3-convnext-tiny-pretrain-lvd1689m',
        'embedding_dim': None,  # ConvNeXt has different output structure
        'params': '29M',
    },
    'dinov3-convnext-small': {
        'hf_name': 'facebook/dinov3-convnext-small-pretrain-lvd1689m',
        'embedding_dim': None,
        'params': '50M',
    },
    'dinov3-convnext-base': {
        'hf_name': 'facebook/dinov3-convnext-base-pretrain-lvd1689m',
        'embedding_dim': None,
        'params': '89M',
    },
    'dinov3-convnext-large': {
        'hf_name': 'facebook/dinov3-convnext-large-pretrain-lvd1689m',
        'embedding_dim': None,
        'params': '198M',
    },
}

# Default model: ViT-L/16 offers excellent quality with reasonable compute
DEFAULT_MODEL = 'dinov3-vitl16'


def load_model_and_processor(
    model_name: str, 
    device: torch.device
) -> Tuple[AutoModel, AutoImageProcessor]:
    """
    Load DINOv3 model and processor from HuggingFace.
    
    Models are automatically cached to ~/.cache/huggingface/hub/ so subsequent
    loads are fast.
    
    Args:
        model_name: Short name (e.g., 'dinov3-vitl16') or full HF name
        device: Device to load model on
        
    Returns:
        Tuple of (model, processor)
    """
    # Get HuggingFace model name
    if model_name in MODEL_CONFIGS:
        config = MODEL_CONFIGS[model_name]
        hf_name = config['hf_name']
        print(f"Loading {model_name} ({config['params']} parameters)")
    else:
        # Assume it's a full HF model name
        hf_name = model_name
        print(f"Loading {hf_name}")
    
    print(f"  HuggingFace model: {hf_name}")
    
    # Load processor (handles preprocessing: resize, normalize)
    # DINOv3 uses standard ImageNet normalization:
    #   mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
    processor = AutoImageProcessor.from_pretrained(hf_name)
    
    # Load model
    model = AutoModel.from_pretrained(hf_name)
    model = model.to(device)
    model.eval()
    
    # Get embedding dimension from model config
    if hasattr(model.config, 'hidden_size'):
        embedding_dim = model.config.hidden_size
    elif model_name in MODEL_CONFIGS and MODEL_CONFIGS[model_name]['embedding_dim']:
        embedding_dim = MODEL_CONFIGS[model_name]['embedding_dim']
    else:
        # For ConvNeXt, we'll determine from output
        embedding_dim = None
    
    print(f"  Embedding dimension: {embedding_dim}")
    print(f"  Processor image size: {processor.size}")
    
    return model, processor


def get_scene_images(scene_path: Path, images_folder: Optional[str] = None) -> List[Path]:
    """
    Get all image paths for a scene.
    
    Args:
        scene_path: Path to scene directory
        images_folder: Specific images folder to use (e.g., 'images_4' for downscaled).
                      If None, auto-detect in order of preference.
        
    Returns:
        Sorted list of image paths
    """
    # If specific folder requested, use it
    if images_folder:
        image_dir = scene_path / images_folder
        if not image_dir.exists():
            # Also try under dslr/ for ScanNet++ format
            image_dir = scene_path / 'dslr' / images_folder
        if not image_dir.exists():
            raise FileNotFoundError(f"Requested image folder not found: {images_folder} in {scene_path}")
    else:
        # Auto-detect image folder
        image_dir = scene_path / 'dslr' / 'resized_undistorted_images'
        
        if not image_dir.exists():
            # Try alternative paths
            alt_paths = [
                scene_path / 'dslr' / 'undistorted_images',
                scene_path / 'images',
            ]
            for alt in alt_paths:
                if alt.exists():
                    image_dir = alt
                    break
            else:
                raise FileNotFoundError(
                    f"Could not find image directory in {scene_path}. "
                    f"Tried: resized_undistorted_images, undistorted_images, images"
                )
    
    # Find all JPG/PNG images
    patterns = ['*.JPG', '*.jpg', '*.jpeg', '*.PNG', '*.png']
    images = []
    for pattern in patterns:
        images.extend(image_dir.glob(pattern))
    
    images = sorted(images)
    
    if not images:
        raise FileNotFoundError(f"No images found in {image_dir}")
    
    return images


def extract_features_for_scene(
    model: AutoModel,
    processor: AutoImageProcessor,
    scene_path: Path,
    device: torch.device,
    batch_size: int = 8,
    model_name: str = DEFAULT_MODEL,
    images_folder: Optional[str] = None,
) -> Dict[str, torch.Tensor]:
    """
    Extract DINOv3 features for all images in a scene.
    
    Args:
        model: DINOv3 model
        processor: Image processor
        scene_path: Path to scene directory
        device: Device to run inference on
        batch_size: Batch size for inference
        model_name: Name of model (for metadata)
        images_folder: Specific images folder to use (e.g., 'images_4')
        
    Returns:
        Dictionary mapping image names to feature tensors
    """
    images = get_scene_images(scene_path, images_folder)
    print(f"  Found {len(images)} images in {images[0].parent if images else 'unknown'}")
    
    features = {}
    embedding_dim = None
    
    # Process in batches
    for i in tqdm(range(0, len(images), batch_size), desc="  Extracting"):
        batch_paths = images[i:i + batch_size]
        batch_images = []
        batch_names = []
        
        for img_path in batch_paths:
            try:
                img = Image.open(img_path).convert('RGB')
                batch_images.append(img)
                # Store without extension for easier matching with camera names
                batch_names.append(img_path.stem)
            except Exception as e:
                print(f"  Warning: Failed to load {img_path}: {e}")
                continue
        
        if not batch_images:
            continue
        
        # Process images using AutoImageProcessor
        # This handles: resize, normalize with ImageNet stats
        inputs = processor(images=batch_images, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Extract features
        with torch.no_grad():
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == 'cuda'):
                outputs = model(**inputs)
                
                # Get pooled output (CLS token)
                # For ViT models: outputs.pooler_output is the CLS token
                # For ConvNeXt: outputs.pooler_output is the global average pool
                if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
                    cls_features = outputs.pooler_output
                elif hasattr(outputs, 'last_hidden_state'):
                    # Fallback: use CLS token from last hidden state
                    cls_features = outputs.last_hidden_state[:, 0, :]
                else:
                    raise ValueError(f"Could not extract CLS token from model output: {type(outputs)}")
        
        # Determine embedding dim from first batch
        if embedding_dim is None:
            embedding_dim = cls_features.shape[1]
            print(f"  Detected embedding dimension: {embedding_dim}")
        
        # Store features (move to CPU to save GPU memory)
        for name, feat in zip(batch_names, cls_features):
            features[name] = feat.cpu().float()  # Convert from bfloat16 if needed
    
    # Add metadata
    if model_name in MODEL_CONFIGS:
        hf_name = MODEL_CONFIGS[model_name]['hf_name']
    else:
        hf_name = model_name
    
    features['_model'] = hf_name
    features['_embedding_dim'] = embedding_dim
    
    return features


def save_features(features: Dict[str, torch.Tensor], output_path: Path) -> None:
    """
    Save features to disk.
    
    Args:
        features: Dictionary of features
        output_path: Path to save features
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(features, output_path)
    
    # Calculate file size
    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"  Saved to {output_path} ({size_mb:.2f} MB)")


def find_scenes(data_root: Path) -> List[Path]:
    """
    Find all scene directories under data root.
    
    Args:
        data_root: Root directory containing scenes
        
    Returns:
        List of scene paths
    """
    scenes = []
    
    # Look for directories that contain dslr subfolder
    for item in data_root.iterdir():
        if item.is_dir():
            if (item / 'dslr').exists():
                scenes.append(item)
            elif (item / 'images').exists():
                scenes.append(item)
    
    return sorted(scenes)


def main():
    parser = argparse.ArgumentParser(
        description='Extract DINOv3 features for ScanNet++ scenes using HuggingFace',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        '--data_root', 
        type=str, 
        required=True,
        help='Root directory containing scene folders (e.g., ~/data/scenes/data)'
    )
    parser.add_argument(
        '--scene_id',
        type=str,
        default=None,
        help='Specific scene ID to process (default: process all scenes)'
    )
    parser.add_argument(
        '--model',
        type=str,
        default=DEFAULT_MODEL,
        choices=list(MODEL_CONFIGS.keys()),
        help=f'DINOv3 model to use (default: {DEFAULT_MODEL})'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=8,
        help='Batch size for inference (default: 8)'
    )
    parser.add_argument(
        '--output_name',
        type=str,
        default='features.pt',
        help='Name of output file (default: features.pt)'
    )
    parser.add_argument(
        '--images',
        type=str,
        default=None,
        help='Images folder to use (e.g., images_4 for pre-downscaled). Default: auto-detect'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu',
        help='Device to use (default: cuda if available)'
    )
    parser.add_argument(
        '--force',
        action='store_true',
        help='Overwrite existing features'
    )
    parser.add_argument(
        '--cache_dir',
        type=str,
        default=None,
        help='HuggingFace cache directory (default: ~/.cache/huggingface/hub)'
    )
    
    args = parser.parse_args()
    
    # Set cache directory if specified
    if args.cache_dir:
        os.environ['HF_HOME'] = args.cache_dir
        os.environ['TRANSFORMERS_CACHE'] = args.cache_dir
    
    # Expand paths
    data_root = Path(args.data_root).expanduser().resolve()
    if not data_root.exists():
        raise FileNotFoundError(f"Data root not found: {data_root}")
    
    device = torch.device(args.device)
    print(f"Using device: {device}")
    
    # Load model and processor
    model, processor = load_model_and_processor(args.model, device)
    
    # Find scenes to process
    if args.scene_id:
        scene_path = data_root / args.scene_id
        if not scene_path.exists():
            raise FileNotFoundError(f"Scene not found: {scene_path}")
        scenes = [scene_path]
    else:
        scenes = find_scenes(data_root)
        if not scenes:
            raise FileNotFoundError(f"No scenes found in {data_root}")
    
    print(f"\nFound {len(scenes)} scene(s) to process")
    
    # Process each scene
    for scene_path in scenes:
        scene_id = scene_path.name
        # Support multiple dataset layouts:
        # - ScanNet++: <scene>/dslr/... (we store embeddings under <scene>/dslr/dino_features)
        # - COLMAP / mip-nerf-360: <scene>/images + <scene>/sparse (store under <scene>/dino_features)
        base_dir = (scene_path / 'dslr') if (scene_path / 'dslr').exists() else scene_path
        output_dir = base_dir / 'dino_features'
        output_path = output_dir / args.output_name
        
        print(f"\nProcessing scene: {scene_id}")
        
        # Check if already exists
        if output_path.exists() and not args.force:
            print(f"  Features already exist at {output_path}")
            print(f"  Use --force to overwrite")
            continue
        
        try:
            features = extract_features_for_scene(
                model=model,
                processor=processor,
                scene_path=scene_path,
                device=device,
                batch_size=args.batch_size,
                model_name=args.model,
                images_folder=args.images,
            )
            
            save_features(features, output_path)
            n_images = len([k for k in features.keys() if not k.startswith('_')])
            print(f"  Extracted {n_images} image features")
            
        except Exception as e:
            print(f"  Error processing {scene_id}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print("\nDone!")
    print(f"\nTo use these features with DINOSelector:")
    print(f"  config = {{'embeddings_path': '<scene_path>/dino_features/{args.output_name}'}}")
    print(f"  selector = build_selector('dino', config=config)")


if __name__ == '__main__':
    main()