Erstelle view_selection/vggt_selector.py

=== KONZEPT ===
VGGT (Visual Geometry Grounded Transformer) generiert aus einem Bild eine 3D-Punktwolke.
Wir nutzen das als "Prior" - wenn die VGGT-Punktwolke stark von unseren Gaussians abweicht,
ist diese Region wahrscheinlich noch nicht gut rekonstruiert.

Score = Chamfer Distance zwischen VGGT-Punktwolke und nahegelegenen Gaussians
Hoher Score = hohe Diskrepanz = informative View

=== DEPENDENCIES ===
VGGT muss separat installiert werden:
  pip install vggt  # oder von GitHub klonen
  
Das Modell ist groß (~1GB), daher:
- Lazy loading beim ersten Aufruf
- Caching der VGGT-Vorhersagen pro Kamera
- Nur periodisch neu berechnen (update_frequency)

=== IMPLEMENTIERUNG ===

class VGGTSelector(ViewSelector):
    def __init__(self, config: dict = None, ...):
        """
        Config-Parameter:
        - model_name (str): VGGT model variant. Default: 'vggt-1b'
        - update_frequency (int): Wie oft VGGT neu evaluieren. Default: 1000
        - temperature (float): Softmax temperature. Default: 1.0
        - chamfer_k (int): K nearest neighbors für Chamfer. Default: 1
        - device (str): 'cuda' oder 'cpu'. Default: 'cuda'
        - cache_predictions (bool): Cache VGGT outputs. Default: True
        """
        self.model = None  # Lazy loading
        self.vggt_cache = {}  # cam_uid -> predicted point cloud
        self.chamfer_scores = {}  # cam_uid -> score
        
    def _load_model(self):
        """Lazy load VGGT model."""
        if self.model is None:
            try:
                from vggt import VGGT
                self.model = VGGT.from_pretrained(self.model_name)
                self.model = self.model.to(self.device)
                self.model.eval()
            except ImportError:
                raise ImportError(
                    "VGGT not installed. Install with: pip install vggt "
                    "or clone from https://github.com/facebookresearch/vggt"
                )
    
    def _predict_pointcloud(self, camera) -> torch.Tensor:
        """
        Get VGGT point cloud prediction for a camera.
        
        Args:
            camera: Camera object with original_image
            
        Returns:
            Tensor of shape [N, 3] with predicted 3D points
        """
        if self.cache_predictions and camera.uid in self.vggt_cache:
            return self.vggt_cache[camera.uid]
        
        self._load_model()
        
        with torch.no_grad():
            # Get image from camera
            image = camera.original_image  # [3, H, W]
            
            # VGGT expects [B, 3, H, W]
            image_batch = image.unsqueeze(0).to(self.device)
            
            # Predict (VGGT returns dict with 'points', 'depth', etc.)
            output = self.model(image_batch)
            points = output['points'][0]  # [N, 3] in camera space
            
            # Transform to world space using camera pose
            # points_world = R^T @ points + t
            R = torch.tensor(camera.R, device=self.device, dtype=points.dtype)
            T = torch.tensor(camera.T, device=self.device, dtype=points.dtype)
            points_world = (R.T @ points.T).T + camera.camera_center.to(self.device)
        
        if self.cache_predictions:
            self.vggt_cache[camera.uid] = points_world
        
        return points_world
    
    def _compute_chamfer_distance(
        self, 
        pred_points: torch.Tensor, 
        gaussian_positions: torch.Tensor
    ) -> float:
        """
        Compute one-sided Chamfer distance from predicted points to Gaussians.
        
        Args:
            pred_points: [N, 3] VGGT predicted points
            gaussian_positions: [M, 3] current Gaussian positions
            
        Returns:
            Mean distance from predicted points to nearest Gaussian
        """
        # For each predicted point, find distance to nearest Gaussian
        # Use torch.cdist for efficiency
        
        # Subsample if too many points (for performance)
        max_points = 10000
        if pred_points.shape[0] > max_points:
            indices = torch.randperm(pred_points.shape[0])[:max_points]
            pred_points = pred_points[indices]
        
        # Compute pairwise distances [N, M]
        dists = torch.cdist(pred_points, gaussian_positions)
        
        # Get distance to k nearest Gaussians
        k = min(self.chamfer_k, gaussian_positions.shape[0])
        nearest_dists, _ = torch.topk(dists, k, dim=1, largest=False)
        
        # Mean of nearest distances
        chamfer = nearest_dists.mean().item()
        
        return chamfer
    
    def _compute_all_chamfer_scores(self, gaussians) -> None:
        """Compute Chamfer scores for all cameras."""
        gaussian_positions = gaussians.get_xyz.detach()
        
        for cam in self.all_cameras:
            try:
                pred_points = self._predict_pointcloud(cam)
                score = self._compute_chamfer_distance(pred_points, gaussian_positions)
                self.chamfer_scores[cam.uid] = score
            except Exception as e:
                if self.verbose:
                    print(f"[VGGTSelector] Warning: Failed for camera {cam.uid}: {e}")
                self.chamfer_scores[cam.uid] = 0.0
        
        if self.verbose:
            scores = list(self.chamfer_scores.values())
            print(f"[VGGTSelector] Chamfer scores: min={min(scores):.4f}, max={max(scores):.4f}")
    
    def compute_probabilities(self, gaussians, iteration: int) -> Dict[int, float]:
        """
        Compute probabilities based on VGGT Chamfer distance.
        
        Higher Chamfer distance = more disagreement = higher probability.
        """
        # Update scores periodically
        if iteration - self.last_update_iteration >= self.update_frequency:
            self._compute_all_chamfer_scores(gaussians)
            self.last_update_iteration = iteration
        
        # Convert scores to probabilities
        # Higher Chamfer = higher score = more likely to select
        scores = np.array([self.chamfer_scores.get(cam.uid, 0.0) for cam in self.all_cameras])
        uids = [cam.uid for cam in self.all_cameras]
        
        # Softmax
        from scipy.special import softmax
        probs = softmax(scores / self.temperature)
        
        return {uid: float(prob) for uid, prob in zip(uids, probs)}

=== REGISTRIERUNG ===
In __init__.py:
    from .vggt_selector import VGGTSelector
    
    SELECTOR_REGISTRY['vggt'] = VGGTSelector

=== FALLBACK ===
Falls VGGT nicht verfügbar ist, sollte der Selector graceful degraden:
- Beim Import-Error: Warnung ausgeben und auf uniform sampling zurückfallen
- Optional: Lightweight alternative (z.B. Depth Anything) als config option

=== USAGE ===
config = {
    'model_name': 'vggt-1b',
    'update_frequency': 2000,  # VGGT ist teuer, selten updaten
    'temperature': 1.0,
    'cache_predictions': True,
}
selector = build_selector('vggt', config=config)

=== HYBRID INTEGRATION ===
Im ScheduledHybridSelector kann VGGT als späte Phase verwendet werden:
{
    "selectors": ["clustering", "loss_based", "vggt"],
    "weights_start": [0.5, 0.4, 0.1],
    "weights_end": [0.1, 0.3, 0.6],
    "schedule_type": "linear",
    "max_iterations": 30000,
    "vggt_config": {
        "update_frequency": 2000,
        "cache_predictions": True
    }
}