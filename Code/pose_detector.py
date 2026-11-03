import numpy as np
import torch
from ultralytics import YOLO
import config


class PoseDetector:
    def __init__(self, model_path=config.YOLO_MODEL_PATH):
        self.model = YOLO(model_path)
        self.prev_kpts = None

        # Parametri pentru Filtrul Adaptiv (Viteză-dependent)
        self.min_alpha = 0.15  # Netezire maximă (când piciorul e pe sol, taie tremuratul)
        self.max_alpha = 0.85  # Netezire minimă (când piciorul se balansează, păstrează amplitudinea)
        # Viteza în pixeli/cadru de la care mișcarea este considerată intenționată și rapidă
        self.speed_threshold = 25.0

    def process_frame(self, frame):
        """Rulează detecția și aplică un filtru dinamic, independent per articulație."""
        results = self.model(frame, conf=config.CONF_THRESHOLD, verbose=False)
        person_detected = False
        keypoints = None
        annotated_frame = frame.copy()

        if len(results) > 0 and len(results[0].boxes) > 0:
            person_detected = True

            if results[0].keypoints is not None:
                current_kpts = results[0].keypoints.data  # Shape: (1, 17, 3) pt YOLO Pose

                if self.prev_kpts is not None and self.prev_kpts.shape == current_kpts.shape:
                    # 1. Calculăm distanța (viteza) parcursă de fiecare din cele 17 articulații (doar X și Y)
                    dist = torch.norm(current_kpts[..., :2] - self.prev_kpts[..., :2], dim=-1, keepdim=True)

                    # 2. Generăm un factor de netezire specific pentru FIECARE punct cheie
                    # Dacă se mișcă încet (sub prag), alpha e mic -> stabilizează și oprește Jitter-ul.
                    # Dacă se mișcă repede (peste prag), alpha urcă spre max -> lasă genunchiul să se îndoaie complet.
                    adaptive_ratio = torch.clamp(dist / self.speed_threshold, 0.0, 1.0)
                    dynamic_alpha = self.min_alpha + (self.max_alpha - self.min_alpha) * adaptive_ratio

                    # Replicăm factorul pentru a acoperi X, Y și gradul de Confidență
                    dynamic_alpha = dynamic_alpha.repeat(1, 1, 3)

                    # 3. Aplicăm netezirea, per articulație
                    smoothed_kpts = dynamic_alpha * current_kpts + (1.0 - dynamic_alpha) * self.prev_kpts
                else:
                    smoothed_kpts = current_kpts.clone()

                self.prev_kpts = smoothed_kpts.clone()
                results[0].keypoints.data = smoothed_kpts

                annotated_frame = results[0].plot()
                keypoints = smoothed_kpts.cpu().numpy()
        else:
            self.prev_kpts = None

        return person_detected, keypoints, annotated_frame