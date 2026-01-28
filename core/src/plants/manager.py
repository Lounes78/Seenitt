import csv
from pathlib import Path
from geometry import get_iou
from quality import view_quality_score
from datetime import datetime

from datetime import datetime
from pathlib import Path

class CatalogManager:
    def __init__(
        self,
        base_dir="catalogue_plantes",
        min_expert_confidence=0.10,
        iou_same_object=0.15,
        max_object_age_frames=90,   # ~3s @30fps
        min_view_quality=3_000_000,
    ):
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")

        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

        # Nouveau sous-dossier pour ce lot/vidéo
        self.session_dir = self.base_dir / f"detections_{ts}"
        self.session_dir.mkdir(parents=True, exist_ok=True)

        # Journal CSV dans ce sous-dossier
        self.log_file = self.session_dir / f"journal_detections_{ts}.csv"

        self.min_expert_confidence = min_expert_confidence
        self.iou_same_object = iou_same_object
        self.max_object_age_frames = max_object_age_frames
        self.min_view_quality = min_view_quality

        # obj_id -> metadata
        self.id_registry = {}
        self._next_id = 0

        self._init_csv()
        print(f"📂 [Manager] Catalogue prêt dans : {self.session_dir}")


    def _init_csv(self):
        if not self.log_file.exists():
            with open(self.log_file, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(
                    ["Frame", "ID_SAM", "Espece", "Confiance", "Qualite", "Chemin_Image"]
                )

    def _new_object_id(self):
        self._next_id += 1
        return self._next_id

    def _match_existing_object(self, label, box, frame_idx):
        for obj_id, data in self.id_registry.items():
            if data["label"] != label:
                continue

            age_frames = frame_idx - data["last_seen_frame"]
            if age_frames > self.max_object_age_frames:
                continue

            if get_iou(box, data["last_box"]) > self.iou_same_object:
                return obj_id

        return None

    def process_detection(self, crop, frame, label, confidence, box, frame_idx):
        """
        Main entry point.
        Returns obj_id if saved, else None.
        """

        if confidence < self.min_expert_confidence:
            return None

        view_quality = view_quality_score(box, frame)
        if view_quality < self.min_view_quality:
            return None

        matched_id = self._match_existing_object(label, box, frame_idx)

        # Case 1 — New object
        if matched_id is None:
            obj_id = self._new_object_id()
            self._save(
                obj_id, crop, label, confidence, view_quality, box, frame_idx
            )
            return obj_id

        # Case 2 — Same object, better view
        old_quality = self.id_registry[matched_id]["best_quality"]
        if view_quality > old_quality:
            self._save(
                matched_id, crop, label, confidence, view_quality, box, frame_idx
            )
            return matched_id

        # Case 3 — Duplicate, ignore
        return None

    def _save(self, obj_id, crop, label, confidence, quality, box, frame_idx):
        """Internal save + registry update."""

        self.id_registry[obj_id] = {
            "label": label,
            "confidence": confidence,
            "best_quality": quality,
            "last_box": box,
            "last_seen_frame": frame_idx,
            "center": ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2),
        }

        species_dir = self.session_dir / label.replace(" ", "_")
        species_dir.mkdir(exist_ok=True)

        filename = f"ID{obj_id}_F{frame_idx}.jpg"
        save_path = species_dir / filename

        try:
            crop.save(save_path)
        except Exception as e:
            print(f"❌ [Manager] Erreur sauvegarde image : {e}")
            return

        with open(self.log_file, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                frame_idx,
                obj_id,
                label,
                f"{confidence:.2%}",
                int(quality),
                str(save_path),
            ])
