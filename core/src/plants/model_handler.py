import json
import numpy as np
import torch
import onnxruntime as ort
from PIL import Image
from scipy.ndimage import binary_opening
from transformers import AutoTokenizer, AutoImageProcessor

from geometry import get_iou, get_ios, get_union_box, capped_union


class PlantIDModel:
    def __init__(self, model_paths, device="cuda"):
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]

        base_path = "/home/matcha/seenit/Seenitt/core/src/plants/"
        print("🚀 Loading ONNX models...")

        self.v_enc = ort.InferenceSession(
            f"{base_path}/sam3_onnx/vision-encoder.onnx", providers=providers
        )
        self.t_enc = ort.InferenceSession(
            f"{base_path}/sam3_onnx/text-encoder.onnx", providers=providers
        )
        self.m_dec = ort.InferenceSession(
            f"{base_path}/sam3_onnx/decoder.onnx", providers=providers
        )
        self.expert = ort.InferenceSession(
            f"{base_path}/expert_vit_gold.onnx", providers=providers
        )

        self.tokenizer = AutoTokenizer.from_pretrained("facebook/sam3")
        self.expert_proc = AutoImageProcessor.from_pretrained(
            "juppy44/plant-identification-2m-vit-b"
        )

        with open(f"{base_path}/labels.json", "r") as f:
            self.id2label = json.load(f)

        self.TARGET_SIZE = 1008

    # ------------------------------------------------------------

    def _nms_candidates(self, candidates, iou_thr=0.4):
        """Simple NMS on mask bounding boxes (mask coords space)."""
        kept = []
        for cand in candidates:
            keep = True
            for k in kept:
                if get_iou(cand["coords"], k["coords"]) > iou_thr:
                    keep = False
                    break
            if keep:
                kept.append(cand)
        return kept

    # ------------------------------------------------------------

    def process_frame(self, frame_pil, prompt="tree", max_entities=4):
        orig_w, orig_h = frame_pil.size

        # --- Vision encoder ---
        img_fixed = frame_pil.resize((self.TARGET_SIZE, self.TARGET_SIZE))
        px_vals = (
            np.array(img_fixed)
            .transpose(2, 0, 1)[None]
            .astype(np.float32)
            / 255.0
        )

        v_out = self.v_enc.run(
            None, {self.v_enc.get_inputs()[0].name: px_vals}
        )
        vision_data = {
            self.v_enc.get_outputs()[i].name: v_out[i]
            for i in range(len(v_out))
        }

        # --- Text encoder ---
        t_in = self.tokenizer(
            prompt, return_tensors="np", padding="max_length", max_length=32
        )

        t_embeds = self.t_enc.run(
            None,
            {
                "input_ids": t_in["input_ids"].astype(np.int64),
                "attention_mask": t_in["attention_mask"].astype(np.int64),
            },
        )[0]

        # --- Decoder ---
        m_outs = self.m_dec.run(
            None,
            {
                "fpn_feat_0": vision_data["fpn_feat_0"],
                "fpn_feat_1": vision_data["fpn_feat_1"],
                "fpn_feat_2": vision_data["fpn_feat_2"],
                "fpn_pos_2": vision_data["fpn_pos_2"],
                "prompt_features": t_embeds.astype(np.float32),
                "prompt_mask": np.zeros((1, 32), dtype=bool),
            },
        )

        all_masks, iou_scores = m_outs[0], m_outs[1]
        mask_h, mask_w = all_masks.shape[2], all_masks.shape[3]

        # ------------------------------------------------------------
        # STEP A — candidate extraction
        # ------------------------------------------------------------

        best_iou = iou_scores[0].max(axis=1)
        valid_idx = np.where(best_iou > 0.65)[0]

        candidates = []

        for idx in valid_idx:
            mask_raw = (all_masks[0, idx] > 0.5).squeeze()
            mask_clean = binary_opening(mask_raw, structure=np.ones((3, 3)))

            ys, xs = np.where(mask_clean)
            if len(xs) < 1000:
                continue

            x1m, x2m = xs.min(), xs.max()
            y1m, y2m = ys.min(), ys.max()

            if (x2m - x1m) > (mask_w * 0.95):
                continue

            candidates.append(
                {
                    "coords": (x1m, y1m, x2m, y2m),
                    "mask": mask_clean,
                    "area": (x2m - x1m) * (y2m - y1m),
                }
            )

        # sort + NMS
        candidates.sort(key=lambda x: x["area"], reverse=True)
        candidates = self._nms_candidates(candidates)

        # ------------------------------------------------------------
        # STEP B — expert ID + in-frame fusion
        # ------------------------------------------------------------

        entities = []

        for cand in candidates:
            x1m, y1m, x2m, y2m = cand["coords"]

            x1 = x1m * orig_w / mask_w
            x2 = x2m * orig_w / mask_w
            y1 = y1m * orig_h / mask_h
            y2 = y2m * orig_h / mask_h

            # minimum size filter (important)
            if (x2 - x1) < 40 or (y2 - y1) < 40:
                continue

            box = (x1, y1, x2, y2)

            # --- Expert crop ---
            margin = 0.12
            pw, ph = (x2 - x1) * margin, (y2 - y1) * margin
            crop_box = (
                max(0, x1 - pw),
                max(0, y1 - ph),
                min(orig_w, x2 + pw),
                min(orig_h, y2 + ph),
            )

            crop = frame_pil.crop(crop_box)

            vit_in = self.expert_proc(images=crop, return_tensors="np")
            logits = self.expert.run(
                None, {"pixel_values": vit_in["pixel_values"]}
            )[0]

            probs = torch.softmax(torch.from_numpy(logits), dim=-1)[0]
            conf, cls_idx = torch.max(probs, dim=-1)

            if conf.item() < 0.40:
                continue

            label = self.id2label[str(cls_idx.item())]

            merged = False
            for ent in entities:
                if ent["label"] != label:
                    continue
            
                overlap = max(
                    get_iou(box, ent["box"]),
                    get_ios(box, ent["box"]),
                )
            
                if overlap > 0.25:
                    ent["box"] = capped_union(ent["box"], box)
                    ent["confidence"] = max(ent["confidence"], conf.item())
                    merged = True
                    break
            
            if not merged and len(entities) < max_entities:
                entities.append(
                    {
                        "box": box,
                        "label": label,
                        "confidence": conf.item(),
                        "crop": crop,
                    }
                )
        return entities