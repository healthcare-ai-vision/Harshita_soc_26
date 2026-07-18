"""
Model loading & inference for your 3 Ultralytics YOLOv8 models:
  - skin classification:   yolov8m-cls.pt  -> imgsz=224  (SoC-Cls.py)
  - skin segmentation:     yolov8s-seg.pt  -> imgsz=640  (SoC-Seg.py)
  - throat classification: yolov8s-cls.pt  -> imgsz=640  (SoC-Det-cls.py)

After training on Colab, Ultralytics saves the final weights at:
  runs/classify/<name>/weights/best.pt
  runs/segment/<name>/weights/best.pt

Download those best.pt files and place them here, renamed:
  models/skin_classifier.pt
  models/skin_segmentation.pt
  models/throat_classifier.pt

Class names don't need to be typed out manually — Ultralytics stores them
inside the model checkpoint (taken from your training folder names), so
they come back automatically in result.names.
"""

import numpy as np
from PIL import Image
from ultralytics import YOLO

SKIN_CLF_IMGSZ = 224     # matches imgsz=224 in your skin classification training
SKIN_SEG_IMGSZ = 640     # matches imgsz=640 in your segmentation training
THROAT_CLF_IMGSZ = 640   # matches imgsz=640 in your throat classification training

skin_clf_model = YOLO("models/skin_classifier.pt")
skin_seg_model = YOLO("models/skin_segmentation.pt")
throat_clf_model = YOLO("models/throat_classifier.pt")


def _classify(model: YOLO, image: Image.Image, imgsz: int):
    result = model.predict(image, imgsz=imgsz, verbose=False)[0]
    probs = result.probs
    names = result.names  # {index: label} embedded in the checkpoint
    idx = int(probs.top1)
    confidence = float(probs.top1conf)
    all_scores = {names[i]: float(probs.data[i]) for i in range(len(names))}
    return names[idx], confidence, all_scores


def predict_skin_classification(image: Image.Image):
    return _classify(skin_clf_model, image, SKIN_CLF_IMGSZ)


def predict_throat_classification(image: Image.Image):
    return _classify(throat_clf_model, image, THROAT_CLF_IMGSZ)


def predict_skin_segmentation(image: Image.Image):
    """
    Runs segmentation and returns:
      - mask_img: a single 0/255 numpy array (original image size) that's the
        union of all detected "affected area" instances, ready for st.image
      - affected_pct: % of the image's pixels flagged as affected skin
    """
    result = skin_seg_model.predict(image, imgsz=SKIN_SEG_IMGSZ, verbose=False)[0]

    w, h = image.size
    combined_mask = np.zeros((h, w), dtype=np.uint8)

    if result.masks is not None:
        # result.masks.data: (num_detected_instances, mask_h, mask_w) at model resolution
        masks = result.masks.data.cpu().numpy()
        for m in masks:
            m_resized = np.array(
                Image.fromarray((m * 255).astype(np.uint8)).resize((w, h))
            )
            combined_mask = np.maximum(combined_mask, m_resized)
    # if result.masks is None, nothing was detected as "affected" -> stays all zeros

    affected_pct = float((combined_mask > 127).sum()) / combined_mask.size * 100
    return combined_mask, affected_pct
