"""Smoke test: build model + dataset, run one detection forward/backward, time it."""
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch
from torch.utils.data import DataLoader

from config import FULL_CONFIG, LESION_CLASSES
from data.prepare_labeled_data import scan_labeled_dataset
from data.dataset import EczemaDataset
from data.transforms import get_train_transforms
from models.multi_task_model import build_model
from train_labeled import _collate_fn

LABELED = "D:/hexy/Dermatology_classification/labeled"

def main():
    print("Scanning...")
    samples = scan_labeled_dataset(LABELED)
    # use a small subset to smoke-test quickly
    sub = samples[:12]
    cfg = {**FULL_CONFIG, "image_size": (512, 512), "batch_size": 4, "num_workers": 0,
           "detection_enabled": True, "use_amp": False, "device": "cpu"}

    print("Building model (convnext_tiny, pretrained)...")
    t0 = time.time()
    model = build_model(backbone_name="convnext_tiny", pretrained=True,
                        detection_enabled=True)
    print(f"  model built in {time.time()-t0:.1f}s")
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  params: {n_params/1e6:.1f}M")

    print("Building dataset + loader...")
    tfm = get_train_transforms(image_size=cfg["image_size"], level="medical")
    ds = EczemaDataset(sub, transform=tfm)
    dl = DataLoader(ds, batch_size=cfg["batch_size"], shuffle=False,
                    num_workers=0, collate_fn=_collate_fn)

    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)

    print("Running one forward+backward step (detection enabled)...")
    batch = next(iter(dl))
    images = batch["image"]
    print(f"  batch images: {images.shape}")
    n_boxes = sum(len(b) for b in batch["bboxes"])
    print(f"  boxes in batch: {n_boxes}")
    print(f"  lesion_labels: {[b.tolist() for b in batch['lesion_labels']]}")

    targets = []
    has = False
    for i in range(len(images)):
        targets.append({"boxes": batch["bboxes"][i], "labels": batch["lesion_labels"][i]})
        if len(batch["bboxes"][i]) > 0:
            has = True
    print(f"  has_det_targets: {has}")

    t0 = time.time()
    out = model(images, targets=targets)
    print(f"  forward took {time.time()-t0:.1f}s")
    print("  output keys:", list(out.keys()))
    det_losses = out.get("detection", {}).get("losses", {})
    print("  det losses:", {k: round(v.item(), 4) for k, v in det_losses.items()})

    total = out["disease_logits"].sum() * 0.0
    for v in det_losses.values():
        total = total + v
    total = total + out["disease_logits"].mean()
    t0 = time.time()
    total.backward()
    print(f"  backward took {time.time()-t0:.1f}s")

    # inference
    model.eval()
    with torch.no_grad():
        t0 = time.time()
        out2 = model(images)
        print(f"  inference forward took {time.time()-t0:.1f}s")
        det = out2.get("detection", {})
        boxes = det.get("boxes", [])
        print(f"  inference detections per image: {[len(b) for b in boxes]}")
        if boxes and len(boxes[0]):
            print(f"    img0 boxes: {boxes[0][:3].tolist()}")
            print(f"    img0 labels: {det['labels'][0][:3].tolist()}")
            print(f"    img0 scores: {[round(float(s),3) for s in det['scores'][0][:3]]}")

    print("SMOKE TEST PASSED")

if __name__ == "__main__":
    main()
