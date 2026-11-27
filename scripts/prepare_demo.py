"""Подготовить комплект BERT-Tiny и SST-2 для автономного запуска.

Скрипт выполняется только при сборке Docker-образа, а не при запуске приложения.
Веса опубликованной модели и данные GLUE скачиваются с Hugging Face.
"""

import hashlib
import json
import os
from pathlib import Path

import onnx
import torch
from datasets import load_dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL_ID = "M-FAC/bert-tiny-finetuned-sst2"
MODEL_REVISION = "3545b172246737cf07975f902be6bf73adff900d"
DATASET_ID = "nyu-mll/glue"
DATASET_CONFIG = "sst2"
SEED = 42
CALIBRATION_COUNT = 256
MAX_LENGTH = 128
OUT_DIR = Path(os.environ.get("QUANTLAB_DEMO_OUT", "/build/examples"))


class ClassifierExport(torch.nn.Module):
    """Передавать в ONNX Runtime только два логита классификации."""

    def __init__(self, classifier):
        super().__init__()
        self.classifier = classifier

    def forward(self, input_ids, attention_mask):
        return self.classifier(input_ids=input_ids, attention_mask=attention_mask).logits


def serialize_samples(dataset, path):
    """Сохранить примеры в переносимом JSON-формате для конкретной задачи."""
    rows = [{"text": str(row["sentence"]), "label": int(row["label"])} for row in dataset]
    path.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    return len(rows)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(SEED)
    torch.set_num_threads(2)

    print(f"[DEMO] Exporting {MODEL_ID} at {MODEL_REVISION}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID, revision=MODEL_REVISION, trust_remote_code=False
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_ID, revision=MODEL_REVISION, trust_remote_code=False
    )
    model.eval()
    tokenizer.save_pretrained(OUT_DIR / "tokenizer")

    inputs = tokenizer(
        ["An excellent movie.", "A terrible movie."],
        padding=True, truncation=True, max_length=MAX_LENGTH, return_tensors="pt",
    )
    wrapper = ClassifierExport(model).eval()
    model_path = OUT_DIR / "bert_tiny_fp32.onnx"
    with torch.inference_mode():
        torch.onnx.export(
            wrapper,
            (inputs["input_ids"], inputs["attention_mask"]),
            str(model_path),
            input_names=["input_ids", "attention_mask"],
            output_names=["logits"],
            dynamic_axes={
                "input_ids": {0: "batch", 1: "sequence"},
                "attention_mask": {0: "batch", 1: "sequence"},
                "logits": {0: "batch"},
            },
            opset_version=17,
            dynamo=False,
        )
    onnx.checker.check_model(str(model_path))

    print("[DEMO] Loading fixed SST-2 calibration/evaluation samples", flush=True)
    train = load_dataset(DATASET_ID, DATASET_CONFIG, split="train")
    val = load_dataset(DATASET_ID, DATASET_CONFIG, split="validation")
    calibration = train.shuffle(seed=SEED).select(range(min(CALIBRATION_COUNT, len(train))))
    num_calibration = serialize_samples(calibration, OUT_DIR / "calibration_samples.json")
    num_eval = serialize_samples(val, OUT_DIR / "evaluation_samples.json")
    if num_calibration != CALIBRATION_COUNT or num_eval != 872:
        raise RuntimeError(f"Unexpected SST-2 sample counts: {num_calibration}, {num_eval}")

    manifest = {
        "model": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "model_sha256": hashlib.sha256(model_path.read_bytes()).hexdigest(),
        "model_bytes": model_path.stat().st_size,
        "dataset": f"{DATASET_ID}/{DATASET_CONFIG}",
        "calibration_split": "train",
        "calibration_seed": SEED,
        "calibration_count": num_calibration,
        "evaluation_split": "validation",
        "evaluation_count": num_eval,
        "max_length": MAX_LENGTH,
        "format": "ONNX FP32, opset 17",
    }
    (OUT_DIR / "demo_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"[DEMO] Ready: {model_path.stat().st_size / 1024 ** 2:.2f} MiB / "
          f"calibration={num_calibration} / evaluation={num_eval}", flush=True)


if __name__ == "__main__":
    main()
