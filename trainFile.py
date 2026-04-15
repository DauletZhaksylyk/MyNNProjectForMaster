import argparse
import datetime
import json
import os
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForMultipleChoice,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

from AugFile import augment_text
from text_utils import FEATURE_COLUMNS, build_model_text, extract_structured_features, normalize_text


DEFAULT_MODEL_NAME = "DeepPavlov/rubert-base-cased"
DEFAULT_DATASET = "fraud_dataset_clean_final.csv"
DEFAULT_ARCHITECTURE = "multiple_choice"
CLASS_ORDER = ["normal", "fraud"]
MULTIPLE_CHOICE_PROMPTS = [
    "Этот текст или разговор является обычным безопасным сообщением без признаков мошенничества.",
    "Этот текст или разговор является мошенничеством, скамом или социальной инженерией.",
]


def load_dataset(dataset_path: str) -> pd.DataFrame:
    df = pd.read_csv(dataset_path)
    df = df.dropna(subset=["text", "label"]).copy()
    df["text"] = df["text"].astype(str).str.strip()
    df["label"] = df["label"].astype(str).str.strip().str.lower()
    df = df[df["label"].isin(["fraud", "normal"])].copy()
    for column in FEATURE_COLUMNS:
        if column not in df.columns:
            df[column] = None

    inferred_features = [extract_structured_features(text, label) for text, label in zip(df["text"], df["label"])]
    inferred_df = pd.DataFrame(inferred_features)

    for column in FEATURE_COLUMNS:
        df[column] = df[column].where(df[column].notna(), inferred_df[column])
        if column.startswith("has_") or column == "victim_confused_or_resisting" or column == "risk_markers_count":
            df[column] = df[column].astype(int)
        else:
            df[column] = df[column].astype(str)

    df["model_text"] = [
        build_model_text(text, features)
        for text, features in zip(df["text"], df[FEATURE_COLUMNS].to_dict("records"))
    ]
    df["group_key"] = df["text"].map(lambda value: normalize_text(value, drop_speaker_prefix=False))
    df = df[df["model_text"].str.len() > 0].reset_index(drop=True)
    return df


def split_dataset(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame]:
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_idx, test_idx = next(splitter.split(df["model_text"], df["label"], groups=df["group_key"]))
    return df.iloc[train_idx].reset_index(drop=True), df.iloc[test_idx].reset_index(drop=True)


def augment_training_data(train_df: pd.DataFrame, target_size_per_class: int) -> pd.DataFrame:
    augmented_rows: List[Dict[str, str]] = []
    methods = ["del", "swap", "noise"]

    for label in sorted(train_df["label"].unique()):
        class_df = train_df[train_df["label"] == label]
        need_more = max(0, target_size_per_class - len(class_df))
        if need_more == 0:
            continue

        source_rows = class_df.sample(frac=1.0, random_state=42).to_dict("records")
        while need_more > 0:
            produced_this_round = 0
            for row in source_rows:
                augmented = augment_text(row["text"], methods=methods, max_try_per_method=1)
                for text in augmented:
                    feature_values = {column: row[column] for column in FEATURE_COLUMNS}
                    augmented_rows.append(
                        {
                            "text": text,
                            "label": label,
                            **feature_values,
                            "model_text": build_model_text(text, feature_values),
                            "group_key": row["group_key"],
                        }
                    )
                    need_more -= 1
                    produced_this_round += 1
                    if need_more <= 0:
                        break
                if need_more <= 0:
                    break
            if produced_this_round == 0:
                break

    if not augmented_rows:
        return train_df

    return pd.concat([train_df, pd.DataFrame(augmented_rows)], ignore_index=True)


class SequenceClassificationDataset(Dataset):
    def __init__(self, encodings: Dict[str, torch.Tensor], labels: List[int]):
        self.encodings = encodings
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = {key: value[idx] for key, value in self.encodings.items()}
        item["labels"] = self.labels[idx]
        return item


class MultipleChoiceDataset(Dataset):
    def __init__(self, encodings: Dict[str, torch.Tensor], labels: List[int]):
        self.encodings = encodings
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = {key: value[idx] for key, value in self.encodings.items()}
        item["labels"] = self.labels[idx]
        return item


def encode_sequence_classification(tokenizer, texts: List[str], max_length: int) -> Dict[str, torch.Tensor]:
    return tokenizer(texts, truncation=True, padding=True, max_length=max_length, return_tensors="pt")


def encode_multiple_choice(tokenizer, texts: List[str], max_length: int) -> Dict[str, torch.Tensor]:
    first_sentences = [[text] * len(MULTIPLE_CHOICE_PROMPTS) for text in texts]
    second_sentences = [MULTIPLE_CHOICE_PROMPTS for _ in texts]
    flat_first = [item for group in first_sentences for item in group]
    flat_second = [item for group in second_sentences for item in group]
    tokenized = tokenizer(
        flat_first,
        flat_second,
        truncation=True,
        padding=True,
        max_length=max_length,
        return_tensors="pt",
    )
    num_choices = len(MULTIPLE_CHOICE_PROMPTS)
    return {key: value.view(len(texts), num_choices, -1) for key, value in tokenized.items()}


def build_dataloaders(
    tokenizer,
    architecture: str,
    train_texts: List[str],
    test_texts: List[str],
    y_train: List[int],
    y_test: List[int],
    max_length: int,
    batch_size: int,
) -> Tuple[DataLoader, DataLoader]:
    if architecture == "multiple_choice":
        train_encodings = encode_multiple_choice(tokenizer, train_texts, max_length=max_length)
        test_encodings = encode_multiple_choice(tokenizer, test_texts, max_length=max_length)
        train_dataset = MultipleChoiceDataset(train_encodings, y_train)
        test_dataset = MultipleChoiceDataset(test_encodings, y_test)
    else:
        train_encodings = encode_sequence_classification(tokenizer, train_texts, max_length=max_length)
        test_encodings = encode_sequence_classification(tokenizer, test_texts, max_length=max_length)
        train_dataset = SequenceClassificationDataset(train_encodings, y_train)
        test_dataset = SequenceClassificationDataset(test_encodings, y_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader


def create_model(model_name: str, architecture: str):
    try:
        if architecture == "multiple_choice":
            return AutoModelForMultipleChoice.from_pretrained(model_name)
        return AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=len(CLASS_ORDER))
    except Exception as exc:
        raise RuntimeError(
            "Failed to load pretrained model weights. Check internet access or cache the model locally."
        ) from exc


def train_one_epoch(model, dataloader, optimizer, scheduler, device):
    model.train()
    total_loss = 0.0
    for batch in dataloader:
        batch = {key: value.to(device) for key, value in batch.items()}
        optimizer.zero_grad()
        output = model(**batch)
        loss = output.loss
        loss.backward()
        optimizer.step()
        scheduler.step()
        total_loss += float(loss.item())
    return total_loss / max(1, len(dataloader))


def predict_logits(model, dataloader, device):
    model.eval()
    all_logits = []
    all_labels = []
    total_loss = 0.0
    with torch.no_grad():
        for batch in dataloader:
            labels = batch["labels"]
            batch = {key: value.to(device) for key, value in batch.items()}
            output = model(**batch)
            total_loss += float(output.loss.item())
            all_logits.append(output.logits.cpu().numpy())
            all_labels.append(labels.numpy())
    logits = np.concatenate(all_logits, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    return logits, labels, total_loss / max(1, len(dataloader))


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, fraud_scores: np.ndarray) -> Dict[str, float]:
    fraud_label_id = CLASS_ORDER.index("fraud")
    report = classification_report(y_true, y_pred, target_names=CLASS_ORDER, digits=4, zero_division=0, output_dict=True)
    cm = confusion_matrix(y_true, y_pred, labels=[CLASS_ORDER.index("normal"), fraud_label_id])
    tn, fp, fn, tp = [int(value) for value in cm.ravel()]
    metrics = {
        "accuracy": float((y_true == y_pred).mean()),
        "fraud_precision": float(precision_score(y_true, y_pred, pos_label=fraud_label_id, zero_division=0)),
        "fraud_recall": float(recall_score(y_true, y_pred, pos_label=fraud_label_id, zero_division=0)),
        "fraud_f1": float(f1_score(y_true, y_pred, pos_label=fraud_label_id, zero_division=0)),
        "macro_f1": float(report["macro avg"]["f1-score"]),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }
    if len(np.unique(y_true)) > 1:
        metrics["roc_auc"] = float(roc_auc_score(y_true, fraud_scores))
        metrics["pr_auc"] = float(average_precision_score(y_true, fraud_scores))
    else:
        metrics["roc_auc"] = None
        metrics["pr_auc"] = None
    return metrics


def train(
    dataset_path: str = DEFAULT_DATASET,
    model_name: str = DEFAULT_MODEL_NAME,
    architecture: str = DEFAULT_ARCHITECTURE,
    output_dir: str = "",
    use_augmentation: bool = True,
    target_size: int = 220,
    max_length: int = 256,
    batch_size: int = 8,
    learning_rate: float = 2e-5,
    epochs: int = 3,
):
    print("=" * 80)
    print("LOADING DATASET")
    print("=" * 80)

    df = load_dataset(dataset_path)
    print(f"Dataset: {dataset_path}")
    print(f"Samples: {len(df)}")
    print(df["label"].value_counts().to_string())
    print("\nScenario distribution:")
    print(df["scenario_type"].value_counts().head(10).to_string())

    train_df, test_df = split_dataset(df)
    print("\nSplit completed before augmentation.")
    print(f"Train size: {len(train_df)}")
    print(f"Test size: {len(test_df)}")

    if use_augmentation:
        train_df = augment_training_data(train_df, target_size_per_class=target_size)
        print(f"Train size after augmentation: {len(train_df)}")
    else:
        print("Augmentation disabled.")

    class_to_id = {label: idx for idx, label in enumerate(CLASS_ORDER)}
    y_train = [class_to_id[label] for label in train_df["label"].tolist()]
    y_test = [class_to_id[label] for label in test_df["label"].tolist()]

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    except Exception as exc:
        raise RuntimeError(
            "Failed to load tokenizer/model from Hugging Face. Check internet access or download the model to local cache first."
        ) from exc

    train_loader, test_loader = build_dataloaders(
        tokenizer=tokenizer,
        architecture=architecture,
        train_texts=train_df["model_text"].tolist(),
        test_texts=test_df["model_text"].tolist(),
        y_train=y_train,
        y_test=y_test,
        max_length=max_length,
        batch_size=batch_size,
    )

    print("\n" + "=" * 80)
    print("TRAINING MODEL")
    print("=" * 80)
    print(f"Model: {model_name}")
    print(f"Architecture: {architecture}")
    print(f"Classes: {class_to_id}")

    model = create_model(model_name, architecture=architecture)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optimizer = AdamW(model.parameters(), lr=learning_rate)
    total_steps = max(1, len(train_loader) * epochs)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=max(1, total_steps // 10), num_training_steps=total_steps)

    history = {"train_loss": [], "val_loss": []}
    best_val_loss = float("inf")
    best_state = None

    for epoch in range(epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, scheduler, device)
        _, _, val_loss = predict_logits(model, test_loader, device)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        print(f"Epoch {epoch + 1}/{epochs}: train_loss={train_loss:.4f} val_loss={val_loss:.4f}")
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {key: value.cpu().clone() for key, value in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    print("\n" + "=" * 80)
    print("EVALUATION")
    print("=" * 80)

    logits, y_true, loss = predict_logits(model, test_loader, device)
    y_pred = np.argmax(logits, axis=1)
    fraud_scores = torch.softmax(torch.tensor(logits), dim=1).numpy()[:, class_to_id["fraud"]]
    metrics = compute_metrics(y_true, y_pred, fraud_scores)
    metrics["loss"] = float(loss)

    report_text = classification_report(y_true, y_pred, target_names=CLASS_ORDER, digits=4, zero_division=0)
    report_dict = classification_report(y_true, y_pred, target_names=CLASS_ORDER, digits=4, zero_division=0, output_dict=True)
    cm = confusion_matrix(y_true, y_pred).tolist()

    print(report_text)
    print("Confusion matrix:")
    for row in cm:
        print(row)
    print("Metrics:")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    model_dir = output_dir or f"fraud_call_model_{architecture}_{timestamp}"
    os.makedirs(model_dir, exist_ok=True)
    model.save_pretrained(model_dir)
    tokenizer.save_pretrained(model_dir)
    joblib.dump(CLASS_ORDER, os.path.join(model_dir, "label_encoder.joblib"))

    metadata = {
        "task": "post_call_fraud_detection",
        "architecture": architecture,
        "dataset_file": dataset_path,
        "model_name": model_name,
        "classes": CLASS_ORDER,
        "class_to_id": class_to_id,
        "multiple_choice_prompts": MULTIPLE_CHOICE_PROMPTS if architecture == "multiple_choice" else None,
        "max_length": max_length,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "epochs_requested": epochs,
        "epochs_trained": len(history["train_loss"]),
        "train_size": len(train_df),
        "test_size": len(test_df),
        "use_augmentation": use_augmentation,
        "target_size_per_class": target_size if use_augmentation else None,
        "metrics": metrics,
        "classification_report": report_dict,
        "confusion_matrix": cm,
        "feature_columns": FEATURE_COLUMNS,
        "created_at": timestamp,
    }

    with open(os.path.join(model_dir, "metadata.json"), "w", encoding="utf-8") as file:
        json.dump(metadata, file, ensure_ascii=False, indent=2)

    with open(os.path.join(model_dir, "metrics.json"), "w", encoding="utf-8") as file:
        json.dump(metrics, file, ensure_ascii=False, indent=2)

    with open(os.path.join(model_dir, "classification_report.txt"), "w", encoding="utf-8") as file:
        file.write(report_text)

    with open(os.path.join(model_dir, "confusion_matrix.json"), "w", encoding="utf-8") as file:
        json.dump(
            {
                "labels": CLASS_ORDER,
                "matrix": cm,
            },
            file,
            ensure_ascii=False,
            indent=2,
        )

    with open(os.path.join(model_dir, "history.json"), "w", encoding="utf-8") as file:
        json.dump(history, file, ensure_ascii=False, indent=2)

    print(f"\nSaved model to: {model_dir}")
    return model, history, metadata


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train fraud detector with sequence classification or multiple choice.")
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--architecture", choices=["multiple_choice", "sequence_classification"], default=DEFAULT_ARCHITECTURE)
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--no-augment", action="store_true")
    parser.add_argument("--target-size", type=int, default=220)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--epochs", type=int, default=3)
    args = parser.parse_args()

    train(
        dataset_path=args.dataset,
        model_name=args.model_name,
        architecture=args.architecture,
        output_dir=args.output_dir,
        use_augmentation=not args.no_augment,
        target_size=args.target_size,
        max_length=args.max_length,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        epochs=args.epochs,
    )
