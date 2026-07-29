"""
LLM Resume Classifier — Fine-Tuned DistilBERT for Placement Prediction
========================================================================

Fine-tunes a pre-trained DistilBERT transformer model on placement data.
Converts student profiles to natural language paragraphs and trains the model
to classify them as "Placed" or "Not Placed".

This module provides:
  - Training pipeline (generate data → tokenise → fine-tune → save)
  - Inference pipeline (load model → predict from resume text)
"""

import os
import sys
import logging
from typing import Optional

import pandas as pd

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# 1. Generate Resume Text from CSV Row
# ═══════════════════════════════════════════════════════════════════════════
def generate_resume_text(row) -> str:
    """Convert a DataFrame row to a natural language resume paragraph.

    Args:
        row: A pandas Series (one row of the DataFrame).

    Returns:
        A descriptive paragraph summarising the student's profile.
    """
    parts = []

    age = row.get("Age", "unknown age")
    gender = row.get("Gender", "")
    degree = row.get("Degree", "")
    branch = row.get("Branch", "")
    parts.append(f"A {age} year old {gender} student pursuing {degree} in {branch}.")

    cgpa = row.get("CGPA", 0)
    backlogs = row.get("Backlogs", 0)
    parts.append(f"Has a CGPA of {cgpa} with {backlogs} backlogs.")

    internships = row.get("Internships", 0)
    projects = row.get("Projects", 0)
    parts.append(f"Completed {internships} internships and {projects} projects.")

    coding = row.get("Coding_Skills", 0)
    comm = row.get("Communication_Skills", 0)
    parts.append(f"Coding skills rated {coding} out of 10. Communication skills rated {comm} out of 10.")

    aptitude = row.get("Aptitude_Test_Score", 0)
    soft = row.get("Soft_Skills_Rating", 0)
    parts.append(f"Aptitude test score: {aptitude} out of 100. Soft skills rating: {soft} out of 10.")

    certs = row.get("Certifications", 0)
    parts.append(f"Holds {certs} certifications.")

    return " ".join(parts)


# ═══════════════════════════════════════════════════════════════════════════
# 2. PyTorch Dataset
# ═══════════════════════════════════════════════════════════════════════════
def _create_dataset_class():
    """Create and return the ResumeDataset class (imports torch lazily)."""
    import torch
    from torch.utils.data import Dataset

    class ResumeDataset(Dataset):
        """PyTorch Dataset for tokenised resume texts."""

        def __init__(self, texts: list[str], labels: list[int], tokenizer, max_length: int = 128):
            self.texts = texts
            self.labels = labels
            self.tokenizer = tokenizer
            self.max_length = max_length

        def __len__(self):
            return len(self.texts)

        def __getitem__(self, idx):
            encoding = self.tokenizer(
                self.texts[idx],
                max_length=self.max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            return {
                "input_ids": encoding["input_ids"].squeeze(0),
                "attention_mask": encoding["attention_mask"].squeeze(0),
                "labels": torch.tensor(self.labels[idx], dtype=torch.long),
            }

    return ResumeDataset


# ═══════════════════════════════════════════════════════════════════════════
# 3. Training Pipeline
# ═══════════════════════════════════════════════════════════════════════════
def train_llm_model(
    train_csv: str,
    model_dir: str,
    epochs: int = 3,
    batch_size: int = 16,
    learning_rate: float = 2e-5,
    max_length: int = 128,
    max_samples: int = 5000,
) -> None:
    """Fine-tune DistilBERT on placement resume data.

    Args:
        train_csv: Path to the training CSV file.
        model_dir: Directory to save the fine-tuned model.
        epochs: Number of training epochs.
        batch_size: Training batch size.
        learning_rate: Learning rate for AdamW optimiser.
        max_length: Maximum token sequence length.
        max_samples: Maximum samples to use (for speed on CPU).
    """
    import torch
    from torch.utils.data import DataLoader
    from transformers import DistilBertTokenizer, DistilBertForSequenceClassification

    logger.info("=" * 70)
    logger.info("   LLM TRAINING PIPELINE — DistilBERT Fine-Tuning")
    logger.info("=" * 70)

    # ── Step 1: Load and prepare data ───────────────────────────────────
    logger.info("\n▶ Step 1: Loading training data...")
    df = pd.read_csv(train_csv)

    # Subsample for CPU training speed
    if len(df) > max_samples:
        df = df.sample(n=max_samples, random_state=42).reset_index(drop=True)
        logger.info(f"  Subsampled to {max_samples} records for training speed")

    # Generate resume texts
    logger.info("▶ Step 2: Generating resume paragraphs...")
    texts = [generate_resume_text(row) for _, row in df.iterrows()]

    # Encode labels
    label_map = {"Placed": 1, "Not Placed": 0}
    labels = [label_map.get(str(val), 0) for val in df["Placement_Status"]]

    logger.info(f"  Generated {len(texts)} resume paragraphs")
    logger.info(f"  Class distribution: Placed={sum(labels)}, Not Placed={len(labels) - sum(labels)}")

    # ── Step 2: Load pre-trained DistilBERT ─────────────────────────────
    logger.info("\n▶ Step 3: Loading pre-trained DistilBERT model...")
    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
    model = DistilBertForSequenceClassification.from_pretrained(
        "distilbert-base-uncased",
        num_labels=2,
    )

    # ── Step 3: Create dataset and dataloader ───────────────────────────
    logger.info("▶ Step 4: Tokenising and creating dataset...")
    ResumeDataset = _create_dataset_class()

    # Split into train (90%) and validation (10%)
    split_idx = int(len(texts) * 0.9)
    train_dataset = ResumeDataset(texts[:split_idx], labels[:split_idx], tokenizer, max_length)
    val_dataset = ResumeDataset(texts[split_idx:], labels[split_idx:], tokenizer, max_length)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    logger.info(f"  Train samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}")

    # ── Step 4: Training setup ──────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"\n▶ Step 5: Training on device: {device}")
    if device.type == "cuda":
        logger.info(f"  GPU: {torch.cuda.get_device_name(0)}")
    else:
        logger.info("  (No GPU detected — training on CPU. This will take longer.)")

    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    # ── Step 5: Training loop ───────────────────────────────────────────
    logger.info(f"\n▶ Step 6: Fine-tuning for {epochs} epochs...")

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0

        for batch_idx, batch in enumerate(train_loader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            batch_labels = batch["labels"].to(device)

            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=batch_labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            preds = outputs.logits.argmax(dim=-1)
            correct += (preds == batch_labels).sum().item()
            total += batch_labels.size(0)

            # Progress every 20 batches
            if (batch_idx + 1) % 20 == 0:
                logger.info(
                    f"  Epoch {epoch + 1}/{epochs} | "
                    f"Batch {batch_idx + 1}/{len(train_loader)} | "
                    f"Loss: {loss.item():.4f}"
                )

        train_acc = correct / total
        avg_loss = total_loss / len(train_loader)

        # ── Validation ──────────────────────────────────────────────────
        model.eval()
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                batch_labels = batch["labels"].to(device)

                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                preds = outputs.logits.argmax(dim=-1)
                val_correct += (preds == batch_labels).sum().item()
                val_total += batch_labels.size(0)

        val_acc = val_correct / val_total

        logger.info(
            f"\n  ✅ Epoch {epoch + 1}/{epochs} Complete | "
            f"Train Loss: {avg_loss:.4f} | "
            f"Train Acc: {train_acc:.4f} | "
            f"Val Acc: {val_acc:.4f}"
        )

    # ── Step 6: Save model ──────────────────────────────────────────────
    save_dir = os.path.join(model_dir, "llm_resume_model")
    os.makedirs(save_dir, exist_ok=True)

    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)

    logger.info(f"\n▶ Step 7: Model saved to {save_dir}")
    logger.info("=" * 70)
    logger.info("   LLM TRAINING COMPLETE")
    logger.info("=" * 70)


# ═══════════════════════════════════════════════════════════════════════════
# 4. Load Trained Model
# ═══════════════════════════════════════════════════════════════════════════
def load_llm_model(model_path: str):
    """Load a fine-tuned DistilBERT model and tokenizer.

    Args:
        model_path: Path to the saved model directory.

    Returns:
        Tuple of (model, tokenizer, device).
    """
    import torch
    from transformers import DistilBertTokenizer, DistilBertForSequenceClassification

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = DistilBertTokenizer.from_pretrained(model_path)
    model = DistilBertForSequenceClassification.from_pretrained(model_path)
    model.to(device)
    model.eval()

    logger.info(f"LLM model loaded from {model_path} (device: {device})")
    return model, tokenizer, device


# ═══════════════════════════════════════════════════════════════════════════
# 5. Predict from Resume Text
# ═══════════════════════════════════════════════════════════════════════════
def predict_resume_llm(
    text: str,
    model=None,
    tokenizer=None,
    device=None,
    model_path: Optional[str] = None,
) -> dict:
    """Classify resume text using the fine-tuned DistilBERT model.

    Args:
        text: Resume text to classify.
        model: Loaded model (optional, will load from model_path if None).
        tokenizer: Loaded tokenizer.
        device: PyTorch device.
        model_path: Path to load model from (if model is None).

    Returns:
        Dictionary with 'label' ("Placed"/"Not Placed") and 'confidence' (0-1).
    """
    import torch

    if model is None:
        if model_path is None:
            raise ValueError("Either model or model_path must be provided")
        model, tokenizer, device = load_llm_model(model_path)

    encoding = tokenizer(
        text,
        max_length=128,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )

    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        probs = torch.softmax(outputs.logits, dim=-1)
        pred_class = probs.argmax(dim=-1).item()
        confidence = probs[0][pred_class].item()

    label_map = {0: "Not Placed", 1: "Placed"}

    return {
        "label": label_map[pred_class],
        "confidence": round(confidence, 4),
        "placed_probability": round(probs[0][1].item(), 4),
    }
