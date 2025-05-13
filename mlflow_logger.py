import mlflow
import mlflow.transformers
from urllib.parse import urlparse
import os
import pandas as pd
import soundfile as sf
import numpy as np
import torch
from datasets import Dataset, Audio
from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments
from dataclasses import dataclass
from typing import Any, Dict, List, Union
from transformers import pipeline
import mlflow.pytorch

# --------------------------
# 1. Load and filter dataset
# --------------------------

LJS_PATH = "LJSpeech-1.1"
metadata_path = os.path.join(LJS_PATH, "metadata.csv")
print(f"Loading metadata from {metadata_path}")

# Load metadata
df = pd.read_csv(metadata_path, sep='|', header=None, names=["file_id", "normalized_text", "unused"])
print("Metadata loaded successfully.")

# Filter rows: keep only LJ001-0001 to LJ001-0186
allowed_ids = [f"LJ001-{i:04d}" for i in range(1, 187)]
df = df[df["file_id"].isin(allowed_ids)].reset_index(drop=True)

# Add full path to .wav files
df["audio_path"] = df["file_id"].apply(lambda x: os.path.join(LJS_PATH, "wavs", f"{x}.wav"))

# Filter out corrupted files BEFORE casting
valid_rows = []
for _, row in df.iterrows():
    try:
        sf.read(row["audio_path"])
        valid_rows.append(row)
    except Exception as e:
        print(f"Skipping corrupted file: {row['audio_path']} ({e})")

# Use valid files only
df = pd.DataFrame(valid_rows)

# Create Hugging Face Dataset
ds = Dataset.from_pandas(df[["audio_path", "normalized_text"]])
ds = ds.rename_columns({"audio_path": "audio", "normalized_text": "normalized_text"})
ds = ds.cast_column("audio", Audio(sampling_rate=16000))

# Preview
print(ds[0])
print(f"Dataset size: {len(ds)}")

# --------------------------
# 2. Preprocessing
# --------------------------

processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
DUMMY_SPEAKER_EMBEDDING = np.zeros(512)

def prepare_dataset(example):
    audio = example["audio"]
    example = processor(
        text=example["normalized_text"],
        audio_target=audio["array"],
        sampling_rate=audio["sampling_rate"],
        return_attention_mask=False,
    )
    example["labels"] = example["labels"][0]  # Remove batch dim
    example["speaker_embeddings"] = DUMMY_SPEAKER_EMBEDDING
    return example

ds = ds.map(prepare_dataset, remove_columns=ds.column_names)
dataset = ds.train_test_split(test_size=0.1)

# --------------------------
# 3. Data Collator
# --------------------------

@dataclass
class TTSDataCollatorWithPadding:
    processor: Any

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        input_ids = [{"input_ids": feature["input_ids"]} for feature in features]
        label_features = [{"input_values": feature["labels"]} for feature in features]
        speaker_features = [feature["speaker_embeddings"] for feature in features]

        batch = self.processor.pad(input_ids=input_ids, labels=label_features, return_tensors="pt")

        batch["labels"] = batch["labels"].masked_fill(
            batch["decoder_attention_mask"].unsqueeze(-1).ne(1), -100
        )
        del batch["decoder_attention_mask"]

        reduction_factor = getattr(model.config, "reduction_factor", 1)
        if reduction_factor > 1:
            lengths = [len(feature["input_values"]) for feature in label_features]
            trimmed_length = min(length - (length % reduction_factor) for length in lengths)
            batch["labels"] = batch["labels"][:, :trimmed_length]

        batch["speaker_embeddings"] = torch.tensor(speaker_features)
        return batch

data_collator = TTSDataCollatorWithPadding(processor=processor)

# --------------------------
# 4. Load Model
# --------------------------

model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts")
model.config.use_cache = False

# --------------------------
# 5. Training Arguments
# --------------------------

training_args = Seq2SeqTrainingArguments(
    output_dir="speechT5_finetuned_ljspeech",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=2,
    learning_rate=1e-4,
    warmup_steps=100,
    max_steps=300,
    gradient_checkpointing=False,
    fp16=False,
    evaluation_strategy="steps",
    per_device_eval_batch_size=2,
    save_steps=300,
    eval_steps=300,
    logging_steps=50,
    report_to=["tensorboard"],
    load_best_model_at_end=True,
    greater_is_better=False,
    label_names=["labels"],
)

trainer = Seq2SeqTrainer(
    args=training_args,
    model=model,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    data_collator=data_collator,
    tokenizer=processor,
)

# --------------------------
# 6. Trainer & Run
# --------------------------

mlflow.set_tracking_uri("http://127.0.0.1:8080")
mlflow.set_experiment("SpeechT5_FineTuning")

with mlflow.start_run(run_name="speechT5_run"):

    mlflow.log_params({
        "learning_rate": training_args.learning_rate,
        "batch_size": training_args.per_device_train_batch_size,
        "max_steps": training_args.max_steps,
        "gradient_accumulation_steps": training_args.gradient_accumulation_steps,
        "warmup_steps": training_args.warmup_steps
    })

    trainer.train()
    trainer.save_model("speechT5_finetuned_ljspeech")

    tracking_type = urlparse(mlflow.get_tracking_uri()).scheme

    mlflow.pytorch.log_model(
        pytorch_model=model,
        artifact_path="speechT5_model",
        registered_model_name="SpeechT5ThaiModel" if tracking_type != "file" else None
    )

    processor.save_pretrained("speechT5_finetuned_ljspeech")
    mlflow.log_artifacts("speechT5_finetuned_ljspeech", artifact_path="processor")

    mlflow.log_metric("final_loss", trainer.state.log_history[-1].get("loss", -1))
