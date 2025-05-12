import os
import pandas as pd
import torchaudio
from datasets import Dataset, Audio

# Set your local path to LJSpeech
LJS_PATH = "LJSpeech-1.1"

# Load metadata
metadata_path = os.path.join(LJS_PATH, "metadata.csv")
df = pd.read_csv(metadata_path, sep='|', header=None, names=["file_id", "normalized_text", "unused"])

# Filter rows: keep only files LJ001-0001 to LJ001-0186
allowed_ids = [f"LJ001-{i:04d}" for i in range(1, 187)]  # 001 to 186
df = df[df["file_id"].isin(allowed_ids)].reset_index(drop=True)

# Create full paths to audio files
df["audio_path"] = df["file_id"].apply(lambda x: os.path.join(LJS_PATH, "wavs", f"{x}.wav"))

# Create Hugging Face Dataset from pandas
ds = Dataset.from_pandas(df[["audio_path", "normalized_text"]])

# Rename columns to expected format
ds = ds.rename_columns({"audio_path": "audio", "normalized_text": "normalized_text"})

# Cast to Hugging Face Audio format (with resampling)
ds = ds.cast_column("audio", Audio(sampling_rate=16000))

# Preview
print(ds[0])
len(ds)

import numpy as np
from transformers import SpeechT5Processor

processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
DUMMY_SPEAKER_EMBEDDING = np.zeros(512)  # Or use random: np.random.rand(512)

def prepare_dataset(example):
    audio = example["audio"]
    example = processor(
        text=example["normalized_text"],
        audio_target=audio["array"],
        sampling_rate=audio["sampling_rate"],
        return_attention_mask=False,
    )
    example["labels"] = example["labels"][0]  # Strip batch dim
    example["speaker_embeddings"] = DUMMY_SPEAKER_EMBEDDING
    return example

dataset = ds.map(prepare_dataset, remove_columns=ds.column_names)


dataset = dataset.train_test_split(test_size=0.1)


from dataclasses import dataclass
from typing import Any, Dict, List, Union
import torch

@dataclass
class TTSDataCollatorWithPadding:
    processor: Any

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        input_ids = [{"input_ids": feature["input_ids"]} for feature in features]
        label_features = [{"input_values": feature["labels"]} for feature in features]
        speaker_features = [feature["speaker_embeddings"] for feature in features]

        batch = processor.pad(input_ids=input_ids, labels=label_features, return_tensors="pt")

        batch["labels"] = batch["labels"].masked_fill(
            batch.decoder_attention_mask.unsqueeze(-1).ne(1), -100
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


from transformers import SpeechT5ForTextToSpeech

model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts")
model.config.use_cache = False


from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer

training_args = Seq2SeqTrainingArguments(
    output_dir="speechT5_finetuned_ljspeech",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=8,
    learning_rate=1e-5,
    warmup_steps=500,
    max_steps=4000,
    gradient_checkpointing=True,
    fp16=False,
    eval_strategy="steps",
    per_device_eval_batch_size=2,
    save_steps=1000,
    eval_steps=1000,
    logging_steps=25,
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


trainer.train()