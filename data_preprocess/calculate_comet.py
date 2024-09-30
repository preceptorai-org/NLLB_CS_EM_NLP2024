import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM, pipeline
import pandas as pd
from comet import download_model, load_from_checkpoint
import tqdm
tqdm.tqdm.pandas()
# Set torch matmul precision to medium
torch.set_float32_matmul_precision('medium')

model = load_from_checkpoint("/home/parin/.cache/huggingface/hub/models--Unbabel--wmt23-cometkiwi-da-xl/snapshots/247f80c250e569fb011dbd906af24f8afe3e8d58/checkpoints/model.ckpt")

# df = pd.read_csv("data/machine_annotated_chunkify.csv")
df = pd.read_csv("data/augmented.csv")
df

# Remove rows with no text
df = df[df["text"].notna()]
df = df[df["text_target"] != ""]

# Heuristics: text length diff
df["text_len"] = df["text"].str.len()
df["text_target_len"] = df["text_target"].str.len()
df["text_len_diff"] = df["text_len"] - df["text_target_len"]
df["text_len_diff"] = df["text_len_diff"].abs()
df

data = df[["text", "text_target"]].rename(columns={"text": "src", "text_target": "mt"}).to_dict("records")
data

model_output = model.predict(data, batch_size=4, gpus=1)
model_output

data = pd.DataFrame(data)
data

data["scores"] = model_output["scores"]
data

# data.to_csv("filtset1.csv", index=False)
data.to_csv("augmented_filtset1.csv", index=False)