import torch
torch.set_float32_matmul_precision('medium')

# Comet metrics
from comet import download_model, load_from_checkpoint

import os
import re
import ast
from copy import copy

import pandas as pd
from tqdm import tqdm

# Pythainlp
from pythainlp.tokenize import word_tokenize
from pythainlp.util import normalize

# NLTK metrics
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.chrf_score import sentence_chrf
from nltk.translate.meteor_score import single_meteor_score

# JIWER
from jiwer import wer, cer

nltk.download("wordnet", quiet=True)
nltk.download("punkt", quiet=True)
nltk.download("omw", quiet=True)

### Config
TARGET_COL = "text_target_clean_final"

MODELS_TO_EVAL = [
    "google_translate_no_mask",
    "google_translate_mask",
    "gpt-4-1106-preview_cs_no_mask",
    "gpt-4-1106-preview_mn_no_mask",
    "gpt-4-1106-preview_cs_mask",
    "gpt-4-1106-preview_mn_mask",
    "gpt-3.5-turbo-1106_cs_no_mask",
    "gpt-3.5-turbo-1106_mn_no_mask",
    "gpt-3.5-turbo-1106_cs_mask",
    "gpt-3.5-turbo-1106_mn_mask",
    "gemini_cs_no_mask",
    "gemini_mn_no_mask",
    "gemini_cs_mask",
    "gemini_mn_mask",
    "nllb_normal_mask",
    "nllb_normal_no_mask",
    "nllb_mach_mask",
    "nllb_mach_no_mask",
    "nllb_augmented_mask",
    "nllb_augmented_no_mask",
    "nllb_mach_augmented_mask",
    "nllb_mach_augmented_no_mask",
    "nllb_mach_filt_mask",
    "nllb_mach_filt_no_mask",
    "nllb_augmented_filt_mask",
    "nllb_augmented_filt_no_mask",
    "nllb_mach_augmented_filt_mask",
    "nllb_mach_augmented_filt_no_mask",
    "llama2_7b_cs_mask",
    "llama2_7b_cs_no_mask",
    "llama2_7b_mn_mask",
    "llama2_7b_mn_no_mask",
    "llama2_13b_cs_mask",
    "llama2_13b_cs_no_mask",
    "llama2_13b_mn_mask",
    "llama2_13b_mn_no_mask",
    "seallm_7b_cs_mask",
    "seallm_7b_cs_no_mask",
    "seallm_7b_mn_mask",
    "seallm_7b_mn_no_mask",
    "typhoon_7b_cs_mask",
    "typhoon_7b_cs_no_mask",
    "typhoon_7b_mn_mask",
    "typhoon_7b_mn_no_mask",
    "openthaigpt_13b_cs_mask",
    "openthaigpt_13b_cs_no_mask",
    "openthaigpt_13b_mn_mask",
    "openthaigpt_13b_mn_no_mask",
    "openthaigpt_7b_cs_mask",
    "openthaigpt_7b_cs_no_mask",
    "openthaigpt_7b_mn_mask",
    "openthaigpt_7b_mn_no_mask",
]

###

def code_switch_boundary_score(hyp, masked_dict):
    """
    Calculate code-switch boundary score
    """

    hyp = hyp.lower()

    cs_words = set(masked_dict.keys())

    tp = 0
    fp = 0
    fn = 0

    # Check whether cs_word is inside hyp
    for cs_word in cs_words:
        cs_word = str(cs_word)
        if cs_word in hyp:
            tp += 1
        else:
            fn += 1

    # Check for incorrectly code-switched words
    for word in word_tokenize(
        hyp, keep_whitespace=False, join_broken_num=True, engine="newmm"
    ):
        # Check if the word is an English word
        if re.match(r"[a-zA-Z]+", word):
            if word not in cs_words:
                fp += 1

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = (
        (2 * precision * recall) / (precision + recall)
        if (precision + recall) > 0
        else 0
    )

    return f1, precision, recall


if __name__ == "__main__":
    # Check if inference_scores.parquet exists
    if os.path.exists("data/inference_final_temp.parquet"):
        print("Loading existing data...")
        # Load existing data
        data = pd.read_parquet("data/inference_final_temp.parquet")
    else:
        # Load data from CSV as it's the first run
        data = pd.read_csv("data/inference_final.csv")

    models_to_evaluate = MODELS_TO_EVAL
    print(f"Summary:")
    print(f"Data shape: {data.shape}")
    print(f"Models to evaluate: {models_to_evaluate}")
    print(f"Target column: {TARGET_COL}")

    # Init comet
    comet_model_path =  download_model("Unbabel/XCOMET-XL")
    comet = load_from_checkpoint(comet_model_path)

    # Preprocess
    for col in models_to_evaluate + [TARGET_COL]:
        data[col] = data[col].astype(str)
        data[col] = data[col].str.strip()
        data[col] = data[col].apply(lambda x: normalize(x))

        # Lowercase
        data[col] = data[col].str.lower()

    for model in tqdm(models_to_evaluate, desc="Evaluating Models", dynamic_ncols=True):
        model_bleu2_col = model + "_bleu2_lower"
        model_bleu3_col = model + "_bleu3_lower"
        model_bleu4_col = model + "_bleu4_lower"
        model_chrf_col = model + "_chrf_lower"
        model_meteor_col = model + "_meteor_lower"
        model_wer_col = model + "_wer_lower"
        model_cer_col = model + "_cer_lower"
        model_cs_f1_col = model + "_cs_f1_lower"
        model_cs_precision_col = model + "_cs_precision_lower"
        model_cs_recall_col = model + "_cs_recall_lower"
        model_comet_col = model + "_comet"
        model_comet_error_spans_col = model + "_comet_error_spans"

        if not (
            model_bleu2_col in data.columns
            and model_bleu3_col in data.columns
            and model_bleu4_col in data.columns
            and model_chrf_col in data.columns
        ):
            # Compute BLEU
            bleu2_scores = []
            bleu_3_scores = []
            bleu_4_scores = []
            for i in tqdm(
                range(len(data)), desc="Computing BLEU", dynamic_ncols=True, leave=False
            ):
                ref = word_tokenize(
                    str(data[TARGET_COL].iloc[i]),
                    keep_whitespace=False,
                    join_broken_num=True,
                    engine="newmm",
                )
                hyp = word_tokenize(
                    str(data[model].iloc[i]),
                    keep_whitespace=False,
                    join_broken_num=True,
                    engine="newmm",
                )
                bleu2_scores.append(
                    sentence_bleu(
                        [ref],
                        hyp,
                        weights=(0.5, 0.5),
                        smoothing_function=SmoothingFunction().method1,
                    )
                )
                bleu_3_scores.append(
                    sentence_bleu(
                        [ref],
                        hyp,
                        weights=(1 / 3, 1 / 3, 1 / 3),
                        smoothing_function=SmoothingFunction().method1,
                    )
                )
                bleu_4_scores.append(
                    sentence_bleu(
                        [ref],
                        hyp,
                        weights=(0.25, 0.25, 0.25, 0.25),
                        smoothing_function=SmoothingFunction().method1,
                    )
                )

            data[model_bleu2_col] = bleu2_scores
            data[model_bleu3_col] = bleu_3_scores
            data[model_bleu4_col] = bleu_4_scores

        if not model_chrf_col in data.columns:
            # Compute CHRF
            chrf_scores = []
            for i in tqdm(
                range(len(data)), desc="Computing CHRF", dynamic_ncols=True, leave=False
            ):
                ref = word_tokenize(
                    str(data[TARGET_COL].iloc[i]),
                    keep_whitespace=False,
                    join_broken_num=True,
                    engine="newmm",
                )
                hyp = word_tokenize(
                    str(data[model].iloc[i]),
                    keep_whitespace=False,
                    join_broken_num=True,
                    engine="newmm",
                )
                chrf_scores.append(sentence_chrf(ref, hyp))

            data[model_chrf_col] = chrf_scores

        if not model_meteor_col in data.columns:
            # Compute METEOR
            meteor_scores = []
            for i in tqdm(
                range(len(data)), desc="Computing METEOR", dynamic_ncols=True, leave=False
            ):
                ref = word_tokenize(
                    str(data[TARGET_COL].iloc[i]),
                    keep_whitespace=False,
                    join_broken_num=True,
                    engine="newmm",
                )
                hyp = word_tokenize(
                    str(data[model].iloc[i]),
                    keep_whitespace=False,
                    join_broken_num=True,
                    engine="newmm",
                )
                meteor_scores.append(single_meteor_score(ref, hyp))
            
            data[model_meteor_col] = meteor_scores
        
        if not (model_wer_col in data.columns):
            # Compute WER
            wer_scores = []
            for i in tqdm(
                range(len(data)), desc="Computing WER", dynamic_ncols=True, leave=False
            ):
                ref = " ".join(word_tokenize(str(data[TARGET_COL].iloc[i]), keep_whitespace=False, join_broken_num=True, engine="newmm"))
                hyp = " ".join(word_tokenize(str(data[model].iloc[i]), keep_whitespace=False, join_broken_num=True, engine="newmm"))
                wer_scores.append(wer(ref, hyp))
            
            data[model_wer_col] = wer_scores
        
        if not (model_cer_col in data.columns):
            # Compute CER
            cer_scores = []
            for i in tqdm(
                range(len(data)), desc="Computing CER", dynamic_ncols=True, leave=False
            ):
                ref = " ".join(word_tokenize(str(data[TARGET_COL].iloc[i]), keep_whitespace=False, join_broken_num=True, engine="newmm"))
                hyp = " ".join(word_tokenize(str(data[model].iloc[i]), keep_whitespace=False, join_broken_num=True, engine="newmm"))
                cer_scores.append(cer(ref, hyp))
            
            data[model_cer_col] = cer_scores

        if not (model_cs_f1_col in data.columns and model_cs_precision_col in data.columns and model_cs_recall_col in data.columns):
            # Compute Code-Switch Boundary Score
            cs_scores_f1, cs_scores_precision, cs_scores_recall = [], [], []
            for i in tqdm(
                range(len(data)),
                desc="Computing Code-Switch Boundary Score",
                dynamic_ncols=True,
                leave=False,
            ):
                hyp = str(data[model].iloc[i])
                f1, precision, recall = code_switch_boundary_score(
                    hyp, ast.literal_eval(data["mask_dict"].iloc[i])
                )
                cs_scores_f1.append(f1)
                cs_scores_precision.append(precision)
                cs_scores_recall.append(recall)
            
            data[model_cs_f1_col] = cs_scores_f1
            data[model_cs_precision_col] = cs_scores_precision
            data[model_cs_recall_col] = cs_scores_recall
        
        if not (model_comet_col in data.columns and model_comet_error_spans_col in data.columns):
            # COMET
            ## Format data into COMET format
            comet_data = []
            for i in tqdm(
                range(len(data)),
                desc="Formatting COMET Data",
                dynamic_ncols=True,
                leave=False,
            ):
                comet_data.append(
                    {
                        "src": str(data["text"].iloc[i]),
                        "mt": str(data[model].iloc[i]),
                        "ref": str(data[TARGET_COL].iloc[i]),
                    }
                )

            ## Infer COMET scores
            output = comet.predict(comet_data, progress_bar=True, accelerator="cuda", batch_size=8)
            comet_scores = output.scores
            comet_error_spans = output.metadata.error_spans
        
            data[model_comet_col] = comet_scores
            data[model_comet_error_spans_col] = comet_error_spans

        # Defragment dataframe
        data = data.copy()

        # Save dataframe to temporary parquet file
        data.to_parquet("data/inference_final_temp.parquet", index=False)

    # Save dataframe to CSV
    data.to_csv("data/inference_final.csv", index=False)
