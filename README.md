# English-Thai Code-switched Machine Translation in Medical Domain

[![arxiv](https://img.shields.io/badge/arXiv-2410.16221-red)](https://doi.org/10.48550/arXiv.2410.16221)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](https://github.com/preceptorai-org/NLLB_CS_EM_NLP2024/blob/release/LICENSE)

This repository contains the code, data, and evaluation scripts for our paper "On Creating an English-Thai Code-switched Machine Translation in Medical Domain".


## Abstract

Machine translation (MT) in the medical domain plays a pivotal role in enhancing healthcare quality and disseminating medical knowledge. Despite advancements in English-Thai MT technology, common MT approaches often underperform in the medical field due to their inability to precisely translate medical terminologies. Our research prioritizes not merely improving translation accuracy but also maintaining medical terminology in English within the translated text through code-switched (CS) translation. We developed a method to produce CS medical translation data, fine-tuned a CS translation model with this data, and evaluated its performance against strong baselines, such as Google Neural Machine Translation (NMT) and GPT-3.5/GPT-4. Our model demonstrated competitive performance in automatic metrics and was highly favored in human preference evaluations. Our evaluation result also shows that medical professionals significantly prefer CS translations that maintain critical English terms accurately, even if it slightly compromises fluency. Our code and test set are publicly available https://github.com/preceptorai-org/NLLB_CS_EM_NLP2024.

## Key Features

* **Code-Switched Translation:** Our system specifically focuses on preserving English medical terminology within the translated Thai text, catering to the preferences of medical professionals.
* **Novel Data Generation:** We introduce a unique masking-based approach to generate pseudo-CS medical translation data, addressing the lack of readily available resources.
* **Rigorous Evaluation:** We conduct comprehensive evaluations using both automatic metrics (BLEU, chrF, METEOR, CER, WER, COMET, CS boundary F1) and human evaluations by medical professionals.
* **Open-Source Resources:** We publicly release our code, test set, and evaluation scripts to facilitate further research in this critical domain.


## Repository Structure

* **inference/ :** Contains scripts for:
    * Generating translations using various LLM-based translators.
    * Implementing the Pseudo-translation Masking technique (Section 3.2 of the paper).
* **data_preprocess/:**
    * **clean_human.py:** Cleans and preprocesses the human-annotated dataset (Section 3.3).
    * **augment.py:** Augments the training data using back-translation (Section 3.4).
    * **calculate_comet.py, filter_comet.ipynb:** Filter generated translations based on COMET score (Section 3.4).
* **finetune/:**  Includes scripts for fine-tuning the NLLB model on the generated CS data (Section 4.1).
-  **data/:** Contains test set data used for evaluation
* **eval/:** Contains scripts for evaluating translation models using various metrics (Section 4.2.1).
* **glicko/:** Contains scripts for analyzing Glicko rating data from human evaluations (Section 4.2.2).

## Requirements

We performed our experiments on Google Colaboratory with additionals dependencies listed below:

- Python 3.8+
- Libraries:
    - `requests`
    - `httpx`
    - `tqdm`
    - `pandas`
    - `seaborn`
    - `matplotlib`
    - `unbabel-comet`
    - `nltk`
    - `jiwer`
    - `pythainlp`


## Released Weights

- [NLLB-1](https://drive.google.com/drive/folders/1-T843S5tqDn9vMLX6SjwzGflitDUYjHB?usp=drive_link)

- [NLLB-3](https://drive.google.com/drive/folders/10IcZ4K1w5RVQXz3JWEHhokuGvoLFifui?usp=sharing)


## Citation

If you use this code or data in your research, please cite our paper:

```bibtex
@inproceedings{pengpun-etal-2024-on,
      title={On Creating an English-Thai Code-switched Machine Translation in Medical Domain}, 
      author={Parinthapat Pengpun and Krittamate Tiankanon and Amrest Chinkamol and Jiramet Kinchagawat and Pitchaya Chairuengjitjaras and Pasit Supholkhan and Pubordee Aussavavirojekul and Chiraphat Boonnag and Kanyakorn Veerakanjana and Hirunkul Phimsiri and Boonthicha Sae-jia and Nattawach Sataudom and Piyalitt Ittichaiwong and Peerat Limkonchotiwat},
      year={2024},
      eprint={2410.16221},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2410.16221}, 
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details. 
