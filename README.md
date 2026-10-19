# Topic Classification for Southwestern Dinka (SIB-200)

[![Open the NLLB demo in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/SLICKMAN-TYRUS/dinka-sib200-topic-classification/blob/main/demo/Dinka_Demo_Colab.ipynb)

Individual summative project: an NLP application for an African language, covering data analysis, baselines, fine-tuning, experiments, error analysis and deployment (Streamlit live app, Colab demo for the fine-tuned model).

| Resource | Link |
|---|---|
| Live app (Streamlit Community Cloud, TF-IDF model) | <https://dinka-topic-classifier.streamlit.app/> |
| Fine-tuned NLLB demo (Colab notebook) | [demo/Dinka_Demo_Colab.ipynb](demo/Dinka_Demo_Colab.ipynb) (open it with the badge above) |
| Demo video | <https://drive.google.com/drive/folders/1xQkZr-DKaElYO-hQ_Dv2iLcDoYoMNL6p?usp=sharing> |
| Model weights (Hugging Face Hub) | https://huggingface.co/slickman-tyrus/dinka-sib200-classifier |
| Report | PDF submitted on the course platform (all numbers in the report come from `results/runs.csv` and `results/preds/`) |

> **Dialect note.** The data is Flores-200 `dik_Latn`, which is **Southwestern (Rek) Dinka**. The author's mother tongue is Bor (Southeastern) Dinka, so Bor input may be classified less reliably. The system is a Southwestern Dinka classifier.

## 1. Problem
Given one Dinka sentence, predict one of 7 topics: science/technology, travel, politics, sports, health, entertainment, geography. Dinka is a low-resource language with very little labelled NLP data. Topic classification is a simple natural-language-understanding task that shows how well pretrained models handle it.

## 2. Data
- **Source:** SIB-200 (Adelani et al., 2024), built on Flores-200. Files: `github.com/dadelani/sib-200`, folder `data/annotated/dik_Latn/`.
- **Size:** 701 train / 99 dev / 204 test sentences (official split, not changed). Class counts in train: science/technology 176, travel 138, politics 102, sports 85, health 77, entertainment 65, geography 58.
- **Nature of the data:** professional human translations of English sentences. The topic labels were annotated on the English text and projected to Dinka. **No LLM-generated data is used.**
- **Preprocessing:** Unicode NFC normalisation and whitespace stripping only. Dinka letters and diacritics are kept.
- **Limitations:** translated (not naturally occurring) text, small size, projected labels with some label noise, one dialect, English loanwords inside Dinka sentences.

## 3. Methods
| Model | Description |
|---|---|
| Majority class | Sanity floor |
| TF-IDF + logistic regression | Character n-grams (`char_wb`), 9 configurations tuned on dev, plus a word-level variant |
| BiLSTM (from scratch) | Word embeddings (128) trained from random initialisation, BiLSTM (128), max-pool, dropout 0.5, class-weighted loss |
| XLM-R base (fine-tuned) | Pretrained encoder, mean pooling, linear head |
| NLLB-200-distilled-600M encoder (fine-tuned) | Encoder only, mean pooling, linear head, token embeddings frozen, source-language tag `dik_Latn` |

Training details for the transformers: AdamW (encoder lr 2e-5 for XLM-R, 3e-5 for NLLB; head lr 1e-3; weight decay 0.01), 10% linear warm-up then linear decay, batch size 16, at most 20 epochs with early stopping (patience 6) on dev macro-F1, fp16 autocast, gradient clipping 1.0, class-weighted cross-entropy, max length 128. Three seeds (0, 1, 2) per neural model. Model selection uses **dev** only; the headline numbers are **test**.

## 4. Results (test set, 204 sentences)
| Model | Test macro-F1 | Test accuracy |
|---|---|---|
| Majority class | 0.057 | 0.250 |
| BiLSTM from scratch (3 seeds) | 0.557 ± 0.038 | 0.596 |
| XLM-R base (3 seeds) | 0.593 ± 0.024 | 0.619 |
| XLM-R base, frozen embeddings (3 seeds) | 0.600 ± 0.037 | 0.628 |
| TF-IDF word 1-2 | 0.680 | 0.726 |
| TF-IDF char 2-5, C=10 | 0.724 | 0.760 |
| **NLLB-600M encoder (3 seeds)** | **0.720 ± 0.014** | 0.732 |
| NLLB-600M encoder, lr 5e-5 (3 seeds) | 0.731 ± 0.001 | 0.744 |

Full table: `results/results_table.csv`. Findings:
1. NLLB and TF-IDF are statistically tied (paired bootstrap difference -0.001, 95% CI [-0.076, +0.073]).
2. NLLB is significantly better than XLM-R (+0.157, CI [+0.074, +0.236]), and TF-IDF is significantly better than the BiLSTM (+0.124, CI [+0.057, +0.191]).
3. Freezing embeddings does not explain the XLM-R vs NLLB gap.
4. Dev scores are optimistic (5-fold CV of TF-IDF on train+dev: 0.678 ± 0.040 vs dev 0.764). Report test numbers.
5. Zero-shot transfer to Dinka with the NLLB encoder: trained on Dholuo 0.674, English 0.635, Nuer 0.634, versus 0.720 trained on Dinka. Pooling Dinka + Nuer gives no gain (0.714).

## 5. Error analysis
Confusion matrix: `results/figures/confusion_matrix_nllb600m_seed0.png`. 53 of 204 test sentences are misclassified by the selected NLLB model; 15 are wrong in all 15 main models. Entertainment has the lowest recall (0.47). I annotated six of the 53 errors against their English sources (`results/errors_annotated.csv`, columns `error_type` and `my_notes`): 2 translation problems (IDs 8, 1947), 2 Rek-dialect differences (IDs 922, 1221) and 2 model errors (IDs 400, 1360). The other 47 rows are not annotated. See Section 7 of the report.

### Extra analysis scripts (src/)
- `explain_tfidf.py`: inspects the TF-IDF model (n-gram weights, one prediction explained, live variants).
- `ngram_lookup.py`: shows how key n-grams are used in the training sentences, with English sources.
- `english_reference.py`: same TF-IDF baseline on the English sentences (0.729 vs 0.724 on Dinka).

## 6. Deployment and how to run it
**Live app (`streamlit_app/`).** A Streamlit app that serves the character n-gram TF-IDF + logistic regression model (the primary baseline). It downloads the SIB-200 Dinka training file at start-up, fits the model in about two seconds and shows the predicted topic with probabilities. It is deployed on Streamlit Community Cloud with main file `streamlit_app/streamlit_app.py`. Free apps hibernate when idle, so the first request may need a wake-up click.

*Why TF-IDF is the live model:* it is statistically tied with the fine-tuned NLLB encoder (paired bootstrap difference -0.001, 95% CI [-0.076, +0.073]) and is cheap to host. The NLLB encoder needs about 0.8 GB of weights plus several GB of RAM, which does not fit the free hosting tiers.

**Fine-tuned NLLB demo (`demo/Dinka_Demo_Colab.ipynb`).** Open it in Google Colab and run all cells. It downloads the fine-tuned weights from the Hugging Face model repo (link above) and serves them with Gradio through a temporary public link.

**Reproduce the experiments.** Open `notebooks/Dinka_SIB200_Colab.ipynb` in Google Colab, choose a T4 GPU and run the cells in order. Results are appended to `runs.csv` on Google Drive after every run, and finished runs are skipped when you re-run. Order: data loading, baselines, BiLSTM, transformer fine-tuning, results table, error analysis, export, transfer experiments.
```bash
pip install -r requirements-notebook.txt      # only needed outside Colab
```
**Run the live app locally (inference):**
```bash
pip install -r streamlit_app/requirements.txt
streamlit run streamlit_app/streamlit_app.py
```
**Check the reported numbers without retraining:** `results/runs.csv` has the dev and test scores of every run and `results/preds/` has the test predictions, so every table in the report can be rebuilt from these files.

Environment used: Python 3.13, torch 2.11, transformers 5.18.0, scikit-learn 1.6.1, Google Colab T4 GPU.

## Repository layout
```
notebooks/Dinka_SIB200_Colab.ipynb   experiments (with outputs)
src/model_def.py                     model class + inference helper shared by notebook and demo
streamlit_app/                       live app (TF-IDF)
demo/Dinka_Demo_Colab.ipynb          NLLB demo
results/                             results_table.csv, runs.csv (every run), preds/ (test predictions of every run), figures/, errors_annotated.csv
```

## 7. Existing resources vs my own work
**Existing:** SIB-200 and Flores-200 data; pretrained XLM-R base and NLLB-200-distilled-600M weights; Hugging Face Transformers, PyTorch, scikit-learn, Streamlit, Gradio.
**My own work:** data analysis, baseline and BiLSTM implementation, fine-tuning pipeline and experiment design, ablations, transfer experiments, statistical comparisons, error analysis, deployment. AI assistance (Claude, Anthropic) was used for code scaffolding and for drafting; I ran the experiments, checked the outputs and am responsible for the content.

## References
Adelani et al. (2024) SIB-200. EACL. | NLLB Team et al. (2024) Scaling neural machine translation to 200 languages. Nature. | Conneau et al. (2020) XLM-R. ACL. Full list in the report.