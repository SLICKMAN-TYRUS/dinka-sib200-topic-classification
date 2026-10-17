"""Model definition + inference helper shared by the training notebook and the Gradio app."""
import json
import os
import unicodedata

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel, AutoTokenizer


class Clf(nn.Module):
    """Pretrained encoder -> mean pooling -> dropout -> linear layer.

    nllb=True keeps only the ENCODER of an NLLB (M2M100) seq2seq model.
    freeze_emb=True freezes the (huge) token-embedding matrix.
    """

    def __init__(self, name, n_labels, nllb=False, freeze_emb=False, drop=0.1, backbone=None):
        super().__init__()
        full = backbone if backbone is not None else AutoModel.from_pretrained(name)
        self.enc = full.get_encoder() if nllb else full
        if freeze_emb:
            for p in self.enc.get_input_embeddings().parameters():
                p.requires_grad = False
        cfg = self.enc.config
        hid = getattr(cfg, "hidden_size", None) or cfg.d_model
        self.drop = nn.Dropout(drop)
        self.head = nn.Linear(hid, n_labels)

    def forward(self, ids, mask):
        h = self.enc(input_ids=ids, attention_mask=mask).last_hidden_state
        m = mask.unsqueeze(-1).to(h.dtype)
        pooled = (h * m).sum(1) / m.sum(1).clamp(min=1)
        return self.head(self.drop(pooled))


def encode(tok, texts, lang=None, maxlen=128, nllb=False):
    """Tokenise a list of strings (NFC-normalised). For NLLB the source-language tag is set first."""
    if nllb and lang:
        tok.src_lang = lang
    texts = [unicodedata.normalize("NFC", str(t)).strip() for t in texts]
    return tok(texts, truncation=True, max_length=maxlen)["input_ids"]


def collate(batch_ids, pad_id, device):
    L = max(len(x) for x in batch_ids)
    ids = torch.full((len(batch_ids), L), pad_id, dtype=torch.long)
    mask = torch.zeros((len(batch_ids), L), dtype=torch.long)
    for i, x in enumerate(batch_ids):
        ids[i, : len(x)] = torch.tensor(x)
        mask[i, : len(x)] = 1
    return ids.to(device), mask.to(device)


def load_predictor(path, device=None):
    """Load meta.json + weights.pt from `path`; return predict(texts, lang=None) -> list of {label: prob}."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    meta = json.load(open(os.path.join(path, "meta.json")))
    nllb = meta["nllb"]
    tok = AutoTokenizer.from_pretrained(meta["tokenizer"], **({"src_lang": meta["src_lang"]} if nllb else {}))
    backbone = AutoModel.from_config(AutoConfig.from_pretrained(meta["name"]))  # architecture only, no download of weights
    model = Clf(meta["name"], len(meta["labels"]), nllb=nllb, backbone=backbone)
    sd = torch.load(os.path.join(path, "weights.pt"), map_location="cpu")
    model.load_state_dict({k: (v.float() if v.is_floating_point() else v) for k, v in sd.items()})
    model.to(device).eval()
    pad = tok.pad_token_id

    @torch.no_grad()
    def predict(texts, lang=None):
        out = []
        for i in range(0, len(texts), 32):
            chunk = texts[i : i + 32]
            ids, mask = collate(encode(tok, chunk, lang or meta["src_lang"], meta["maxlen"], nllb), pad, device)
            probs = torch.softmax(model(ids, mask).float(), -1).cpu().tolist()
            out += [dict(zip(meta["labels"], p)) for p in probs]
        return out

    return predict
