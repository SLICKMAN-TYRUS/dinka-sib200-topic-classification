"""Reference point: the same TF-IDF baseline trained and tested on the English sentences of SIB-200, compared with Dinka."""
import unicodedata, numpy as np, pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import f1_score

B = "https://raw.githubusercontent.com/dadelani/sib-200/main/data/annotated/"
def rd(lang, split):
    d = pd.read_csv(f"{B}{lang}/{split}.tsv", sep="\t", quoting=3)
    d["text"] = d["text"].astype(str).map(lambda t: unicodedata.normalize("NFC", t).strip()); return d

D = {l: {s: rd(l, s) for s in ("train", "dev", "test")} for l in ("dik_Latn", "eng_Latn")}

def fit_predict(lang):
    m = make_pipeline(TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), sublinear_tf=True, lowercase=True),
                      LogisticRegression(max_iter=3000, C=10, class_weight="balanced"))
    m.fit(D[lang]["train"].text, D[lang]["train"].category)
    return m.predict(D[lang]["test"].text)

y = D["dik_Latn"]["test"].category.values
assert (D["eng_Latn"]["test"].index_id.values == D["dik_Latn"]["test"].index_id.values).all(), "test sentences not aligned"
p_dik, p_eng = fit_predict("dik_Latn"), fit_predict("eng_Latn")
print("test macro-F1  Dinka: %.3f | English: %.3f" % (f1_score(y, p_dik, average="macro"), f1_score(y, p_eng, average="macro")))

w_dik, w_eng = p_dik != y, p_eng != y
print("errors  Dinka:", w_dik.sum(), "| English:", w_eng.sum(), "| wrong in both:", (w_dik & w_eng).sum())

rng = np.random.default_rng(0); diffs = []
for _ in range(2000):
    ix = rng.integers(0, len(y), len(y))
    diffs.append(f1_score(y[ix], p_eng[ix], average="macro") - f1_score(y[ix], p_dik[ix], average="macro"))
print("English minus Dinka: %+.3f, 95%% CI [%+.3f, %+.3f]" % (np.mean(diffs), *np.percentile(diffs, [2.5, 97.5])))
