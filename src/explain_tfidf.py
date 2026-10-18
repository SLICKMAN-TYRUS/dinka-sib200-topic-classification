import unicodedata, numpy as np, pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import f1_score

TFIDF_URL = "https://raw.githubusercontent.com/dadelani/sib-200/main/data/annotated/dik_Latn/"
def rd(s):
    d = pd.read_csv(f"{TFIDF_URL}{s}.tsv", sep="\t", quoting=3)
    d["text"] = d["text"].astype(str).map(lambda t: unicodedata.normalize("NFC", t).strip()); return d
tr, dv, te = rd("train"), rd("dev"), rd("test")   # short names so nothing else in the notebook is overwritten

SENT = "Raan Mexico cɔl Arly Velasquez aci thök namar thiar ku dhiec kenɛ röör yic Super-G. Kene New Zealand cɔl Adam ɣöl aci thök namr dhoŋuan 9th kene röör yic Super-G."

def build(ngram=(2, 5), C=10, balanced=True):
    return make_pipeline(TfidfVectorizer(analyzer="char_wb", ngram_range=ngram, sublinear_tf=True, lowercase=True),
                         LogisticRegression(max_iter=3000, C=C, class_weight="balanced" if balanced else None))

tfidf_model = build().fit(tr.text, tr.category)
vec, lr = tfidf_model.named_steps["tfidfvectorizer"], tfidf_model.named_steps["logisticregression"]
feat_names = np.array(vec.get_feature_names_out())
print("features (distinct character n-grams):", len(vec.vocabulary_), "| classes:", list(lr.classes_))

# 1) what does one sentence look like to the model?
xv = vec.transform([SENT])
print("sentence -> sparse vector of shape", xv.shape, "with", xv.nnz, "non-zero n-grams")

# 2) which n-grams pushed the prediction? (contribution = tf-idf value x class weight)
probs = tfidf_model.predict_proba([SENT])[0]; kk = probs.argmax()
contrib = xv.toarray()[0] * lr.coef_[kk]
top_idx = contrib.argsort()[::-1][:8]
print("predicted:", lr.classes_[kk], f"({probs[kk]:.0%})")
print("n-grams that pushed it towards that class:", [(str(feat_names[i]), round(float(contrib[i]), 3)) for i in top_idx])

# 3) the strongest n-grams for every class
for j, c in enumerate(lr.classes_):
    print(f"{c:20s}", [str(n) for n in feat_names[lr.coef_[j].argsort()[::-1][:8]]])

# 4) live change: edit the settings below and re-run to see the effect on dev and test
for conf in [dict(ngram=(2, 5), C=10), dict(ngram=(1, 3), C=10), dict(ngram=(2, 5), C=10, balanced=False)]:
    mm = build(**conf).fit(tr.text, tr.category)
    print(conf, "dev macro-F1 %.3f | test macro-F1 %.3f" % (f1_score(dv.category, mm.predict(dv.text), average="macro"),
                                                        f1_score(te.category, mm.predict(te.text), average="macro")))
