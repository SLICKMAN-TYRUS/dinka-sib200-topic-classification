"""Look up how a few n-grams are used in the SIB-200 Dinka training sentences (with their English sources)."""
import unicodedata, pandas as pd

B = "https://raw.githubusercontent.com/dadelani/sib-200/main/data/annotated/"
def rd(lang, split):
    d = pd.read_csv(f"{B}{lang}/{split}.tsv", sep="\t", quoting=3)
    d["text"] = d["text"].astype(str).map(lambda t: unicodedata.normalize("NFC", t).strip().lower()); return d

tr = rd("dik_Latn", "train")
eng = pd.concat([rd("eng_Latn", s) for s in ["train", "dev", "test"]]); ENG = dict(zip(eng.index_id, eng.text))

# Dinka letters written as escapes so no special keyboard is needed:  \u00f6 = ö   \u014b = ŋ   \u00eb = ë
for pat in ["l\u00f6\u014b", "tuan", "thu\u00eb\u00ebc"]:
    sub = tr[tr.text.str.contains(pat, regex=False)]
    print(f"\n=== '{pat}': {len(sub)} training sentences | topics: {sub.category.value_counts().to_dict()}")
    for _, r in sub.head(5).iterrows():
        print(f"  [{r.category}] {r.text[:110]}\n      EN: {ENG[r.index_id][:110]}")
