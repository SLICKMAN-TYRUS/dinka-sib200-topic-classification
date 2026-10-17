"""Streamlit app: topic classifier for Southwestern Dinka (SIB-200).

The model (character n-gram TF-IDF + logistic regression, the primary baseline of the project) is trained at
start-up from the public SIB-200 training file, so no model file has to be stored in the repository.
"""
import unicodedata

import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

TRAIN_URL = "https://raw.githubusercontent.com/dadelani/sib-200/main/data/annotated/dik_Latn/train.tsv"

st.set_page_config(page_title="Dinka topic classifier", page_icon="🌍")


@st.cache_resource(show_spinner="Training the model on the SIB-200 training set (a few seconds)...")
def load_model():
    df = pd.read_csv(TRAIN_URL, sep="\t", quoting=3)
    df["text"] = df["text"].astype(str).map(lambda t: unicodedata.normalize("NFC", t).strip())
    model = make_pipeline(
        TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), sublinear_tf=True, lowercase=True),
        LogisticRegression(max_iter=3000, C=10, class_weight="balanced"),
    )
    model.fit(df["text"], df["category"])
    return model, len(df)


model, n_train = load_model()

EXAMPLES = {
    "(type your own)": "",
    "Geography example": "Pan cɔl Turkey aci piu war adekdiet guɔ̈l piny ɣon kadiakic, Aegean tɔ ciem, Black sea tɔ ye nhom tueŋ ku Meditreanean tɔ ye cok piiny.",
    "Health example": "Dek mɔ̈u amäth. Mɔ̈u akɔ̈c thöŋ gup tɛ̈ loi thok, ku ŋïcŋïc adan ye dek ake riil tör",
}

st.title("Dinka topic classifier")
st.caption("Classifies one Southwestern Dinka sentence into one of 7 topics: science/technology, travel, politics, "
           "sports, health, entertainment, geography.")

choice = st.selectbox("Try an example", list(EXAMPLES))
text = st.text_area("Dinka sentence", value=EXAMPLES[choice], height=130)

if st.button("Classify", type="primary"):
    if not text.strip():
        st.warning("Please enter a sentence.")
    else:
        t = unicodedata.normalize("NFC", text).strip()
        probs = pd.Series(model.predict_proba([t])[0], index=model.classes_).sort_values(ascending=False)
        st.subheader(f"Predicted topic: {probs.index[0]}")
        st.caption(f"Probability {probs.iloc[0]:.0%}. A low top probability means the model is unsure.")
        st.bar_chart(probs)

with st.sidebar:
    st.header("About")
    st.write(f"Model: character n-gram TF-IDF (2 to 5) + logistic regression, trained on {n_train} SIB-200 sentences.")
    st.write("Test macro-F1 on the 204 held-out sentences: about 0.72. A fine-tuned NLLB encoder scores the same within noise.")
    st.write("**Limitations:** trained on Southwestern (Rek) Dinka, so other dialects such as Bor may be classified less reliably. "
             "Small training set; news/travel-style single sentences only; it cannot tell whether the input is Dinka.")
