"""Review sentiment classification + complaint topic discovery -> BI-ready summary tables."""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)

POS = ["great service and friendly staff", "fresh produce, really happy", "fast delivery, arrived on time",
       "good prices this week", "easy to find everything, clean store", "loved the bakery section"]
NEG = {"delivery": ["delivery was late again", "driver never showed up", "order arrived two hours late"],
       "freshness": ["bread was stale", "milk was past its date", "vegetables were mouldy"],
       "price": ["prices keep going up", "way too expensive now", "specials weren't honoured at checkout"],
       "staff": ["staff were rude", "nobody at the counter to help", "checkout queue was huge, one lane open"]}
STORES = ["Riccarton", "Rolleston", "Hornby", "Papanui"]


def make_data(n=3000, seed=9):
    rng = np.random.default_rng(seed)
    rows = []
    months = pd.period_range("2026-01", "2026-09", freq="M")
    for i in range(n):
        store = rng.choice(STORES)
        month = months[rng.integers(len(months))]
        p_neg = 0.25 + (0.2 if store == "Hornby" and month >= pd.Period("2026-06", "M") else 0)
        if rng.random() < p_neg:
            theme = rng.choice(list(NEG), p=[0.4, 0.2, 0.25, 0.15] if store == "Hornby" else None)
            text = rng.choice(NEG[theme]) + ". " + rng.choice(["not impressed", "disappointed", "won't be back"])
            label = "negative"
        else:
            theme, label = None, "positive"
            text = rng.choice(POS) + ". " + rng.choice(["will be back", "thanks team", "recommend"])
        rows.append({"review_id": i, "store": store, "month": str(month), "text": text,
                     "label": label, "true_theme": theme})
    return pd.DataFrame(rows)


def main():
    df = make_data()
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, stop_words="english")
    X = vec.fit_transform(df.text)
    Xtr, Xte, ytr, yte = train_test_split(X, df.label, test_size=0.25, random_state=0, stratify=df.label)
    clf = LogisticRegression(max_iter=1000).fit(Xtr, ytr)
    print(classification_report(yte, clf.predict(Xte), digits=3))
    df["pred_sentiment"] = clf.predict(X)

    neg = df[df.pred_sentiment == "negative"]
    # drop generic sign-off words so topics reflect the actual complaint
    filler = ["impressed", "disappointed", "won", "don", "weren", "really", "way", "again"]
    stop = list(TfidfVectorizer(stop_words="english").get_stop_words()) + filler
    vec_n = TfidfVectorizer(stop_words=stop, min_df=2)
    Xn = vec_n.fit_transform(neg.text)
    nmf = NMF(n_components=4, random_state=0, init="nndsvda").fit(Xn)
    terms = np.array(vec_n.get_feature_names_out())
    topics = pd.DataFrame({"topic": range(4),
                           "top_terms": [", ".join(terms[c.argsort()[::-1][:5]]) for c in nmf.components_]})
    topics.to_csv(OUT / "topics.csv", index=False)
    print(topics.to_string(index=False))
    df.loc[neg.index, "topic"] = nmf.transform(Xn).argmax(axis=1)

    summary = (df.groupby(["store", "month"])
               .agg(reviews=("review_id", "size"), pct_negative=("pred_sentiment", lambda s: (s == "negative").mean() * 100),
                    top_topic=("topic", lambda s: s.mode().iloc[0] if s.notna().any() else None))
               .round(1).reset_index())
    summary.to_csv(OUT / "store_month_summary.csv", index=False)
    df.to_csv(OUT / "review_scored.csv", index=False)
    print("\nHighest % negative store-months:\n", summary.nlargest(4, "pct_negative").to_string(index=False))


if __name__ == "__main__":
    main()
