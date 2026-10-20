"""Review sentiment classification + complaint topic discovery on real customer reviews -> BI-ready tables.

Data: Women's E-Commerce Clothing Reviews (Kaggle: nicapotato/womens-ecommerce-clothing-reviews). 23,486 real,
anonymised reviews with a 1-5 star rating, product department/class and reviewer age. Stored in data/.

Label: rating 4-5 = positive, 1-2 = negative. 3-star reviews are mixed, so they are left out of training and
scored afterwards to show how the model reads them.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "womens_reviews.csv.gz"
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)
N_TOPICS = 6
# names given by reading the top terms and sample reviews of each topic (fixed random_state keeps the order stable)
TOPIC_NAMES = {0: "Material quality vs price", 1: "Runs small / large", 2: "Length and proportions",
               3: "Doesn't look like the photo", 4: "Unflattering fabric or cut", 5: "Size inconsistency"}


def load():
    df = pd.read_csv(DATA, index_col=0)
    df.columns = ["clothing_id", "age", "title", "review", "rating", "recommended", "helpful_votes",
                  "division", "department", "class_name"]
    df = df.dropna(subset=["review"]).copy()
    df["text"] = df.title.fillna("") + ". " + df.review
    df["label"] = np.select([df.rating >= 4, df.rating <= 2], ["positive", "negative"], "mixed")
    df["age_band"] = pd.cut(df.age, [0, 29, 39, 49, 59, 120], labels=["<30", "30s", "40s", "50s", "60+"])
    return df


def main():
    df = load()
    print(f"{len(df):,} reviews with text. Ratings: {df.rating.value_counts().sort_index().to_dict()}\n")
    lab = df[df.label != "mixed"]

    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=3, sublinear_tf=True)
    X = vec.fit_transform(lab.text)
    y = (lab.label == "negative").astype(int)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=0, stratify=y)
    clf = LogisticRegression(max_iter=2000, C=4, class_weight="balanced").fit(Xtr, ytr)
    p = clf.predict_proba(Xte)[:, 1]
    print(classification_report(yte, (p >= 0.5).astype(int), target_names=["positive", "negative"], digits=3))
    print(f"ROC-AUC {roc_auc_score(yte, p):.3f}\n")

    # most telling words, straight from the model coefficients
    terms = np.array(vec.get_feature_names_out())
    order = np.argsort(clf.coef_[0])
    words = pd.DataFrame({"most_negative": terms[order[::-1][:15]], "most_positive": terms[order[:15]]})
    words.to_csv(OUT / "telling_words.csv", index=False)
    print("Words that push hardest towards each class:\n" + words.head(10).to_string(index=False) + "\n")

    df["p_negative"] = clf.predict_proba(vec.transform(df.text))[:, 1].round(3)
    df["pred_sentiment"] = np.where(df.p_negative >= 0.5, "negative", "positive")
    mixed = df[df.label == "mixed"]
    print(f"3-star reviews read as negative by the model: {(mixed.pred_sentiment == 'negative').mean():.0%}\n")

    # topics inside complaints (1-3 stars), with clothing words that carry no complaint removed
    neg = df[df.rating <= 3]
    filler = {"dress", "top", "shirt", "sweater", "just", "really", "like", "love", "wanted", "looked", "look",
              "bought", "ordered", "would", "im", "ve", "did", "didn", "wasn", "don", "doesn", "isn", "fit", "fits",
              "retailer", "store", "online", "item", "one", "got", "beautiful", "cute", "pretty", "nice", "great"}
    vec_n = TfidfVectorizer(stop_words=list(ENGLISH_STOP_WORDS | filler), min_df=10, max_df=0.5)
    Xn = vec_n.fit_transform(neg.review)
    nmf = NMF(n_components=N_TOPICS, random_state=0, init="nndsvda", max_iter=500).fit(Xn)
    t = np.array(vec_n.get_feature_names_out())
    topics = pd.DataFrame({"topic": range(N_TOPICS),
                           "top_terms": [", ".join(t[c.argsort()[::-1][:8]]) for c in nmf.components_]})
    df.loc[neg.index, "topic"] = pd.Series(nmf.transform(Xn).argmax(axis=1), index=neg.index).map(TOPIC_NAMES)
    topics.insert(1, "name", topics.topic.map(TOPIC_NAMES))
    topics["reviews"] = topics.name.map(df.topic.value_counts())
    topics.to_csv(OUT / "complaint_topics.csv", index=False)
    print("Complaint topics (1-3 star reviews):\n" + topics.to_string(index=False) + "\n")

    summary = (df.groupby(["department", "class_name"])
               .agg(reviews=("rating", "size"), avg_rating=("rating", "mean"),
                    pct_negative=("pred_sentiment", lambda s: (s == "negative").mean() * 100),
                    pct_recommended=("recommended", lambda s: s.mean() * 100),
                    top_complaint_topic=("topic", lambda s: s.mode().iloc[0] if s.notna().any() else None))
               .round(1).reset_index())
    summary.to_csv(OUT / "class_summary.csv", index=False)
    by_age = df.groupby("age_band", observed=True).agg(reviews=("rating", "size"), avg_rating=("rating", "mean"),
                                                       pct_negative=("pred_sentiment", lambda s: (s == "negative").mean() * 100)).round(1)
    by_age.to_csv(OUT / "age_band_summary.csv")
    df.drop(columns=["text"]).to_csv(OUT / "reviews_scored.csv.gz", index=False)
    print("Product classes with the most negative reviews (min 100 reviews):\n"
          + summary[summary.reviews >= 100].nlargest(5, "pct_negative").to_string(index=False))


if __name__ == "__main__":
    main()
