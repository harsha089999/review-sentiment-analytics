# Customer Review Sentiment Analytics

[![Run project](https://github.com/harsha089999/review-sentiment-analytics/actions/workflows/run.yml/badge.svg)](https://github.com/harsha089999/review-sentiment-analytics/actions/workflows/run.yml)

**Classify 22k real customer reviews, find what people complain about, and roll it up into BI tables by product class and age band.**

> **About the data:** this project runs on a real public dataset, **Women's E-Commerce Clothing Reviews**, included in `data/` so it runs straight after cloning. Sources and licence are listed below and in [`data/README.md`](data/README.md).

## The business question

How do customers feel about our products, what exactly are they unhappy about, and which product lines need attention?

## What this project does

Star ratings say how unhappy people are but not why. This project trains a sentiment classifier on real clothing reviews, uses topic modelling to group complaints into themes, and produces summary tables a merchandising team can filter in Power BI.

## The data

23,486 real, anonymised reviews from a women's clothing retailer (the company name is replaced with 'retailer'). 22,641 have review text. Ratings lean positive: 12,540 five-star against 821 one-star. Labels come from the rating: 4 to 5 stars is positive, 1 to 2 is negative. 3-star reviews are mixed, so they're left out of training and scored afterwards.

**Source:**
- [Kaggle: Women's E-Commerce Clothing Reviews (nicapotato)](https://www.kaggle.com/datasets/nicapotato/womens-ecommerce-clothing-reviews)

**Licence:** Published on Kaggle under CC0 (public domain).

## How it works

1. Combine title and review text, and label from the star rating.
2. TF-IDF on words and two-word phrases, then logistic regression with balanced class weights, because negatives are only 12% of labelled reviews.
3. Report precision and recall per class plus ROC-AUC on a held-out 25%, and pull the most telling words from the model's coefficients.
4. Topic modelling (NMF) on 1 to 3 star reviews, with generic clothing words removed so topics show the complaint, not the item. The six topics are then named by reading their top terms and sample reviews.
5. Summary tables by department and class, and by age band: average rating, % negative, % recommended and the most common complaint topic.

## Run it

```bash
git clone https://github.com/harsha089999/review-sentiment-analytics
cd review-sentiment-analytics
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python src/main.py
```

Outputs are written to `outputs/`. Python 3.10 or newer is recommended.

## Results from the included run

This is the actual console output when the script is run as shipped:

```text
22,641 reviews with text. Ratings: {1: 821, 2: 1549, 3: 2823, 4: 4908, 5: 12540}

              precision    recall  f1-score   support

    positive      0.979     0.962     0.971      4362
    negative      0.755     0.852     0.800       593

    accuracy                          0.949      4955
   macro avg      0.867     0.907     0.886      4955
weighted avg      0.953     0.949     0.950      4955

ROC-AUC 0.979

Words that push hardest towards each class:
most_negative most_positive
 disappointed          love
          not         great
        cheap        little
 unflattering     beautiful
         poor   comfortable
         huge       perfect
      to love           bit
disappointing          soft
       wanted       not too
    wanted to        lovely

3-star reviews read as negative by the model: 57%

Complaint topics (1-3 star reviews):
 topic                        name                                                         top_terms  reviews
     0   Material quality vs price material, quality, color, price, wear, white, disappointed, loved     1104
     1          Runs small / large                small, large, runs, medium, wear, usually, big, xs      729
     2      Length and proportions            short, work, length, long, petite, arms, sleeves, wide     1217
     3 Doesn't look like the photo             looks, model, picture, person, shape, boxy, way, does      644
     4  Unflattering fabric or cut     fabric, skirt, flattering, design, felt, colors, cheap, waist      873
     5          Size inconsistency            size, smaller, true, sizes, tried, order, tight, usual      626

Product classes with the most negative reviews (min 100 reviews):
department class_name  reviews  avg_rating  pct_negative  pct_recommended        top_complaint_topic
     Trend      Trend      118         3.8          30.5             74.6 Unflattering fabric or cut
      Tops    Blouses     2983         4.1          21.7             80.7     Length and proportions
      Tops   Sweaters     1380         4.2          21.1             79.7     Length and proportions
   Dresses    Dresses     6145         4.1          20.7             80.5     Length and proportions
  Intimate       Swim      332         4.2          19.6             79.5         Runs small / large
```

Overall accuracy is 95%, but the number that matters is the negative class: 0.76 precision, 0.85 recall, 0.80 F1, with 0.98 ROC-AUC. The model reads 57% of 3-star reviews as negative, which fits how 3-star reviews usually read ('I wanted to love this, but...'). The complaint topics are concrete and usable: length and proportions (1,217 reviews, often from petite customers), material quality versus price, unflattering cut, runs small or large, size inconsistency, and 'doesn't look like the photo'. Blouses, sweaters and dresses have the most negative reviews among big classes, and their top complaint is length and proportions, which points at a petite range or better size guidance.

## Output files

| File | What it contains |
|---|---|
| `outputs/reviews_scored.csv.gz` | Every review with predicted sentiment, probability and complaint topic. |
| `outputs/complaint_topics.csv` | Six topics with names, top terms and review counts. |
| `outputs/class_summary.csv` | By department and class: reviews, average rating, % negative, % recommended, top complaint. |
| `outputs/age_band_summary.csv` | The same metrics by reviewer age band. |
| `outputs/telling_words.csv` | Words and phrases that push hardest towards negative and positive. |

## Limitations and honest caveats

- Labels come from star ratings, which don't always match the text (some 4-star reviews read like complaints).
- NMF topics overlap. A review can be about size and fabric at once, but it gets one topic here.
- Topic names are a human judgement on fixed model output. Changing the random seed or settings will reorder them.
- TF-IDF misses context and sarcasm. A transformer model would do better, at more cost.

## Turning it into a dashboard

A matrix of department and class against % negative with conditional formatting, a complaint topic bar chart that cross-filters, and a review text table so people can read the real comments behind each number.

## Ideas to extend it

- Fine-tune a small transformer (for example DistilBERT) and compare against this baseline.
- Use an LLM to tag reviews with several topics at once and compare with NMF.
- Track % negative by product over time to catch a bad batch early.

## Skills demonstrated

NLP, text classification, imbalanced metrics, topic modelling, turning unstructured text into BI tables.

**Tools:** Python, pandas, scikit-learn

## Project structure

```text
review-sentiment-analytics/
├── src/main.py                 # the whole pipeline
├── data/                       # real source data (gzipped) + data/README.md with source and licence
├── outputs/                    # files from the run shown above
├── requirements.txt
├── .github/workflows/run.yml   # CI: installs and runs the project on every push
└── LICENSE
```

## Licence

Code: MIT, see [LICENSE](LICENSE). The data in `data/` keeps its original licence (see above).
