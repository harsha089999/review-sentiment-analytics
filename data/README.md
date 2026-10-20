# Data

Real data, not simulated. Files are the original CSVs, gzip-compressed to keep the repo small. pandas and DuckDB read
`.csv.gz` directly, so nothing needs unzipping.

**Dataset:** Women's E-Commerce Clothing Reviews

- [Kaggle: Women's E-Commerce Clothing Reviews (nicapotato)](https://www.kaggle.com/datasets/nicapotato/womens-ecommerce-clothing-reviews)

**Licence:** Published on Kaggle under CC0 (public domain).

| File | Contents |
|---|---|
| `womens_reviews.csv.gz` | 23,486 reviews: clothing ID, reviewer age, title, text, 1-5 rating, recommended flag, helpful votes, division, department, class |

To refresh from Kaggle instead, download the dataset with the Kaggle CLI (`kaggle datasets download <owner/dataset>`),
then gzip the needed files into this folder with the same names.
