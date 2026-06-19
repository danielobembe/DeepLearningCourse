# Data Exploration & Preparation Guide for Tabular Data → PyTorch

This guide walks through every step of EDA and preprocessing as a generalizable framework — applicable to any tabular dataset, regardless of domain.

Each step includes: what it is, why it matters, simple code, what issues to watch for, and how to resolve them.

---

# PHASE 1: EXPLORATORY DATA ANALYSIS (EDA)

The goal of this phase is understanding, not fixing. You're building a mental model of your dataset so that every preprocessing decision in Phase 2 is informed rather than guessed.

## Step 1.1 — Get the Basic Shape of the Data

**Explanation**
Before anything else, you need to know how big your dataset is, what columns exist, and what type each column is (numeric, text, date, etc.). This orients you and catches obvious problems immediately (e.g., a numeric column that got loaded as text).

**Example Code**
```python
import pandas as pd

df = pd.read_csv("data.csv")

print(df.shape)        # (rows, columns)
print(df.dtypes)       # data type of each column
print(df.head())       # first 5 rows
print(df.info())       # combines dtype + non-null counts + memory usage
```

**What to look for**
- A column that should be numeric (like "age" or "price") but shows up as `object` (text) — usually means it contains stray characters like `$`, `,`, or inconsistent entries like `"N/A"` mixed with numbers.
- Date columns loaded as plain strings.
- An unexpectedly small or large row/column count compared to what you expected from the data source.

**How to resolve**
```python
# Force a column to numeric, turning unparseable entries into NaN (so you can find them)
df["price"] = pd.to_numeric(df["price"].str.replace("$", "").str.replace(",", ""), errors="coerce")

# Convert a column to datetime
df["signup_date"] = pd.to_datetime(df["signup_date"], errors="coerce")

# See which rows failed to parse (now NaN) so you can inspect the original values
print(df[df["price"].isna()])
```

---

## Step 1.2 — Check for Missing Data

**Explanation**
Missing values affect almost every downstream step — which features you keep, how you impute, and even how you interpret patterns. You want a precise picture: which columns, how much, and ideally why.

**Example Code**
```python
missing_counts = df.isnull().sum()
missing_pct = (df.isnull().sum() / len(df)) * 100

missing_summary = pd.DataFrame({"count": missing_counts, "percent": missing_pct})
print(missing_summary[missing_summary["count"] > 0].sort_values("percent", ascending=False))
```

A quick visual is often clearer than a table:
```python
import seaborn as sns
import matplotlib.pyplot as plt

sns.heatmap(df.isnull(), cbar=False)
plt.show()
```

**What to look for**
- Columns with a small amount missing (a few percent) — usually safe to impute simply.
- Columns with a large amount missing (40%+) — worth asking whether the column is even usable.
- Missingness that follows a pattern — e.g., column B is only missing when column A has a specific value. This suggests the missingness is *informative* rather than random (e.g., "discount_pct" is missing because no discount was applied, which is itself a meaningful fact, not a random gap).

**How to resolve**
This step is diagnostic — full resolution happens in Phase 2 (Step 2.2). But you can start tagging columns now:
```python
# Flag candidate columns to drop because they're missing too much to be useful
cols_to_consider_dropping = missing_summary[missing_summary["percent"] > 50].index.tolist()
print(cols_to_consider_dropping)

# Check whether missingness in one column correlates with values in another
df.groupby(df["discount_pct"].isna())["purchase_type"].value_counts()
```

---

## Step 1.3 — Understand the Target Variable

**Explanation**
The shape of your target determines your loss function, evaluation metric, and sometimes your preprocessing (e.g., transforming a skewed regression target). It's worth understanding before you look at anything else, since it frames why every other feature matters.

**Example Code**

*Classification target:*
```python
print(df["target"].value_counts())
print(df["target"].value_counts(normalize=True))  # as proportions

sns.countplot(x="target", data=df)
plt.show()
```

*Regression target:*
```python
print(df["target"].describe())  # mean, std, min, max, quartiles

sns.histplot(df["target"], kde=True)
plt.show()
```

**What to look for**
- **Classification:** class imbalance — e.g., 95% of rows belong to one class. Accuracy becomes a misleading metric in this case.
- **Regression:** skewness — a long tail to one side — or unrealistic outliers (e.g., a negative value where only positive values make sense).

**How to resolve**
```python
# Classification: check the imbalance ratio
counts = df["target"].value_counts()
imbalance_ratio = counts.max() / counts.min()
print(imbalance_ratio)  # if this is large (e.g. >10), plan for class weighting or resampling later

# Regression: a log transform often helps right-skewed targets
import numpy as np
df["target_log"] = np.log1p(df["target"])  # log1p handles zero values safely
sns.histplot(df["target_log"], kde=True)
plt.show()
```

---

## Step 1.4 — Examine Numerical Features

**Explanation**
For each numerical column, you want to understand its central tendency, spread, and shape. This tells you whether scaling is needed, whether outliers exist, and whether the feature even varies enough to be useful.

**Example Code**
```python
numerical_cols = df.select_dtypes(include="number").columns.tolist()

print(df[numerical_cols].describe())

# Histograms for all numerical columns at once
df[numerical_cols].hist(figsize=(12, 8), bins=30)
plt.tight_layout()
plt.show()

# Box plots are great for spotting outliers
for col in numerical_cols:
    sns.boxplot(x=df[col])
    plt.title(col)
    plt.show()
```

**What to look for**
- **Outliers**: points far outside the main cluster in a box plot (beyond the "whiskers").
- **Skew**: a histogram with a long tail on one side rather than a roughly symmetric bell shape.
- **Near-zero variance**: a column where almost every value is identical — it likely won't help the model.
- **Suspicious values**: negative values where only positive makes sense, or values like `999` or `-1` that look like placeholder codes for missing data rather than real measurements.

**How to resolve**
This is diagnostic for now, but you can quantify what you see:
```python
# Quantify skew
print(df[numerical_cols].skew())  # values far from 0 indicate skew

# Flag near-constant columns
low_variance_cols = [col for col in numerical_cols if df[col].nunique() <= 1]
print(low_variance_cols)

# Identify outliers using the IQR method (just to see how many, not to remove yet)
Q1 = df["feature"].quantile(0.25)
Q3 = df["feature"].quantile(0.75)
IQR = Q3 - Q1
outliers = df[(df["feature"] < Q1 - 1.5*IQR) | (df["feature"] > Q3 + 1.5*IQR)]
print(len(outliers))
```

---

## Step 1.5 — Examine Categorical Features

**Explanation**
For categorical columns, the key question is cardinality (how many unique categories) — this drives your encoding strategy in Phase 2. You also want to check for inconsistent labeling, which is a very common real-world data issue.

**Example Code**
```python
categorical_cols = df.select_dtypes(include="object").columns.tolist()

for col in categorical_cols:
    print(f"--- {col} ---")
    print(f"Unique values: {df[col].nunique()}")
    print(df[col].value_counts().head(10))
    print()
```

**What to look for**
- **High cardinality**: hundreds or thousands of unique values (e.g., zip codes, user IDs, free-text fields) — one-hot encoding this would create an enormous number of columns.
- **Inconsistent labels**: `"NY"`, `"N.Y."`, `"New York"` all meaning the same category, or differences in casing/whitespace like `"Yes"` vs `"yes "`.
- **Rare categories**: categories that appear only once or a handful of times — these can cause problems when splitting into train/test (a category might appear in test but never in train).

**How to resolve**
```python
# Standardize text formatting (catches whitespace/casing inconsistencies)
df["state"] = df["state"].str.strip().str.lower()

# Map known synonyms to a single canonical value
state_mapping = {"ny": "new york", "n.y.": "new york", "new york": "new york"}
df["state"] = df["state"].map(state_mapping).fillna(df["state"])

# Check cardinality precisely, to decide encoding strategy later
cardinality = df[categorical_cols].nunique().sort_values(ascending=False)
print(cardinality)

# Group rare categories into "Other" (useful for high-cardinality columns)
value_counts = df["city"].value_counts()
rare_categories = value_counts[value_counts < 10].index
df["city"] = df["city"].replace(rare_categories, "Other")
```

---

## Step 1.6 — Feature-to-Target Relationships

**Explanation**
This step starts answering "which features actually matter?" It's not about finalizing feature selection yet, but about building intuition and catching surprises (a feature with zero relationship to the target might be noise; a feature with too perfect a relationship might be a data leak).

**Example Code**

*Numerical feature vs. numerical/binary target:*
```python
print(df[numerical_cols + ["target"]].corr()["target"].sort_values(ascending=False))
```

*Numerical feature vs. categorical target:*
```python
sns.boxplot(x="target", y="feature", data=df)
plt.show()
```

*Categorical feature vs. target:*
```python
df.groupby("category_feature")["target"].mean().sort_values(ascending=False)
```

**What to look for**
- A feature with near-zero correlation/relationship to the target — possibly unhelpful (but don't discard purely on this basis; nonlinear relationships won't show up in a simple correlation).
- A feature with a *suspiciously perfect* relationship to the target — this is a red flag for **data leakage** (e.g., a column that was actually computed *from* the target, or only known *after* the target is determined).

**How to resolve**
```python
# Investigate a suspiciously strong correlation before trusting it
print(df[["suspicious_feature", "target"]].corr())
# Ask: could this feature only be known AFTER the target outcome happens?
# If yes, drop it — it would leak information the model wouldn't have at prediction time.
df = df.drop(columns=["suspicious_feature"])
```

---

## Step 1.7 — Feature-to-Feature Relationships

**Explanation**
Highly correlated (redundant) features add noise and computation without adding much new information. This matters less for neural networks than for linear models, but it's still worth knowing about, especially for interpretability and dataset size reduction.

**Example Code**
```python
corr_matrix = df[numerical_cols].corr()

sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", center=0)
plt.show()
```

**What to look for**
- Pairs of features with very high correlation (e.g., above 0.9) — these are often redundant (e.g., "total_price" and "price_per_unit × quantity").
- Clusters of correlated features that might represent the same underlying concept.

**How to resolve**
```python
# Find highly correlated pairs programmatically
corr_pairs = corr_matrix.abs().unstack().sort_values(ascending=False)
corr_pairs = corr_pairs[corr_pairs < 1.0]  # remove self-correlation
print(corr_pairs.head(10))

# Drop one of a redundant pair (keep the one more interpretable or more complete)
df = df.drop(columns=["price_per_unit"])
```

---

## Step 1.8 — Data Quality Checks

**Explanation**
A final sweep for structural issues: duplicate rows, impossible values, and inconsistent formats that didn't get caught in earlier, more targeted checks.

**Example Code**
```python
# Duplicate rows
print(df.duplicated().sum())

# Impossible values (domain-specific — adjust the condition to your data)
print(df[df["age"] < 0])
print(df[df["age"] > 120])
```

**What to look for**
- Exact duplicate rows (could be genuine repeated events, or could be a data export error).
- Logically impossible values (negative ages, dates in the future for a "date of birth" field, etc.).

**How to resolve**
```python
# Drop exact duplicates
df = df.drop_duplicates()

# Fix or remove impossible values (treat as missing, then handle in Phase 2)
df.loc[df["age"] < 0, "age"] = None
df.loc[df["age"] > 120, "age"] = None
```

---

# PHASE 2: PREPARING DATA FOR PYTORCH

The order below is intentional — each step assumes a clean version of the previous one. The single most important rule across this whole phase: **anything that "learns" a statistic from data must learn it only from the training set.**

## Step 2.1 — Split the Data

**Explanation**
This happens early — before imputation, scaling, or encoding — because every one of those steps must be "fit" only on the training data. If you split *after* computing statistics on the full dataset, you leak information from validation/test into training, and your evaluation results become unreliable (too optimistic).

**Example Code**
```python
from sklearn.model_selection import train_test_split

X = df.drop(columns=["target"])
y = df["target"]

# First split off the test set
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y  # stratify keeps class balance consistent (classification only)
)

# Then split the remainder into validation and test
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
)

print(len(X_train), len(X_val), len(X_test))
```

**What to look for**
- Class proportions that look very different between train/val/test — a sign you forgot `stratify` (classification problems).
- A test or validation set so small that performance metrics will be noisy.

**How to resolve**
```python
# Verify class balance is consistent across splits
print(y_train.value_counts(normalize=True))
print(y_val.value_counts(normalize=True))
print(y_test.value_counts(normalize=True))
```

---

## Step 2.2 — Handle Missing Values (Imputation)

**Explanation**
Now you act on what Step 1.2 found. Common strategies: drop the column (if mostly missing and not informative), drop the rows (if very few are missing and you can afford to lose them), or impute (fill with a statistic, ideally computed from train only).

**Example Code**
```python
from sklearn.impute import SimpleImputer

# Numerical: median is often safer than mean (robust to outliers)
num_imputer = SimpleImputer(strategy="median")
X_train[numerical_cols] = num_imputer.fit_transform(X_train[numerical_cols])
X_val[numerical_cols] = num_imputer.transform(X_val[numerical_cols])
X_test[numerical_cols] = num_imputer.transform(X_test[numerical_cols])

# Categorical: most frequent value, or a constant like "missing"
cat_imputer = SimpleImputer(strategy="most_frequent")
X_train[categorical_cols] = cat_imputer.fit_transform(X_train[categorical_cols])
X_val[categorical_cols] = cat_imputer.transform(X_val[categorical_cols])
X_test[categorical_cols] = cat_imputer.transform(X_test[categorical_cols])
```

**What to look for**
- Imputing with the **mean** when the column is skewed or has outliers — this can distort the imputed value significantly. Median is usually more robust.
- Calling `.fit_transform()` on validation or test data by mistake — this is the data leakage bug again, just in a different step.
- Imputation that erases a meaningful signal — e.g., if "missing" itself was informative (per Step 1.2's pattern check), consider adding a separate binary "was_missing" indicator column instead of just filling the gap silently.

**How to resolve**
```python
# Preserve the information that a value was missing, if it seemed meaningful in EDA
X_train["income_was_missing"] = X_train["income"].isna().astype(int)
X_val["income_was_missing"] = X_val["income"].isna().astype(int)
X_test["income_was_missing"] = X_test["income"].isna().astype(int)
# then impute "income" as normal afterward
```

---

## Step 2.3 — Handle Outliers

**Explanation**
Only act here if Step 1.4 revealed outliers that are likely errors (not just rare-but-real values). Whether to clip, transform, or leave outliers alone is a judgment call based on whether they reflect genuine variation or data problems.

**Example Code**
```python
# Capping (clipping) extreme values to a reasonable range, based on train statistics
lower = X_train["feature"].quantile(0.01)
upper = X_train["feature"].quantile(0.99)

X_train["feature"] = X_train["feature"].clip(lower, upper)
X_val["feature"] = X_val["feature"].clip(lower, upper)    # reuse train's bounds
X_test["feature"] = X_test["feature"].clip(lower, upper)
```

**What to look for**
- Over-aggressive clipping that removes genuine signal (e.g., clipping income data so aggressively that all high earners look the same).
- Outliers that are actually data entry errors (e.g., a height of 800 cm) vs. outliers that are real but rare (e.g., a celebrity's net worth in a dataset of average people).

**How to resolve**
```python
# If outliers are errors, treat as missing instead of clipping, then impute
X_train.loc[X_train["height_cm"] > 250, "height_cm"] = None
# then re-run imputation (Step 2.2) on this column

# If outliers are real but skew the distribution heavily, a log transform can help
X_train["income_log"] = np.log1p(X_train["income"])
```

---

## Step 2.4 — Encode Categorical Variables

**Explanation**
Neural networks require numeric input, so categories like `"red"`, `"blue"`, `"green"` need to become numbers. The two common approaches: **one-hot encoding** (good for low cardinality) and **integer/label encoding paired with an embedding layer** (good for high cardinality, and standard practice in PyTorch).

**Example Code**

*One-hot encoding (low cardinality, e.g., < ~15 categories):*
```python
from sklearn.preprocessing import OneHotEncoder

ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
ohe.fit(X_train[["color"]])  # fit only on train

train_encoded = ohe.transform(X_train[["color"]])
val_encoded = ohe.transform(X_val[["color"]])
test_encoded = ohe.transform(X_test[["color"]])
```

*Label/integer encoding (for high cardinality, to be used with an embedding layer later):*
```python
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
X_train["city_encoded"] = le.fit_transform(X_train["city"])

# IMPORTANT: handle categories in val/test that never appeared in train
X_val["city_encoded"] = X_val["city"].map(
    lambda x: le.transform([x])[0] if x in le.classes_ else -1  # -1 = "unknown" bucket
)
```

**What to look for**
- **Unseen categories**: a category that appears in validation/test but never in train. `LabelEncoder.transform()` will throw an error on this by default — `OneHotEncoder` with `handle_unknown="ignore"` handles it gracefully, but plain `LabelEncoder` does not.
- One-hot encoding a high-cardinality column by accident, creating hundreds of sparse columns and bloating your input size.

**How to resolve**
```python
# Reserve an explicit "unknown" index for unseen categories when using label encoding + embeddings
# Common pattern: index 0 = unknown/unseen, all real categories start from 1
city_to_idx = {city: idx + 1 for idx, city in enumerate(X_train["city"].unique())}
X_train["city_idx"] = X_train["city"].map(city_to_idx)
X_val["city_idx"] = X_val["city"].map(city_to_idx).fillna(0).astype(int)   # unseen -> 0
X_test["city_idx"] = X_test["city"].map(city_to_idx).fillna(0).astype(int)
```

---

## Step 2.5 — Scale Numerical Features

**Explanation**
Neural networks generally train faster and more stably when input features are on a similar scale (rather than one feature ranging 0-1 and another ranging 0-1,000,000). Standardization (mean 0, std 1) is the most common default; min-max scaling (0 to 1) is another option, often used when you want bounded inputs.

**Example Code**
```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaler.fit(X_train[numerical_cols])  # fit on train only

X_train[numerical_cols] = scaler.transform(X_train[numerical_cols])
X_val[numerical_cols] = scaler.transform(X_val[numerical_cols])
X_test[numerical_cols] = scaler.transform(X_test[numerical_cols])
```

**What to look for**
- Scaling **after** one-hot encoding by accident, which would scale binary 0/1 columns unnecessarily (harmless but pointless — typically you scale only the genuinely numeric columns).
- Fitting the scaler on the full dataset instead of train only (the recurring leakage issue).
- Extreme outliers (if not handled in Step 2.3) distorting `StandardScaler`'s mean/std — consider `RobustScaler` if this is a concern.

**How to resolve**
```python
from sklearn.preprocessing import RobustScaler

# RobustScaler uses median/IQR instead of mean/std, less sensitive to outliers
robust_scaler = RobustScaler()
robust_scaler.fit(X_train[numerical_cols])
X_train[numerical_cols] = robust_scaler.transform(X_train[numerical_cols])
```

---

## Step 2.6 — Convert to PyTorch Tensors and Build a Dataset/DataLoader

**Explanation**
The final step: turning your cleaned, encoded, scaled pandas/numpy data into PyTorch tensors, wrapped in a `Dataset` and `DataLoader` so you can iterate over mini-batches during training.

**Example Code**
```python
import torch
from torch.utils.data import Dataset, DataLoader

class TabularDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X.values, dtype=torch.float32)
        self.y = torch.tensor(y.values, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_dataset = TabularDataset(X_train, y_train)
val_dataset = TabularDataset(X_val, y_val)
test_dataset = TabularDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Sanity check
X_batch, y_batch = next(iter(train_loader))
print(X_batch.shape, y_batch.shape)
```

**What to look for**
- `shuffle=True` on validation/test loaders — unnecessary, and can make debugging confusing since order should be irrelevant for eval anyway, but it's not "wrong," just non-standard. Convention is `shuffle=False` for val/test.
- Wrong dtype — classification targets for `CrossEntropyLoss` need to be `torch.long`, not `float32`.
- Leftover non-numeric columns (e.g., an ID column) accidentally still in `X` — this will throw a tensor conversion error or silently feed garbage into the model.

**How to resolve**
```python
# For classification with CrossEntropyLoss, targets must be long (integer class indices), not float
self.y = torch.tensor(y.values, dtype=torch.long)

# Drop ID-like or leftover non-feature columns before tensor conversion
X_train = X_train.drop(columns=["customer_id"], errors="ignore")

# If you have a mix of numeric and embedding-index categorical columns, separate them
class TabularDatasetWithEmbeddings(Dataset):
    def __init__(self, X_num, X_cat, y):
        self.X_num = torch.tensor(X_num.values, dtype=torch.float32)
        self.X_cat = torch.tensor(X_cat.values, dtype=torch.long)
        self.y = torch.tensor(y.values, dtype=torch.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X_num[idx], self.X_cat[idx], self.y[idx]
```

---

## Quick Reference: The Full Pipeline in Order

1. Load data, check shape/dtypes
2. Check missing data patterns
3. Understand the target (balance/skew)
4. Explore numerical features (distribution, outliers)
5. Explore categorical features (cardinality, inconsistent labels)
6. Check feature-target relationships (and watch for leakage)
7. Check feature-feature relationships (redundancy)
8. Clean structural issues (duplicates, impossible values)
9. **Split train/val/test**
10. Impute missing values (fit on train only)
11. Handle outliers (using train statistics)
12. Encode categoricals (fit on train only)
13. Scale numerical features (fit on train only)
14. Convert to tensors → `Dataset` → `DataLoader`
