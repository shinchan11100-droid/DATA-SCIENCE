# Exploratory Data Analysis (EDA)

> **"Before you model, you must understand."**  
> EDA is the process of investigating a dataset to summarise its main characteristics, uncover patterns, spot anomalies, and test assumptions — using statistical summaries and visualisations.

---

## Table of Contents

1. [What is EDA?](#1-what-is-eda)
2. [Why EDA Matters](#2-why-eda-matters)
3. [Types of EDA](#3-types-of-eda)
4. [EDA Workflow](#4-eda-workflow)
5. [Step 1 — Load & Inspect the Data](#5-step-1--load--inspect-the-data)
6. [Step 2 — Univariate Analysis](#6-step-2--univariate-analysis)
7. [Step 3 — Bivariate & Multivariate Analysis](#7-step-3--bivariate--multivariate-analysis)
8. [Step 4 — Missing Data Analysis](#8-step-4--missing-data-analysis)
9. [Step 5 — Outlier Detection](#9-step-5--outlier-detection)
10. [Step 6 — Feature Relationships & Correlation](#10-step-6--feature-relationships--correlation)
11. [EDA for Different Data Types](#11-eda-for-different-data-types)
12. [Automated EDA Libraries](#12-automated-eda-libraries)
13. [Full EDA Template](#13-full-eda-template)
14. [Common Mistakes in EDA](#14-common-mistakes-in-eda)
15. [EDA vs Data Cleaning](#15-eda-vs-data-cleaning)
16. [Key Takeaways](#16-key-takeaways)

---

## 1. What is EDA?

**Exploratory Data Analysis (EDA)** is an approach to analysing datasets to summarise their main characteristics — often using visual methods. The term was coined by statistician **John W. Tukey** in his 1977 book *Exploratory Data Analysis*.

EDA is not about proving a hypothesis. It is about:

- Understanding **what the data looks like** (shape, size, types)
- Finding **patterns and relationships** between variables
- Detecting **problems** such as missing values, duplicates, and outliers
- Generating **hypotheses** to test later with statistical models

EDA sits at the beginning of every data science project and directly influences every downstream decision — feature engineering, model selection, and evaluation strategy.

---

## 2. Why EDA Matters

Skipping EDA leads to models built on dirty, misunderstood data. Here is what EDA prevents:

| Problem | What happens without EDA | How EDA helps |
|---|---|---|
| Missing values | Model crashes or gives biased results | Detected and handled early |
| Outliers | Skews model training | Identified and investigated |
| Wrong data types | Silent errors in computation | Caught during inspection |
| Class imbalance | Model biased toward majority class | Revealed in value counts |
| Irrelevant features | Noise slows training | Removed after correlation check |
| Data leakage | Inflated accuracy | Spotted by examining feature relationships |

---

## 3. Types of EDA

EDA can be categorised in two ways:

### By method

| Type | Description |
|---|---|
| **Graphical EDA** | Uses plots and charts — histograms, scatter plots, box plots, heatmaps |
| **Non-graphical EDA** | Uses statistics — mean, median, std dev, percentiles, frequency tables |

### By number of variables

| Type | Variables Involved | Example Techniques |
|---|---|---|
| **Univariate** | One variable at a time | Histogram, box plot, value counts |
| **Bivariate** | Two variables together | Scatter plot, correlation, grouped box plot |
| **Multivariate** | Three or more variables | Pair plot, heatmap, parallel coordinates |

---

## 4. EDA Workflow

A structured EDA follows these six steps:

```
Raw Data
   │
   ▼
Step 1: Load & Inspect       ← shape, dtypes, head(), info()
   │
   ▼
Step 2: Univariate Analysis  ← distributions, counts, stats
   │
   ▼
Step 3: Bivariate Analysis   ← scatter, correlation, grouped plots
   │
   ▼
Step 4: Missing Data         ← isnull(), heatmap, imputation
   │
   ▼
Step 5: Outlier Detection    ← IQR, Z-score, box plots
   │
   ▼
Step 6: Insights & Actions   ← document findings, prepare for modelling
```

---

## 5. Step 1 — Load & Inspect the Data

The very first step is to load your data and get a broad overview of what you are working with.

### Import libraries

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Optional but useful
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['figure.figsize'] = (10, 6)
sns.set_style('whitegrid')
```

### Load the dataset

```python
df = pd.read_csv('data.csv')

# Other formats
df = pd.read_excel('data.xlsx')
df = pd.read_json('data.json')
df = pd.read_sql(query, connection)
```

### Inspect shape and structure

```python
print("Shape:", df.shape)           # (rows, columns)
print("Columns:", df.columns.tolist())
print("\nData Types:\n", df.dtypes)
```

**Example output:**
```
Shape: (1000, 8)
Columns: ['age', 'gender', 'score', 'grade', 'study_hours', 'income', 'city', 'passed']

Data Types:
age             int64
gender         object
score         float64
grade          object
study_hours   float64
income        float64
city           object
passed           bool
```

### View sample rows

```python
df.head()       # first 5 rows
df.tail()       # last 5 rows
df.sample(5)    # 5 random rows
```

### Full summary

```python
df.info()          # non-null counts, dtypes, memory usage
df.describe()      # stats for numeric cols: count, mean, std, min, quartiles, max
df.describe(include='all')   # includes categorical cols too
```

### Check for duplicates

```python
print("Duplicate rows:", df.duplicated().sum())
df.drop_duplicates(inplace=True)   # remove if found
```

---

## 6. Step 2 — Univariate Analysis

Univariate analysis examines **one variable at a time** to understand its distribution and properties.

### Numeric columns

```python
# Summary statistics
df['score'].describe()
# count, mean, std, min, 25%, 50%, 75%, max

# Central tendency
print("Mean:  ", df['score'].mean())
print("Median:", df['score'].median())
print("Mode:  ", df['score'].mode()[0])

# Spread
print("Std Dev:", df['score'].std())
print("Variance:", df['score'].var())
print("Range:", df['score'].max() - df['score'].min())

# Shape
print("Skewness:", df['score'].skew())
# Positive skew = tail on right; Negative = tail on left; 0 = symmetric
print("Kurtosis:", df['score'].kurt())
# High kurtosis = heavy tails (more outliers)
```

#### Histogram — distribution shape

```python
plt.figure(figsize=(8, 5))
df['score'].hist(bins=30, color='#7F77DD', edgecolor='white')
plt.title('Distribution of Scores')
plt.xlabel('Score')
plt.ylabel('Frequency')
plt.axvline(df['score'].mean(), color='red', linestyle='--', label='Mean')
plt.axvline(df['score'].median(), color='orange', linestyle='--', label='Median')
plt.legend()
plt.show()
```

#### Box plot — spread and outliers

```python
sns.boxplot(x=df['score'], color='#CECBF6')
plt.title('Box Plot of Scores')
plt.show()
```

**Reading a box plot:**

```
Min    Q1      Median   Q3     Max
|------|========|========|------|
       ←  IQR  →
                                  ●  ← outlier
```

- **Box** = middle 50% of data (Q1 to Q3)
- **Line inside box** = median (Q2)
- **Whiskers** = data within 1.5 × IQR
- **Dots beyond whiskers** = outliers

### Categorical columns

```python
# Count of each category
print(df['grade'].value_counts())
print(df['grade'].value_counts(normalize=True) * 100)  # as percentages

# Unique categories
print("Unique values:", df['grade'].nunique())
print(df['grade'].unique())
```

#### Bar chart

```python
df['grade'].value_counts().plot(kind='bar', color='#7F77DD', edgecolor='white')
plt.title('Grade Distribution')
plt.xlabel('Grade')
plt.ylabel('Count')
plt.xticks(rotation=0)
plt.show()
```

#### Pie chart (use sparingly, only for few categories)

```python
df['grade'].value_counts().plot(kind='pie', autopct='%1.1f%%')
plt.title('Grade Proportions')
plt.ylabel('')
plt.show()
```

---

## 7. Step 3 — Bivariate & Multivariate Analysis

Bivariate analysis looks at **relationships between two variables**.

### Numeric vs Numeric — scatter plot

```python
sns.scatterplot(data=df, x='study_hours', y='score', alpha=0.6, color='#7F77DD')
plt.title('Study Hours vs Score')
plt.show()
```

Add a regression line to see the trend:

```python
sns.regplot(data=df, x='study_hours', y='score', scatter_kws={'alpha':0.4})
plt.title('Study Hours vs Score with Trend Line')
plt.show()
```

### Numeric vs Categorical — grouped box plot

```python
sns.boxplot(data=df, x='grade', y='score', palette='Set2')
plt.title('Score Distribution by Grade')
plt.show()
```

### Categorical vs Categorical — crosstab / heatmap

```python
ct = pd.crosstab(df['grade'], df['passed'])
print(ct)

sns.heatmap(ct, annot=True, fmt='d', cmap='Blues')
plt.title('Grade vs Passed Crosstab')
plt.show()
```

### Multivariate — pair plot

Plots all numeric columns against each other in one grid:

```python
sns.pairplot(df[['score', 'study_hours', 'age', 'income']], 
             diag_kind='kde', 
             plot_kws={'alpha': 0.5})
plt.suptitle('Pair Plot', y=1.02)
plt.show()
```

---

## 8. Step 4 — Missing Data Analysis

Missing data is one of the most common problems in real-world datasets. It must be understood before being treated.

### Detect missing values

```python
# Count per column
print(df.isnull().sum())

# Percentage per column
missing_pct = df.isnull().mean() * 100
print(missing_pct[missing_pct > 0].sort_values(ascending=False))

# Total missing in entire dataset
print("Total missing cells:", df.isnull().sum().sum())
print("Total cells:", df.size)
print("Missing %:", round(df.isnull().sum().sum() / df.size * 100, 2))
```

### Visualise missing data

```python
import missingno as msno   # pip install missingno

msno.bar(df)         # bar chart of non-null counts
msno.matrix(df)      # white = missing, black = present
msno.heatmap(df)     # correlations between missingness of columns
```

Or with seaborn (no extra library):

```python
sns.heatmap(df.isnull(), cbar=False, cmap='viridis', yticklabels=False)
plt.title('Missing Data Map (yellow = missing)')
plt.show()
```

### Types of missing data

| Type | Meaning | Example |
|---|---|---|
| **MCAR** — Missing Completely At Random | Missing has no pattern | Random sensor failure |
| **MAR** — Missing At Random | Missing depends on other observed variables | Income missing more for younger people |
| **MNAR** — Missing Not At Random | Missing depends on the missing value itself | High earners skip income field |

MCAR is the safest. MNAR is the most dangerous — you cannot simply impute it.

### Handle missing values

```python
# Option 1: Drop rows with any missing
df.dropna(inplace=True)

# Option 2: Drop rows only if missing in specific column
df.dropna(subset=['income'], inplace=True)

# Option 3: Drop columns with > 30% missing
thresh = int(0.7 * len(df))
df.dropna(axis=1, thresh=thresh, inplace=True)

# Option 4: Fill numeric with mean/median
df['age'].fillna(df['age'].median(), inplace=True)
df['income'].fillna(df['income'].mean(), inplace=True)

# Option 5: Fill categorical with mode or constant
df['city'].fillna(df['city'].mode()[0], inplace=True)
df['gender'].fillna('Unknown', inplace=True)

# Option 6: Forward / backward fill (for time series)
df['price'].fillna(method='ffill', inplace=True)

# Option 7: sklearn imputer (recommended for pipelines)
from sklearn.impute import SimpleImputer, KNNImputer

imputer = SimpleImputer(strategy='median')   # or 'mean', 'most_frequent', 'constant'
df[['age', 'income']] = imputer.fit_transform(df[['age', 'income']])

# KNN imputer — uses nearest neighbours (better accuracy, slower)
knn_imp = KNNImputer(n_neighbors=5)
df_imputed = knn_imp.fit_transform(df)
```

**Rule of thumb:**

| Missing % | Recommended action |
|---|---|
| < 5% | Safe to impute with mean/median or drop rows |
| 5–30% | Impute carefully; check if MCAR/MAR/MNAR |
| > 30% | Consider dropping the column; flag to stakeholders |

---

## 9. Step 5 — Outlier Detection

An **outlier** is a data point that differs significantly from other observations. It can be a genuine extreme value or a data entry error.

### Method 1: IQR (Interquartile Range)

The most commonly used method — robust to skewed distributions.

```python
Q1 = df['score'].quantile(0.25)
Q3 = df['score'].quantile(0.75)
IQR = Q3 - Q1

lower_fence = Q1 - 1.5 * IQR
upper_fence = Q3 + 1.5 * IQR

print(f"Q1: {Q1}, Q3: {Q3}, IQR: {IQR}")
print(f"Lower fence: {lower_fence}, Upper fence: {upper_fence}")

outliers = df[(df['score'] < lower_fence) | (df['score'] > upper_fence)]
print(f"Number of outliers: {len(outliers)}")
print(outliers)
```

### Method 2: Z-score

Best for normally distributed data. Flags values more than 3 standard deviations from the mean.

```python
from scipy import stats

z_scores = np.abs(stats.zscore(df['score']))
outliers_z = df[z_scores > 3]

print(f"Outliers by Z-score: {len(outliers_z)}")
```

### Method 3: Isolation Forest (machine learning approach)

Useful for multivariate outlier detection.

```python
from sklearn.ensemble import IsolationForest

iso = IsolationForest(contamination=0.05, random_state=42)
df['outlier_flag'] = iso.fit_predict(df[['score', 'study_hours', 'income']])
# -1 = outlier, 1 = normal
outliers_iso = df[df['outlier_flag'] == -1]
```

### Visualise outliers

```python
# Box plots for all numeric columns
df.select_dtypes(include='number').plot(kind='box', subplots=True, layout=(2, 4), figsize=(14, 8))
plt.suptitle('Box Plots for Outlier Detection')
plt.tight_layout()
plt.show()

# Scatter with outlier highlight
colors = np.where(z_scores > 3, 'red', '#7F77DD')
plt.scatter(df.index, df['score'], c=colors, alpha=0.6)
plt.title('Score — Outliers Highlighted in Red')
plt.show()
```

### What to do with outliers

```
Outlier found
     │
     ├─► Is it a data entry error?     YES → Fix or remove
     │
     ├─► Is it physically impossible?  YES → Remove
     │
     ├─► Is it real but extreme?       YES → Investigate further
     │        │
     │        ├─► Keep if it carries meaningful information
     │        ├─► Cap it (Winsorize) to the fence value
     │        └─► Use robust models (tree-based, median regression)
     │
     └─► Never blindly delete without understanding why
```

#### Winsorization — capping outliers

```python
from scipy.stats.mstats import winsorize

df['score_capped'] = winsorize(df['score'], limits=[0.05, 0.05])
# Caps the bottom 5% and top 5%
```

---

## 10. Step 6 — Feature Relationships & Correlation

### Pearson correlation matrix

```python
corr_matrix = df.corr()
print(corr_matrix)

plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, 
            annot=True, 
            fmt='.2f', 
            cmap='coolwarm',
            center=0,
            square=True,
            linewidths=0.5)
plt.title('Correlation Heatmap')
plt.show()
```

**Interpreting correlation:**

| Correlation value | Strength |
|---|---|
| 0.9 to 1.0 | Very strong positive |
| 0.7 to 0.9 | Strong positive |
| 0.5 to 0.7 | Moderate positive |
| 0.3 to 0.5 | Weak positive |
| -0.3 to 0.3 | Little to no linear relationship |
| -0.5 to -0.3 | Weak negative |
| -0.7 to -0.5 | Moderate negative |
| -1.0 to -0.7 | Strong negative |

> **Important:** Correlation ≠ Causation. Always interpret in context.

### Spearman correlation (for ordinal or non-normal data)

```python
spearman_corr = df.corr(method='spearman')
sns.heatmap(spearman_corr, annot=True, cmap='coolwarm')
plt.title('Spearman Correlation Heatmap')
plt.show()
```

### Correlation with target variable

```python
target_corr = df.corr()['score'].drop('score').sort_values(ascending=False)
print(target_corr)

target_corr.plot(kind='bar', color='#7F77DD', edgecolor='white')
plt.title('Correlation of Features with Score')
plt.axhline(0, color='black', linewidth=0.8)
plt.show()
```

---

## 11. EDA for Different Data Types

### Time series data

```python
df['date'] = pd.to_datetime(df['date'])
df.set_index('date', inplace=True)

df['value'].plot()                            # line plot
df['value'].resample('M').mean().plot()       # monthly average
df['value'].rolling(window=7).mean().plot()   # 7-day rolling average
```

### Text data

```python
df['text'].str.len().hist()                          # text length distribution
df['text'].str.split().apply(len).describe()          # word count stats

from collections import Counter
all_words = ' '.join(df['text']).lower().split()
word_freq = Counter(all_words).most_common(20)
```

### Image data

```python
import os
print("Image count:", len(os.listdir('images/')))

# Check image sizes
from PIL import Image
sizes = [Image.open(f'images/{f}').size for f in os.listdir('images/')]
widths, heights = zip(*sizes)
print(f"Width range: {min(widths)} – {max(widths)}")
print(f"Height range: {min(heights)} – {max(heights)}")
```

---

## 12. Automated EDA Libraries

These libraries can generate a full EDA report with a single line of code.

### ydata-profiling (formerly pandas-profiling)

```python
pip install ydata-profiling

from ydata_profiling import ProfileReport
profile = ProfileReport(df, title='EDA Report')
profile.to_file('eda_report.html')
```

Generates an interactive HTML report with distributions, correlations, missing data, and duplicates.

### Sweetviz

```python
pip install sweetviz

import sweetviz as sv
report = sv.analyze(df)
report.show_html('sweetviz_report.html')

# Compare train vs test
report = sv.compare([train, 'Train'], [test, 'Test'])
report.show_html('compare_report.html')
```

### D-Tale

```python
pip install dtale

import dtale
dtale.show(df)   # opens interactive GUI in browser
```

### AutoViz

```python
pip install autoviz

from autoviz.AutoViz_Class import AutoViz_Class
AV = AutoViz_Class()
AV.AutoViz('data.csv')
```

| Library | Best for | Output |
|---|---|---|
| ydata-profiling | Comprehensive reports | HTML |
| Sweetviz | Train vs test comparison | HTML |
| D-Tale | Interactive exploration | Web GUI |
| AutoViz | Quick visual overview | Plots |

---

## 13. Full EDA Template

Copy this template at the start of any new project:

```python
# ============================================================
# EXPLORATORY DATA ANALYSIS — FULL TEMPLATE
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings

warnings.filterwarnings('ignore')
plt.rcParams['figure.figsize'] = (10, 6)
sns.set_style('whitegrid')

# ── 1. LOAD DATA ─────────────────────────────────────────────
df = pd.read_csv('data.csv')

# ── 2. SHAPE & STRUCTURE ─────────────────────────────────────
print("Shape:", df.shape)
print("\nColumn names:", df.columns.tolist())
print("\nData types:\n", df.dtypes)
print("\nFirst 5 rows:")
df.head()

# ── 3. SUMMARY STATISTICS ────────────────────────────────────
print("\nNumerical summary:")
df.describe()

print("\nCategorical summary:")
df.describe(include='object')

# ── 4. DUPLICATES ────────────────────────────────────────────
print("\nDuplicate rows:", df.duplicated().sum())
df.drop_duplicates(inplace=True)

# ── 5. MISSING DATA ──────────────────────────────────────────
missing = df.isnull().mean() * 100
print("\nMissing % per column:")
print(missing[missing > 0].sort_values(ascending=False))

sns.heatmap(df.isnull(), cbar=False, cmap='viridis', yticklabels=False)
plt.title('Missing Data Map')
plt.show()

# ── 6. UNIVARIATE — NUMERIC ──────────────────────────────────
numeric_cols = df.select_dtypes(include='number').columns

for col in numeric_cols:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    df[col].hist(bins=30, ax=axes[0], color='#7F77DD', edgecolor='white')
    axes[0].set_title(f'{col} — Histogram')
    sns.boxplot(x=df[col], ax=axes[1], color='#CECBF6')
    axes[1].set_title(f'{col} — Box Plot')
    plt.suptitle(f'Analysis of: {col}')
    plt.tight_layout()
    plt.show()
    print(f"\n{col} — Skewness: {df[col].skew():.3f}, Kurtosis: {df[col].kurt():.3f}")

# ── 7. UNIVARIATE — CATEGORICAL ──────────────────────────────
cat_cols = df.select_dtypes(include='object').columns

for col in cat_cols:
    print(f"\n{col} — Value Counts:")
    print(df[col].value_counts())
    df[col].value_counts().plot(kind='bar', color='#7F77DD', edgecolor='white')
    plt.title(f'{col} — Distribution')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# ── 8. BIVARIATE — CORRELATION HEATMAP ───────────────────────
plt.figure(figsize=(12, 8))
sns.heatmap(df[numeric_cols].corr(), annot=True, fmt='.2f',
            cmap='coolwarm', center=0, linewidths=0.5)
plt.title('Correlation Heatmap')
plt.show()

# ── 9. BIVARIATE — PAIR PLOT ─────────────────────────────────
sns.pairplot(df[numeric_cols], diag_kind='kde', plot_kws={'alpha': 0.4})
plt.suptitle('Pair Plot', y=1.02)
plt.show()

# ── 10. OUTLIER DETECTION (IQR) ──────────────────────────────
print("\n── Outlier Detection (IQR Method) ──")
for col in numeric_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    outliers = df[(df[col] < Q1 - 1.5 * IQR) | (df[col] > Q3 + 1.5 * IQR)]
    print(f"{col}: {len(outliers)} outliers "
          f"(fence: [{Q1 - 1.5*IQR:.2f}, {Q3 + 1.5*IQR:.2f}])")

# ── 11. SUMMARY INSIGHTS ─────────────────────────────────────
print("\n── Key Findings ──")
print(f"Dataset: {df.shape[0]} rows × {df.shape[1]} columns")
print(f"Numeric columns: {len(numeric_cols)}")
print(f"Categorical columns: {len(cat_cols)}")
print(f"Total missing values: {df.isnull().sum().sum()}")
print(f"Duplicate rows removed: {df.duplicated().sum()}")
```

---

## 14. Common Mistakes in EDA

| Mistake | Why it is a problem | Better approach |
|---|---|---|
| Skipping EDA entirely | Leads to wrong models and wrong conclusions | Always start with EDA |
| Looking only at mean | Hides skewness and outliers | Always check median, std, and distribution shape |
| Ignoring data types | Causes silent bugs in computation | Check `df.dtypes` and convert where needed |
| Blindly deleting outliers | Removes genuine extreme values | Investigate before removing |
| Treating all missing as random | MNAR requires different treatment | Check the pattern of missingness |
| Over-engineering at EDA stage | EDA is for understanding, not fixing everything | Document findings, fix in preprocessing |
| Only checking pairwise correlation | Misses multivariate relationships | Use pair plots and feature importance |

---

## 15. EDA vs Data Cleaning

These two are often confused but serve different purposes:

| Aspect | EDA | Data Cleaning |
|---|---|---|
| **Purpose** | Understand the data | Fix the data |
| **Output** | Insights and findings | Clean dataset |
| **When** | Before any transformation | After EDA informs the issues |
| **Tools** | Plots, statistics | Imputation, type casting, dropping |
| **Nature** | Exploratory, investigative | Prescriptive, transformative |

EDA tells you **what** is wrong. Data cleaning **fixes** it.

---

## 16. Key Takeaways

- **EDA is not optional** — it is the foundation of every reliable data science project.
- Always check **shape, types, missing values, distributions, and correlations** before anything else.
- Use **visuals and statistics together** — a chart without numbers can mislead, numbers without charts can hide patterns.
- **Skewness and outliers** affect model performance significantly — always investigate them.
- Missing data is not just a count problem — the **type of missingness** (MCAR, MAR, MNAR) determines the right treatment.
- **Correlation does not imply causation** — always interpret relationships in domain context.
- Automated EDA tools like `ydata-profiling` and `sweetviz` save time but do not replace understanding.

---

## References & Further Reading

- Tukey, J. W. (1977). *Exploratory Data Analysis*. Addison-Wesley.
- [Pandas documentation](https://pandas.pydata.org/docs/)
- [Seaborn documentation](https://seaborn.pydata.org/)
- [ydata-profiling](https://github.com/ydataai/ydata-profiling)
- [Kaggle EDA tutorials](https://www.kaggle.com/learn/data-visualization)

---

*This README was written to serve as a complete reference for Exploratory Data Analysis. Feel free to extend each section with your own dataset-specific findings.*
