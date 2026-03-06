#!/usr/bin/env python
# coding: utf-8

# # CAS Case Competition (Code)

# ## Data Cleaning

# In[1]:


import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import patsy
import statsmodels.api as sm
import statsmodels.formula.api as smf

from statsmodels.discrete.discrete_model import NegativeBinomial, Logit
from statsmodels.discrete.count_model import ZeroInflatedPoisson
from sklearn.model_selection import KFold
from sklearn.metrics import mean_poisson_deviance

# Load Data
file_path = "06 - CAS Predictive Modeling Case Competition- Dataset.xlsx"

# If you know the sheet name, set it here; otherwise load the first sheet.
try:
    df_raw = pd.read_excel(file_path, sheet_name="4 - Predictive Modeling Case Co")
except ValueError:
    df_raw = pd.read_excel(file_path)


# In[2]:


df = df_raw.copy()

# 1.1 Remove duplicates
# ===============================
# Exact duplicates
df = df.drop_duplicates()

# Key-based duplicates
key_cols = ["student_id", "coverage", "claim_id"]
missing_keys = [c for c in key_cols if c not in df.columns]
if missing_keys:
    raise KeyError(f"Missing key columns needed for de-duplication: {missing_keys}")

df = (
    df.sort_values(key_cols)
      .drop_duplicates(subset=key_cols, keep="first")
)


# The reasons why we removed some duplicates are that we need to make sure we won't overestimate **claim rate**, also making sure the **severity** won't be calculated multiple times.

# In[3]:


# 1.2 Basic type normalization
# ===============================
for bcol in ["sprinklered", "holdout"]:
    if bcol in df.columns:
        df[bcol] = df[bcol].astype("boolean")

# Strip whitespace in key string columns
for scol in ["coverage", "class", "study", "greek", "off_campus", "gender", "name"]:
    if scol in df.columns:
        df[scol] = df[scol].astype(str).str.strip()


# In[4]:


# 1.3 Correct unreasonable values
# ===============================
# amount >= 0
if "amount" in df.columns:
    df.loc[df["amount"] < 0, "amount"] = np.nan

# distance_to_campus >= 0
if "distance_to_campus" in df.columns:
    df.loc[df["distance_to_campus"] < 0, "distance_to_campus"] = np.nan

# gpa in [0, 4.33]
if "gpa" in df.columns:
    df.loc[(df["gpa"] < 0) | (df["gpa"] > 4.33), "gpa"] = np.nan

# risk_tier in {1,2,3}
if "risk_tier" in df.columns:
    df.loc[~df["risk_tier"].isin([1, 2, 3]), "risk_tier"] = np.nan


# In[5]:


# 1.4 Cap losses at coverage limits
# ===============================
coverage_limits = {
    "Personal Property": 10000.0,
    "Liability": 500000.0,
    "Guest Medical": 150000.0,
}

if "coverage" in df.columns and "amount" in df.columns:
    for cov, lim in coverage_limits.items():
        m = (df["coverage"] == cov) & (df["amount"] > lim)
        df.loc[m, "amount"] = lim


# In[6]:


# 1.5 Handle missing values (NO missing indicators)
# ===============================
numeric_cols = [c for c in ["gpa", "distance_to_campus", "amount"] if c in df.columns]
cat_cols = [c for c in ["class", "study", "greek", "off_campus", "gender", "sprinklered"]
            if c in df.columns]

# Numeric imputation:
# group mean by (study, class) → fallback to global median
group_cols = [c for c in ["study", "class"] if c in df.columns]
for c in numeric_cols:
    if group_cols:
        df[c] = df[c].fillna(df.groupby(group_cols)[c].transform("mean"))
    df[c] = df[c].fillna(df[c].median())

# Categorical imputation:
for c in cat_cols:
    df[c] = df[c].fillna("Unknown").astype(str).str.strip()


# In[7]:


df_clean = df


# In[8]:


df_clean


# # Exploratory Data Analysis

# ### Step 1: Is the claim sparse?

# In[9]:


# EDA 1: Is claim sparse?
# =========================
# Assumes you already have df_clean in memory.

# --- 0) Quick checks ---
print("Shape:", df_clean.shape)
print("Columns:", list(df_clean.columns))

# --- 1) Create helper flags ---
# has_claim: whether a row has a claim (claim_id > 0)
df_clean["has_claim"] = df_clean["claim_id"].astype(int) > 0

# paid_amount: amount for claimed rows only (NaN for no-claim rows)
df_clean["paid_amount"] = np.where(df_clean["has_claim"], df_clean["amount"], np.nan)

# --- 2) Overall claim sparsity (row-level) ---
n_rows = len(df_clean)
n_claim_rows = int(df_clean["has_claim"].sum())
pct_claim_rows = n_claim_rows / n_rows

print("\n=== Row-level sparsity (each row = student_id x coverage x claim_id record) ===")
print(f"Total rows: {n_rows:,}")
print(f"Rows with claim_id > 0: {n_claim_rows:,} ({pct_claim_rows:.2%})")
print(f"Rows with claim_id == 0: {n_rows - n_claim_rows:,} ({1 - pct_claim_rows:.2%})")


# In[10]:


# --- 3) Amount distribution among claim rows ---
claim_amounts = df_clean.loc[df_clean["has_claim"], "amount"].dropna()

print("\n=== Severity among claim rows (amount | claim_id > 0) ===")
print(claim_amounts.describe(percentiles=[0.5, 0.75, 0.9, 0.95, 0.99]))


# In[11]:


# --- 4) How many unique students have >=1 claim? (person-level sparsity) ---
# Because dataset is long format with multiple coverages per student,
# we check at student level: did the student have any claim in any coverage?
student_any_claim = (
    df_clean.groupby("student_id")["has_claim"]
    .any()
)

n_students = student_any_claim.shape[0]
n_students_with_claim = int(student_any_claim.sum())
pct_students_with_claim = n_students_with_claim / n_students

print("\n=== Student-level sparsity (any claim across all coverages) ===")
print(f"Unique students: {n_students:,}")
print(f"Students with >=1 claim: {n_students_with_claim:,} ({pct_students_with_claim:.2%})")
print(f"Students with 0 claim: {n_students - n_students_with_claim:,} ({1 - pct_students_with_claim:.2%})")


# In[12]:


# --- 5) Basic plots (no modeling) ---
# 5.1 Bar: claim vs no-claim (row-level)
counts = df_clean["has_claim"].value_counts().sort_index()
plt.figure()
plt.bar(["No claim", "Has claim"], [counts.get(False, 0), counts.get(True, 0)])
plt.title("Claim Sparsity (Row-level)")
plt.ylabel("Count of rows")
plt.show()


# Since there are a lot of zeros in the column of claim_id, we can consider to use the zero-inflated Poisson model to model it.

# ### Step 2: What about 4 different coverage?

# In[13]:


# Step 0) Create helper flags
# -----------------------------
# Row-level claim indicator (claim record vs non-claim record)
df_clean["has_claim"] = df_clean["claim_id"].astype(int) > 0

# Optional sanity checks
print("Rows:", len(df_clean))
print("Unique students:", df_clean["student_id"].nunique())
print("Coverages:", df_clean["coverage"].nunique(), sorted(df_clean["coverage"].dropna().unique()))


# In[14]:


# 1) Overall claim rate by coverage (P(claim_id > 0))
# ============================================================
coverage_claim_rate = (
    df_clean.groupby("coverage")["has_claim"]
    .agg(rows="size", claim_rows="sum", claim_rate="mean")
    .sort_values("claim_rate", ascending=False)
)

print("\n=== (1) Overall claim rate by coverage (row-level) ===")
print(coverage_claim_rate)

# Plot: claim rate by coverage
plt.figure()
plt.bar(coverage_claim_rate.index.astype(str), coverage_claim_rate["claim_rate"].values)
plt.title("Claim Rate by Coverage (Row-level: P(claim_id > 0))")
plt.ylabel("Claim rate")
plt.xticks(rotation=30, ha="right")
plt.show()


# In[15]:


# 2) Zero inflation / sparsity by coverage (share of zeros)
#    (This is just 1 - claim_rate, but we print it explicitly.)
# ============================================================
coverage_sparsity = coverage_claim_rate.copy()
coverage_sparsity["zero_share"] = 1 - coverage_sparsity["claim_rate"]

print("\n=== (2) Zero share by coverage (row-level) ===")
print(coverage_sparsity[["rows", "claim_rows", "claim_rate", "zero_share"]])

# Plot: zero share by coverage
plt.figure()
plt.bar(coverage_sparsity.index.astype(str), coverage_sparsity["zero_share"].values)
plt.title("Zero Share by Coverage (Row-level: P(claim_id == 0))")
plt.ylabel("Zero share")
plt.xticks(rotation=30, ha="right")
plt.show()


# In[16]:


# 3) Claim count levels (person-level): 0 vs 1 vs 2+ claims
#    per (student_id, coverage)
#    - This helps you see if a coverage is mostly single-loss or repeat claims.
# ============================================================
# Count number of claim records per student & coverage
# (since claim_id==0 indicates no claim; claim_id>0 indicates a claim record)
claim_counts = (
    df_clean.groupby(["student_id", "coverage"])["has_claim"]
    .sum()
    .rename("n_claims")
    .reset_index()
)

# Bucket into 0 / 1 / 2+
claim_counts["claim_bucket"] = pd.cut(
    claim_counts["n_claims"],
    bins=[-0.1, 0.5, 1.5, np.inf],
    labels=["0", "1", "2+"]
)

bucket_dist = (
    claim_counts.groupby(["coverage", "claim_bucket"])
    .size()
    .unstack(fill_value=0)
)

# Convert to proportions within coverage
bucket_prop = bucket_dist.div(bucket_dist.sum(axis=1), axis=0)

print("\n=== (3) Claim count buckets per student (within each coverage): counts ===")
print(bucket_dist)

print("\n=== (3) Claim count buckets per student (within each coverage): proportions ===")
print(bucket_prop)

# Plot: stacked bars of proportions (0/1/2+)
plt.figure()
bottom = np.zeros(len(bucket_prop))
for bucket in ["0", "1", "2+"]:
    vals = bucket_prop[bucket].values if bucket in bucket_prop.columns else np.zeros(len(bucket_prop))
    plt.bar(bucket_prop.index.astype(str), vals, bottom=bottom, label=bucket)
    bottom += vals
plt.title("Claim Count Buckets by Coverage (Student-level)")
plt.ylabel("Proportion of students")
plt.xticks(rotation=30, ha="right")
plt.legend(title="Claims per student")
plt.show()


# In[17]:


# 4) Frequency x key risk factors (2–3 variables) BY coverage

key_factors = ["greek", "off_campus", "sprinklered"] 
key_factors = [c for c in key_factors if c in df_clean.columns]

print("\nKey factors available:", key_factors)


# In[18]:


for factor in key_factors:
    # Claim rate within each coverage by factor level
    tmp = (
        df_clean.groupby(["coverage", factor])["has_claim"]
        .mean()
        .reset_index(name="claim_rate")
    )
    print(f"\n=== (4) Claim rate by coverage and {factor} ===")
    print(tmp.sort_values(["coverage", "claim_rate"], ascending=[True, False]))

    # Simple plot: for each coverage, show bars by factor level
    # (Keeps it readable; avoids overcomplicating.)
    coverages = tmp["coverage"].dropna().unique()
    plt.figure()
    for cov in coverages:
        sub = tmp[tmp["coverage"] == cov]
        x = sub[factor].astype(str).values
        y = sub["claim_rate"].values
        plt.plot(x, y, marker="o", linestyle="-", label=str(cov))
    plt.title(f"Claim Rate by {factor} (Lines = Coverage)")
    plt.ylabel("Claim rate")
    plt.xlabel(factor)
    plt.legend()
    plt.show()


# At this stage, our EDA focuses on identifying ***structural differences*** in claim frequency across coverage types, rather than explaining individual-level behavioral variation. Therefore, we prioritize variables such as Greek affiliation, off-campus residence, and sprinkler protection, which directly relate to the underlying risk environment and loss mechanisms and are highly interpretable at the coverage level. Variables like major, academic year, GPA, and gender, while potentially predictive, primarily capture individual heterogeneity and require additional controls to interpret properly. Including them at this stage would add complexity without improving our understanding of how risk fundamentally differs by coverage. These variables are more appropriately examined in within-coverage analyses or incorporated later as control variables in the modeling stage.

# ***From the EDA above, we decide to seperate 4 coverages!***

# ### Step 3: Can we tell something from the claim amount?

# In[19]:


# Histogram: claim amounts (linear scale)
plt.figure()
plt.hist(claim_amounts, bins=40)
plt.title("Claim Amounts (Linear Scale) - Claim Rows Only")
plt.xlabel("Amount")
plt.ylabel("Frequency")
plt.show()


# The distribution of **Claim Amount** is severly right_skewed. So modelling it by using Gamma model or lognormal model will be a good choice!

# ### Step 4: Plots of each predictor vs. target (claim rate, claim amount)

# ### Plots of each predictor vs. claim rate by coverage

# In[20]:


# =========================
# Preconditions
# =========================
df = df_clean.copy()

# Make sure has_claim exists
if "has_claim" not in df.columns:
    if "claim_id" in df.columns:
        df["has_claim"] = (df["claim_id"].astype(float) > 0)
    else:
        raise KeyError("Need 'has_claim' or 'claim_id'.")

# -------------------------
# Define key predictors
# -------------------------
continuous_vars = [c for c in ["gpa", "distance_to_campus"] if c in df.columns]
categorical_vars = [c for c in ["gender", "class", "study", "greek", "off_campus", "sprinklered"]
                    if c in df.columns]

print("Continuous predictors:", continuous_vars)
print("Categorical predictors:", categorical_vars)

# =========================
# Helper functions
# =========================

def plot_binned_claim_rate(data, x, bins=10, title=None):
    """
    Plot binned mean of has_claim vs continuous predictor x.
    """
    d = data[[x, "has_claim"]].dropna().copy()
    if d.empty or d[x].nunique() < 3:
        print(f"[skip] {x}: not enough variation")
        return

    # quantile bins are robust
    try:
        d["bin"] = pd.qcut(d[x], q=bins, duplicates="drop")
    except ValueError:
        d["bin"] = pd.cut(d[x], bins=min(bins, d[x].nunique()))

    g = (
        d.groupby("bin", observed=True)
         .agg(
             n=("has_claim", "size"),
             claim_rate=("has_claim", "mean"),
             x_mid=(x, "median")
         )
         .reset_index()
    )

    plt.figure()
    plt.plot(g["x_mid"], g["claim_rate"], marker="o")
    plt.xlabel(x)
    plt.ylabel("Claim Rate")
    plt.title(title)
    plt.grid(alpha=0.3)
    plt.show()


def plot_categorical_claim_rate(data, x, top_k=15, title=None):
    """
    Bar plot of claim rate by categorical predictor x.
    """
    d = data[[x, "has_claim"]].dropna().copy()
    d[x] = d[x].astype(str)

    g = (
        d.groupby(x)
         .agg(
             n=("has_claim", "size"),
             claim_rate=("has_claim", "mean")
         )
         .reset_index()
         .sort_values("n", ascending=False)
    )

    g_plot = g.head(top_k)

    plt.figure()
    plt.bar(g_plot[x], g_plot["claim_rate"])
    plt.xticks(rotation=45, ha="right")
    plt.xlabel(x)
    plt.ylabel("Claim Rate")
    plt.title(title)
    plt.tight_layout()
    plt.show()

    # Print table for interpretation
    print(g_plot.to_string(index=False))


# In[21]:


# =========================
# Main loop: BY coverage
# =========================
coverages = sorted(df["coverage"].dropna().unique())

for cov in coverages:
    d_cov = df[df["coverage"] == cov].copy()

    print("\n" + "=" * 80)
    print(f"Coverage: {cov} | n = {len(d_cov)} | claim rate = {d_cov['has_claim'].mean():.4f}")
    print("=" * 80)

    # ---- Continuous predictors ----
    for x in continuous_vars:
        plot_binned_claim_rate(
            d_cov,
            x=x,
            bins=10,
            title=f"[{cov}] Claim Rate vs {x} (binned)"
        )

    # ---- Categorical predictors ----
    for x in categorical_vars:
        plot_categorical_claim_rate(
            d_cov,
            x=x,
            top_k=15,
            title=f"[{cov}] Claim Rate by {x}"
        )


# Since the exposure of each observation is 1 (i.e. no terms like claim length or something like this). We use the claim rate is for the sake of generalization, and the claim count is just claim rate here because exposure = 1 for each observation.

# ### Plots of each predictor vs. claim amount by coverage 

# In[22]:


# -------------------------
# Helper: binned mean severity (continuous predictors)
# -------------------------
def plot_binned_mean_severity(data, x, bins=10, title=None):
    """
    Plot binned mean of claim amount vs continuous predictor x.
    Uses raw claim amount (no log).
    """
    d = data[[x, "amount"]].dropna().copy()
    d = d[d["amount"] > 0]

    if d.empty or d[x].nunique() < 3:
        print(f"[skip] {x}: not enough severity data")
        return

    # Quantile bins to handle skew
    try:
        d["bin"] = pd.qcut(d[x], q=bins, duplicates="drop")
    except ValueError:
        d["bin"] = pd.cut(d[x], bins=min(bins, d[x].nunique()))

    g = (
        d.groupby("bin", observed=True)
         .agg(
             n=("amount", "size"),
             mean_amount=("amount", "mean"),
             median_amount=("amount", "median"),
             x_mid=(x, "median")
         )
         .reset_index()
    )

    plt.figure()
    plt.plot(g["x_mid"], g["mean_amount"], marker="o", label="Mean")
    plt.plot(g["x_mid"], g["median_amount"], marker="x", linestyle="--", label="Median")
    plt.xlabel(x)
    plt.ylabel("Claim Amount")
    plt.title(title)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()


# -------------------------
# Helper: categorical severity summary
# -------------------------
def plot_categorical_severity(data, x, top_k=15, title=None):
    """
    Bar plot of mean claim amount by categorical predictor x.
    """
    d = data[[x, "amount"]].dropna().copy()
    d = d[d["amount"] > 0]
    d[x] = d[x].astype(str)

    g = (
        d.groupby(x)
         .agg(
             n=("amount", "size"),
             mean_amount=("amount", "mean"),
             median_amount=("amount", "median")
         )
         .reset_index()
         .sort_values("n", ascending=False)
    )

    g_plot = g.head(top_k)

    plt.figure()
    plt.bar(g_plot[x], g_plot["mean_amount"])
    plt.xticks(rotation=45, ha="right")
    plt.xlabel(x)
    plt.ylabel("Mean Claim Amount")
    plt.title(title)
    plt.tight_layout()
    plt.show()

    # Print table for interpretation
    print(g_plot.to_string(index=False))


# In[23]:


# ============================================================
# Run Severity EDA BY coverage
# ============================================================
for cov in coverages:
    d_cov = df[df["coverage"] == cov].copy()
    d_cov = d_cov[d_cov["amount"] > 0]

    print("\n" + "=" * 80)
    print(f"Coverage: {cov} | severity n = {len(d_cov)}")
    print("=" * 80)

    # ---- Continuous predictors ----
    for x in continuous_vars:
        plot_binned_mean_severity(
            d_cov,
            x=x,
            bins=10,
            title=f"[{cov}] Claim Amount vs {x} (binned, raw scale)"
        )

    # ---- Categorical predictors ----
    for x in categorical_vars:
        plot_categorical_severity(
            d_cov,
            x=x,
            top_k=15,
            title=f"[{cov}] Mean Claim Amount by {x}"
        )


# ### Step 5: Check correlation between continuous predictors (flag high correlation >0.7)

# In[24]:


# 1. Identify continuous predictors
continuous_vars = [c for c in ["gpa", "distance_to_campus"] if c in df_clean.columns]

print("Continuous predictors used for correlation check:")
print(continuous_vars)

# 2. Correlation matrix (pairwise complete observations)
corr_matrix = (
    df_clean[continuous_vars]
    .corr(method="pearson")
)

print("\nCorrelation matrix:")
print(corr_matrix.round(3))

# 3. Flag high correlations
threshold = 0.7
high_corr_pairs = []

for i in range(len(continuous_vars)):
    for j in range(i + 1, len(continuous_vars)):
        v1 = continuous_vars[i]
        v2 = continuous_vars[j]
        corr_val = corr_matrix.loc[v1, v2]
        if abs(corr_val) > threshold:
            high_corr_pairs.append((v1, v2, corr_val))

# 4. Report results
if high_corr_pairs:
    print("\n⚠️ High correlation pairs (|corr| > 0.7):")
    for v1, v2, c in high_corr_pairs:
        print(f"  {v1} vs {v2}: corr = {c:.3f}")
else:
    print("\n✅ No continuous predictor pairs with |corr| > 0.7")


# ### Step 6: Summarize categorical variables (counts per level, mean target by level)

# In[25]:


categorical_vars = [
    "gender",
    "class",
    "study",
    "greek",
    "off_campus",
    "sprinklered"
]

categorical_vars = [c for c in categorical_vars if c in df_clean.columns]

print("Categorical variables included:")
print(categorical_vars)


# -------------------------
# Loop by coverage
# -------------------------
for cov in df_clean["coverage"].dropna().unique():
    df_cov = df_clean[df_clean["coverage"] == cov]

    print("\n" + "=" * 90)
    print(f"Coverage: {cov}")
    print("=" * 90)

    for var in categorical_vars:
        print(f"\n--- {var} ---")

        # Frequency summary
        freq_summary = (
            df_cov
            .groupby(var)
            .agg(
                n_obs=("has_claim", "size"),
                claim_rate=("has_claim", "mean")
            )
            .reset_index()
            .sort_values("n_obs", ascending=False)
        )

        print("\nFrequency (claim rate):")
        print(freq_summary.to_string(index=False))

        # Severity summary (conditional on claim)
        sev_df = df_cov[df_cov["amount"] > 0]

        if sev_df.empty:
            print("\nSeverity: no claims for this coverage.")
            continue

        sev_summary = (
            sev_df
            .groupby(var)
            .agg(
                n_claims=("amount", "size"),
                mean_amount=("amount", "mean"),
                median_amount=("amount", "median")
            )
            .reset_index()
            .sort_values("n_claims", ascending=False)
        )

        print("\nSeverity (conditional on claim):")
        print(sev_summary.to_string(index=False))


# ### Step 7: Assumptions check for GLM

# In[26]:


df = df_clean.copy()

# -----------------------------
# 0) Build FREQUENCY target as COUNT per (student_id, coverage)
#    This is the right target for Poisson/NB GLMs (not claim_id itself).
# -----------------------------
if "claim_id" not in df.columns:
    raise KeyError("df_clean must contain 'claim_id' to build claim counts.")

df["has_claim_row"] = df["claim_id"].astype(float) > 0

freq = (
    df.groupby(["student_id", "coverage"], as_index=False)["has_claim_row"]
      .sum()
      .rename(columns={"has_claim_row": "claim_count"})
)

# Merge predictors at student_id-coverage level (take first non-null; most are constant per student)
# Keep only columns you might later use in GLM (edit as needed)
predictor_cols = [c for c in [
    "gpa", "distance_to_campus",
    "gender", "class", "study", "greek", "off_campus", "sprinklered"
] if c in df.columns]

pred = (
    df.groupby(["student_id", "coverage"], as_index=False)[predictor_cols]
      .first()
)

freq = freq.merge(pred, on=["student_id", "coverage"], how="left")

# -----------------------------
# Helper: Poisson overdispersion diagnostics
# -----------------------------
def poisson_overdispersion(y, X):
    """
    Fit Poisson GLM and return dispersion metrics:
      - Pearson chi2 / df_resid
      - Deviance / df_resid
    """
    model = sm.GLM(y, X, family=sm.families.Poisson())
    res = model.fit()
    df_resid = res.df_resid
    pearson_disp = res.pearson_chi2 / df_resid if df_resid > 0 else np.nan
    deviance_disp = res.deviance / df_resid if df_resid > 0 else np.nan
    return res, pearson_disp, deviance_disp

# -----------------------------
# Helper: Severity variance vs mean (Gamma-like behavior)
#   - bin a predictor and compute mean(amount), var(amount)
# -----------------------------
def plot_severity_mean_var_by_bins(data_pos, x, bins=10, title=None):
    d = data_pos[[x, "amount"]].dropna().copy()
    if d.empty or d[x].nunique() < 3:
        print(f"[skip] {x}: not enough data/variation for mean-variance check")
        return

    try:
        d["bin"] = pd.qcut(d[x], q=bins, duplicates="drop")
    except ValueError:
        d["bin"] = pd.cut(d[x], bins=min(bins, d[x].nunique()))

    g = (d.groupby("bin", observed=True)
           .agg(
               n=("amount", "size"),
               mean_amt=("amount", "mean"),
               var_amt=("amount", lambda s: np.var(s, ddof=1) if len(s) > 1 else np.nan)
           )
           .reset_index())

    # scatter mean vs variance
    plt.figure()
    plt.scatter(g["mean_amt"], g["var_amt"])
    plt.xlabel("Mean(claim amount) in bin")
    plt.ylabel("Var(claim amount) in bin")
    plt.title(title if title else f"Severity mean-variance by {x} bins")
    plt.grid(alpha=0.3)
    plt.show()

# ============================================================
# Run diagnostics BY coverage
# ============================================================
coverages = sorted(freq["coverage"].dropna().unique())

freq_diag_rows = []
sev_diag_rows = []

for cov in coverages:
    d_cov = freq[freq["coverage"] == cov].copy()
    y = d_cov["claim_count"].astype(int).values

    print("\n" + "=" * 90)
    print(f"Coverage: {cov} (Frequency diagnostics) | n={len(d_cov)}")
    print("=" * 90)

    # ---- A1) Basic frequency distribution checks ----
    mean_y = np.mean(y)
    var_y = np.var(y, ddof=1) if len(y) > 1 else np.nan
    zero_rate = np.mean(y == 0)
    one_rate = np.mean(y == 1)
    ge2_rate = np.mean(y >= 2)

    print(f"Mean(claim_count): {mean_y:.4f}")
    print(f"Var(claim_count):  {var_y:.4f}")
    print(f"Var/Mean:          {(var_y/mean_y) if mean_y>0 else np.nan:.4f}")
    print(f"Zero rate P(Y=0):   {zero_rate:.4f}")
    print(f"P(Y=1):            {one_rate:.4f}")
    print(f"P(Y>=2):           {ge2_rate:.4f}")

    # ---- A2) Overdispersion check via intercept-only Poisson ----
    X0 = np.ones((len(y), 1))  # intercept only
    res0, pearson0, dev0 = poisson_overdispersion(y, X0)

    print("\nIntercept-only Poisson dispersion:")
    print(f"  Pearson chi2 / df:  {pearson0:.3f}")
    print(f"  Deviance / df:      {dev0:.3f}")

    # Expected zero rate under Poisson with lambda = mean_y
    pois_zero_expected = np.exp(-mean_y) if mean_y >= 0 else np.nan
    print(f"  Observed P(Y=0):    {zero_rate:.3f}")
    print(f"  Poisson E[P(Y=0)]:  {pois_zero_expected:.3f}")

    freq_diag_rows.append({
        "coverage": cov,
        "n": len(y),
        "mean": mean_y,
        "var": var_y,
        "var_to_mean": (var_y/mean_y) if mean_y > 0 else np.nan,
        "zero_rate": zero_rate,
        "poisson_zero_expected": pois_zero_expected,
        "pearson_disp_intercept_only": pearson0,
        "deviance_disp_intercept_only": dev0
    })

    # Plot frequency distribution (counts)
    plt.figure()
    # show up to, say, 6+ grouped as 6
    y_plot = np.clip(y, 0, 6)
    plt.hist(y_plot, bins=np.arange(-0.5, 7.5, 1))
    plt.xticks(range(0, 7), ["0", "1", "2", "3", "4", "5", "6+"])
    plt.xlabel("Claim count (capped at 6+ for display)")
    plt.ylabel("Number of student-coverage records")
    plt.title(f"[{cov}] Claim count distribution (student-level)")
    plt.grid(alpha=0.3)
    plt.show()

    # ============================================================
    # Severity diagnostics (positive amounts only) BY coverage
    # ============================================================
    sev_cov = df[(df["coverage"] == cov) & (pd.to_numeric(df["amount"], errors="coerce") > 0)].copy()
    sev_cov["amount"] = pd.to_numeric(sev_cov["amount"], errors="coerce")
    sev_cov = sev_cov[sev_cov["amount"] > 0]

    print("\n" + "-" * 90)
    print(f"Coverage: {cov} (Severity diagnostics) | n_pos={len(sev_cov)}")
    print("-" * 90)

    if sev_cov.empty:
        print("No positive severities for this coverage.")
        continue

    amt = sev_cov["amount"].values
    desc = pd.Series(amt).describe(percentiles=[0.5, 0.75, 0.9, 0.95, 0.99])
    print(desc)

    # Skew/kurtosis (heavy-tail indicators)
    skew = pd.Series(amt).skew()
    kurt = pd.Series(amt).kurt()
    print(f"Skewness:  {skew:.3f}")
    print(f"Kurtosis:  {kurt:.3f}")

    sev_diag_rows.append({
        "coverage": cov,
        "n_pos": len(amt),
        "mean": float(np.mean(amt)),
        "median": float(np.median(amt)),
        "p90": float(np.quantile(amt, 0.90)),
        "p99": float(np.quantile(amt, 0.99)),
        "max": float(np.max(amt)),
        "skew": float(skew),
        "kurtosis": float(kurt),
    })

    # Plot raw severity distribution
    plt.figure()
    plt.hist(amt, bins=50)
    plt.xlabel("Claim amount (raw)")
    plt.ylabel("Frequency")
    plt.title(f"[{cov}] Severity distribution (raw scale, amount>0)")
    plt.grid(alpha=0.3)
    plt.show()

    # Optional: variance-vs-mean check using bins of a continuous predictor (if available)
    for x in [c for c in ["gpa", "distance_to_campus"] if c in sev_cov.columns]:
        plot_severity_mean_var_by_bins(
            sev_cov, x=x, bins=10,
            title=f"[{cov}] Severity mean-variance by {x} bins (raw)"
        )

# ============================================================
# Summary tables (nice to paste into report)
# ============================================================
freq_diag = pd.DataFrame(freq_diag_rows)
sev_diag = pd.DataFrame(sev_diag_rows)

print("\n" + "=" * 90)
print("FREQUENCY DIAGNOSTICS SUMMARY (BY coverage)")
print("=" * 90)
print(freq_diag.to_string(index=False))

print("\n" + "=" * 90)
print("SEVERITY DIAGNOSTICS SUMMARY (BY coverage)")
print("=" * 90)
print(sev_diag.to_string(index=False))

print("\nInterpretation:")
print("- Overdispersion signals: var_to_mean >> 1 and/or dispersion (Pearson/Deviance) >> 1.")
print("- Excess zeros signal: observed zero_rate >> exp(-mean).")
print("- Severity heavy-tail signals: high skewness/kurtosis, p99 >> median, long right tail in histogram.")


# **Claim Frequency (Count Model Assumptions)**
# 
# 1. The variance of claim counts consistently exceeds the mean across coverages, indicating overdispersion relative to the Poisson assumption.
# 
# 2. Dispersion diagnostics from intercept-only Poisson GLMs (Pearson and deviance dispersion statistics) are greater than 1, further confirming that the equi-dispersion assumption of Poisson models is violated.
# 
# 3. The observed proportion of zero claims is substantially higher than the zero probability implied by a Poisson distribution with the same mean, suggesting the presence of excess zeros.
# 
# Conclusion:
# 
# The standard Poisson GLM is unlikely to adequately capture claim frequency variability. Overdispersion and excess zeros must be accounted for in subsequent frequency modeling.

# **Claim Severity (Positive Amount Model Assumptions)**
# 
# 1. The distribution of positive claim amounts is highly right-skewed, with extreme values in the upper tail.
# 
# 2. Measures of skewness and kurtosis are large, indicating strong deviation from normality.
# 
# 3. Variance increases with the mean across subsets of the data, which is inconsistent with constant-variance assumptions.
# 
# Conclusion:
# 
# Gaussian assumptions for claim severity are inappropriate. Severity modeling requires distributions that can accommodate heavy-tailed behavior and mean–variance dependence.

# # Data Modelling (Frequency)

# In[27]:


# ------------------------------------------------------------
# Create a binary claim indicator
# claim_id == 0  -> no claim
# claim_id != 0  -> one claim occurrence
# ------------------------------------------------------------
df["has_claim"] = (df["claim_id"] != 0).astype(int)

# ------------------------------------------------------------
# Aggregate to claim count by student_id × coverage
# Each row represents one exposure unit (exposure = 1)
# ------------------------------------------------------------
group_cols = ["student_id", "coverage"]

# Variables that are constant within each student-coverage unit
static_features = [
    "class",
    "study",
    "gpa",
    "greek",
    "off_campus",
    "distance_to_campus",
    "gender",
    "sprinklered",
    "risk_tier",
    "holdout"
]

freq_df = (
    df
    .groupby(group_cols, as_index=False)
    .agg(
        # Frequency target: total number of claims
        claim_count=("has_claim", "sum"),

        # Carry forward static covariates
        **{col: (col, "first") for col in static_features}
    )
)

# ------------------------------------------------------------
# freq_df is now the modeling dataset for frequency models
# Target variable: claim_count
# Exposure: implicitly equal to 1 for all observations
# ------------------------------------------------------------

freq_df.head()


# In[28]:


def frequency_diagnostics_by_coverage(df):
    """
    Compute basic frequency diagnostics for claim counts
    separately by coverage.

    Diagnostics include:
    - Mean and variance of claim counts
    - Variance-to-mean ratio
    - Observed zero rate
    - Expected zero probability under Poisson
    - Dispersion statistics from intercept-only Poisson model
    """

    results = []

    for coverage, d in df.groupby("coverage"):
        y = d["claim_count"].values
        n = len(y)

        mean_y = y.mean()
        var_y = y.var(ddof=1)
        var_to_mean = var_y / mean_y if mean_y > 0 else np.nan

        # Observed proportion of zero counts
        zero_rate = np.mean(y == 0)

        # Expected zero probability under Poisson assumption
        poisson_zero_expected = np.exp(-mean_y)

        # ----------------------------------------------------
        # Intercept-only Poisson model for dispersion checks
        # ----------------------------------------------------
        X_intercept = np.ones((n, 1))
        poisson_model = sm.GLM(y, X_intercept, family=sm.families.Poisson())
        poisson_res = poisson_model.fit()

        # Pearson and deviance dispersion statistics
        pearson_dispersion = poisson_res.pearson_chi2 / poisson_res.df_resid

        results.append({
            "coverage": coverage,
            "n": n,
            "mean": mean_y,
            "var": var_y,
            "var_to_mean": var_to_mean,
            "zero_rate": zero_rate,
            "poisson_zero_expected": poisson_zero_expected,
            "pearson_disp_intercept_only": pearson_dispersion
        })

    return pd.DataFrame(results)


# ------------------------------------------------------------
# Run diagnostics on frequency dataset
# ------------------------------------------------------------
freq_diagnostics = frequency_diagnostics_by_coverage(freq_df)

# Display diagnostics
freq_diagnostics


# Diagnostics suggest Poisson is adequate at the aggregate level, but alternative count models are explored to account for potential latent heterogeneity and coverage-specific loss mechanisms. Generally, no overdispersions for claim count!

# **Now, for each coverage, we try Poisson Model because no overdispersion here.**

# ## Model with respect to Personal Property Protection

# In[29]:


# ------------------------------------------------------------
# Filter frequency dataset for Personal Property Protection
# ------------------------------------------------------------
ppp_df = freq_df[freq_df["coverage"] == "Personal Property"].copy()

# Separate training and holdout samples
ppp_train = ppp_df[ppp_df["holdout"] == False].copy()
ppp_test = ppp_df[ppp_df["holdout"] == True].copy()


# In[30]:


# Capping the claim_count, so that we have a more stable and robust model
cap_value = int(ppp_train["claim_count"].quantile(0.975))

ppp_train["claim_count_capped"] = ppp_train["claim_count"].clip(upper=cap_value)

print("Chosen cap value:", cap_value)
print(ppp_train[["claim_count", "claim_count_capped"]].describe())


# In[31]:


(ppp_train["claim_count"] > cap_value).mean()


# In[32]:


ppp_train["claim_count"].mean(), ppp_train["claim_count_capped"].mean()


# In[33]:


formula = """
claim_count_capped ~
C(Q("class")) +
C(study) +
C(greek) +
C(off_campus) +
C(gender) +
C(sprinklered) +
C(risk_tier) +
gpa +
distance_to_campus
"""

poisson_model = smf.glm(
    formula=formula,
    data=ppp_train,
    family=sm.families.Poisson()
)

poisson_results = poisson_model.fit()
print(poisson_results.summary())


# ### Variable Selection

# **Do inclusions of certain variable in a rating plan conform the acturial standards of practice and regulatory requirements?**

# Some candidate variables raise concerns regarding compliance with actuarial standards of practice and regulatory requirements. In particular, **GPA and gender** are considered potentially discriminatory variables.
# Gender is generally regarded as a protected characteristic under regulatory frameworks, while GPA may serve as a proxy for socioeconomic background or other protected attributes. As a result, despite any observed statistical significance, these variables were excluded from the rating plan to ensure fairness, transparency, and regulatory compliance.

# **Can the electronic quotation system be easily modified to handle the inclusion of relevant variables in the rating formula?**

# Yes, the electronic quotation system can be readily modified to incorporate the selected variables. Variables such as **risk tier and sprinklered** are already available in the existing database. Specifically, risk tier is derived from historical scoring information and can be directly extracted without additional system changes, while sprinklered status is a standard property attribute stored in the database. As a result, the inclusion of these variables does not require significant modifications to the quotation infrastructure.

# **Are the relevant variables cost-effective to collect the value when writing new and renewal business?**

# Yes, the selected variables are cost-effective to collect for both new and renewal business. Variables such as **study, class, and off-campus status** can be obtained through standard application forms prior to policy issuance, requiring minimal additional effort from policyholders. Since these variables are either already collected during the underwriting process or easily accessible at the point of sale, their inclusion does not impose significant administrative or operational costs.

# **Now, we have our new model!**

# In[34]:


formula = """
claim_count_capped ~
C(Q("class")) +
C(study) +
C(greek) +
C(off_campus) +
C(sprinklered) +
C(risk_tier) +
distance_to_campus
"""

poisson_model = smf.glm(
    formula=formula,
    data=ppp_train,
    family=sm.families.Poisson()
)

poisson_results = poisson_model.fit()
print(poisson_results.summary())


# ### Let's detect non-linearity with partial residual plots

# We diagnose the variable **distance_to_campus** because it is the only continuous variable in our current model. Partial residual plots don't apply to categorical variables.

# In[35]:


# Extract fitted values and residual components
mu_hat = poisson_results.mu                 # fitted mean μ̂
y = ppp_train["claim_count_capped"]         # response
beta_dist = poisson_results.params["distance_to_campus"]

# Partial residual for distance
ppp_train["pr_distance"] = (y - mu_hat) + beta_dist * ppp_train["distance_to_campus"]


# In[36]:


from statsmodels.nonparametric.smoothers_lowess import lowess

x = ppp_train["distance_to_campus"]
pr = ppp_train["pr_distance"]

# LOWESS smooth
lowess_fit = lowess(pr, x, frac=0.3)

plt.figure(figsize=(7, 5))
plt.scatter(x, pr, alpha=0.3, s=15, label="Partial residuals")
plt.plot(lowess_fit[:, 0], lowess_fit[:, 1], color="red", linewidth=2, label="LOWESS")

plt.xlabel("Distance to campus")
plt.ylabel("Partial residual")
plt.title("Partial Residual Plot: Distance to Campus")
plt.legend()
plt.show()


# Since there are two trends here in the partial residual plot, so it looks like the behavior of categorical variable. So we consider to bin this continuous predictor.

# In[37]:


# Create quantile-based bins
ppp_train["distance_bin"] = pd.qcut(
    ppp_train["distance_to_campus"],
    q=5,                 # try 4–6; 5 is a good default
    duplicates="drop"
)

# Check bin counts
ppp_train["distance_bin"].value_counts().sort_index()


# In[38]:


formula_bin = """
claim_count_capped ~
C(Q("class")) +
C(study) +
C(greek) +
C(off_campus) +
C(sprinklered) +
C(risk_tier) +
C(distance_bin)
"""

poisson_bin = smf.glm(
    formula=formula_bin,
    data=ppp_train,
    family=sm.families.Poisson()
).fit()

print(poisson_bin.summary())


# In[39]:


print("Linear AIC:", poisson_results.aic)
print("Binned AIC:", poisson_bin.aic)


# **Distance to campus exhibits a highly skewed distribution with a large mass near zero. Quantile-based binning results in only two effective distance groups. Comparing linear and binned specifications yields nearly identical AIC values, indicating no meaningful improvement from discretization. To avoid some random noise, we therefore retain a linear specification for distance to campus.**

# In[40]:


formula = """
claim_count_capped ~
C(Q("class")) +
C(study) +
C(greek) +
C(off_campus) +
C(sprinklered) +
C(risk_tier) +
distance_to_campus
"""

poisson_model = smf.glm(
    formula=formula,
    data=ppp_train,
    family=sm.families.Poisson()
)

poisson_results = poisson_model.fit()
print(poisson_results.summary())


# ### Interactions?

# Greek affiliation and off-campus status were identified as important predictors of claim frequency. Rather than including their interaction solely based on statistical significance, we considered an interaction term to capture potential differences in how off-campus residence affects risk across Greek and non-Greek students. Due to distinct social and exposure patterns, the impact of off-campus status is expected to vary by Greek affiliation, making the interaction term actuarially meaningful.

# In[41]:


formula = """
claim_count_capped ~
C(Q("class")) +
C(study) +
C(greek) * C(off_campus)+
C(sprinklered) +
C(risk_tier) +
distance_to_campus
"""

poisson_model_interaction_1 = smf.glm(
    formula=formula,
    data=ppp_train,
    family=sm.families.Poisson()
)

poisson_results_interaction_1 = poisson_model_interaction_1.fit()
print(poisson_results_interaction_1.summary())


# In[42]:


formula = """
claim_count_capped ~
C(Q("class")) +
C(study) +
C(greek) * distance_to_campus +
C(off_campus) +
C(sprinklered) +
C(risk_tier)
"""

poisson_model_interaction_2 = smf.glm(
    formula=formula,
    data=ppp_train,
    family=sm.families.Poisson()
)

poisson_results_interaction_2 = poisson_model_interaction_2.fit()
print(poisson_results_interaction_2.summary())


# In[43]:


formula = """
claim_count_capped ~
C(Q("class")) +
C(study) +
C(greek) * C(off_campus) +
C(sprinklered) +
C(risk_tier) +
C(greek) * distance_to_campus
"""

poisson_model_interaction_3 = smf.glm(
    formula=formula,
    data=ppp_train,
    family=sm.families.Poisson()
)

poisson_results_interaction_3 = poisson_model_interaction_3.fit()
print(poisson_results_interaction_3.summary())


# In[44]:


print("Base model AIC:        ", poisson_results.aic)
print("Greek × Off-campus AIC:", poisson_results_interaction_1.aic)
print("Greek × Distance AIC:  ", poisson_results_interaction_2.aic)
print("Both interactions AIC: ", poisson_results_interaction_3.aic)


# Additional interaction terms were considered; however, only interactions with clear actuarial and business justification were evaluated. These interactions did not result in material improvements in model fit or interpretability. Therefore, no interaction terms were retained in the final model to preserve parsimony and rate stability. **Retaining our poisson_model!**

# ## Model Refinement

# In[45]:


influence = poisson_results.get_influence()
cooks_d = influence.cooks_distance[0]

# quick summary
print("Max Cook's D:", cooks_d.max())
print("Mean Cook's D:", cooks_d.mean())

# rule-of-thumb threshold
threshold = 4 / len(ppp_train)
print("Threshold:", threshold)

# count influential points
print("Num influential:", (cooks_d > threshold).sum())


# Since the largest Cook's distance is 0.0044 which is very small, meaning these influecnial points had very small influences on the estimate of the respones!

# In[46]:


df_diag = ppp_train.copy()
df_diag["deviance_resid"] = poisson_results.resid_deviance
df_diag["fitted"] = poisson_results.fittedvalues


# In[47]:


# residuals by key categorical variable
df_diag.groupby("greek")["deviance_resid"].mean()
df_diag.groupby("off_campus")["deviance_resid"].mean()


# In[48]:


pearson_chi2 = poisson_results.pearson_chi2
df_resid = poisson_results.df_resid
print("Pearson chi2 / df:", pearson_chi2 / df_resid)


# In[49]:


dev_resid = df_diag["deviance_resid"]
plt.figure(figsize=(8, 5))
plt.hist(dev_resid, bins=40, density=True, alpha=0.7)
plt.axvline(0, color="red", linestyle="--")
plt.xlabel("Deviance residual")
plt.ylabel("Density")
plt.title("Density of Deviance Residuals")
plt.show()


# The right-skewness observed in the deviance residuals is expected due to the discrete and capped nature of the response variable. Since the claim count was capped at 1, the response effectively becomes binary, leading to asymmetric residual behavior under a Poisson GLM. This skewness reflects the imbalance between zero and one outcomes rather than model misspecification. Given the discrete nature of the response, strict normality of deviance residuals is not expected.

# In[50]:


formula_main = """
claim_count_capped ~
C(Q("class")) +
C(study) +
C(greek) +
C(off_campus) +
C(sprinklered) +
C(risk_tier) +
distance_to_campus
"""

B = 500 
coef_list = []
fail = 0

for b in range(B):
    sample = ppp_train.sample(frac=1, replace=True)

    try:
        m = smf.glm(
            formula=formula_main,
            data=sample,
            family=sm.families.Poisson()
        ).fit()

        coef_list.append(m.params)

    except Exception:
        fail += 1
        continue

boot_df = pd.DataFrame(coef_list)

print("Bootstrap fits:", len(boot_df), "Failed:", fail)

# summary: mean/std + 95% percentile CI
summary = pd.DataFrame({
    "mean": boot_df.mean(),
    "std": boot_df.std(),
    "p2.5": boot_df.quantile(0.025),
    "p97.5": boot_df.quantile(0.975),
})

# sign stability (how often coefficient > 0)
summary["Pr(>0)"] = (boot_df > 0).mean()

summary.sort_values("std", ascending=False)


# Bootstrap resampling was used to assess parameter stability. Key main effects such as Greek affiliation and off-campus status exhibited consistent direction and magnitude across resamples. In contrast, interaction terms showed substantial variability and lacked stability, indicating that they are not structurally meaningful. These findings further support retaining a parsimonious main-effects model.

# In[51]:


formula = """
claim_count_capped ~
C(Q("class")) +
C(study) +
C(greek) +
C(off_campus) +
C(sprinklered) +
C(risk_tier) +
distance_to_campus
"""

logistic_model = smf.glm(
    formula=formula,
    data=ppp_train,
    family=sm.families.Binomial()
)

logistic_results = logistic_model.fit()
print(logistic_results.summary())


# In[52]:


print("Poisson model:", poisson_results.aic)
print("Logistic regression:", logistic_results.aic)


# Logistic regression did not make any difference between poisson regression, since their AIC are almost the same!

# In[53]:


from statsmodels.stats.outliers_influence import variance_inflation_factor

# Build design matrix X from formula (drop intercept for VIF)
y, X = patsy.dmatrices(formula_main, data=ppp_train, return_type="dataframe")

# drop intercept if present
if "Intercept" in X.columns:
    X_vif = X.drop(columns=["Intercept"])
else:
    X_vif = X.copy()

vif_df = pd.DataFrame({
    "variable": X_vif.columns,
    "VIF": [variance_inflation_factor(X_vif.values, i) for i in range(X_vif.shape[1])]
}).sort_values("VIF", ascending=False)

vif_df


# The infinite VIF values observed for certain categorical predictors arise from dummy-variable encoding （structural issue).
# Dummy variables corresponding to the same categorical factor are linearly dependent by construction, which makes the computation of VIF for individual dummy levels not meaningful.
# 
# Importantly, no evidence of problematic multicollinearity is found among the continuous or binary predictors, as their VIF values remain well below conventional cutoffs (e.g. VIF > 10).
# Therefore, multicollinearity is not a concern for this model.

# In[54]:


# Choose your fitted poisson results object here:
res = poisson_results

df_work = ppp_train.copy()
df_work["mu_hat"] = res.fittedvalues
df_work["y"] = df_work["claim_count_capped"]

# working residuals for Poisson
df_work["working_resid"] = (df_work["y"] - df_work["mu_hat"]) / np.sqrt(df_work["mu_hat"].clip(lower=1e-12))

# bin by fitted mean
n_bins = 20
df_work["bin"] = pd.qcut(df_work["mu_hat"], q=n_bins, duplicates="drop")

bin_stats = df_work.groupby("bin").agg(
    mu_hat_mean=("mu_hat", "mean"),
    wr_mean=("working_resid", "mean"),
    wr_std=("working_resid", "std"),
    n=("working_resid", "size")
).reset_index(drop=True)

# standard error for mean
bin_stats["wr_se"] = bin_stats["wr_std"] / np.sqrt(bin_stats["n"])

plt.figure(figsize=(8,5))
plt.errorbar(bin_stats["mu_hat_mean"], bin_stats["wr_mean"], yerr=1.96*bin_stats["wr_se"], fmt="o")
plt.axhline(0, color="red", linestyle="--")
plt.xlabel("Mean fitted value (binned)")
plt.ylabel("Mean working residual (± 95% CI)")
plt.title("Working Residuals by Fitted Value Bins")
plt.show()


# The binned working residual plot indicates that the residuals are centered around zero across the range of fitted values, suggesting no systematic bias in the model predictions.
# Additionally, there is no evidence of fanning out, implying that the variance structure assumed by the Poisson model is appropriate.
# 
# Overall, the working residual analysis supports the adequacy of the model’s functional form and variance assumptions.

# In[55]:


import scipy.stats as st

res = poisson_results

y_obs = ppp_train["claim_count_capped"].to_numpy()
mu_hat = res.fittedvalues.to_numpy()

# Poisson CDF at y and y-1
F_y = st.poisson.cdf(y_obs, mu_hat)
F_y_minus = st.poisson.cdf(y_obs - 1, mu_hat)  # for y=0, cdf(-1)=0 automatically

# randomized u in [F(y-1), F(y)]
rng = np.random.default_rng(123)
u = F_y_minus + rng.random(len(y_obs)) * (F_y - F_y_minus)

# avoid 0/1 for numerical stability
eps = 1e-12
u = np.clip(u, eps, 1 - eps)

rqr = st.norm.ppf(u)

# QQ plot
plt.figure(figsize=(6,6))
st.probplot(rqr, dist="norm", plot=plt)
plt.title("QQ Plot of Randomized Quantile Residuals")
plt.xlabel("Theoretical Quantiles")
plt.ylabel("Sample Quantiles")
plt.show()


# The QQ plot of randomized quantile residuals shows good agreement with the standard normal distribution in the central region, indicating an overall satisfactory model fit.
# 
# Minor deviations in the tails are observed, which are expected given the discrete and capped nature of the response variable.
# Such tail behavior does not suggest systematic model misspecification and is commonly observed in count data models.
# 
# Therefore, the randomized quantile residual analysis supports the adequacy of the Poisson regression model.

# ## Model Validation

# In[56]:


# test set prediction
ppp_test["mu_hat"] = poisson_results.predict(ppp_test)


# In[57]:


# Assessing Fit with Plots of Actual vs. Predicted
def actual_vs_predicted_plot(
    df,
    model,
    y_col="claim_count",
    n_bins=20,
    title="Actual vs Predicted"
):
    df = df.copy()

    # 1. Predict
    df["mu_hat"] = model.predict(df)

    # 2. Sort by prediction
    df = df.sort_values("mu_hat")

    # 3. Create quantile bins
    df["bin"] = pd.qcut(df["mu_hat"], q=n_bins, duplicates="drop")

    # 4. Aggregate
    agg = df.groupby("bin").agg(
        actual_mean=(y_col, "mean"),
        pred_mean=("mu_hat", "mean"),
        count=(y_col, "size")
    ).reset_index()

    # 5. Plot
    plt.figure(figsize=(6, 6))
    plt.scatter(agg["pred_mean"], agg["actual_mean"])
    max_val = max(agg["pred_mean"].max(), agg["actual_mean"].max())
    plt.plot([0, max_val], [0, max_val], "r--", label="Perfect fit")

    plt.xlabel("Mean Predicted Claim Count")
    plt.ylabel("Mean Actual Claim Count")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()

    return agg


# In[58]:


agg_train = actual_vs_predicted_plot(
    df=ppp_train,
    model=poisson_results,
    y_col="claim_count",
    n_bins=20,
    title="Training Set: Actual vs Predicted"
)


# On the training set, the aggregated actual claim counts closely track the predicted values across quantiles, indicating that the Poisson model provides a good in-sample fit without obvious systematic bias.

# In[59]:


agg_test = actual_vs_predicted_plot(
    df=ppp_test,
    model=poisson_results,
    y_col="claim_count",
    n_bins=20,
    title="Test Set: Actual vs Predicted"
)


# When evaluated on the holdout (test) set, the actual versus predicted plot remains close to the 45-degree line, suggesting that the model generalizes well to unseen data and does not suffer from material overfitting.

# In[60]:


ppp_test = ppp_test.sort_values("mu_hat").reset_index(drop=True)

ppp_test["decile"] = pd.qcut(
    ppp_test["mu_hat"],
    q=10,
    labels=False
)

decile_summary = (
    ppp_test
    .groupby("decile")
    .agg(
        mean_predicted=("mu_hat", "mean"),
        mean_actual=("claim_count", "mean"),
        count=("claim_count", "size")
    )
    .reset_index()
)

decile_summary


# In[61]:


plt.figure(figsize=(8, 6))

plt.plot(
    decile_summary["decile"],
    decile_summary["mean_actual"],
    marker="o",
    label="Actual"
)

plt.plot(
    decile_summary["decile"],
    decile_summary["mean_predicted"],
    marker="o",
    linestyle="--",
    label="Predicted"
)

plt.xlabel("Decile (low risk → high risk)")
plt.ylabel("Mean Claim Frequency")
plt.title("Simple Decile Plot (Test Set)")
plt.legend()
plt.grid(True)

plt.show()


# A simple decile plot was constructed on the test set by ranking observations based on predicted claim frequency.
# 
# The predicted frequencies increase monotonically across deciles, demonstrating strong risk differentiation. Observed claim frequencies generally follow the predicted trend, with some variability in the lower deciles due to the low claim rate and inherent randomness in claim occurrence.
# 
# In the highest deciles, observed frequencies slightly exceed predicted values, indicating mild underestimation at the upper tail. Overall, the model exhibits satisfactory calibration and stable performance on unseen data.

# ### Loss Ratio Chart

# In[62]:


# ---------- Decile assignment ----------
ppp_test = ppp_test.copy()
ppp_test["decile"] = pd.qcut(
    ppp_test["mu_hat"],
    q=10,
    labels=False,
    duplicates="drop"
)

# ---------- Aggregate by decile ----------
lr_df = (
    ppp_test
    .groupby("decile")
    .agg(
        actual=("claim_count", "mean"),
        predicted=("mu_hat", "mean")
    )
    .reset_index()
)

lr_df["loss_ratio"] = lr_df["actual"] / lr_df["predicted"]


# In[63]:


plt.figure(figsize=(8, 5))

plt.plot(
    lr_df["decile"],
    lr_df["loss_ratio"],
    marker="o",
    linewidth=2
)

plt.axhline(
    y=1.0,
    color="red",
    linestyle="--",
    label="Perfect calibration (LR = 1)"
)

plt.xlabel("Decile (low risk → high risk)")
plt.ylabel("Loss Ratio (Actual / Predicted)")
plt.title("Loss Ratio by Decile (Test Set)")
plt.legend()
plt.grid(True)

plt.show()


# The loss ratio by decile on the test set fluctuates around 1 with no systematic trend, indicating good overall calibration. While some moderate-risk deciles exhibit mild underprediction, the model remains well calibrated across both low- and high-risk segments.

# ### Gini Index Plot

# In[64]:


def gini_lorenz(df, actual_col, pred_col):
    df = df.sort_values(pred_col).reset_index(drop=True)

    df["cum_actual"] = df[actual_col].cumsum()
    df["cum_actual_share"] = df["cum_actual"] / df["cum_actual"].iloc[-1]

    df["cum_exposure"] = np.arange(1, len(df) + 1)
    df["cum_exposure_share"] = df["cum_exposure"] / len(df)

    # Lorenz area
    area = np.trapz(
        df["cum_actual_share"],
        df["cum_exposure_share"]
    )

    gini = 1 - 2 * area
    return df, gini


# In[65]:


lorenz_df, gini_index = gini_lorenz(
    ppp_test,
    actual_col="claim_count",
    pred_col="mu_hat"
)

gini_index


# In[66]:


plt.figure(figsize=(6, 6))

plt.plot(
    lorenz_df["cum_exposure_share"],
    lorenz_df["cum_actual_share"],
    label="Model Lorenz Curve",
    linewidth=2
)

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    color="gray",
    label="No discrimination"
)

plt.xlabel("Cumulative Share of Exposure")
plt.ylabel("Cumulative Share of Claims")
plt.title(f"Lorenz Curve (Test Set)\nGini Index = {gini_index:.3f}")
plt.legend()
plt.grid(True)

plt.show()


# The Lorenz curve on the test set lies consistently above the line of no discrimination, yielding a Gini index of 0.206. This indicates that the model provides meaningful risk differentiation, particularly given the low-frequency nature of claim counts in this dataset.

# ## Model with respect to Additional Living Expense

# In[67]:


ale_df = freq_df[freq_df["coverage"] == "Additional Living Expense"].copy()

# Separate training and holdout samples
ale_train = ale_df[ale_df["holdout"] == False].copy()
ale_test = ale_df[ale_df["holdout"] == True].copy()


# In[68]:


# Capping the claim_count, so that we have a more stable and robust model
cap_value = int(ale_train["claim_count"].quantile(0.975))

ale_train["claim_count_capped"] = ale_train["claim_count"].clip(upper=cap_value)

print("Chosen cap value:", cap_value)
print(ale_train[["claim_count", "claim_count_capped"]].describe())


# In[69]:


(ale_train["claim_count"] > cap_value).mean()


# In[70]:


ale_train["claim_count"].mean(), ale_train["claim_count_capped"].mean()


# In[71]:


formula = """
claim_count_capped ~
C(Q("class")) +
C(study) +
C(greek) +
C(off_campus) +
C(sprinklered) +
C(risk_tier) +
distance_to_campus
"""

poisson_model_ale = smf.glm(
    formula=formula,
    data=ale_train,
    family=sm.families.Poisson()
)

poisson_results_ale = poisson_model_ale.fit()
print(poisson_results_ale.summary())


# In[72]:


# Extract fitted values and residual components
mu_hat_ale = poisson_results_ale.mu                 # fitted mean μ̂
y = ale_train["claim_count_capped"]         # response
beta_dist_ale = poisson_results_ale.params["distance_to_campus"]

# Partial residual for distance
ale_train["pr_distance"] = (y - mu_hat_ale) + beta_dist_ale * ale_train["distance_to_campus"]


# In[73]:


x = ale_train["distance_to_campus"]
pr = ale_train["pr_distance"]

# LOWESS smooth
lowess_fit = lowess(pr, x, frac=0.3)

plt.figure(figsize=(7, 5))
plt.scatter(x, pr, alpha=0.3, s=15, label="Partial residuals")
plt.plot(lowess_fit[:, 0], lowess_fit[:, 1], color="red", linewidth=2, label="LOWESS")

plt.xlabel("Distance to campus")
plt.ylabel("Partial residual")
plt.title("Partial Residual Plot: Distance to Campus")
plt.legend()
plt.show()


# In[80]:


# Create quantile-based bins
ale_train["distance_bin"] = pd.qcut(
    ale_train["distance_to_campus"],
    q=5,                 # try 4–6; 5 is a good default
    duplicates="drop"
)

# Check bin counts
ale_train["distance_bin"].value_counts().sort_index()


# In[83]:


formula_bin = """
claim_count_capped ~
C(Q("class")) +
C(study) +
C(greek) +
C(off_campus) +
C(sprinklered) +
C(risk_tier) +
C(distance_bin)
"""

poisson_bin_ale = smf.glm(
    formula=formula_bin,
    data=ale_train,
    family=sm.families.Poisson()
).fit()

print(poisson_bin_ale.summary())


# In[84]:


print("Linear AIC:", poisson_results_ale.aic)
print("Binned AIC:", poisson_bin_ale.aic)


# **Distance to campus exhibits a highly skewed distribution with a large mass near zero. Quantile-based binning results in only two effective distance groups. Comparing linear and binned specifications yields nearly identical AIC values, indicating no meaningful improvement from discretization. To avoid some random noise, we therefore retain a linear specification for distance to campus.**

# ### interactions

# In[85]:


formula = """
claim_count_capped ~
C(Q("class")) +
C(study) +
C(greek) * C(off_campus)+
C(sprinklered) +
C(risk_tier) +
distance_to_campus
"""

poisson_model_interaction_1_ale = smf.glm(
    formula=formula,
    data=ale_train,
    family=sm.families.Poisson()
)

poisson_results_interaction_1_ale = poisson_model_interaction_1_ale.fit()
print(poisson_results_interaction_1_ale.summary())


# In[86]:


formula = """
claim_count_capped ~
C(Q("class")) +
C(study) +
C(greek) * distance_to_campus +
C(off_campus) +
C(sprinklered) +
C(risk_tier)
"""

poisson_model_interaction_2_ale = smf.glm(
    formula=formula,
    data=ale_train,
    family=sm.families.Poisson()
)

poisson_results_interaction_2_ale = poisson_model_interaction_2_ale.fit()
print(poisson_results_interaction_2_ale.summary())


# In[87]:


formula = """
claim_count_capped ~
C(Q("class")) +
C(study) +
C(greek) * C(off_campus) +
C(greek) * distance_to_campus +
C(sprinklered) +
C(risk_tier)
"""

poisson_model_interaction_3_ale = smf.glm(
    formula=formula,
    data=ale_train,
    family=sm.families.Poisson()
)

poisson_results_interaction_3_ale = poisson_model_interaction_3_ale.fit()
print(poisson_results_interaction_3_ale.summary())


# In[88]:


print(poisson_results_interaction_2_ale.aic)
print(poisson_results_interaction_1_ale.aic)
print(poisson_results_interaction_3_ale.aic)


# keep the original model!

# ## Model Refinement

# In[89]:


influence_ale = poisson_results_ale.get_influence()
cooks_d = influence_ale.cooks_distance[0]

# quick summary
print("Max Cook's D:", cooks_d.max())
print("Mean Cook's D:", cooks_d.mean())

# rule-of-thumb threshold
threshold_ale = 4 / len(ale_train)
print("Threshold:", threshold)

# count influential points
print("Num influential:", (cooks_d > threshold_ale).sum())


# In[90]:


df_diag = ale_train.copy()
df_diag["deviance_resid"] = poisson_results_ale.resid_deviance
df_diag["fitted"] = poisson_results_ale.fittedvalues


# In[91]:


# residuals by key categorical variable
df_diag.groupby("greek")["deviance_resid"].mean()
df_diag.groupby("off_campus")["deviance_resid"].mean()


# In[92]:


pearson_chi2 = poisson_results_ale.pearson_chi2
df_resid = poisson_results_ale.df_resid
print("Pearson chi2 / df:", pearson_chi2 / df_resid)


# In[93]:


dev_resid = df_diag["deviance_resid"]
plt.figure(figsize=(8, 5))
plt.hist(dev_resid, bins=40, density=True, alpha=0.7)
plt.axvline(0, color="red", linestyle="--")
plt.xlabel("Deviance residual")
plt.ylabel("Density")
plt.title("Density of Deviance Residuals")
plt.show()


# In[94]:


formula_main = """
claim_count_capped ~
C(Q("class")) +
C(study) +
C(greek) +
C(off_campus) +
C(sprinklered) +
C(risk_tier) +
distance_to_campus
"""

B = 500 
coef_list = []
fail = 0

for b in range(B):
    sample = ale_train.sample(frac=1, replace=True)

    try:
        m = smf.glm(
            formula=formula_main,
            data=sample,
            family=sm.families.Poisson()
        ).fit()

        coef_list.append(m.params)

    except Exception:
        fail += 1
        continue

boot_df = pd.DataFrame(coef_list)

print("Bootstrap fits:", len(boot_df), "Failed:", fail)

# summary: mean/std + 95% percentile CI
summary = pd.DataFrame({
    "mean": boot_df.mean(),
    "std": boot_df.std(),
    "p2.5": boot_df.quantile(0.025),
    "p97.5": boot_df.quantile(0.975),
})

# sign stability (how often coefficient > 0)
summary["Pr(>0)"] = (boot_df > 0).mean()

summary.sort_values("std", ascending=False)


# In[95]:


formula = """
claim_count_capped ~
C(Q("class")) +
C(study) +
C(greek) +
C(off_campus) +
C(sprinklered) +
C(risk_tier) +
distance_to_campus
"""

logistic_model_ale = smf.glm(
    formula=formula,
    data=ale_train,
    family=sm.families.Binomial()
)

logistic_results_ale = logistic_model_ale.fit()
print(logistic_results_ale.summary())


# In[96]:


print("Poisson model:", poisson_results_ale.aic)
print("Logistic regression:", logistic_results_ale.aic)


# Possion model wins!

# In[97]:


# Build design matrix X from formula (drop intercept for VIF)
y, X = patsy.dmatrices(formula, data=ale_train, return_type="dataframe")

# drop intercept if present
if "Intercept" in X.columns:
    X_vif = X.drop(columns=["Intercept"])
else:
    X_vif = X.copy()

vif_df = pd.DataFrame({
    "variable": X_vif.columns,
    "VIF": [variance_inflation_factor(X_vif.values, i) for i in range(X_vif.shape[1])]
}).sort_values("VIF", ascending=False)

vif_df


# In[98]:


# Choose your fitted poisson results object here:
res = poisson_results_ale

df_work = ale_train.copy()
df_work["mu_hat"] = res.fittedvalues
df_work["y"] = df_work["claim_count_capped"]

# working residuals for Poisson
df_work["working_resid"] = (df_work["y"] - df_work["mu_hat"]) / np.sqrt(df_work["mu_hat"].clip(lower=1e-12))

# bin by fitted mean
n_bins = 20
df_work["bin"] = pd.qcut(df_work["mu_hat"], q=n_bins, duplicates="drop")

bin_stats = df_work.groupby("bin").agg(
    mu_hat_mean=("mu_hat", "mean"),
    wr_mean=("working_resid", "mean"),
    wr_std=("working_resid", "std"),
    n=("working_resid", "size")
).reset_index(drop=True)

# standard error for mean
bin_stats["wr_se"] = bin_stats["wr_std"] / np.sqrt(bin_stats["n"])

plt.figure(figsize=(8,5))
plt.errorbar(bin_stats["mu_hat_mean"], bin_stats["wr_mean"], yerr=1.96*bin_stats["wr_se"], fmt="o")
plt.axhline(0, color="red", linestyle="--")
plt.xlabel("Mean fitted value (binned)")
plt.ylabel("Mean working residual (± 95% CI)")
plt.title("Working Residuals by Fitted Value Bins")
plt.show()


# In[99]:


res = poisson_results_ale

y_obs = ale_train["claim_count_capped"].to_numpy()
mu_hat = res.fittedvalues.to_numpy()

# Poisson CDF at y and y-1
F_y = st.poisson.cdf(y_obs, mu_hat)
F_y_minus = st.poisson.cdf(y_obs - 1, mu_hat)  # for y=0, cdf(-1)=0 automatically

# randomized u in [F(y-1), F(y)]
rng = np.random.default_rng(123)
u = F_y_minus + rng.random(len(y_obs)) * (F_y - F_y_minus)

# avoid 0/1 for numerical stability
eps = 1e-12
u = np.clip(u, eps, 1 - eps)

rqr = st.norm.ppf(u)

# QQ plot
plt.figure(figsize=(6,6))
st.probplot(rqr, dist="norm", plot=plt)
plt.title("QQ Plot of Randomized Quantile Residuals")
plt.xlabel("Theoretical Quantiles")
plt.ylabel("Sample Quantiles")
plt.show()


# ## Model Validation

# In[100]:


# test set prediction
ale_test["mu_hat"] = poisson_results_ale.predict(ale_test)


# In[101]:


agg_train = actual_vs_predicted_plot(
    df=ale_train,
    model=poisson_results_ale,
    y_col="claim_count",
    n_bins=20,
    title="Training Set: Actual vs Predicted"
)


# In[102]:


agg_test = actual_vs_predicted_plot(
    df=ale_test,
    model=poisson_results_ale,
    y_col="claim_count",
    n_bins=20,
    title="Test Set: Actual vs Predicted"
)


# In[103]:


ale_test = ale_test.sort_values("mu_hat").reset_index(drop=True)

ale_test["decile"] = pd.qcut(
    ale_test["mu_hat"],
    q=10,
    labels=False
)

decile_summary = (
    ale_test
    .groupby("decile")
    .agg(
        mean_predicted=("mu_hat", "mean"),
        mean_actual=("claim_count", "mean"),
        count=("claim_count", "size")
    )
    .reset_index()
)

decile_summary


# In[104]:


plt.figure(figsize=(8, 6))

plt.plot(
    decile_summary["decile"],
    decile_summary["mean_actual"],
    marker="o",
    label="Actual"
)

plt.plot(
    decile_summary["decile"],
    decile_summary["mean_predicted"],
    marker="o",
    linestyle="--",
    label="Predicted"
)

plt.xlabel("Decile (low risk → high risk)")
plt.ylabel("Mean Claim Frequency")
plt.title("Simple Decile Plot (Test Set)")
plt.legend()
plt.grid(True)

plt.show()


# ## Loss Ratio chart

# In[105]:


# ---------- Decile assignment ----------
ale_test = ale_test.copy()
ale_test["decile"] = pd.qcut(
    ale_test["mu_hat"],
    q=10,
    labels=False,
    duplicates="drop"
)

# ---------- Aggregate by decile ----------
lr_df = (
    ale_test
    .groupby("decile")
    .agg(
        actual=("claim_count", "mean"),
        predicted=("mu_hat", "mean")
    )
    .reset_index()
)

lr_df["loss_ratio"] = lr_df["actual"] / lr_df["predicted"]


# In[106]:


plt.figure(figsize=(8, 5))

plt.plot(
    lr_df["decile"],
    lr_df["loss_ratio"],
    marker="o",
    linewidth=2
)

plt.axhline(
    y=1.0,
    color="red",
    linestyle="--",
    label="Perfect calibration (LR = 1)"
)

plt.xlabel("Decile (low risk → high risk)")
plt.ylabel("Loss Ratio (Actual / Predicted)")
plt.title("Loss Ratio by Decile (Test Set)")
plt.legend()
plt.grid(True)

plt.show()


# The predicted claim frequency increases monotonically across deciles, indicating that the model provides reasonable risk ranking on the test set.
# 
# However, the observed claim frequency shows noticeable volatility in the lower deciles, which is expected given the very low claim frequency and the capping of claim counts at 1. In such settings, empirical averages within deciles are highly sensitive to a small number of claims.

# ## Gini Index Plot

# In[107]:


lorenz_df, gini_index = gini_lorenz(
    ale_test,
    actual_col="claim_count",
    pred_col="mu_hat"
)

gini_index


# In[108]:


plt.figure(figsize=(6, 6))

plt.plot(
    lorenz_df["cum_exposure_share"],
    lorenz_df["cum_actual_share"],
    label="Model Lorenz Curve",
    linewidth=2
)

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    color="gray",
    label="No discrimination"
)

plt.xlabel("Cumulative Share of Exposure")
plt.ylabel("Cumulative Share of Claims")
plt.title(f"Lorenz Curve (Test Set)\nGini Index = {gini_index:.3f}")
plt.legend()
plt.grid(True)

plt.show()


# # Model with respect to Guest Medical

# In[111]:


gm_df = freq_df[freq_df["coverage"] == "Guest Medical"].copy()

# Separate training and holdout samples
gm_train = gm_df[gm_df["holdout"] == False].copy()
gm_test = gm_df[gm_df["holdout"] == True].copy()


# In[116]:


# Capping the claim_count, so that we have a more stable and robust model
cap_value = int(gm_train["claim_count"].quantile(0.975))

gm_train["claim_count_capped"] = gm_train["claim_count"].clip(upper=cap_value)

print("Chosen cap value:", cap_value)
print(gm_train[["claim_count", "claim_count_capped"]].describe())


# In[117]:


(gm_train["claim_count"] > cap_value).mean()


# In[118]:


gm_train["claim_count"].mean(), gm_train["claim_count_capped"].mean()


# **DON'T CAP!!!**

# In[239]:


formula = """
claim_count ~
C(Q("class")) +
C(study) +
C(greek) +
C(off_campus) +
C(sprinklered) +
C(risk_tier) +
distance_to_campus
"""

poisson_model_gm = smf.glm(
    formula=formula,
    data=gm_train,
    family=sm.families.Poisson()
)

poisson_results_gm = poisson_model_gm.fit()
print(poisson_results_gm.summary())


# In[120]:


# Extract fitted values and residual components
mu_hat_gm = poisson_results_gm.mu                 # fitted mean μ̂
y = gm_train["claim_count"]         # response
beta_dist_gm = poisson_results_gm.params["distance_to_campus"]

# Partial residual for distance
gm_train["pr_distance"] = (y - mu_hat_gm) + beta_dist_gm * gm_train["distance_to_campus"]


# In[121]:


x = gm_train["distance_to_campus"]
pr = gm_train["pr_distance"]

# LOWESS smooth
lowess_fit = lowess(pr, x, frac=0.3)

plt.figure(figsize=(7, 5))
plt.scatter(x, pr, alpha=0.3, s=15, label="Partial residuals")
plt.plot(lowess_fit[:, 0], lowess_fit[:, 1], color="red", linewidth=2, label="LOWESS")

plt.xlabel("Distance to campus")
plt.ylabel("Partial residual")
plt.title("Partial Residual Plot: Distance to Campus")
plt.legend()
plt.show()


# In[122]:


# Create quantile-based bins
gm_train["distance_bin"] = pd.qcut(
    gm_train["distance_to_campus"],
    q=5,                 # try 4–6; 5 is a good default
    duplicates="drop"
)

# Check bin counts
gm_train["distance_bin"].value_counts().sort_index()


# In[123]:


formula_bin = """
claim_count ~
C(Q("class")) +
C(study) +
C(greek) +
C(off_campus) +
C(sprinklered) +
C(risk_tier) +
C(distance_bin)
"""

poisson_bin_gm = smf.glm(
    formula=formula_bin,
    data=gm_train,
    family=sm.families.Poisson()
).fit()

print(poisson_bin_gm.summary())


# In[124]:


print("Linear AIC:", poisson_results_gm.aic)
print("Binned AIC:", poisson_bin_gm.aic)


# For Guest Medical coverage, we compared a linear specification and a binned specification for the distance variable. The linear model achieved a lower AIC than the binned alternative, indicating no improvement from discretization. Given the extremely sparse claim frequency and limited evidence of nonlinearity, the linear specification was retained to avoid unnecessary model complexity.

# **Keep the original model!**

# ## Interactions

# In[125]:


formula = """
claim_count ~
C(Q("class")) +
C(study) +
C(greek) * C(off_campus)+
C(sprinklered) +
C(risk_tier) +
distance_to_campus
"""

poisson_model_interaction_1_gm = smf.glm(
    formula=formula,
    data=gm_train,
    family=sm.families.Poisson()
)

poisson_results_interaction_1_gm = poisson_model_interaction_1_gm.fit()
print(poisson_results_interaction_1_gm.summary())


# In[126]:


formula = """
claim_count ~
C(Q("class")) +
C(study) +
C(greek) * distance_to_campus +
C(off_campus) +
C(sprinklered) +
C(risk_tier)
"""

poisson_model_interaction_2_gm = smf.glm(
    formula=formula,
    data=gm_train,
    family=sm.families.Poisson()
)

poisson_results_interaction_2_gm = poisson_model_interaction_2_gm.fit()
print(poisson_results_interaction_2_gm.summary())


# In[127]:


formula = """
claim_count ~
C(Q("class")) +
C(study) +
C(greek) * C(off_campus) +
C(greek) * distance_to_campus +
C(sprinklered) +
C(risk_tier)
"""

poisson_model_interaction_3_gm = smf.glm(
    formula=formula,
    data=gm_train,
    family=sm.families.Poisson()
)

poisson_results_interaction_3_gm = poisson_model_interaction_3_gm.fit()
print(poisson_results_interaction_3_gm.summary())


# In[128]:


print(poisson_results_interaction_2_gm.aic)
print(poisson_results_interaction_1_gm.aic)
print(poisson_results_interaction_3_gm.aic)


# **They did not make any differences, so retaining our original model!**

# ## Model Refinement

# In[130]:


influence_gm = poisson_results_gm.get_influence()
cooks_d = influence_gm.cooks_distance[0]

# quick summary
print("Max Cook's D:", cooks_d.max())
print("Mean Cook's D:", cooks_d.mean())

# rule-of-thumb threshold
threshold_gm = 4 / len(gm_train)
print("Threshold:", threshold_gm)

# count influential points
print("Num influential:", (cooks_d > threshold_gm).sum())


# Given the low-frequency nature of claim data, these influential observations correspond to rare but informative claims rather than data errors. Therefore, no observations were removed. Model robustness was instead ensured through claim capping and out-of-sample validation.

# In[132]:


df_diag = gm_train.copy()
df_diag["deviance_resid"] = poisson_results_gm.resid_deviance
df_diag["fitted"] = poisson_results_gm.fittedvalues


# In[133]:


# residuals by key categorical variable
df_diag.groupby("greek")["deviance_resid"].mean()
df_diag.groupby("off_campus")["deviance_resid"].mean()


# In[134]:


# residuals by key categorical variable
df_diag.groupby("greek")["deviance_resid"].mean()
df_diag.groupby("off_campus")["deviance_resid"].mean()


# In[135]:


dev_resid = df_diag["deviance_resid"]
plt.figure(figsize=(8, 5))
plt.hist(dev_resid, bins=40, density=True, alpha=0.7)
plt.axvline(0, color="red", linestyle="--")
plt.xlabel("Deviance residual")
plt.ylabel("Density")
plt.title("Density of Deviance Residuals")
plt.show()


# In[136]:


formula_main = """
claim_count ~
C(Q("class")) +
C(study) +
C(greek) +
C(off_campus) +
C(sprinklered) +
C(risk_tier) +
distance_to_campus
"""

B = 500 
coef_list = []
fail = 0

for b in range(B):
    sample = gm_train.sample(frac=1, replace=True)

    try:
        m = smf.glm(
            formula=formula_main,
            data=sample,
            family=sm.families.Poisson()
        ).fit()

        coef_list.append(m.params)

    except Exception:
        fail += 1
        continue

boot_df = pd.DataFrame(coef_list)

print("Bootstrap fits:", len(boot_df), "Failed:", fail)

# summary: mean/std + 95% percentile CI
summary = pd.DataFrame({
    "mean": boot_df.mean(),
    "std": boot_df.std(),
    "p2.5": boot_df.quantile(0.025),
    "p97.5": boot_df.quantile(0.975),
})

# sign stability (how often coefficient > 0)
summary["Pr(>0)"] = (boot_df > 0).mean()

summary.sort_values("std", ascending=False)


# In[138]:


# Build design matrix X from formula (drop intercept for VIF)
y, X = patsy.dmatrices(formula, data=gm_train, return_type="dataframe")

# drop intercept if present
if "Intercept" in X.columns:
    X_vif = X.drop(columns=["Intercept"])
else:
    X_vif = X.copy()

vif_df = pd.DataFrame({
    "variable": X_vif.columns,
    "VIF": [variance_inflation_factor(X_vif.values, i) for i in range(X_vif.shape[1])]
}).sort_values("VIF", ascending=False)

vif_df


# In[240]:


# Choose your fitted poisson results object here:
res = poisson_results_gm

df_work = gm_train.copy()
df_work["mu_hat"] = res.fittedvalues
df_work["y"] = df_work["claim_count_capped"]

# working residuals for Poisson
df_work["working_resid"] = (df_work["y"] - df_work["mu_hat"]) / np.sqrt(df_work["mu_hat"].clip(lower=1e-12))

# bin by fitted mean
n_bins = 20
df_work["bin"] = pd.qcut(df_work["mu_hat"], q=n_bins, duplicates="drop")

bin_stats = df_work.groupby("bin").agg(
    mu_hat_mean=("mu_hat", "mean"),
    wr_mean=("working_resid", "mean"),
    wr_std=("working_resid", "std"),
    n=("working_resid", "size")
).reset_index(drop=True)

# standard error for mean
bin_stats["wr_se"] = bin_stats["wr_std"] / np.sqrt(bin_stats["n"])

plt.figure(figsize=(8,5))
plt.errorbar(bin_stats["mu_hat_mean"], bin_stats["wr_mean"], yerr=1.96*bin_stats["wr_se"], fmt="o")
plt.axhline(0, color="red", linestyle="--")
plt.xlabel("Mean fitted value (binned)")
plt.ylabel("Mean working residual (± 95% CI)")
plt.title("Working Residuals by Fitted Value Bins")
plt.show()


# In[241]:


res = poisson_results_gm

y_obs = gm_train["claim_count_capped"].to_numpy()
mu_hat = res.fittedvalues.to_numpy()

# Poisson CDF at y and y-1
F_y = st.poisson.cdf(y_obs, mu_hat)
F_y_minus = st.poisson.cdf(y_obs - 1, mu_hat)  # for y=0, cdf(-1)=0 automatically

# randomized u in [F(y-1), F(y)]
rng = np.random.default_rng(123)
u = F_y_minus + rng.random(len(y_obs)) * (F_y - F_y_minus)

# avoid 0/1 for numerical stability
eps = 1e-12
u = np.clip(u, eps, 1 - eps)

rqr = st.norm.ppf(u)

# QQ plot
plt.figure(figsize=(6,6))
st.probplot(rqr, dist="norm", plot=plt)
plt.title("QQ Plot of Randomized Quantile Residuals")
plt.xlabel("Theoretical Quantiles")
plt.ylabel("Sample Quantiles")
plt.show()


# ## Model Validation

# In[142]:


# test set prediction
gm_test["mu_hat"] = poisson_results_gm.predict(gm_test)


# In[143]:


agg_train = actual_vs_predicted_plot(
    df=gm_train,
    model=poisson_results_gm,
    y_col="claim_count",
    n_bins=20,
    title="Training Set: Actual vs Predicted"
)


# In[144]:


agg_test = actual_vs_predicted_plot(
    df=gm_test,
    model=poisson_results_gm,
    y_col="claim_count",
    n_bins=20,
    title="Test Set: Actual vs Predicted"
)


# In[145]:


gm_test = gm_test.sort_values("mu_hat").reset_index(drop=True)

gm_test["decile"] = pd.qcut(
    gm_test["mu_hat"],
    q=10,
    labels=False
)

decile_summary = (
    gm_test
    .groupby("decile")
    .agg(
        mean_predicted=("mu_hat", "mean"),
        mean_actual=("claim_count", "mean"),
        count=("claim_count", "size")
    )
    .reset_index()
)

decile_summary


# In[146]:


plt.figure(figsize=(8, 6))

plt.plot(
    decile_summary["decile"],
    decile_summary["mean_actual"],
    marker="o",
    label="Actual"
)

plt.plot(
    decile_summary["decile"],
    decile_summary["mean_predicted"],
    marker="o",
    linestyle="--",
    label="Predicted"
)

plt.xlabel("Decile (low risk → high risk)")
plt.ylabel("Mean Claim Frequency")
plt.title("Simple Decile Plot (Test Set)")
plt.legend()
plt.grid(True)

plt.show()


# In[147]:


# Loss Ratio Chart
# ---------- Decile assignment ----------
gm_test = gm_test.copy()
gm_test["decile"] = pd.qcut(
    gm_test["mu_hat"],
    q=10,
    labels=False,
    duplicates="drop"
)

# ---------- Aggregate by decile ----------
lr_df = (
    gm_test
    .groupby("decile")
    .agg(
        actual=("claim_count", "mean"),
        predicted=("mu_hat", "mean")
    )
    .reset_index()
)

lr_df["loss_ratio"] = lr_df["actual"] / lr_df["predicted"]


# In[148]:


plt.figure(figsize=(8, 5))

plt.plot(
    lr_df["decile"],
    lr_df["loss_ratio"],
    marker="o",
    linewidth=2
)

plt.axhline(
    y=1.0,
    color="red",
    linestyle="--",
    label="Perfect calibration (LR = 1)"
)

plt.xlabel("Decile (low risk → high risk)")
plt.ylabel("Loss Ratio (Actual / Predicted)")
plt.title("Loss Ratio by Decile (Test Set)")
plt.legend()
plt.grid(True)

plt.show()


# In[149]:


# Gini Index Plot
lorenz_df, gini_index = gini_lorenz(
    gm_test,
    actual_col="claim_count",
    pred_col="mu_hat"
)

gini_index


# In[150]:


plt.figure(figsize=(6, 6))

plt.plot(
    lorenz_df["cum_exposure_share"],
    lorenz_df["cum_actual_share"],
    label="Model Lorenz Curve",
    linewidth=2
)

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    color="gray",
    label="No discrimination"
)

plt.xlabel("Cumulative Share of Exposure")
plt.ylabel("Cumulative Share of Claims")
plt.title(f"Lorenz Curve (Test Set)\nGini Index = {gini_index:.3f}")
plt.legend()
plt.grid(True)

plt.show()


# Although the Poisson GLM captures some directional relationship between covariates and claim frequency, several diagnostic results indicate systematic lack-of-fit.
# 
# Working residual plots exhibit a clear downward trend against fitted values, suggesting that the model consistently overestimates the mean claim count, particularly in higher-risk segments.
# 
# The randomized quantile residual QQ plot further reveals heavy left tails, consistent with the presence of excess zeros that cannot be adequately accommodated under a single-stage Poisson assumption.
# 
# Calibration diagnostics, including decile plots and loss ratios, show substantial volatility across risk groups. This behavior is expected given the extremely low base frequency of claims, where a small number of events can dominate group-level averages.
# 
# Overall, these results suggest that claim occurrence and claim frequency are driven by partially distinct mechanisms. While a two-part model such as a hurdle specification would be more appropriate, model constraints require retaining a single-stage Poisson GLM. Under these constraints, the Poisson model serves as a reasonable baseline but is not fully capable of capturing the underlying data-generating process.

# # Model with respect to Liability

# In[151]:


la_df = freq_df[freq_df["coverage"] == "Liability"].copy()

# Separate training and holdout samples
la_train = la_df[la_df["holdout"] == False].copy()
la_test = la_df[la_df["holdout"] == True].copy()


# In[157]:


# Capping the claim_count, so that we have a more stable and robust model
cap_value = int(la_train["claim_count"].quantile(0.975))
print("Chosen cap value:", cap_value)


# Don't Cap!!!

# In[211]:


formula = """
claim_count ~
C(Q("class")) +
C(study) +
C(greek) +
C(off_campus) +
C(sprinklered) +
C(risk_tier) +
distance_to_campus
"""

poisson_model_la = smf.glm(
    formula=formula,
    data=la_train,
    family=sm.families.Poisson()
)

poisson_results_la = poisson_model_la.fit()
print(poisson_results_la.summary())


# In[212]:


# Extract fitted values and residual components
mu_hat_la = poisson_results_la.mu                 # fitted mean μ̂
y = la_train["claim_count_capped"]         # response
beta_dist_la = poisson_results_la.params["distance_to_campus"]

# Partial residual for distance
la_train["pr_distance"] = (y - mu_hat_la) + beta_dist_la * la_train["distance_to_campus"]


# In[213]:


x = la_train["distance_to_campus"]
pr = la_train["pr_distance"]

# LOWESS smooth
lowess_fit = lowess(pr, x, frac=0.3)

plt.figure(figsize=(7, 5))
plt.scatter(x, pr, alpha=0.3, s=15, label="Partial residuals")
plt.plot(lowess_fit[:, 0], lowess_fit[:, 1], color="red", linewidth=2, label="LOWESS")

plt.xlabel("Distance to campus")
plt.ylabel("Partial residual")
plt.title("Partial Residual Plot: Distance to Campus")
plt.legend()
plt.show()


# **Perfect!!!**

# ## Interactions

# In[214]:


formula = """
claim_count ~
C(Q("class")) +
C(study) +
C(greek) * C(off_campus)+
C(sprinklered) +
C(risk_tier) +
distance_to_campus
"""

poisson_model_interaction_1_la = smf.glm(
    formula=formula,
    data=la_train,
    family=sm.families.Poisson()
)

poisson_results_interaction_1_la = poisson_model_interaction_1_la.fit()
print(poisson_results_interaction_1_la.summary())


# In[215]:


formula = """
claim_count ~
C(Q("class")) +
C(study) +
C(greek) * distance_to_campus +
C(off_campus) +
C(sprinklered) +
C(risk_tier)
"""

poisson_model_interaction_2_la = smf.glm(
    formula=formula,
    data=la_train,
    family=sm.families.Poisson()
)

poisson_results_interaction_2_la = poisson_model_interaction_2_la.fit()
print(poisson_results_interaction_2_la.summary())


# In[216]:


formula = """
claim_count ~
C(Q("class")) +
C(study) +
C(greek) * C(off_campus) +
C(greek) * distance_to_campus +
C(sprinklered) +
C(risk_tier)
"""

poisson_model_interaction_3_la = smf.glm(
    formula=formula,
    data=la_train,
    family=sm.families.Poisson()
)

poisson_results_interaction_3_la = poisson_model_interaction_3_la.fit()
print(poisson_results_interaction_3_la.summary())


# In[217]:


print(poisson_results_interaction_2_la.aic)
print(poisson_results_interaction_1_la.aic)
print(poisson_results_interaction_3_la.aic)
print(poisson_results_la.aic)


# Keep the original model!

# ## Model Refinement

# In[218]:


influence_la = poisson_results_la.get_influence()
cooks_d = influence_la.cooks_distance[0]

# quick summary
print("Max Cook's D:", cooks_d.max())
print("Mean Cook's D:", cooks_d.mean())

# rule-of-thumb threshold
threshold_la = 4 / len(la_train)
print("Threshold:", threshold_la)

# count influential points
print("Num influential:", (cooks_d > threshold_la).sum())


# In[219]:


df_diag = la_train.copy()
df_diag["deviance_resid"] = poisson_results_la.resid_deviance
df_diag["fitted"] = poisson_results_la.fittedvalues


# In[220]:


# residuals by key categorical variable
df_diag.groupby("greek")["deviance_resid"].mean()
df_diag.groupby("off_campus")["deviance_resid"].mean()


# In[221]:


# residuals by key categorical variable
df_diag.groupby("greek")["deviance_resid"].mean()
df_diag.groupby("off_campus")["deviance_resid"].mean()


# In[222]:


dev_resid = df_diag["deviance_resid"]
plt.figure(figsize=(8, 5))
plt.hist(dev_resid, bins=40, density=True, alpha=0.7)
plt.axvline(0, color="red", linestyle="--")
plt.xlabel("Deviance residual")
plt.ylabel("Density")
plt.title("Density of Deviance Residuals")
plt.show()


# In[223]:


formula_main = """
claim_count ~
C(Q("class")) +
C(study) +
C(greek) +
C(off_campus) +
C(sprinklered) +
C(risk_tier) +
distance_to_campus
"""

B = 500 
coef_list = []
fail = 0

for b in range(B):
    sample = la_train.sample(frac=1, replace=True)

    try:
        m = smf.glm(
            formula=formula_main,
            data=sample,
            family=sm.families.Poisson()
        ).fit()

        coef_list.append(m.params)

    except Exception:
        fail += 1
        continue

boot_df = pd.DataFrame(coef_list)

print("Bootstrap fits:", len(boot_df), "Failed:", fail)

# summary: mean/std + 95% percentile CI
summary = pd.DataFrame({
    "mean": boot_df.mean(),
    "std": boot_df.std(),
    "p2.5": boot_df.quantile(0.025),
    "p97.5": boot_df.quantile(0.975),
})

# sign stability (how often coefficient > 0)
summary["Pr(>0)"] = (boot_df > 0).mean()

summary.sort_values("std", ascending=False)


# In[224]:


# Build design matrix X from formula (drop intercept for VIF)
y, X = patsy.dmatrices(formula, data=la_train, return_type="dataframe")

# drop intercept if present
if "Intercept" in X.columns:
    X_vif = X.drop(columns=["Intercept"])
else:
    X_vif = X.copy()

vif_df = pd.DataFrame({
    "variable": X_vif.columns,
    "VIF": [variance_inflation_factor(X_vif.values, i) for i in range(X_vif.shape[1])]
}).sort_values("VIF", ascending=False)

vif_df


# In[225]:


# Choose your fitted poisson results object here:
res = poisson_results_la

df_work = la_train.copy()
df_work["mu_hat"] = res.fittedvalues
df_work["y"] = df_work["claim_count_capped"]

# working residuals for Poisson
df_work["working_resid"] = (df_work["y"] - df_work["mu_hat"]) / np.sqrt(df_work["mu_hat"].clip(lower=1e-12))

# bin by fitted mean
n_bins = 20
df_work["bin"] = pd.qcut(df_work["mu_hat"], q=n_bins, duplicates="drop")

bin_stats = df_work.groupby("bin").agg(
    mu_hat_mean=("mu_hat", "mean"),
    wr_mean=("working_resid", "mean"),
    wr_std=("working_resid", "std"),
    n=("working_resid", "size")
).reset_index(drop=True)

# standard error for mean
bin_stats["wr_se"] = bin_stats["wr_std"] / np.sqrt(bin_stats["n"])

plt.figure(figsize=(8,5))
plt.errorbar(bin_stats["mu_hat_mean"], bin_stats["wr_mean"], yerr=1.96*bin_stats["wr_se"], fmt="o")
plt.axhline(0, color="red", linestyle="--")
plt.xlabel("Mean fitted value (binned)")
plt.ylabel("Mean working residual (± 95% CI)")
plt.title("Working Residuals by Fitted Value Bins")
plt.show()


# In[226]:


res = poisson_results_la

y_obs = la_train["claim_count_capped"].to_numpy()
mu_hat = res.fittedvalues.to_numpy()

# Poisson CDF at y and y-1
F_y = st.poisson.cdf(y_obs, mu_hat)
F_y_minus = st.poisson.cdf(y_obs - 1, mu_hat)  # for y=0, cdf(-1)=0 automatically

# randomized u in [F(y-1), F(y)]
rng = np.random.default_rng(123)
u = F_y_minus + rng.random(len(y_obs)) * (F_y - F_y_minus)

# avoid 0/1 for numerical stability
eps = 1e-12
u = np.clip(u, eps, 1 - eps)

rqr = st.norm.ppf(u)

# QQ plot
plt.figure(figsize=(6,6))
st.probplot(rqr, dist="norm", plot=plt)
plt.title("QQ Plot of Randomized Quantile Residuals")
plt.xlabel("Theoretical Quantiles")
plt.ylabel("Sample Quantiles")
plt.show()


# ## Model Validation

# In[227]:


# test set prediction
la_test["mu_hat"] = poisson_results_la.predict(la_test)


# In[228]:


agg_train = actual_vs_predicted_plot(
    df=la_train,
    model=poisson_results_la,
    y_col="claim_count",
    n_bins=20,
    title="Training Set: Actual vs Predicted"
)


# In[229]:


agg_test = actual_vs_predicted_plot(
    df=la_test,
    model=poisson_results_la,
    y_col="claim_count",
    n_bins=20,
    title="Test Set: Actual vs Predicted"
)


# In[230]:


la_test = la_test.sort_values("mu_hat").reset_index(drop=True)

la_test["decile"] = pd.qcut(
    la_test["mu_hat"],
    q=10,
    labels=False
)

decile_summary = (
    la_test
    .groupby("decile")
    .agg(
        mean_predicted=("mu_hat", "mean"),
        mean_actual=("claim_count", "mean"),
        count=("claim_count", "size")
    )
    .reset_index()
)

decile_summary


# In[231]:


plt.figure(figsize=(8, 6))

plt.plot(
    decile_summary["decile"],
    decile_summary["mean_actual"],
    marker="o",
    label="Actual"
)

plt.plot(
    decile_summary["decile"],
    decile_summary["mean_predicted"],
    marker="o",
    linestyle="--",
    label="Predicted"
)

plt.xlabel("Decile (low risk → high risk)")
plt.ylabel("Mean Claim Frequency")
plt.title("Simple Decile Plot (Test Set)")
plt.legend()
plt.grid(True)

plt.show()


# ## Loss Ratio Chart

# In[232]:


# ---------- Decile assignment ----------
la_test = la_test.copy()
la_test["decile"] = pd.qcut(
    la_test["mu_hat"],
    q=10,
    labels=False,
    duplicates="drop"
)

# ---------- Aggregate by decile ----------
lr_df = (
    la_test
    .groupby("decile")
    .agg(
        actual=("claim_count", "mean"),
        predicted=("mu_hat", "mean")
    )
    .reset_index()
)

lr_df["loss_ratio"] = lr_df["actual"] / lr_df["predicted"]


# In[233]:


plt.figure(figsize=(8, 5))

plt.plot(
    lr_df["decile"],
    lr_df["loss_ratio"],
    marker="o",
    linewidth=2
)

plt.axhline(
    y=1.0,
    color="red",
    linestyle="--",
    label="Perfect calibration (LR = 1)"
)

plt.xlabel("Decile (low risk → high risk)")
plt.ylabel("Loss Ratio (Actual / Predicted)")
plt.title("Loss Ratio by Decile (Test Set)")
plt.legend()
plt.grid(True)

plt.show()


# ## Gini Index

# In[234]:


lorenz_df, gini_index = gini_lorenz(
    la_test,
    actual_col="claim_count",
    pred_col="mu_hat"
)

gini_index


# In[235]:


plt.figure(figsize=(6, 6))

plt.plot(
    lorenz_df["cum_exposure_share"],
    lorenz_df["cum_actual_share"],
    label="Model Lorenz Curve",
    linewidth=2
)

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    color="gray",
    label="No discrimination"
)

plt.xlabel("Cumulative Share of Exposure")
plt.ylabel("Cumulative Share of Claims")
plt.title(f"Lorenz Curve (Test Set)\nGini Index = {gini_index:.3f}")
plt.legend()
plt.grid(True)

plt.show()


# In[ ]:




