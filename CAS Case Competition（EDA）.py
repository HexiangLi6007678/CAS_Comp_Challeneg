#!/usr/bin/env python
# coding: utf-8

# # CAS Case Competition (Code)

# ## Data Cleaning

# In[42]:


import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import statsmodels.api as sm

# Load Data
file_path = "06 - CAS Predictive Modeling Case Competition- Dataset.xlsx"

# If you know the sheet name, set it here; otherwise load the first sheet.
try:
    df_raw = pd.read_excel(file_path, sheet_name="4 - Predictive Modeling Case Co")
except ValueError:
    df_raw = pd.read_excel(file_path)


# In[12]:


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

# In[13]:


# 1.2 Basic type normalization
# ===============================
for bcol in ["sprinklered", "holdout"]:
    if bcol in df.columns:
        df[bcol] = df[bcol].astype("boolean")

# Strip whitespace in key string columns
for scol in ["coverage", "class", "study", "greek", "off_campus", "gender", "name"]:
    if scol in df.columns:
        df[scol] = df[scol].astype(str).str.strip()


# In[14]:


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


# In[15]:


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


# In[16]:


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


# In[17]:


df_clean = df


# In[18]:


df_clean


# # Exploratory Data Analysis

# ### Step 1: Is the claim sparse?

# In[20]:


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


# In[21]:


# --- 3) Amount distribution among claim rows ---
claim_amounts = df_clean.loc[df_clean["has_claim"], "amount"].dropna()

print("\n=== Severity among claim rows (amount | claim_id > 0) ===")
print(claim_amounts.describe(percentiles=[0.5, 0.75, 0.9, 0.95, 0.99]))


# In[22]:


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


# In[23]:


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

# In[26]:


# Step 0) Create helper flags
# -----------------------------
# Row-level claim indicator (claim record vs non-claim record)
df_clean["has_claim"] = df_clean["claim_id"].astype(int) > 0

# Optional sanity checks
print("Rows:", len(df_clean))
print("Unique students:", df_clean["student_id"].nunique())
print("Coverages:", df_clean["coverage"].nunique(), sorted(df_clean["coverage"].dropna().unique()))


# In[27]:


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


# In[28]:


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


# In[29]:


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


# In[31]:


# 4) Frequency x key risk factors (2–3 variables) BY coverage

key_factors = ["greek", "off_campus", "sprinklered"] 
key_factors = [c for c in key_factors if c in df_clean.columns]

print("\nKey factors available:", key_factors)


# In[32]:


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

# In[41]:


# Histogram: claim amounts (linear scale)
plt.figure()
plt.hist(claim_amounts, bins=40)
plt.title("Claim Amounts (Linear Scale) - Claim Rows Only")
plt.xlabel("Amount")
plt.ylabel("Frequency")
plt.show()


# The distribution of **Claim Amount** is severly right_skewed. So modelling it by using Gamma model or lognormal model will be a good choice!

# ### Step 4: Plots of each predictor vs. target (claim rate, claim amount)

# ### Plots of each predictor vs. claim rate by covergae

# In[53]:


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


# In[54]:


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


# ### Plots of each predictor vs. claim amount by coverage 

# In[55]:


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


# In[56]:


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

# In[57]:


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

# In[58]:


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

# In[59]:


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

print("\nInterpretation cheat-sheet:")
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

# In[ ]:




