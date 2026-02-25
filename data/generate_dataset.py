"""
Loan Applications Dataset Generator
=====================================
Generates the synthetic dataset used across all 5 labs in Module 3.

Feature availability per lab:
  Lab 1:  credit_score, annual_income, loan_amount, num_defaults
  Lab 2:  + employment_years, age, loan_purpose
  Lab 3:  + neighborhood_risk_score
  Lab 4:  (same 9 features)
  Lab 5:  + voice_transcript, context_score, bank_id

Label  : approved (0 = denied, 1 = approved)
Balance: ~65% approved, ~35% denied  (intentional — exposes the accuracy trap)
Seed   : 42  (fully reproducible)
"""

import numpy as np
import pandas as pd
from pathlib import Path

np.random.seed(42)
N = 1000

# ── Lab 1 features ──────────────────────────────────────────────────────────
credit_score = np.clip(
    np.random.normal(650, 90, N), 300, 850
).astype(int)

annual_income = np.clip(
    np.random.lognormal(np.log(52_000), 0.45, N), 18_000, 240_000
)

loan_amount = np.clip(
    np.random.lognormal(np.log(15_000), 0.60, N), 3_000, 80_000
)

num_defaults = np.clip(
    np.random.poisson(0.35, N), 0, 5
).astype(int)

# ── Lab 2 features ──────────────────────────────────────────────────────────
employment_years = np.clip(
    np.random.normal(7.0, 5.0, N), 0, 35
).round(1)

age = np.clip(
    np.random.normal(37, 11, N), 21, 70
).astype(int)

loan_purpose = np.random.choice(
    ['car', 'home', 'education', 'business', 'personal'],
    N,
    p=[0.25, 0.30, 0.18, 0.12, 0.15],
)

# ── Lab 3 feature — deliberately biased proxy ───────────────────────────────
# Correlated with income (lower income areas → higher risk score).
# Geographically correlated with demographics.
# Improves model accuracy but is legally and morally problematic.
# Students discover this via feature importance in Lab 3 and remove it.
income_norm = (annual_income - annual_income.min()) / (
    annual_income.max() - annual_income.min()
)
neighborhood_risk_score = np.clip(
    10 - 5 * income_norm + np.random.normal(0, 1.5, N), 1, 10
).round(1)

# ── Lab 5 features ──────────────────────────────────────────────────────────
bank_id = np.random.choice(['A', 'B', 'C'], N, p=[0.40, 0.35, 0.25])
voice_transcript = [''] * N      # populated for 3 applicants in Lab 5
context_score    = [np.nan] * N  # extracted by LLM in Lab 5

# ── Label generation ────────────────────────────────────────────────────────
# Decision formula (ground truth):
#   + credit score (strong positive)
#   - debt-to-income ratio (strong negative — high loan relative to income = risky)
#   - number of past defaults (each default is a major red flag)
#   + employment stability (minor positive)
#   - neighborhood risk (biased proxy, minor negative — intentionally included)
#   + Gaussian noise (makes the task non-trivial)

credit_norm  = (credit_score - 300) / 550          # scaled 0 → 1
dti          = loan_amount / annual_income          # debt-to-income
emp_norm     = np.minimum(employment_years / 15, 1) # capped at 1

decision_score = (
     1.40 * credit_norm
    - 2.00 * dti
    - 0.50 * num_defaults
    + 0.15 * emp_norm
    - 0.10 * (neighborhood_risk_score / 10)
    + np.random.normal(0, 0.12, N)
)

# Calibrate threshold so that ~65% of applicants are approved
threshold = np.percentile(decision_score, 35)
approved = (decision_score > threshold).astype(int)

approval_rate = approved.mean()
print(f"Approval rate: {approval_rate:.1%}  (target ~65%)")

# ── Assemble ─────────────────────────────────────────────────────────────────
df = pd.DataFrame({
    'applicant_id':            [f'APP-{i:04d}' for i in range(N)],
    # Lab 1
    'credit_score':            credit_score,
    'annual_income':           annual_income.round(2),
    'loan_amount':             loan_amount.round(2),
    'num_defaults':            num_defaults,
    # Lab 2
    'employment_years':        employment_years,
    'age':                     age,
    'loan_purpose':            loan_purpose,
    # Lab 3 (biased proxy)
    'neighborhood_risk_score': neighborhood_risk_score,
    # Lab 5
    'bank_id':                 bank_id,
    'voice_transcript':        voice_transcript,
    'context_score':           context_score,
    # Label
    'approved':                approved,
})

# ── Save ─────────────────────────────────────────────────────────────────────
out_path = Path(__file__).parent / 'loan_applications.csv'
df.to_csv(out_path, index=False)
print(f"Saved {N} rows  →  {out_path}")
print(f"\nClass balance:\n{df['approved'].value_counts(normalize=True).rename({0: 'denied', 1: 'approved'}).to_string()}")
print(f"\nFeature dtypes:\n{df.dtypes.to_string()}")
