# SHAP Explainability -- Concept Guide

## What is SHAP?

SHAP = SHapley Additive exPlanations. It answers: "WHY did the model predict this?"

Think of it like a cricket team. Each player (feature) contributes some runs (prediction).
SHAP tells you exactly how many runs each player scored, and whether they helped win or lose.

## Why Explainability Matters (especially in banking)

Banks can't just say "our AI rejected your loan." RBI and regulators require explanations.
SHAP provides legally defensible, mathematically grounded explanations for each prediction.

**Interview answer:** "I added SHAP explainability because in regulated industries like
banking, model transparency is mandatory. SHAP decomposes each prediction into per-feature
contributions, making it auditable."

## Key Concepts

### 1. Shapley Values (from Game Theory)
Originally from economics (Lloyd Shapley, Nobel Prize 2012). The idea:
- Try every possible combination of features
- Measure how much adding a feature changes the prediction
- Average across all combinations = that feature's SHAP value

**Interview Q:** "What are Shapley values?"
**Answer:** "They come from cooperative game theory. Each feature is a 'player' and the
prediction is the 'payout'. Shapley values fairly distribute the payout among players
based on their marginal contribution across all possible coalitions."

### 2. TreeExplainer
SHAP's fast algorithm specifically for tree-based models (Random Forest, XGBoost).
Instead of testing all feature combinations (exponentially slow), it exploits the
tree structure to compute exact SHAP values in polynomial time.

**Interview Q:** "Why TreeExplainer and not KernelExplainer?"
**Answer:** "TreeExplainer gives exact SHAP values in O(TLD²) time for tree models.
KernelExplainer is model-agnostic but uses sampling and is much slower. Since we use
Random Forest, TreeExplainer is the right choice."

### 3. Waterfall Plot
Shows how each feature pushes the prediction from the base value (average prediction)
to the final prediction. Red bars push toward churn, blue bars push away from churn.

### 4. Global vs Local Explanations
- **Local:** "For THIS customer, age was the biggest factor" (waterfall plot)
- **Global:** "Across ALL customers, balance is the most important feature" (bar chart)

**Interview Q:** "Difference between local and global interpretability?"
**Answer:** "Global explains what the model learned overall. Local explains a specific
prediction. A doctor needs both -- general medical knowledge AND this patient's diagnosis."

## Experiments to Try
1. Set age=25, low balance -- see SHAP says low churn risk. Then change age=60 -- see
   how the waterfall plot shifts
2. Compare active vs inactive member -- see which features flip
3. Try a high-balance German customer -- SHAP will likely show country as a risk factor

## Resources
- **SHAP docs:** shap.readthedocs.io
- **Christoph Molnar's book (free):** christophm.github.io/interpretable-ml-book/shap.html
- **Krish Naik SHAP video:** Search "Krish Naik SHAP" on YouTube
