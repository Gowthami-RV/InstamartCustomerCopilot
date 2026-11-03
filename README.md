# Instamart Customer Intelligence & Support Copilot

## Overview
A reproducible, offline capstone implementation using synthetic data. It cleans order exports, builds customer segments and a lapse model, redacts and triages tickets, and provides TF-IDF policy retrieval plus a tool-using refund agent. All monetary values are INR. No network or paid model is used.

## Reproduce
Run these commands from the repository root in a clean Python 3.11+ environment:

1. `python -m venv .venv`
2. Windows: `.venv\\Scripts\\activate` (macOS/Linux: `source .venv/bin/activate`)
3. `python -m pip install --upgrade pip`
4. `pip install -r requirements.txt`
5. `python generate_data.py`
6. `python run_all.py`


The generator and mock model are local and deterministic. `src/parta_datavalidation.py` writes clean CSVs and SQLite tables; Part B/C/D scripts consume those outputs in order.

## Part A
Customers: trim and title-case city; map gender spellings to male/female/unknown; treat ages outside 10–100 as missing, then impute by city median and finally global median if a city has no valid age; parse signup dates. Orders: parse timestamps and numeric fields; remove exact duplicate rows; keep latest row per order ID; recompute missing amount from catalogue price × quantity × (1 − discount/100); set negative delivery durations missing; impute completed-order delivery time using product-category median while leaving cancelled delivery times missing. Clean tables are exported to CSV and SQLite.

`sql/queries.sql` contains the seven requested query patterns: WHERE/HAVING, three-table join, scalar subquery, LEFT JOIN, CTE, ranking windows, and running total. Run queries using SQLite or `pandas.read_sql_query`. `src/parta_datavalidation.py` asserts core cleaning invariants and prints `ALL CHECKPOINTS PASSED`.

## Part B
RFM is calculated from completed orders at 2025-10-01. K-means metrics for K=2…8 and a four-cluster segment file are saved under `outputs/`; PCA variance ratios are saved as well. Lapse labels use the September 2025 window and features are built from pre-cutoff orders only. Encoding/scaling and hyperparameter tuning are inside sklearn pipelines and grid search uses training rows only; test rows are held out for final metrics. The chosen model is persisted and reloaded to verify prediction equality. The Premium retention list is illustrative and not a leakage-free evaluation because the Premium flag is a post-hoc whole-window label.

## Part C
Tickets are processed in file order. Empty and >400-token tickets are skipped/flagged before any model call. Email, phone and order identifiers are redacted before prompting; re-redaction assertions guard against leakage. The pipeline retries transient mock failures with exponential backoff and jitter, validates JSON or a single category keyword, and routes invalid/guardrail cases to human review. Outputs include zero-shot comparison, few-shot final triage and a token-budget demo.

## Part D
Policy files are chunked into 40-word passages with 10-word overlap and indexed with TF-IDF cosine similarity. The policy assistant answers only from retrieved context. The refund agent uses five tools: `get_order`, `search_policy`, `calculate_refund`, `issue_refund`, and `escalate_to_human`; actions are appended to a JSONL ledger and checkpoints are written under `state/`. Guardrails provide iteration, token and time fallbacks. Policy evaluation and scenario outputs are written under `outputs/`.

TF is the frequency of a term within a passage, while IDF downweights terms common across passages. TF-IDF is lexical, so semantically equivalent questions with different wording may retrieve poorly.

## Limitations and Next Steps
This is a synthetic educational dataset and the supplied mock LLM is deliberately simplistic; results do not establish production performance. Improve monitoring, human audit, tests, and real-world privacy/security review before deployment. The retention list is illustrative and not a leakage-free evaluation because Premium status uses the full RFM window. The lapse model's own features use only pre-2025-09-01 orders, and segment labels are not model features. No encoder, scaler or resampler is fitted on test rows.

### Reported run results

#### Part A checkpoint table
| Checkpoint | Result |
|---|---:|
RAW DATA AUDIT SUMMARY
                          Audit measure  Observed value
                        Raw orders rows           13084
             Exact duplicate order rows             119
                     Pending order rows             656
               Distinct order_id values           12309
       Non-numeric order_amount entries              67
              Negative delivery_minutes             106
                 Blank delivery_minutes            2541
                    Blank customer ages              61
                    Ages outside 10-100               6
            Distinct raw city spellings              24
Distinct non-blank raw gender spellings              10
                 Blank customer genders              30
--- Cleaned Orders Summary ---
Final shape: (12309, 9)

Orders by status:
status
completed    11110
cancelled     1199
Completed-order revenue: 2998668.8

The SQL file includes all seven required query forms. Ties can give equal RANK/DENSE_RANK values while ROW_NUMBER still assigns a unique position to every row, so those functions can differ.

Exploratory Data Analysis

What is this data about?
The data represents Instamart completed orders, including order amounts, quantities, discounts, delivery times, customer cities, and product categories.

What do its distributions and statistics look like?
The mean order amount is INR 269.91, while the median is INR 135.00.Q1 is INR 64.80 and Q3 is INR 297.00. Skewness is approximately 4.61, indicating a strongly right-skewed distribution. 

Are there outliers or anomalies?What relationships exist between variables?
The IQR upper fence is INR 645.30, with 897 order amounts above it and none below the lower fence. These unusually large values warrant investigation but are not necessarily errors.
The strongest absolute correlation is between order_amount and discount_pct, at approximately −0.095; order_amount and quantity correlate at approximately +0.087. Both relationships are weak. Correlation of this size does not establish causation.

#### Part B run metrics
| Metric | Result |
|---|---:|
| Lapse cohort | 1156 |
| Lapsed customers | 443 |
| Train / test | 924 / 232 |
| Always-not-lapsed baseline accuracy | 0.616 |
| Logistic regression best parameters | `{'model__C': 100}` |
| Random forest best parameters | `{'model__max_depth': 8, 'model__min_samples_leaf': 10}` |
| Random forest test precision / recall / F1 | 0.604 / 0.652 / 0.627 |
| Random forest confusion matrix | `[[111, 32], [31, 58]]` |

The scaler/encoder are in the pipeline and are fitted during cross-validation on training folds only; the held-out test set is used only for final evaluation. The saved pipeline reloads with identical predictions.

#### Part C run metrics
 T0024       C0486 classified_json     fraud      At-Risk Customers     HIGH security_team          100
    T0073       C0080 classified_json   billing Loyal Active Customers   NORMAL support_queue           96
    T0095       C1087 classified_json technical   Occasional Customers   NORMAL support_queue           98
    T0159       C1172   skipped_empty           Loyal Active Customers   NORMAL  human_review            0
    T0080       C0910 classified_json     fraud   Occasional Customers     HIGH security_team          100


Redaction substitutions across all tickets: 55 emails, 91 phones, 132 order IDs (278 total). The budget demo used a 6,000-token budget and stopped 23 tickets at the budget boundary. Few-shot used about 1.49× as many tokens per classified ticket and improved accuracy by 14.3 percentage points in this synthetic run, so the additional cost is justified when classification quality is the priority.

#### Part D scenario results
chunks: 15

Policy evaluation: the overlap-10 run is recorded in `outputs/policy_eval_results.csv`; the zero-overlap run is in `outputs/policy_eval_zero_overlap.csv`. The latter demonstrates that retrieval/grounding can fail when a relevant sentence is split across non-overlapping chunks; compare the question-level rows in the two files. Evaluation files include the two out-of-scope questions and their fallback answers.

For the policy evaluation specifically, the overlap-10 configuration returns the expected fact for all 12 answerable questions and the fallback for both out-of-scope questions. With zero overlap, 10/12 answerable expected facts are present: the refund eligibility threshold question and the failed-payment timing question break because their relevant facts cross chunk boundaries. TF-IDF can still retrieve a document, but the retrieved fragment no longer contains the complete answer.
