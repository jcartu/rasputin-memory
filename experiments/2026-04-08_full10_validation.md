# Full 10-Conversation Validation — Two-Lane 45w+15f with Cross-Encoder

**Date:** 2026-04-08
**Dataset:** LoCoMo all 10 conversations, 1986 questions
**Config:** Two-lane search (45 dense windows + 15 dense facts), cross-encoder ON, w5s2 windows

## Purpose

Validate the two-lane search improvement (69.7% on conv-0) across all 10
LoCoMo conversations to confirm it's not an outlier.

## Setup

Ran 3 parallel batches on separate ports to complete in ~7 hours instead of ~20:
- Batch A: convs 0,1,2,3 (port 7779)
- Batch B: convs 4,5,6 (port 7782)
- Batch C: convs 7,8,9 (port 7781)

## Per-Conversation Results

| Conversation | Non-Adv | Total Qs |
|-------------|---------|----------|
| conv-26 | 65.8% | 199 |
| conv-30 | **75.3%** | 105 |
| conv-41 | **71.1%** | 193 |
| conv-42 | 66.3% | 260 |
| conv-43 | 65.7% | 242 |
| conv-44 | 62.6% | 158 |
| conv-47 | **71.3%** | 190 |
| conv-48 | **70.7%** | 239 |
| conv-49 | 64.7% | 196 |
| conv-50 | 63.9% | 204 |
| **Average** | **67.5%** | **1986** |

Range: 62.6% (conv-44) to 75.3% (conv-30). Standard deviation ~3.9pp.

## Per-Category Results (All 10 Conversations)

| Category | Questions | Correct | Accuracy |
|----------|-----------|---------|----------|
| open-domain | 841 | 689 | **81.9%** |
| temporal | 321 | 208 | **64.8%** |
| single-hop | 282 | 105 | 37.2% |
| multi-hop | 96 | 37 | 38.5% |
| adversarial | 446 | 11 | 2.5% |
| **Non-Adv** | **1540** | **1039** | **67.5%** |

## Comparison with Single-Conversation Result

| Metric | Conv-0 Only (exp10) | Full 10-Conv | Delta |
|--------|-------------------|-------------|-------|
| Non-Adv | 69.7% | 67.5% | -2.2pp |
| Open-domain | 82.9% | 81.9% | -1.0pp |
| Temporal | 73.0% | 64.8% | -8.2pp |
| Single-hop | 43.8% | 37.2% | -6.6pp |
| Multi-hop | 53.8% | 38.5% | -15.3pp |

Conv-0 was above average for all categories. The full 10-conv result is
lower but still shows clear improvement over the original baseline.

## Key Findings

1. **Two-lane search holds across conversations.** 67.5% non-adv across 1986
   questions is a real improvement. No conversation scored below 62.6%.

2. **Open-domain is rock-solid.** 81.9% across 841 questions with very low
   variance. This is the most reliable category.

3. **Temporal shows more variance.** 64.8% across 10 convs vs 73.0% on conv-0.
   Temporal questions are sensitive to conversation structure and date resolution.

4. **Single-hop and multi-hop are the weakest categories.** 37.2% and 38.5%
   respectively. These need fundamentally different retrieval approaches
   (better embeddings, entity-based lookup, or graph traversal for multi-hop).

5. **Adversarial is near-zero (2.5%).** This is expected — adversarial questions
   are designed to trick the system and aren't the optimization target.

## Verdict: VALIDATED — Two-lane search is the production configuration

67.5% non-adv across all 10 conversations confirms the two-lane approach
works broadly, not just on a single favorable conversation. The per-conversation
range (62.6%-75.3%) shows healthy variance without outliers.
