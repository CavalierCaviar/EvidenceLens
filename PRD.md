# EvidenceLens

## Research-Grade Product Requirements Document

### Contradiction-Aware, Evidence-Weighted Academic Research Assistant

**Version:** 2.0
**Status:** Proposed
**Document Type:** Product Requirements Document + Research Specification
**Primary Objective:** Build a usable academic research assistant whose contradiction detection and evidence-strength estimation components can be independently evaluated and compared against established baselines.

---

# 1. Executive Summary

EvidenceLens is a multi-document academic research assistant designed to improve upon conventional Retrieval-Augmented Generation (RAG) systems for literature understanding.

A conventional academic RAG system generally performs:

```text
Research Question
       ↓
Retrieve Relevant Passages
       ↓
LLM
       ↓
Summary + Citations
```

EvidenceLens adds two explicit analytical layers before synthesis:

```text
Research Question
       ↓
Evidence Retrieval
       ↓
Claim Extraction
       ↓
Claim Normalization
       ↓
┌───────────────────────────────┐
│                               │
▼                               ▼
Contradiction Analysis    Evidence Analysis
│                               │
▼                               ▼
Claim Relationship         Evidence Strength
Classification             Estimation
│                               │
└───────────────┬───────────────┘
                ▼
       Evidence Aggregation
                ↓
       Evidence-Aware LLM
          Synthesis
                ↓
      Answer + Citations
      + Conflicts
      + Evidence Weights
      + Uncertainty
```

The system is not intended to determine scientific truth.

Its purpose is to:

1. retrieve relevant academic evidence;
2. identify and normalize research claims;
3. detect whether claims support, contradict, or fail to meaningfully compare with one another;
4. estimate the relative strength or certainty of individual pieces of evidence using observable study characteristics and an explicitly defined annotation framework;
5. aggregate evidence across studies;
6. produce a traceable synthesis that preserves disagreement rather than hiding it.

The project is designed simultaneously as:

* a usable research product;
* an ML/NLP project;
* an experimental research platform;
* a reproducible benchmark;
* a potential foundation for a future journal publication.

---

# 2. Proposed Research Positioning

## 2.1 Core research problem

Academic literature often contains heterogeneous and apparently conflicting findings.

A conventional RAG system may retrieve those findings but leave the LLM responsible for interpreting them.

This creates a structural weakness:

> Retrieval tells the model what evidence exists; it does not explicitly tell the model how the evidence relates or how much confidence should be placed in each finding.

EvidenceLens addresses this by introducing two explicit analytical tasks:

### Task A — Scientific claim relationship classification

Given two related research claims:

```text
Claim A
Claim B
```

classify their relationship as:

```text
CONTRADICTION
SUPPORT / AGREEMENT
NEUTRAL / NOT COMPARABLE
```

### Task B — Evidence-strength estimation

Given a study and a specific claim/outcome:

```text
Study characteristics
+
Claim characteristics
+
Statistical information
```

estimate a continuous evidence-strength score and an interpretable categorical label.

These outputs are then supplied to the synthesis layer.

---

# 3. Research Hypotheses

The system should be developed around explicit hypotheses rather than simply demonstrating that the application works.

## H1 — Contradiction-aware analysis

Adding explicit claim-level contradiction analysis will improve the system's ability to identify conflicting findings compared with conventional RAG.

## H2 — Evidence-aware synthesis

Adding evidence-strength information will improve human-rated usefulness and reliability of literature synthesis compared with contradiction-aware RAG without evidence weighting.

## H3 — Structured claim representation

Representing claims using structured semantic attributes will improve contradiction classification compared with raw sentence-pair classification alone.

## H4 — Interpretable evidence estimation

A transparent evidence-strength estimator based on study characteristics and expert-labelled targets will produce scores that correlate with expert assessment.

## H5 — Grounded synthesis

Providing the LLM with structured supporting and contradictory evidence will reduce unsupported synthesis and improve citation-groundedness compared with standard RAG.

These hypotheses define the eventual experimental program.

---

# 4. Research Questions

The project should investigate the following research questions.

### RQ1

Can explicit claim-level contradiction detection identify conflicting findings in academic literature more reliably than conventional RAG?

### RQ2

Does structured claim normalization improve scientific contradiction detection?

### RQ3

Can study characteristics and claim-level evidence characteristics be used to estimate evidence strength in a way that agrees with expert assessment?

### RQ4

Does evidence-aware aggregation improve the quality of generated literature syntheses?

### RQ5

Does contradiction-aware and evidence-aware synthesis outperform conventional RAG in identifying uncertainty and disagreement?

### RQ6

Which components contribute most to final system performance?

These questions should map directly to the experiments reported in the eventual research paper.

---

# 5. Product Vision

EvidenceLens should answer a question that ordinary academic summarizers do not answer explicitly:

> **What does the literature say, where does it disagree, and how strong is the evidence behind each side?**

The system should therefore not optimize for producing a single confident conclusion.

It should optimize for:

* evidence retrieval;
* disagreement visibility;
* source traceability;
* evidence-quality transparency;
* calibrated uncertainty;
* faithful synthesis.

---

# 6. Target Users

## Primary users

### Students

Students performing:

* literature reviews;
* research projects;
* thesis preparation;
* academic assignments;
* research paper preparation.

### Researchers

Researchers exploring:

* conflicting findings;
* emerging topics;
* large collections of papers;
* competing hypotheses;
* research gaps.

### Secondary users

* research assistants;
* academic labs;
* faculty;
* technical reviewers.

---

# 7. Example Use Case

User uploads 30 papers and asks:

> Does intermittent fasting improve insulin sensitivity?

EvidenceLens should not simply answer:

> Yes, intermittent fasting improves insulin sensitivity.

Instead it should produce something closer to:

```text
OVERALL FINDING

The available evidence generally favors an improvement,
but the literature is heterogeneous and several studies
report null or context-dependent effects.

SUPPORTING EVIDENCE

Study A
Positive effect
Evidence strength: High

Study B
Positive effect
Evidence strength: Moderate

CONTRADICTORY EVIDENCE

Study C
No significant effect
Evidence strength: Moderate

Study D
Negative effect
Evidence strength: Low

INTERPRETATION

The strongest evidence favors a positive association,
but the contradictory evidence indicates that the effect
may depend on population, intervention duration, or study design.
```

Every statement should be traceable to source material.

---

# 8. Scope

## 8.1 Version 1 — Required

The first research-grade version must include:

* PDF ingestion;
* document parsing;
* metadata extraction;
* semantic chunking;
* embeddings;
* vector retrieval;
* optional lexical retrieval;
* reranking;
* claim extraction;
* claim normalization;
* structured claim representation;
* candidate claim pairing;
* scientific NLI;
* logistic-regression contradiction classifier;
* study-characteristic extraction;
* evidence-strength annotation;
* linear-regression evidence-strength estimator;
* transparent evidence rubric;
* cross-study aggregation;
* evidence-aware synthesis;
* citations;
* source passage traceability;
* contradiction explorer;
* evidence breakdown;
* evaluation mode;
* experiment logging;
* reproducible test datasets.

---

# 9. Explicit Non-Goals for Version 1

The following should not become implementation distractions:

* automatic systematic-review registration;
* autonomous paper discovery from the entire internet;
* automated peer review;
* automated scientific fact adjudication;
* automated medical recommendations;
* citation graph analysis;
* automated retraction detection;
* full multimodal figure interpretation;
* automated meta-analysis;
* automated statistical re-analysis of arbitrary papers;
* autonomous research conclusions without source evidence.

These can be future research directions.

---

# 10. Core System Architecture

The complete architecture is:

```text
                    ┌───────────────────────┐
                    │   Academic Papers     │
                    └───────────┬───────────┘
                                ↓
                    ┌───────────────────────┐
                    │ Document Processing   │
                    └───────────┬───────────┘
                                ↓
                    ┌───────────────────────┐
                    │ Text + Metadata       │
                    │ + Page Information    │
                    └───────────┬───────────┘
                                ↓
                    ┌───────────────────────┐
                    │ Semantic Chunking      │
                    └───────────┬───────────┘
                                ↓
                    ┌───────────────────────┐
                    │ Embedding Generation  │
                    └───────────┬───────────┘
                                ↓
                    ┌───────────────────────┐
                    │ Hybrid Retrieval      │
                    │ Vector + Keyword      │
                    └───────────┬───────────┘
                                ↓
                    ┌───────────────────────┐
                    │ Reranking             │
                    └───────────┬───────────┘
                                ↓
                    ┌───────────────────────┐
                    │ Relevant Evidence     │
                    └───────────┬───────────┘
                                ↓
                    ┌───────────────────────┐
                    │ Claim Extraction      │
                    └───────────┬───────────┘
                                ↓
                    ┌───────────────────────┐
                    │ Claim Normalization   │
                    └───────────┬───────────┘
                                ↓
                    ┌───────────────────────┐
                    │ Structured Claims     │
                    └───────────┬───────────┘
                                ↓
             ┌──────────────────┴──────────────────┐
             │                                     │
             ▼                                     ▼
┌─────────────────────────┐          ┌─────────────────────────┐
│ Contradiction Engine    │          │ Evidence Engine         │
│                         │          │                         │
│ Semantic Pairing        │          │ Study Design            │
│ NLI Features            │          │ Sample Size             │
│ Direction Features      │          │ Effect Characteristics  │
│ Logistic Regression     │          │ Precision               │
│ Calibration             │          │ Directness              │
└────────────┬────────────┘          │ Annotated Target        │
             │                       │ Linear Regression       │
             │                       └────────────┬────────────┘
             │                                    │
             └──────────────────┬─────────────────┘
                                ▼
                    ┌───────────────────────┐
                    │ Evidence Graph /      │
                    │ Claim Graph            │
                    └───────────┬───────────┘
                                ↓
                    ┌───────────────────────┐
                    │ Evidence Aggregation  │
                    └───────────┬───────────┘
                                ↓
                    ┌───────────────────────┐
                    │ Evidence-Aware LLM    │
                    │ Synthesis              │
                    └───────────┬───────────┘
                                ↓
              ┌─────────────────────────────────────┐
              │ Final Research Answer               │
              │                                     │
              │ Overall finding                     │
              │ Supporting evidence                 │
              │ Contradictory evidence              │
              │ Evidence strength                   │
              │ Uncertainty                         │
              │ Citations                           │
              └─────────────────────────────────────┘
```

---

# 11. Architectural Principle

The LLM must not be responsible for every task.

The system should deliberately separate:

```text
Retrieval
Understanding
Classification
Scoring
Aggregation
Generation
```

This allows each stage to be tested independently.

It also makes the system suitable for scientific evaluation.

---

# 12. Module A — Document Ingestion

## FR-A01 — Upload

The user must be able to upload multiple PDF files.

Each file receives:

```text
paper_id
document_hash
filename
ingestion_timestamp
```

The document hash allows duplicate detection.

---

# 13. Document Parsing

The parser should extract:

* title;
* authors;
* abstract;
* publication year;
* DOI;
* section headings;
* page numbers;
* paragraphs;
* tables where text extraction is available;
* references where detectable.

The system must preserve page-level provenance.

Example:

```json
{
  "paper_id": "P001",
  "page": 7,
  "section": "Results",
  "text": "..."
}
```

---

# 14. Provenance Requirement

Every downstream object must retain its ancestry.

```text
Answer
 ↓
Evidence Group
 ↓
Claim
 ↓
Chunk
 ↓
Page
 ↓
Paper
```

No claim should exist without source provenance.

No generated answer statement should be accepted as grounded unless it can be traced through this chain.

---

# 15. Module B — Retrieval

## FR-B01 — Research Question

The user enters a natural-language research question.

Example:

> Does exercise improve cognitive performance in older adults?

---

# 16. Retrieval Pipeline

Initial implementation:

```text
Question
   ↓
Query embedding
   ↓
Dense retrieval
   +
Keyword retrieval
   ↓
Candidate passages
   ↓
Reranker
   ↓
Top-K evidence
```

The system should prefer evidence distributed across multiple papers.

A single paper should not dominate retrieval merely because it contains many semantically similar passages.

---

# 17. Retrieval Diversity

The retrieval engine should support document diversity.

For example:

```text
Top 10 passages

Paper A → 3
Paper B → 2
Paper C → 2
Paper D → 2
Paper E → 1
```

rather than:

```text
Paper A → 10
```

where appropriate.

This prevents the synthesis layer from accidentally treating one paper as the entire literature.

---

# 18. Retrieval Evaluation

The retrieval subsystem must be independently evaluable.

Metrics:

* Recall@K;
* Precision@K where gold evidence exists;
* MRR;
* nDCG;
* document recall;
* evidence recall.

A small manually annotated retrieval benchmark should be created.

---

# 19. Module C — Claim Extraction

The system identifies research claims from retrieved evidence.

Example:

Source:

> Participants receiving intervention X experienced a statistically significant 15% improvement in memory scores.

Extracted claim:

```text
Intervention X improves memory performance.
```

The claim must retain:

```text
claim_id
paper_id
chunk_id
page
source_text
```

---

# 20. Claim Types

The system should initially distinguish:

```text
RESULT
ASSOCIATION
CAUSAL CLAIM
NULL FINDING
METHODOLOGICAL CLAIM
BACKGROUND CLAIM
```

This classification should be treated as metadata, not as scientific truth.

---

# 21. Module D — Claim Normalization

Different wording may represent the same finding.

Example:

```text
Treatment improved memory.

Memory performance increased following treatment.

Participants demonstrated improved memory after treatment.
```

Normalize to:

```text
Intervention → Treatment
Outcome      → Memory performance
Direction    → Positive
```

---

# 22. Structured Claim Representation

Each claim should ideally contain:

```json
{
  "population": "...",
  "intervention": "...",
  "comparator": "...",
  "outcome": "...",
  "direction": "positive",
  "effect": "...",
  "statistical_status": "significant",
  "magnitude": "...",
  "claim_type": "result"
}
```

Fields may be null.

The system must never fabricate missing values.

---

# 23. PICO/PECO-Aware Representation

Where appropriate, the claim engine should map information to:

```text
Population
Intervention / Exposure
Comparator
Outcome
```

For observational studies, exposure can replace intervention.

This allows the system to distinguish:

```text
Same intervention
+
Different population
```

from:

```text
Same intervention
+
Same population
+
Opposite outcome
```

The latter is much stronger evidence of contradiction.

---

# 24. Module E — Candidate Claim Pairing

Comparing every claim with every other claim is computationally inefficient and increases false positives.

If there are:

```text
N claims
```

naive pairwise comparison requires:

```text
N(N-1)/2
```

comparisons.

Instead:

```text
All claims
   ↓
Embedding similarity
   ↓
Structured-field filtering
   ↓
Candidate pairs
   ↓
NLI
```

Candidate pairs should generally have:

* similar intervention/exposure;
* similar outcome;
* compatible population;
* sufficient semantic similarity.

---

# 25. Module F — Contradiction Detection

This is one of the two primary research components.

The system classifies pairs into:

```text
CONTRADICTION
SUPPORT / AGREEMENT
NEUTRAL / NOT COMPARABLE
```

---

# 26. Why Three Classes?

Binary classification is insufficient.

Example:

```text
A: Exercise improves cognition in adults over 60.

B: Exercise improves cognition in adults under 30.
```

These are not necessarily contradictory.

They are:

```text
NEUTRAL / DIFFERENT POPULATION
```

Similarly:

```text
A: Exercise improves cognition.

B: Exercise improves cognition substantially.
```

is agreement, not contradiction.

---

# 27. Contradiction Taxonomy

The system should identify at least:

### Direct contradiction

```text
A: X improves Y.
B: X does not improve Y.
```

### Directional contradiction

```text
A: X increases Y.
B: X decreases Y.
```

### Statistical contradiction

```text
A: Significant positive effect.
B: No significant effect.
```

### Magnitude disagreement

```text
A: Large improvement.
B: Small or negligible improvement.
```

This should initially be treated cautiously because different estimates can both be compatible depending on confidence intervals and study populations.

---

# 28. Contradiction vs Heterogeneity

This distinction is critical.

Different findings do not automatically constitute contradictions.

The system should explicitly consider:

```text
Population
Intervention
Comparator
Outcome
Measurement
Time period
Study design
```

before labelling a pair contradictory.

Example:

```text
Study A:
Effect positive in adults.

Study B:
Effect negative in children.
```

Output:

```text
Relationship: NOT DIRECTLY COMPARABLE
Reason:
Different population.
```

This is preferable to a false contradiction.

---

# 29. Contradiction Model

The contradiction classifier should use a hybrid approach.

```text
Claim A
Claim B
   ↓
Transformer/NLI model
   ↓
Semantic features
   +
Structured claim differences
   +
Direction features
   +
Statistical features
   ↓
Logistic Regression
   ↓
Probability distribution
```

Output:

```json
{
  "relationship": "contradiction",
  "p_contradiction": 0.91,
  "p_support": 0.06,
  "p_neutral": 0.03
}
```

---

# 30. Role of Logistic Regression

Logistic Regression is not being used because it "understands language."

It is used as an interpretable supervised classifier over NLP-derived features.

Example features:

```text
semantic_similarity
nli_contradiction_probability
nli_entailment_probability
direction_match
direction_conflict
population_similarity
intervention_similarity
outcome_similarity
statistical_status_difference
effect_direction_difference
```

This gives the project a legitimate classical ML component.

---

# 31. Contradiction Classifier Baselines

The research implementation should compare:

### Baseline A

Embedding cosine similarity only.

### Baseline B

Transformer/NLI model directly.

### Baseline C

Logistic Regression using engineered NLP features.

### Model D

NLI features + structured claim features + Logistic Regression.

The final model should only be considered justified if it improves the evaluation metrics.

---

# 32. Contradiction Confidence

The classifier must output calibrated probabilities.

UI categories may be:

```text
0.85–1.00 → High confidence
0.65–0.84 → Moderate confidence
<0.65     → Low / possible
```

These thresholds are initial engineering thresholds, not universal scientific standards.

Thresholds must be tuned on validation data.

---

# 33. Probability Calibration

The system should evaluate whether:

```text
0.90 probability
```

actually corresponds approximately to:

```text
90% correctness
```

Calibration methods may include:

* Platt scaling;
* isotonic regression.

Metrics:

* Brier score;
* Expected Calibration Error;
* reliability diagrams.

This is important because the UI must not display misleading confidence numbers.

---

# 34. Module G — Study Characteristic Extraction

The Evidence Engine requires structured study information.

Initial fields:

```text
study_design
sample_size
population
intervention
comparator
outcome
study_duration
effect_size
confidence_interval
p_value
```

Optional fields:

```text
number_of_groups
follow_up_duration
attrition
replication_status
```

---

# 35. Study Design Taxonomy

Initial taxonomy:

```text
Meta-analysis
Systematic Review
Randomized Controlled Trial
Cohort Study
Case-Control Study
Cross-Sectional Study
Case Study
Pilot Study
Other
Unknown
```

The taxonomy should be configurable because evidence hierarchies differ across research domains.

---

# 36. Unknown Handling

If the model cannot confidently identify:

```text
study design
sample size
effect size
p-value
confidence interval
```

the value must be:

```text
UNKNOWN
```

not an inferred value.

This is a hard reliability requirement.

---

# 37. Evidence Strength — Conceptual Correction

EvidenceLens must distinguish:

### Evidence strength

How much confidence the system should place in a particular finding given the available study characteristics and assessment criteria.

from:

### Scientific truth

Whether the finding is actually true in the real world.

EvidenceLens estimates the first.

It does not establish the second.

---

# 38. Evidence Strength Model

The system should use two complementary models.

## Model 1 — Transparent baseline

A deterministic evidence rubric.

## Model 2 — Learned estimator

A Linear Regression model trained against expert-labelled evidence-strength targets.

This distinction is important for research.

The transparent model provides interpretability.

The regression model tests whether observable study features can reproduce expert assessments.

---

# 39. Evidence Assessment Dimensions

The evidence rubric should consider, where applicable:

```text
Study design
Sample size / information size
Precision
Directness
Risk-of-bias indicators
Statistical evidence
Effect magnitude
Consistency with independent studies
```

These dimensions are inspired by established evidence-assessment methodology, but EvidenceLens must not claim to implement GRADE unless the complete methodology is actually implemented. GRADE explicitly distinguishes certainty of evidence from recommendation strength and considers domains such as risk of bias, inconsistency, indirectness and imprecision.

---

# 40. Evidence Feature Categories

### A. Study design

Encoded categorically or numerically.

### B. Sample size

Use a transformed value where appropriate, such as:

```text
log(1 + n)
```

to prevent enormous studies from dominating purely through sample size.

### C. Precision

Where confidence intervals are available:

```text
CI width
```

may serve as an input.

### D. Statistical evidence

Potential features:

```text
p-value
effect estimate
confidence interval
```

### E. Directness

How closely the study matches the user's research question:

```text
population match
intervention match
outcome match
```

### F. Consistency

Whether independent studies produce similar directional findings.

---

# 41. Critical Evidence Rule

Evidence weighting must never overwrite the original finding.

If a paper reports:

```text
No significant effect
```

the system must preserve:

```text
Finding: No significant effect
Evidence strength: Low
```

It must never transform that into:

```text
The treatment probably works.
```

Evidence strength modifies the interpretation of evidence.

It does not modify the evidence itself.

---

# 42. Evidence Strength Target

For supervised learning, the project requires a labelled target.

Each annotated study/claim pair receives:

```text
0.00–1.00
```

representing expert-assessed evidence strength under the project's explicit rubric.

Additionally:

```text
Very Weak
Weak
Moderate
Strong
```

may be derived from the continuous score for UI purposes.

The continuous target is the research target.

The categorical label is primarily a presentation layer.

---

# 43. Human Annotation Protocol

Annotators should receive:

```text
Research question
Claim
Source passage
Study characteristics
```

They should independently assess evidence strength.

They should not see:

* the model prediction;
* other annotators' ratings;
* the final LLM synthesis.

This prevents confirmation bias.

---

# 44. Annotation Rubric

Annotators should evaluate defined dimensions rather than simply asking:

> "Does this paper look good?"

Example:

| Dimension               | Score |
| ----------------------- | ----: |
| Study design            |   0–1 |
| Information size        |   0–1 |
| Precision               |   0–1 |
| Directness              |   0–1 |
| Risk-of-bias indicators |   0–1 |
| Statistical support     |   0–1 |

The final expert target can be generated using a predefined annotation protocol or expert overall rating.

The exact aggregation rule must be frozen before testing.

---

# 45. Annotator Agreement

At least two independent annotators should label a substantial subset.

Measure:

* Cohen's kappa for categorical labels;
* Spearman correlation for ordinal/continuous ratings;
* intraclass correlation where appropriate.

Disagreements should be adjudicated separately.

The test set must not be silently changed after seeing model results.

---

# 46. Linear Regression Evidence Estimator

Input:

```text
Study characteristics
+
Claim characteristics
+
Evidence characteristics
```

Output:

```text
Predicted evidence strength ∈ [0,1]
```

The model should be trained only on the training set.

Example:

```text
X =
[
  study_design_score,
  log_sample_size,
  precision_score,
  directness_score,
  risk_of_bias_score,
  statistical_support_score,
  ...
]

y =
expert_evidence_strength
```

---

# 47. Why Linear Regression?

Linear Regression provides:

* interpretability;
* coefficient inspection;
* reproducibility;
* a strong classical ML baseline;
* a clear way to test whether the chosen features have predictive value.

The model should not be presented as the universally correct scientific evidence model.

It is an experimentally testable estimator.

---

# 48. Evidence Model Baselines

Compare:

### Baseline A

Study-design-only heuristic.

### Baseline B

Transparent weighted rubric.

### Model C

Linear Regression.

### Optional Model D

A nonlinear model such as Random Forest or Gradient Boosting.

The optional nonlinear model should be included only if project scope permits.

This creates a scientifically meaningful comparison:

> Does a learned model improve upon a transparent evidence rubric?

---

# 49. Regression Metrics

Use:

* MAE;
* RMSE;
* R²;
* Spearman correlation;
* calibration/error plots.

If the target distribution is ordinal rather than genuinely continuous, the paper should report that limitation rather than overstate the interpretation of R².

---

# 50. Evidence Strength and Contradiction Are Separate

A weak study contradicting a strong study is still a contradiction.

Example:

```text
Study A:
Positive result
Evidence strength = 0.91

Study B:
Negative result
Evidence strength = 0.31
```

The system should output:

```text
Contradiction detected.

Higher-weight evidence:
Study A

Lower-weight contradictory evidence:
Study B
```

It must not output:

```text
No contradiction because Study B is weak.
```

This distinction is central to the architecture.

---

# 51. Module H — Cross-Study Evidence Aggregation

Once claims and evidence weights exist, the system creates evidence groups.

Example:

```text
Research question
      ↓
Outcome: cognitive performance
      ↓
┌───────────────┬───────────────┐
│               │               │
Positive       Null            Negative
claims         claims          claims
│               │               │
↓               ↓               ↓
Weights         Weights         Weights
```

The system should report:

* number of supporting studies;
* number of contradictory studies;
* aggregate evidence weight;
* distribution of study designs;
* major sources of heterogeneity.

---

# 52. Evidence Aggregation Must Not Be a Vote

The system must not use:

```text
7 papers say yes
3 papers say no
therefore yes
```

as its primary logic.

It should instead preserve:

```text
direction
+
study quality
+
independence
+
comparability
+
precision
+
consistency
```

A large number of weak studies should not automatically outweigh fewer strong studies.

---

# 53. Independence

If multiple papers use the same dataset, they should not automatically count as independent replications.

Where detectable, the system should record:

```text
dataset_overlap
author_overlap
study_cohort_overlap
```

For Version 1, these fields may be:

```text
Unknown
```

unless reliable evidence exists.

The architecture must nevertheless permit them.

---

# 54. Evidence Graph

The system should internally represent relationships as a graph.

```text
                 Claim C
                   ▲
                   │ supports
                   │
Paper A ─────── Claim A ─────── contradicts ─────── Claim B ───── Paper B
                   │
                   │ supports
                   ▼
                 Paper C
```

Nodes:

```text
Paper
Claim
Evidence
Research Question
```

Edges:

```text
supports
contradicts
neutral
derived_from
```

This structure will later support graph visualization and advanced research features.

---

# 55. Module I — Evidence-Aware Synthesis

The LLM receives structured evidence.

It should not independently infer evidence strength from raw papers when the Evidence Engine has already produced structured assessments.

Input:

```text
Research question

Supporting claims
Contradictory claims
Neutral claims

Evidence strengths
Study characteristics
Contradiction probabilities
Source passages
```

---

# 56. Synthesis Prompt Contract

The synthesis model must follow strict rules:

1. Use only supplied evidence.
2. Preserve contradictions.
3. Never invent study characteristics.
4. Never modify reported results.
5. Distinguish evidence strength from truth.
6. Cite every substantive research claim.
7. Explicitly state uncertainty.
8. Say when evidence is insufficient.
9. Avoid presenting model confidence as scientific certainty.

---

# 57. Required Final Answer

The final answer should contain:

## Overall Finding

A concise answer.

## Evidence Supporting the Finding

Major supporting studies.

## Contradictory Evidence

Important conflicting findings.

## Evidence Strength

Why some evidence receives more or less weight.

## Sources of Disagreement

Potential differences in:

* population;
* methodology;
* intervention;
* measurement;
* sample size;
* study design.

## Uncertainty

What cannot currently be concluded.

## Sources

Paper-level and passage-level citations.

---

# 58. Example Final Output

```text
OVERALL FINDING

The literature generally favors a positive effect, but the evidence
is heterogeneous and does not establish a universal effect.

SUPPORTING EVIDENCE

Paper A
Positive finding
RCT
n = 1,240
Evidence strength: 0.88

Paper B
Positive finding
RCT
n = 430
Evidence strength: 0.76

CONTRADICTORY EVIDENCE

Paper C
No significant effect
Observational
n = 74
Evidence strength: 0.41

Paper D
Negative association
Pilot study
n = 32
Evidence strength: 0.24

INTERPRETATION

The strongest available studies favor a positive effect. However,
the contradictory studies indicate that the effect may depend on
population or intervention characteristics.

UNCERTAINTY

The available evidence does not establish whether the effect is
consistent across all populations.
```

---

# 59. Citation Grounding

Every generated evidence statement must point to:

```text
Answer statement
      ↓
Claim
      ↓
Source passage
      ↓
Page
      ↓
Paper
```

The UI should allow the user to click a citation and see the exact supporting passage.

---

# 60. Unsupported Claim Detection

Before returning the final answer, the system should run a validation stage.

```text
Generated answer
       ↓
Claim extraction
       ↓
Compare with evidence graph
       ↓
Unsupported claim?
       ↓
YES → revise / remove
NO  → continue
```

This is preferable to assuming that a capable LLM will always remain grounded.

---

# 61. Uncertainty Handling

The system must be able to say:

```text
Insufficient evidence.
```

Examples:

* too few papers;
* contradictory evidence of similar strength;
* poor retrieval coverage;
* missing study characteristics;
* low classifier confidence;
* claims not directly comparable.

This is a required feature.

---

# 62. UI Design

The interface should have five primary areas.

### 1. Research Question

```text
[ Does X improve Y?                     ]
```

### 2. Overall Finding

```text
┌──────────────────────────────────────┐
│ OVERALL FINDING                      │
│                                      │
│ Evidence generally favors X, but     │
│ important contradictory findings     │
│ remain.                              │
└──────────────────────────────────────┘
```

### 3. Evidence Landscape

```text
Supporting       Contradicting       Neutral

██████████       █████              ███
Strong           Moderate           Low
```

### 4. Contradiction Explorer

Side-by-side claims.

### 5. Source Explorer

Exact source passage and page.

---

# 63. Contradiction Explorer

Example:

```text
┌────────────────────────┐
│ PAPER A                │
│                        │
│ X improves Y           │
│                        │
│ RCT                    │
│ n = 1,240              │
│ Evidence: 0.88         │
└───────────┬────────────┘
            │
       CONTRADICTION
       P = 0.93
            │
┌───────────┴────────────┐
│ PAPER B                │
│                        │
│ X does not improve Y   │
│                        │
│ Cohort                 │
│ n = 210                │
│ Evidence: 0.54         │
└────────────────────────┘
```

Clicking either claim opens the original passage.

---

# 64. Evidence Explanation

The user must be able to ask:

> Why is this evidence rated higher?

The system should answer from structured features.

Example:

```text
Evidence strength: 0.82

Primary contributing factors:

+ Randomized controlled study design
+ Large sample size
+ Narrow confidence interval
+ Direct match to research population

Limitations:

- Single study
- No independent replication identified
```

The explanation must not be generated solely from the LLM.

It should be derived from stored model features.

---

# 65. Research / Experiment Mode

EvidenceLens must include a separate evaluation interface.

This is essential for turning the project from a demo into a research platform.

The user should be able to select:

```text
Dataset
Retriever
Embedding model
NLI model
Contradiction classifier
Evidence model
LLM
Top-K
Threshold
```

and run an experiment.

---

# 66. Experiment Configuration

Every experiment receives:

```text
experiment_id
timestamp
dataset_version
code_version
model_versions
hyperparameters
random_seed
evaluation_split
```

This allows experiments to be reproduced.

---

# 67. Experiment Registry

Example:

```text
EXP-001
Baseline RAG

EXP-002
RAG + NLI

EXP-003
RAG + NLI + Logistic Regression

EXP-004
RAG + Evidence Rubric

EXP-005
Full EvidenceLens
```

Results should be stored rather than overwritten.

---

# 68. Benchmark Mode

The system should provide a fixed benchmark dataset containing:

```text
Research question
Paper IDs
Claims
Claim relationships
Evidence annotations
Gold source passages
```

The benchmark must be versioned.

---

# 69. Internal Test Dataset

In addition to public datasets, create a project-specific dataset.

Target:

```text
100–500 claim pairs
```

initially, with expansion if resources permit.

Each pair:

```text
claim_a
claim_b
relationship
paper_a
paper_b
population
intervention
outcome
annotator_ids
adjudicated_label
```

The eventual paper should report the exact dataset size and construction protocol.

---

# 70. Public Benchmark Datasets

The project should investigate suitable scientific NLP benchmarks before finalizing the evaluation dataset.

SciFact is particularly relevant because it contains expert-written scientific claims paired with evidence-containing abstracts and labels/rationales.

SciNLI is relevant for scientific natural-language inference because it was specifically constructed from scientific text rather than ordinary-domain language.

These datasets should be used as external benchmarks where their label definitions match the intended task.

They should not be forced into the project's task if their definitions differ.

---

# 71. Dataset Strategy

The evaluation should contain three levels.

## Level 1 — Synthetic sanity tests

Controlled examples.

Purpose:

* detect obvious implementation failures;
* verify label logic;
* test edge cases.

## Level 2 — Public scientific benchmark

Purpose:

* external comparability;
* avoid evaluating only on internally created examples.

## Level 3 — Project-specific annotated corpus

Purpose:

* evaluate the exact EvidenceLens task;
* test cross-paper contradiction detection;
* evaluate evidence-strength estimation.

---

# 72. Synthetic Test Cases

The system should include deterministic tests such as:

### Test 1

```text
A: X increases Y.
B: X decreases Y.
Expected: CONTRADICTION
```

### Test 2

```text
A: X improves Y in adults.
B: X improves Y in children.
Expected: NEUTRAL / NOT COMPARABLE
```

### Test 3

```text
A: X improves Y.
B: X improves Y.
Expected: SUPPORT
```

### Test 4

```text
A: X has no significant effect on Y.
B: X significantly improves Y.
Expected: CONTRADICTION
```

### Test 5

```text
A: X improves Y.
B: X improves Z.
Expected: NEUTRAL
```

---

# 73. Data Splitting

No random sentence-level split should be used if multiple examples originate from the same paper.

The preferred split is:

```text
TRAIN
Paper 1
Paper 2
Paper 3
...

VALIDATION
Different papers

TEST
Completely different papers
```

All claims from a paper should remain in the same partition whenever feasible.

This prevents paper-level information leakage.

---

# 74. Cross-Domain Evaluation

If feasible, the benchmark should include more than one scientific domain.

For example:

```text
Biomedical
Computer Science
Psychology
Environmental Science
```

The goal is to determine whether the method generalizes.

If only one domain is used, the paper must explicitly state that limitation.

---

# 75. Contradiction Evaluation

Primary metrics:

```text
Macro F1
Contradiction F1
Precision
Recall
Accuracy
```

The primary research metric should be:

> Contradiction F1

because both false positives and false negatives matter.

---

# 76. Per-Class Evaluation

Report:

```text
Support F1
Contradiction F1
Neutral F1
```

This prevents the model from appearing strong merely because the majority class is easy to predict.

---

# 77. Confusion Matrix

Every contradiction experiment should produce:

```text
                 Predicted

              S      C      N

Actual S      ✓      ?      ?
Actual C      ?      ✓      ?
Actual N      ?      ?      ✓
```

This should be included in the research dashboard and eventually the paper.

---

# 78. Evidence Evaluation

Primary metrics:

```text
MAE
RMSE
R²
Spearman correlation
```

For categorical labels:

```text
Macro F1
Weighted F1
Confusion Matrix
```

The paper should emphasize correlation and agreement rather than pretending that expert evidence strength is an objective ground truth.

---

# 79. End-to-End Evaluation

Compare:

```text
                Baseline RAG
                      vs
             EvidenceLens
```

Evaluate:

* answer correctness;
* citation correctness;
* citation completeness;
* evidence coverage;
* contradiction recall;
* contradiction precision;
* evidence-strength agreement;
* hallucination/unsupported-claim rate;
* human-rated usefulness;
* uncertainty calibration.

RAG evaluation should be multidimensional rather than relying on a single score; retrieval relevance and generation faithfulness are distinct properties.

---

# 80. Ablation Study

At minimum:

## Experiment A — Vanilla RAG

```text
Question
 ↓
Retrieval
 ↓
LLM
```

## Experiment B — RAG + Claim Analysis

```text
Question
 ↓
Retrieval
 ↓
Claim Extraction
 ↓
LLM
```

## Experiment C — RAG + Contradiction Detection

```text
Question
 ↓
Retrieval
 ↓
Claims
 ↓
Contradiction Engine
 ↓
LLM
```

## Experiment D — RAG + Evidence Weighting

```text
Question
 ↓
Retrieval
 ↓
Evidence Engine
 ↓
LLM
```

## Experiment E — Full EvidenceLens

```text
Question
 ↓
Retrieval
 ↓
Claims
 ↓
Contradiction
 +
Evidence
 ↓
Aggregation
 ↓
LLM
```

This allows the paper to demonstrate whether each contribution actually matters.

---

# 81. Model Ablation

For contradiction detection:

```text
NLI only
vs
Logistic Regression only
vs
NLI + Logistic Regression
vs
NLI + structured features + Logistic Regression
```

For evidence estimation:

```text
Study-design heuristic
vs
Transparent rubric
vs
Linear Regression
vs
Optional nonlinear model
```

---

# 82. Statistical Significance

If two systems are compared, report uncertainty.

Depending on the metric:

* bootstrap confidence intervals;
* paired tests;
* approximate randomization;
* McNemar's test for paired classification outcomes.

The exact statistical test should be chosen based on the metric and experimental design.

Do not report:

```text
Model A = 82%
Model B = 84%
```

and automatically claim improvement.

---

# 83. Human Evaluation

A human evaluation should assess the final synthesis.

Annotators rate:

```text
Correctness
Completeness
Evidence grounding
Contradiction awareness
Evidence-weighting quality
Usefulness
Clarity
Uncertainty calibration
```

Use blinded evaluation where feasible.

Annotators should not know which system produced the answer.

---

# 84. Human Evaluation Question

One particularly important question:

> Did the system correctly identify the major disagreement present in the supplied literature?

This directly evaluates the project's central contribution.

---

# 85. Evidence-Weighting Human Evaluation

Annotators should answer:

> Given the supplied research question and study information, does the relative ordering of evidence strength appear justified?

Possible:

```text
Strongly justified
Somewhat justified
Neutral
Somewhat unjustified
Strongly unjustified
```

This complements numerical evaluation.

---

# 86. Citation Evaluation

For every generated answer:

```text
Generated claim
      ↓
Citation
      ↓
Retrieved passage
      ↓
Entailment check
```

Measure:

### Citation correctness

Does the cited passage support the statement?

### Citation completeness

Are major factual claims cited?

### Citation source validity

Does the citation actually refer to the claimed paper?

---

# 87. Unsupported Claim Rate

Define:

```text
Unsupported Claim Rate =
unsupported substantive claims
/
total substantive claims
```

This should be one of the principal end-to-end reliability metrics.

---

# 88. Contradiction Coverage

Define:

```text
Contradiction Coverage =
identified gold contradictions
/
total gold contradictions
```

This is useful because a system can produce highly fluent summaries while completely missing important disagreements.

---

# 89. Evidence-Weighted Synthesis Evaluation

Create questions where literature contains:

```text
Strong positive evidence
Weak negative evidence
```

and the inverse:

```text
Weak positive evidence
Strong negative evidence
```

The model should correctly communicate the asymmetry without deleting the weaker finding.

This is a critical controlled experiment.

---

# 90. Stress Tests

The system should be tested against:

### Conflicting sample sizes

```text
n = 12
vs
n = 10,000
```

### Different populations

```text
Adults
vs
Children
```

### Different outcomes

```text
Memory
vs
Reaction time
```

### Different study designs

```text
RCT
vs
Case report
```

### Null vs positive findings

```text
Significant improvement
vs
No significant effect
```

### Ambiguous language

```text
May improve
Could potentially improve
Evidence is insufficient
```

The model should not convert uncertainty into contradiction.

---

# 91. Failure Taxonomy

Every evaluation error should be assigned a category.

```text
Retrieval failure
Claim extraction failure
Claim normalization failure
Population mismatch
Outcome mismatch
NLI failure
Logistic regression failure
Metadata extraction failure
Evidence model failure
Aggregation failure
LLM synthesis failure
Citation failure
```

This allows systematic improvement rather than random debugging.

---

# 92. Reproducibility Requirements

Every research experiment must record:

```text
dataset version
train/validation/test split
model versions
embedding model
NLI model
LLM
hyperparameters
random seed
retrieval K
reranking configuration
classification threshold
evidence model version
```

The system should be capable of recreating a previous experiment from its configuration.

---

# 93. Research Artifact Export

The system should support exporting:

```text
claims.json
claim_pairs.json
contradictions.json
evidence_scores.json
experiments.json
evaluation_results.json
citations.json
```

Optionally:

```text
CSV
JSON
JSONL
```

This makes analysis outside the application straightforward.

---

# 94. Research Dashboard

The evaluation interface should display:

```text
Experiment: EXP-005

Retrieval Recall@10       0.82
Contradiction F1          0.78
Evidence Spearman         0.71
Citation Correctness      0.91
Unsupported Claim Rate    0.07

Baseline RAG
Contradiction Recall      0.42

EvidenceLens
Contradiction Recall      0.78
```

Actual values must only appear after experiments have been run.

No placeholder metrics should be presented as results.

---

# 95. Product Architecture

Recommended initial stack:

```text
Frontend
    ↓
React / Next.js
    ↓
FastAPI
    ↓
Application Services
    │
    ├── Document Service
    ├── Retrieval Service
    ├── Claim Service
    ├── Contradiction Service
    ├── Evidence Service
    ├── Synthesis Service
    └── Evaluation Service
    │
    ├── PostgreSQL
    └── Vector Database
```

The exact database technology may change during implementation.

The important requirement is modularity.

---

# 96. Service Separation

### Document Service

Handles:

* uploads;
* parsing;
* metadata;
* provenance.

### Retrieval Service

Handles:

* embeddings;
* vector search;
* keyword search;
* reranking.

### Claim Service

Handles:

* extraction;
* normalization;
* structured representation.

### Contradiction Service

Handles:

* candidate generation;
* NLI;
* logistic regression;
* calibration.

### Evidence Service

Handles:

* study characteristics;
* evidence features;
* rubric;
* linear regression;
* evidence explanations.

### Synthesis Service

Handles:

* evidence aggregation;
* prompt construction;
* LLM generation;
* grounding validation.

### Evaluation Service

Handles:

* benchmark datasets;
* experiments;
* metrics;
* ablations;
* result storage.

---

# 97. Data Model — Paper

```text
paper_id
document_hash
title
authors
year
doi
abstract
study_design
sample_size
metadata_confidence
source_path
```

---

# 98. Data Model — Chunk

```text
chunk_id
paper_id
page
section
text
embedding_id
retrieval_score
```

---

# 99. Data Model — Claim

```text
claim_id
paper_id
chunk_id

claim_text

population
intervention
comparator
outcome
direction
effect
magnitude

claim_type
statistical_status
effect_size
p_value
confidence_interval

extraction_confidence
```

---

# 100. Data Model — Claim Pair

```text
pair_id
claim_a
claim_b

semantic_similarity

population_similarity
intervention_similarity
outcome_similarity

nli_entailment_probability
nli_contradiction_probability
nli_neutral_probability

direction_conflict
statistical_conflict

relationship
relationship_confidence
```

---

# 101. Data Model — Evidence Assessment

```text
assessment_id
claim_id
study_design
sample_size

precision_score
directness_score
risk_of_bias_score
statistical_support_score
consistency_score

expert_score
rubric_score
linear_regression_score

final_evidence_score
evidence_category
```

---

# 102. Data Model — Experiment

```text
experiment_id
dataset_version
model_configuration
retriever_configuration
classifier_configuration
evidence_model_configuration

random_seed
created_at

metrics
artifact_paths
```

---

# 103. Data Model — Annotation

```text
annotation_id
annotator_id
item_id
task_type
label
score
timestamp
adjudicated
```

Annotator identities should be anonymized in research exports.

---

# 104. Security

Uploaded documents must be isolated by project/user.

Requirements:

* authentication;
* authorization;
* project-level document isolation;
* encrypted credentials;
* no API keys in frontend code;
* secure file handling;
* deletion support.

---

# 105. Privacy

The system should not use uploaded papers for model training unless explicitly authorized.

Research exports should remove:

* user identifiers;
* private filenames where necessary;
* access tokens;
* credentials.

---

# 106. Product Reliability Requirements

The system must never invent:

* sample size;
* study design;
* effect size;
* statistical significance;
* confidence interval;
* citation;
* source passage.

Unknown values must remain:

```text
Unknown
```

---

# 107. Graceful Failure

If retrieval is poor:

```text
The available papers do not provide sufficient evidence
to answer this question confidently.
```

If contradiction confidence is low:

```text
Potential disagreement detected, but the claims may not
be directly comparable.
```

If evidence data is missing:

```text
Evidence strength could not be estimated reliably because
key study characteristics were unavailable.
```

---

# 108. Version 1 Definition of Done

A user must be able to:

1. Create a research collection.
2. Upload multiple academic PDFs.
3. Process the papers.
4. Ask a research question.
5. Retrieve evidence across multiple papers.
6. Extract claims.
7. Normalize claims.
8. Compare relevant claims.
9. Detect support, contradiction, and neutral relationships.
10. View contradiction confidence.
11. View both original source passages.
12. Extract study characteristics.
13. Estimate evidence strength.
14. View why evidence received its score.
15. View aggregate supporting and contradictory evidence.
16. Generate an evidence-aware synthesis.
17. Open every citation to source evidence.
18. Receive an uncertainty response when evidence is insufficient.
19. Run the benchmark.
20. Compare against vanilla RAG.
21. Run ablation experiments.
22. Export experiment results.

---

# 109. Minimum Viable Research Prototype

Before building the full product, implement the smallest experimentally valid pipeline:

```text
10–30 papers
      ↓
PDF extraction
      ↓
Chunking
      ↓
Embeddings
      ↓
Retrieval
      ↓
Claim extraction
      ↓
Claim normalization
      ↓
NLI
      ↓
Logistic Regression
      ↓
Evidence rubric
      ↓
Linear Regression
      ↓
LLM synthesis
```

The prototype should be sufficient to generate preliminary results.

---

# 110. Prototype Dataset

Start with a deliberately small corpus.

Recommended initial structure:

```text
20–50 papers
50–150 claims
100–300 claim pairs
```

This is not the final publication dataset.

It is the engineering/debugging dataset.

---

# 111. Research Dataset Expansion

After the pipeline works:

```text
Phase 1
Engineering dataset

        ↓

Phase 2
Annotated development dataset

        ↓

Phase 3
Frozen test dataset

        ↓

Phase 4
External/public benchmark

        ↓

Phase 5
Cross-domain evaluation
```

The test set must be frozen before final model tuning.

---

# 112. Experimental Workflow

```text
Define hypothesis
      ↓
Create dataset
      ↓
Annotate
      ↓
Measure annotation agreement
      ↓
Freeze test set
      ↓
Train models
      ↓
Tune validation set
      ↓
Run test set
      ↓
Run ablations
      ↓
Statistical analysis
      ↓
Error analysis
      ↓
Document results
```

---

# 113. Publication-Oriented Experiment Structure

The eventual paper should be able to contain:

## Experiment 1

Retrieval performance.

## Experiment 2

Claim extraction/normalization.

## Experiment 3

Contradiction classification.

## Experiment 4

Evidence-strength estimation.

## Experiment 5

End-to-end synthesis.

## Experiment 6

Ablation study.

## Experiment 7

Human evaluation.

## Experiment 8

Cross-domain generalization.

Not all experiments must be implemented immediately, but the architecture should support them.

---

# 114. Expected Research Contribution

The project should not claim:

> "We built an AI research assistant."

That alone is not a meaningful research contribution.

The stronger research framing is:

> **We investigate whether explicit claim-level contradiction modelling and evidence-strength estimation can improve the reliability and usefulness of multi-document academic RAG synthesis.**

The contribution therefore lies in:

1. the task formulation;
2. the structured claim representation;
3. the contradiction-analysis pipeline;
4. the evidence-strength estimation framework;
5. the evidence-aware synthesis architecture;
6. the benchmark/annotation methodology;
7. the empirical comparison against conventional RAG.

---

# 115. What Is Actually Novel?

The project should avoid claiming novelty for:

* RAG itself;
* embeddings;
* vector databases;
* LLM summarization;
* Logistic Regression;
* Linear Regression;
* NLI individually.

These are established techniques.

Potential novelty lies in the **integration and experimental formulation**:

```text
Scientific retrieval
+
structured claims
+
cross-paper relationship modelling
+
evidence-strength estimation
+
evidence-aware synthesis
```

The eventual literature review must determine exactly how novel this combination is relative to existing systems.

---

# 116. Research Contribution Risk

The greatest research risk is:

> The system may simply combine existing techniques without producing a sufficiently novel contribution.

To mitigate this, the project must conduct a literature review before the final paper.

The review should specifically search for systems combining:

```text
RAG
+
scientific contradiction detection
+
claim-level evidence weighting
+
academic synthesis
```

The paper should then explicitly position EvidenceLens against the closest systems.

---

# 117. Important Scientific Limitation

Evidence strength cannot be universally reduced to:

```text
study design
+
sample size
+
p-value
```

Different disciplines use different standards.

Therefore EvidenceLens should use a **domain-configurable evidence framework**.

Example:

```text
Biomedical configuration
Psychology configuration
Computer Science configuration
Environmental Science configuration
```

Version 1 may implement only one configuration.

---

# 118. Domain Configuration

Each domain should eventually define:

```text
study taxonomy
evidence dimensions
feature mappings
annotation instructions
```

The model architecture remains unchanged.

This creates a clean separation between:

```text
general architecture
```

and:

```text
domain-specific evidence methodology
```

---

# 119. Evidence Score Interpretation

The system should never display:

```text
Scientific truth = 82%
```

Instead:

```text
Estimated evidence strength = 0.82
```

or:

```text
Expert-agreement evidence category = Strong
```

The UI should explain that the score represents the system's assessment under the configured framework.

---

# 120. Contradiction Interpretation

Similarly:

```text
Contradiction probability = 0.91
```

means:

> The classifier estimates a high probability that the two normalized claims represent contradictory findings.

It does not mean:

> One paper is wrong.

---

# 121. Evidence Aggregation Interpretation

The final system should distinguish:

```text
Finding
Evidence strength
Agreement
Consistency
Uncertainty
```

For example:

```text
Finding:
Positive association reported.

Evidence strength:
Moderate.

Cross-study consistency:
Low.

Contradiction:
Detected.

Overall confidence:
Moderate.
```

This is considerably more informative than a single confidence score.

---

# 122. Final Product Output Schema

Internally, the final answer should be represented approximately as:

```json
{
  "research_question": "...",
  "overall_finding": "...",

  "supporting_evidence": [],
  "contradictory_evidence": [],
  "neutral_evidence": [],

  "evidence_summary": {
    "overall_strength": 0.0,
    "consistency": 0.0,
    "uncertainty": "..."
  },

  "contradictions": [],

  "limitations": [],

  "citations": []
}
```

The UI may render this differently.

---

# 123. Research Output Export

The system should support a machine-readable research export containing:

```text
research_question
retrieval_results
claims
claim_pairs
relationships
evidence_features
evidence_scores
synthesis
citations
evaluation_metadata
```

This enables independent statistical analysis.

---

# 124. Future Extensions

The architecture should support later additions.

## Phase 2 — Claim-Level Citation Verification

Verify that generated claims are actually entailed by their cited passages.

## Phase 3 — Temporal Evidence Analysis

Track how findings change over time.

## Phase 4 — Methods-Aware Retrieval

Retrieve papers based on methodology as well as topic.

## Phase 5 — Citation Graph

Model:

```text
paper
 ↓ cites
paper
 ↓ challenges
paper
```

## Phase 6 — Table/Figure Extraction

Extract numerical evidence from scientific tables and figures.

## Phase 7 — Domain-Specific Evidence Frameworks

Support different evidence assessment methodologies.

## Phase 8 — Semi-Automated Literature Review

Assist researchers with systematic evidence screening while keeping humans in control.

---

# 125. Publication Roadmap

## Stage 1 — Engineering

Build the minimum working pipeline.

## Stage 2 — Preliminary Experiment

Demonstrate that:

```text
Full system
>
Vanilla RAG
```

on at least one measurable task.

## Stage 3 — Dataset Construction

Create and freeze a research-quality annotation set.

## Stage 4 — Model Evaluation

Run:

* baselines;
* ablations;
* statistical analysis;
* error analysis.

## Stage 5 — Cross-Domain Testing

Test whether the approach generalizes.

## Stage 6 — Paper

Structure the eventual paper as:

```text
Abstract
Introduction
Related Work
Problem Formulation
Methodology
System Architecture
Dataset
Annotation Protocol
Experimental Setup
Results
Ablation Study
Error Analysis
Human Evaluation
Limitations
Conclusion
```

---

# 126. Proposed Paper Contribution Table

| Contribution                               | Demonstrated by                |
| ------------------------------------------ | ------------------------------ |
| Claim-level contradiction framework        | Contradiction benchmark        |
| Structured scientific claim representation | Ablation                       |
| Interpretable contradiction classifier     | Logistic Regression comparison |
| Evidence-strength estimation               | Expert agreement               |
| Evidence-aware synthesis                   | Human evaluation               |
| Improved disagreement visibility           | Contradiction recall           |
| Improved grounded synthesis                | Citation/faithfulness metrics  |
| Generalization                             | Cross-domain evaluation        |

---

# 127. Main Success Criteria

The project should not be declared successful merely because the application produces attractive answers.

Success requires evidence that:

### Contradiction detection

The proposed model reliably identifies genuine conflicting claims.

### Evidence estimation

The proposed evidence model agrees meaningfully with human assessment.

### Synthesis

The final system preserves disagreement and does not collapse heterogeneous evidence into a misleading consensus.

### Grounding

Generated claims remain traceable to source evidence.

### Reproducibility

The reported results can be reproduced from frozen data and configuration.

---

# 128. Product Success Criteria

From a user perspective:

> A researcher should be able to upload a collection of papers, ask a research question, receive a literature synthesis, inspect disagreements between studies, understand why evidence received different weights, and trace every major conclusion back to its source.

---

# 129. Research Success Criteria

From a scientific perspective:

> The project should be able to answer experimentally whether explicit contradiction detection and evidence-strength estimation improve academic RAG beyond conventional retrieval-and-generation pipelines.

That is the core research question.

---

# 130. Final Architecture

The final conceptual architecture is:

```text
                         ACADEMIC PAPERS
                                │
                                ▼
                      DOCUMENT PROCESSING
                                │
                                ▼
                    TEXT + METADATA + PROVENANCE
                                │
                                ▼
                         SEMANTIC CHUNKS
                                │
                                ▼
                         HYBRID RETRIEVAL
                                │
                                ▼
                            RERANKING
                                │
                                ▼
                       RELEVANT EVIDENCE
                                │
                                ▼
                        CLAIM EXTRACTION
                                │
                                ▼
                       CLAIM NORMALIZATION
                                │
                                ▼
                     STRUCTURED CLAIM GRAPH
                                │
                    ┌───────────┴───────────┐
                    │                       │
                    ▼                       ▼
             CONTRADICTION             EVIDENCE
                ENGINE                  ENGINE
                    │                       │
              NLI features            Study features
                    │                  Annotation
              Structured             Transparent rubric
                features                   │
                    │                Linear Regression
                    ▼                       │
            Logistic Regression             │
                    │                       │
                    └───────────┬───────────┘
                                ▼
                      EVIDENCE RELATIONSHIP
                              GRAPH
                                │
                                ▼
                       CROSS-STUDY ANALYSIS
                                │
                                ▼
                     EVIDENCE-AWARE SYNTHESIS
                                │
                                ▼
              ┌─────────────────────────────────┐
              │                                 │
              │ Overall finding                 │
              │ Supporting evidence             │
              │ Contradictory evidence          │
              │ Evidence strength               │
              │ Sources of disagreement         │
              │ Uncertainty                     │
              │ Citations                       │
              │                                 │
              └─────────────────────────────────┘
```

---

# 131. Final Product Definition

## EvidenceLens

> **A contradiction-aware and evidence-weighted academic research assistant that retrieves, compares, evaluates, and synthesizes findings across scientific literature.**

Its defining pipeline is:

```text
RETRIEVE
   ↓
UNDERSTAND
   ↓
COMPARE
   ↓
WEIGH
   ↓
AGGREGATE
   ↓
SYNTHESIZE
   ↓
VERIFY
```

Its defining principle is:

> **Do not hide disagreement. Do not treat all evidence equally. Do not manufacture certainty.**

---

# 132. Final Research Definition

EvidenceLens should ultimately be presented not as:

> "An AI that summarizes papers."

but as:

> **A modular framework for evidence-aware multi-document scientific synthesis that explicitly models claim relationships and evidence strength before language-model generation.**

The eventual empirical question is:

> **Does making disagreement and evidence strength explicit improve the reliability, faithfulness, and usefulness of academic RAG systems?**

That is a research question worth testing.

The application is the experimental instrument.

The benchmark is the evidence.

The ablation study is the argument.

And the final paper should be built around the results rather than around the existence of the product.
