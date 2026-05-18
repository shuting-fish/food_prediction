# Food Prediction Repo-Local AGENTS.md

INSTRUCTION_CLASS: REPO_LOCAL_AGENTS
PROJECT: Food Prediction
DEFAULT_REPO_PATH: C:\Users\simon\food_prediction
STATUS: Non-final
DEFAULT_PHASE: External Data Acquisition + Reference Mapping + Source QA only

## Authority and conflict rules

Use this repo-local instruction only for the Food Prediction repository.

Authority order:
1. System and tool safety rules.
2. Active user task.
3. Current global Codex custom instructions visible to the session.
4. Fresh PowerShell, Git, and file evidence from `C:\Users\simon\food_prediction`.
5. Existing files in this repository only after fresh verification.
6. Prior chat or memory as non-authoritative background only.

If instructions conflict, follow the higher-priority source. If evidence is missing, stale, conflicting, or unverified, stop before guessing and mark the item `TODO-VERIFY` or use exactly: `No reliable evidence`.

Do not use cross-project assumptions.

## Fresh verification requirement

Before repo-dependent conclusions or repository-changing work, verify current state with PowerShell and explicit Windows paths.

Minimum fresh verification:
- repository path
- `.git` existence or repository status
- remotes, when Git is available
- current branch, when Git is available
- tracking and working tree status, when Git is available
- divergence against intended upstream or base, when relevant
- recent commits, when relevant
- relevant file and folder existence

Do not invent evidence, paths, folders, branches, commits, remotes, tests, validation outcomes, repository state, implementation status, source availability, or project status.

Use PowerShell for Windows workflows. Use explicit paths such as `C:\Users\simon\food_prediction`.

## Phase boundary

Default phase:

External Data Acquisition + Reference Mapping + Source QA only.

Allowed work:
- external-data acquisition review
- source documentation
- reproducibility documentation
- registry and evidence-audit consistency QA
- file lineage QA
- reference mapping QA
- geospatial QA
- join feasibility review
- causal availability review
- leakage review
- read-only verification
- documentation-only hardening after explicit approval

Forbidden work:
- ML integration
- model training
- model comparison
- final training feature engineering
- SHAP
- feature importance
- Streamlit
- deployment
- business recommendations
- predictive-value claims
- feature-value claims
- forecast-improvement claims
- model-impact claims
- operational-benefit claims
- business-benefit claims

Stop if the task would cross the phase boundary.

## Data and evidence rules

Canonical raw data are exactly:
- sales
- stores
- weather
- holidays

External data are candidate enrichments only.

Do not create parallel canonical truths. Do not treat undocumented external data as canonical. Do not infer longer temporal coverage from short history.

Preserve `TODO-VERIFY` for unresolved:
- ZIP/postcode mapping
- AGS/Gemeindeschluessel mapping
- municipality mapping
- PLZ centroids
- store coordinates
- OSM/POI lineage
- license
- leakage
- causal availability
- mapping quality
- value claims

Do not resolve `TODO-VERIFY` by assumption. Resolve it only with fresh repository evidence, source evidence, reproducible checks, command output, or explicit user confirmation.

## Language rule

Customer-facing artifacts, repository documentation, code comments, notebooks, tables, reports, and deliverables must be English.

## Finality and status claims

Do not claim `final`, `complete`, `release-ready`, `production-ready`, `QA-passed`, `validated`, or equivalent status unless directly proven by fresh evidence.

Use `Non-final` when unresolved `TODO-VERIFY` items affect correctness, reproducibility, legality, leakage safety, geospatial validity, repository accuracy, or implementation status.

## Machine-readable completion reporting

Do not generate long copy-pasteable completion reports in chat unless explicitly requested.

Prefer strict JSON for structured task-completion feedback. JSON must be parseable, contain no comments, contain no trailing commas, use `null` for unavailable non-text values, and use `TODO-VERIFY` for unresolved verification items.

Minimal final chat response must contain only:
1. created or updated report file path, if any
2. validation result
3. blockers only if blockers exist

Do not invent evidence, paths, commands, tests, branch state, validation outcomes, or status in completion reporting.

Do not make `final`, `complete`, `release-ready`, `production-ready`, `QA-passed`, or `validated` claims unless directly proven by fresh evidence.
