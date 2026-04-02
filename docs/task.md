# TimeLens Extension — Task Tracker

## Phase 1: Housekeeping & Quality ✅

- [x] Fix stale `.compute()` references in `api_structure.md`
- [x] Fix stale `.compute()` references in `inheritance_diagrams.md`
- [x] Fix incorrect statement about ConditionalSHAP not inheriting BaseExplainer
- [x] Add visualization unit tests (`test_visualization.py`) — 14 tests
- [x] Add adapter unit tests (`test_adapters.py`) — 7 tests

## Phase 2: New Explainers ✅

- [x] 2A: SHAP-IQ — `ConditionalSHAPIQ` class — 12 tests, 86% coverage
- [x] 2A: `SHAPIQResult` type (inherits BaseResult)
- [x] 2A: `shapiq` optional dependency in pyproject.toml + ty.toml
- [x] 2A: Unit tests for SHAP-IQ
- [x] 2B: Feature Dropping — `ConditionalDropImportance` class — 12 tests, 96% coverage
- [x] 2B: Unit tests for dropping

## Phase 3: Causal Feature Importance ✅

- [x] Flesh out `CausalExplainer` base class in `base.py`
- [x] `CausalResult` and `RefutationResult` types in `types.py`
- [x] `CausalFeatureImportance` implementation in `importance/causal.py`
  - [x] DoWhy integration (model → identify → estimate → refute)
  - [x] EconML estimators (CausalForestDML, DynamicDML, LinearDML)
  - [x] Series-conditional treatment effects
  - [x] Auto graph discovery via causal-learn
- [x] `causal-discovery` optional dependency in pyproject.toml
- [x] Unit tests for causal — 17 tests, 62% module coverage

## Phase 4: Framework Adapters

- [x] `SklearnAdapter` implementation
- [x] `DartsAdapter` implementation
- [x] Unit tests for new adapters
- [x] Update pyproject.toml optional deps

## Phase 5: Advanced Features

- [x] 5A: Temporal windowed importance
- [x] 5B: Statistical significance testing
- [x] 5C: Comparison utilities
- [x] 5D: Enhanced visualization functions

## Phase 6: Unified Dashboard API (Designed, awaiting approval)

- [x] `Dashboard` class (builder pattern + orchestrator)
- [ ] Dashboard components:
  - [x] `InterpretabilityComponent`
  - [x] `ErrorAnalysisComponent`
  - [x] `CausalComponent`
  - [x] `InteractionComponent`
- [x] `DashboardResult` container
- [x] `ErrorAnalyzer` (per-series, per-window metrics)
- [x] Report generation (HTML via Jinja2)
- [x] `dashboard.show()` interactive view
- [ ] Integration tests for full Dashboard workflow

## Current Stats

- **Tests**: 111 passing
- **Coverage**: 82% (last recorded)
- **All pre-commit checks**: ✅ passing
