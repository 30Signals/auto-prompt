## Current Status: Phase 2 Complete ✓

Phase 2 statistical rigor infrastructure implemented and ready for use.

---

## Issue #2: Improve repo organization and make findings publication-ready

### Phase 1: Foundation (Quick Wins) ✅ COMPLETED
- [x] Reorganize code structure
  - [x] Created `experiments/` directory
  - [x] Created `shared/` utilities
  - [x] Created `case_study/` directory
  - [x] Created `scripts/` directory
- [x] Add visualization tools
  - [x] `plot_field_comparison()`
  - [x] `plot_accuracy_bars()`
  - [x] `plot_improvement_heatmap()`
- [x] Pin dependencies
  - [x] Created `requirements.txt` with pinned versions
- [x] Create case study template
  - [x] `case_study/writeup.md`
  - [x] `case_study/analysis.ipynb`
- [x] Update documentation
  - [x] Updated `.gitignore`
  - [x] Updated `CLAUDE.md` for new structure

**PR:** #3 - https://github.com/30Signals/auto-prompt/pull/3

---

### Phase 2: Statistical Rigor ✅ COMPLETED
- [x] Run multiple trials with seeds
  - [x] `run_multiple_trials()` in `experiments/resume_extraction/run.py`
  - [x] Seed tracking via `config.RANDOM_SEEDS` (5 seeds)
  - [x] Trial-specific result directories (`trial_0/`, `trial_1/`, etc.)
- [x] Add confidence intervals
  - [x] `compute_confidence_interval()` in `shared/evaluation/statistics.py`
  - [x] `aggregate_metrics()` with mean, std_dev, ci_lower, ci_upper
  - [x] Error bar visualizations: `plot_field_comparison_with_ci()`, `plot_accuracy_bars_with_ci()`
- [x] Statistical significance testing
  - [x] `compute_significance()` with paired t-test
  - [x] `compute_effect_size()` (Cohen's d)
  - [x] Result aggregation in `aggregate_results.py`
- [x] Ablation studies infrastructure
  - [x] `ablation_runner.py` with demo count and optimizer ablations
  - [x] Test BootstrapFewShot only (no COPRO)
  - [x] Test different demo counts (4, 8, 16, 32)
  - [ ] **TODO:** Add MIPROv2 optimizer comparison // DO IT LATER
  - [ ] **TODO:** Add temperature variation ablation // DO IT LATER
- [x] Prompt export
  - [x] `save_baseline_prompt()` in `shared/evaluation/prompt_utils.py`
  - [x] `extract_optimized_prompt()` for DSPy optimized prompts
  - [x] `generate_prompt_comparison()` for side-by-side comparison

**Usage:**
```bash
# Run multi-trial experiment
python -m experiments.resume_extraction.run --multi-run 5

# Aggregate results with statistics
python -m experiments.resume_extraction.aggregate_results 5

# Run ablation studies
python -m experiments.resume_extraction.ablation_runner --demo-count
python -m experiments.resume_extraction.ablation_runner --optimizer
```

---


### Phase 3: New Experiments 📋 PLANNED
- [ ] Experiment 2: Medical Entity Extraction
  - [ ] Download NCBI Disease Corpus or i2b2 dataset
  - [ ] Create experiment structure in `experiments/medical_ner/`
  - [ ] Implement baseline and DSPy modules
  - [ ] Run evaluation and comparison
- [ ] Experiment 3: Product Categorization
  - [ ] Source Amazon product dataset
  - [ ] Create experiment structure
  - [ ] Implement classification modules
  - [ ] Evaluate multi-label performance
- [ ] Experiment 4: Legal Contract Analysis
  - [ ] Download CUAD dataset
  - [ ] Create experiment structure
  - [ ] Implement extraction modules
  - [ ] Evaluate on legal text
- [ ] Cross-domain validation
  - [ ] Train on resumes, test on cover letters
  - [ ] Measure transfer learning effectiveness
  - [ ] Document generalization findings

---

### Phase 4: Publication 📝 PLANNED
- [ ] Generate all figures
  - [ ] Run visualization scripts for all experiments
  - [ ] Create comparison charts across experiments
  - [ ] Generate learning curves
  - [ ] Create cost-performance plots
- [ ] Complete case study writeup
  - [ ] Fill in methodology section
  - [ ] Add results with figures
  - [ ] Write discussion and limitations
  - [ ] Add future work section
- [ ] Statistical analysis
  - [ ] Calculate effect sizes (Cohen's d)
  - [ ] Perform power analysis
  - [ ] Apply multiple comparison corrections
- [ ] Reproducibility documentation
  - [ ] Add Docker setup
  - [ ] Document data sources
  - [ ] Add reproducibility checklist
  - [ ] Create code availability statement

---

## Additional Baseline Comparisons (Future)
- [ ] GPT-4 zero-shot baseline
- [ ] GPT-4 with 3-shot manual examples
- [ ] Fine-tuned small model (Llama 3 8B)
- [ ] Traditional NLP (spaCy NER + rules)
- [ ] RAG-based approach

---

## Enhanced Evaluation Metrics (Future)
- [ ] Precision, Recall, F1 per field
- [ ] Confusion matrices
- [ ] Error analysis dashboard
- [ ] Cost analysis (tokens, API costs)
- [ ] Latency measurements
- [ ] Inter-annotator agreement on ground truth

---

## Notes

### Branch: `claude/organize-repo-publication-c9sQj`
- Contains Phase 1 reorganization
- Waiting for review/merge

### Next Actions
1. Review and merge PR #3 - DONE by USER
2. Begin Phase 2: Statistical rigor
3. Run multiple trials with confidence intervals

### Dependencies
- Phase 2 can start immediately after PR #3 merge
- Phase 3 requires dataset sourcing
- Phase 4 requires Phases 2-3 completion

---

Last updated: 2026-02-04
