# Full Dataset Usage Instructions

## 🎯 Research Quality Mandate: Use Complete Datasets

This research uses **FULL EVALUATION DATASETS** to ensure maximum statistical power and research integrity.

### ✅ Full Dataset Specifications

| Dataset | Full Size | Split | Coverage |
|---------|-----------|-------|----------|
| **MMLU** | 14,042 samples | test | 57 academic subjects |
| **GSM8K** | 1,319 samples | test | Grade school math |
| **HumanEval** | 164 samples | test | Python programming |
| **TruthfulQA** | 817 samples | validation | Truthfulness evaluation |
| **TOTAL** | **16,342 samples** | | |

### 🔬 Scientific Justification

**Why Full Datasets?**
1. **Statistical Power**: Maximum sample size = most reliable statistical tests
2. **No Sampling Bias**: Complete coverage eliminates selection effects
3. **Reproducibility**: Other researchers can replicate exact evaluation
4. **Benchmark Consistency**: Results comparable to other full-dataset studies
5. **Publication Standards**: Top venues expect comprehensive evaluation

### ⚡ Computational Requirements

**Estimated Runtime with Qwen2.5 4-stage pipeline:**
- MMLU (14,042): ~2-3 hours
- GSM8K (1,319): ~20-30 minutes  
- HumanEval (164): ~5-10 minutes
- TruthfulQA (817): ~15-20 minutes
- **Total: ~3-4 hours** (fully automated)

**Resource Usage:**
- 8x H100 GPUs (concurrent model loading)
- ~500GB model storage
- ~2GB evaluation data storage
- ~10GB results storage

### 🚫 DO NOT Use Subsets

**Prohibited practices:**
- ❌ Random sampling from full datasets
- ❌ First-N sampling (introduces bias)
- ❌ Cherry-picking "easier" samples
- ❌ Time-based truncation for convenience

**Exception:** Only reduce dataset size if technical limitations prevent full evaluation, and document this clearly as a limitation.

### 📊 Expected Benefits

**Statistical Improvements:**
- Confidence intervals: Much tighter
- Effect size detection: More reliable  
- P-values: More stable across runs
- Cross-validation: More robust

**Research Impact:**
- Higher citation potential
- Stronger publication acceptance
- Better reproducibility
- Benchmark leadership

### 🎖️ Quality Assurance

All experiment scripts MUST:
1. Load full datasets by default
2. Report actual sample sizes in logs
3. Verify dataset completeness before evaluation
4. Document any deviations from full dataset usage

## ✅ Implementation Status

- [x] evaluation.yaml: Updated to full dataset sizes
- [x] qwen2.5_models.yaml: Updated to full dataset sizes  
- [x] CLAUDE.md: Updated requirements
- [x] All experiment scripts: Support full dataset loading
- [x] Documentation: Updated to reflect full dataset usage

**This research now uses the COMPLETE evaluation datasets for maximum scientific rigor.**