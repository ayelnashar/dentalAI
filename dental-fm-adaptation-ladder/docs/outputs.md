# Output files

The notebook writes everything to `$RESULTS_DIR/v5_harmonized_tooth_context/`. The files in `results/summary/` are the summary outputs of the run reported in the paper, with local paths replaced by `<DATA_ROOT>`. Per-crop scores (`per_tooth/`), the ADCD record list and label arrays are produced by the notebook but are not distributed (see the README on data licences).

## Where each table and figure comes from

| Paper item | File(s) |
|---|---|
| Table I, dataset flow | `dataset_flow_v2.csv`, `adcd_parse_stats.json` |
| Table II, ladder on the Dental ViT-L | `E5_table_B.csv`, `E5_table_C.csv` (per seed: `B_frozen_ladder.json`, `C_live_trunk.json`) |
| Table III, ladder on DINOv3 and BiomedCLIP | same files as Table II |
| Table IV, per-condition AUROC with intervals | `E1_auroc_ci.csv`, `E2_paired_diff.csv` |
| Table V, two-center regimes | `D1_joint_budgets.json`, `E5_table_D1.csv` |
| Table VI, transferability test | `D2_ctt.csv` |
| Figures 3 to 9 | `nums.json`, drawn by `analysis/make_figures.py` |
| Box-geometry baseline | `D3_geometry_baseline.csv` |
| Source preservation per run | `E3_source_preservation.csv` |
| Gate refusal on DENTEX, ADCD and the external set | `E6_gate_matrix.csv` |
| AUROC against coverage | `E4_auroc_coverage.csv` |
| Two-backbone fusion | `B2_fusion.json`, `E5_table_F.csv` |
| Duplicate-file sensitivity | `dupsens.json` |
| Healthy-tooth validity check | `G_healthy_teeth_status.json` |
| Environment, package versions, model revisions | `env.json`, `embedding_minutes.json`, `dental_vitl_config.json` |
| Run completeness and integrity checks | `completeness.json`, `completeness_verdict.json`, `checks.json` |

## Per-run records

`B_frozen_ladder.json` and `C_live_trunk.json` hold one record per backbone, technique, label budget and seed. The main fields are:

- `tgt`: ADCD test metrics in several views. `full` is all test crops, `src` the crops accepted by the source-fitted gate, `tgt` the crops accepted by a gate re-fitted on unlabeled ADCD embeddings, `cov20`/`cov50` the 20% and 50% of crops nearest the source by each model's own gate, and `fix20`/`fix50` the same fractions chosen once by the unadapted source gate. Each view gives per-condition AUROC, AUPRC, prevalence, positive and negative counts, coverage, ECE and the macro-average. The macro-average is reported only when all three conditions have at least 5 positives and 5 negatives.
- `tgt_ext`: the same metrics on the ADCD external set.
- `src_before`, `src_after`: DENTEX test metrics before and after adaptation, including the four-condition macro AUROC (`macro4_full`) and the largest residual correction (`max_dev`).
- `controls` (adapted runs): the head trained during adaptation kept as the predictor (`tgt_head`) and the original source envelope applied to the adapted features (`src_env_fixed`).
- `delta_ok`: whether the largest residual correction stayed within δ = 1.5.
- Stage C records also carry the trainable-parameter counts, optimizer update counts, training and total minutes, accelerator memory and a run signature used for resuming.

`D2_ctt.csv` has one row per direction (`D2A` DENTEX to ADCD, `A2D` ADCD to DENTEX), backbone, condition, budget and seed, with the development-partition AUROCs `A_S2T` and `A_T`, the ratio `R`, its bootstrap interval and verdict, and, for `D2A`, the test-split outcome of the locked protocol (`R_outcome`) and whether the two agree.
