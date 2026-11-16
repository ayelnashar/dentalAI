# Adaptation Ladder for Dental Foundation Models Under a Safety Envelope

Code and summary results for the paper

> A. Elnashar, N. Elnashar and A. Emam, "An Adaptation Ladder for Dental Foundation Models Under a Safety Envelope: Transfer from DENTEX to ADCD," manuscript under review, 2026.

The study asks how much adaptation a dental foundation model needs to move from one panoramic-radiograph dataset (DENTEX) to another (ADCD) while a safety envelope stays in force. The envelope consists of temperature-scaled logistic probes, a residual correction clipped to a fixed logit bound (δ = 1.5), and a Mahalanobis out-of-distribution gate. Its readout is re-fitted on source labels after every adaptation, and source performance is held to a prespecified tolerance of 0.02 macro AUROC. The envelope itself is described in [1].

## Repository layout

```
notebooks/adaptation_ladder.ipynb   all experiments, top to bottom (Stages A, B, B2, C, D, G, E)
analysis/aggregate_results.py       collects the reported numbers from a results folder into nums.json
analysis/duplicate_sensitivity.py   recomputes test AUROC with duplicated ADCD files removed
analysis/make_figures.py            Figures 2 to 9 of the paper from nums.json
results/summary/                    summary CSV/JSON files from the run reported in the paper
results/nums.json                   aggregated numbers used in the paper text, tables and figures
results/figures/                    Figures 1 and 3 to 9 (PNG and PDF)
docs/data.md                        where to get the data and the expected folder layout
docs/outputs.md                     which output file feeds which table or figure
```

## Main result

Dental ViT-L, 256 labeled ADCD crops where labels are used, mean over three seeds. ADCD values are macro AUROC over impacted teeth, caries and periapical lesions at full coverage. DENTEX values are four-condition macro AUROC before and after adaptation. "Pass" counts seeds whose DENTEX drop is at most 0.02.

| Technique | ADCD test | ADCD external | DENTEX | Pass |
|---|---|---|---|---|
| Source only (no adaptation) | 0.789 | 0.876 | 0.882 → 0.882 | 3/3 |
| Rung 0: feature-statistics alignment | 0.787 | 0.875 | 0.882 → 0.882 | 3/3 |
| Embedding-space linear correction with replay | 0.847 | 0.918 | 0.882 → 0.874 | 3/3 |
| Rung 2: LoRA (live trunk) | 0.811 | 0.890 | 0.884 → 0.884 | 3/3 |
| Rung 2: adapter (live trunk) | 0.835 | 0.908 | 0.884 → 0.889 | 3/3 |
| Rung 1: head re-fit on ADCD labels | 0.918 | 0.962 | 0.882 → 0.745 | 0/3 |
| Rung 3: full trunk + EWC | 0.916 | 0.963 | 0.884 → 0.888 | 3/3 |
| Rung 3: full trunk + feature anchoring | 0.920 | 0.966 | 0.884 → 0.888 | 3/3 |
| Rung 3: full trunk + rehearsal | 0.936 | 0.977 | 0.884 → 0.899 | 3/3 |

ADCD annotates lesions only and contains no healthy teeth, so every ADCD number measures discrimination among the three lesion types. The full results, including DINOv3, BiomedCLIP, two-center training, the condition-wise transferability test and the box-geometry baseline, are in `results/summary/` and the paper.

## Reproducing the experiments

1. **Environment.** Python 3.12. Install the pinned versions with

   ```
   pip install -r requirements.txt
   ```

   The Dental ViT-L model code is cloned at run time from [jeonsworld/ViT-pytorch](https://github.com/jeonsworld/ViT-pytorch). The commit used is recorded in `results/summary/env.json`.

2. **Data.** Download DENTEX and ADCD as described in `docs/data.md` and point the notebook at them:

   ```
   export DENTEX_DIR=/path/to/DENTEX/training_data/quadrant-enumeration-disease
   export ADCD_DIR=/path/to/ADCD/panoramic_radiography_yolo_dataset_14_classes
   export RESULTS_DIR=outputs        # optional, default "outputs"
   ```

3. **Models.** The three backbones are downloaded from Hugging Face on first use: the periapical-radiograph expert of `Bryceee/Teeth_Visual_Experts_Models` (Dental ViT-L), `microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224` and `facebook/dinov3-vitl16-pretrain-lvd1689m`. DINOv3 is gated: accept its licence on the model page, then run `huggingface-cli login` once.

4. **Run** `notebooks/adaptation_ladder.ipynb` from top to bottom. Results are written to `$RESULTS_DIR/v5_harmonized_tooth_context/`, and the last cell writes a zip archive whose name says `COMPLETE` or `INCOMPLETE`. Stage C saves after every run, so a restarted notebook resumes where it stopped.

5. **Numbers and figures** from a finished run:

   ```
   python analysis/duplicate_sensitivity.py  outputs/v5_harmonized_tooth_context
   python analysis/aggregate_results.py      outputs/v5_harmonized_tooth_context
   python analysis/make_figures.py           outputs/v5_harmonized_tooth_context figures/
   ```

   To redraw the figures from the numbers shipped here, without running the notebook:

   ```
   python analysis/make_figures.py results/summary figures/ results/nums.json
   ```

   Figure 2 (example crops) is drawn only when the notebook output `Figure_crop_examples.png` is present, because it contains dataset images.

## Hardware and runtime

The reported run used a Mac Studio with an Apple M3 Ultra, PyTorch 2.12 on the Metal (MPS) backend. The full notebook took about 8 hours. Stage C (48 live-trunk and full-trunk runs) took most of it: about 11 min per parameter-efficient run and 15 to 17.5 min per full-trunk run on the Dental ViT-L, and 3 to 6 min per run on DINOv3. The largest accelerator memory reported for one Dental ViT-L run was about 63 GB at batch size 8. The notebook also selects CUDA when it is available, but the reported results were produced on MPS only.

## Data licences and privacy

No patient data are distributed here. DENTEX is released under CC BY 4.0. ADCD is distributed under a database usage agreement that restricts use to non-commercial research and education, so the per-crop labels, file lists and per-crop scores derived from it are not included in `results/`. Only aggregate metrics are. Users must obtain both datasets from their original sources and accept their terms.

## Citation

See `CITATION.cff`. Please also cite DENTEX, ADCD and the backbone models you use.

[1] A. Elnashar, N. Elnashar and A. Elemam, "A safety envelope for foundation-model dental diagnosis: Bounded-deviation guarantees and cross-center refusal," Research Square, preprint, 2026, doi: 10.21203/rs.3.rs-10255012/v1.

## Licence

Code: MIT (see `LICENSE`). Datasets and pretrained models remain under their own licences.
