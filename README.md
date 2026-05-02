# Fusarium Virulence Prediction: Multimodal Deep Learning Framework

A comprehensive machine learning pipeline for predicting Fusarium species virulence across multiple host plants using multimodal genomic data integration.

## Overview

Fusarium species are major fungal pathogens causing devastating diseases in agricultural crops worldwide, including wheat, maize, and soybean. This project develops a multimodal machine learning framework that integrates multiple genomic data types:

- **Presence/Absence Variation (PAV)** - Binary gene presence/absence patterns
- **Copy Number Variation (CNV)** - Gene copy number differences across species
- **Protein Language Model Embeddings (ESM-2)** - Deep sequence representations
- **Phylogenetic Correction** - Evolutionary distance-aware modeling using PGLS

## Project Structure

```
Fusarium-Virulence-Prediction/
├── code/                           # Analysis scripts
│   ├── 01.prepare_multimodal_data.py    # Data preprocessing (PAV/CNV matrices)
│   ├── 02.extract_esm2_embeddings.py    # ESM-2 protein embedding extraction
│   ├── 03.final_production_benchmark.py # Model training & evaluation
│   ├── 04.predict_unknown_species.py     # Predict virulence for new species
│   ├── phylo_dist.py                     # Phylogenetic distance calculation
│   └── utils/
│       ├── PNNGS.py                      # Parallel Neural Network for Genomic Selection
│       └── ResGS.py                      # Residual Network for Genomic Selection
├── data/                           # Input data files
│   ├── metadata.csv                # Species phenotype & grouping information
│   ├── pav_matrix.csv              # PAV matrix (binary: 0/1)
│   ├── cnv_matrix.csv              # CNV matrix (copy numbers)
│   ├── protein_embeddings.h5       # ESM-2 embeddings (HDF5 format)
│   ├── og_mapping.tsv              # Protein to Ortholog Group mapping
│   ├── phylo_dist.csv              # Phylogenetic distance matrix
│   └── 102fusarium_phylogeny.nwk  # Newick phylogeny tree
├── model/                          # Pre-trained model weights
│   └── esm2_t33_650M_UR50D/        # ESM-2 model (650M parameters)
├── results/                        # Output results
│   ├── cv_fold_metrics.csv         # 10-fold cross-validation metrics
│   ├── holdout_metrics.csv         # Independent holdout test metrics
│   ├── Unknown_Species_Virulence_Predictions.csv  # Predictions for new species
│   └── saved_models/               # Trained model artifacts
│       └── {Phenotype}_Production_Pipeline.pkl   # Production pipeline models
└── environment.yml                 # Conda environment specification
```

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/your-repo/Fusarium-Virulence-Prediction.git
cd Fusarium-Virulence-Prediction
```

### 2. Create conda environment

```bash
conda env create -f environment.yml
conda activate fusarium-virulence-prediction
```

### 3. Download ESM-2 model

The ESM-2 model will be automatically downloaded when running `02.extract_esm2_embeddings.py`. Alternatively, you can manually download from HuggingFace:

```bash
# The model will be saved to model/esm2_t33_650M_UR50D/
```

## Input Data Format

### Phenotype Metadata (metadata.csv)
| Column | Description |
|--------|-------------|
| Species | Species name (strain identifier) |
| Phenotype_WheatHead | Virulence score on wheat heads (0-6 scale) |
| Phenotype_WheatStem | Virulence score on wheat stems |
| Phenotype_MaizeStem | Virulence score on maize stems |
| Phenotype_SoybeanStem | Virulence score on soybean stems |
| Species_Complex | Fusarium species complex (e.g., FGSC, FOSC, FSSC) |

### PAV Matrix (pav_matrix.csv)
- Rows: Species names
- Columns: Ortholog Group IDs (OG0000000, OG0000001, ...)
- Values: Binary (0 = absent, 1 = present)

### CNV Matrix (cnv_matrix.csv)
- Rows: Species names
- Columns: Ortholog Group IDs
- Values: Integer copy numbers

### ESM-2 Embeddings (protein_embeddings.h5)
HDF5 structure: `h5_file[species_name][protein_id]` → 1280-dimensional embedding vector

### OG Mapping (og_mapping.tsv)
```tsv
Species    Protein_ID    OG_ID
Fusarium_xxx    ProteinA    OG0000001
```

### Phylogenetic Tree (102fusarium_phylogeny.nwk)
Newick format phylogenetic tree with 102 Fusarium species

## Pipeline Description

### Step 1: Data Preparation

Generate multimodal features from OrthoFinder output:

```bash
python code/01.prepare_multimodal_data.py
```

**Required input files:**
- `Orthogroups.GeneCount.tsv` - OrthoFinder gene count results
- `Orthogroups.tsv` - OrthoFinder ortholog assignments
- `102Fusarium_infecting_plants_phenotype.xlsx` - Phenotype data
- `*.faa` files - Protein sequences

**Output:**
- `data/pav_matrix.csv` - Binary presence/absence matrix
- `data/cnv_matrix.csv` - Copy number variation matrix
- `data/og_mapping.tsv` - Protein to ortholog mapping
- `data/metadata.csv` - Processed phenotype data
- `fasta/` - Species-specific protein FASTA files

### Step 2: ESM-2 Embedding Extraction

Extract protein language model embeddings using ESM-2:

```bash
python code/02.extract_esm2_embeddings.py \
    --fasta_dir fasta/ \
    --model_path model/esm2_t33_650M_UR50D \
    --output_h5 data/protein_embeddings.h5 \
    --num_gpus 4
```

**Parameters:**
| Parameter | Default | Description |
|-----------|---------|-------------|
| `--fasta_dir` | required | Directory containing protein FASTA files |
| `--model_path` | required | Path to ESM-2 model |
| `--output_h5` | required | Output HDF5 file path |
| `--num_gpus` | 4 | Number of GPUs for parallel processing |
| `--batch_size` | 32 | Batch size for inference |
| `--max_len` | 1024 | Maximum sequence length |

### Step 3: Phylogenetic Distance Calculation

Convert Newick tree to distance matrix:

```bash
python code/phylo_dist.py
```

This generates `data/phylo_dist.csv` for phylogenetic correction.

### Step 4: Model Training & Benchmarking

Run the complete training and evaluation pipeline:

```bash
python code/03.final_production_benchmark.py \
    --pav data/pav_matrix.csv \
    --cnv data/cnv_matrix.csv \
    --h5_in data/protein_embeddings.h5 \
    --og_map data/og_mapping.tsv \
    --metadata data/metadata.csv \
    --phylo_dist data/phylo_dist.csv \
    --outdir results
```

**Parameters:**
| Parameter | Default | Description |
|-----------|---------|-------------|
| `--pav` | data/pav_matrix.csv | PAV matrix CSV file |
| `--cnv` | data/cnv_matrix.csv | CNV matrix CSV file |
| `--h5_in` | data/protein_embeddings.h5 | ESM-2 embeddings HDF5 file |
| `--og_map` | data/og_mapping.tsv | Ortholog mapping TSV file |
| `--metadata` | data/metadata.csv | Species metadata CSV file |
| `--phylo_dist` | None | Phylogenetic distance matrix |
| `--phylo_dim` | 10 | Phylogenetic embedding dimensions |
| `--outdir` | results | Output directory |
| `--phenotypes` | all 4 | Target phenotypes to predict |
| `--k_grid` | 200 500 1000 | Ortholog group selection grid |
| `--pca_dim` | 16 | PCA dimensions for embeddings |
| `--epochs` | 300 | Training epochs for deep learning |
| `--batch_size` | 128 | Training batch size |
| `--models` | all 8 | Models to evaluate |

### Step 5: Predict Virulence for Unknown Species

Predict virulence for species without phenotype data:

```bash
python code/04.predict_unknown_species.py \
    --pav data/pav_matrix.csv \
    --cnv data/cnv_matrix.csv \
    --h5_in data/protein_embeddings.h5 \
    --og_map data/og_mapping.tsv \
    --phylo_dist data/phylo_dist.csv \
    --model_dir results/saved_models \
    --outdir results
```

## Supported Models

### Traditional Machine Learning
| Model | Description |
|-------|-------------|
| Ridge | L2-regularized linear regression |
| SVR | Support Vector Regression with RBF kernel |
| RandomForest | Ensemble tree-based regressor |
| GBR | Gradient Boosting Regression |

### Deep Learning Architectures
| Model | Description |
|-------|-------------|
| MLP | Multi-layer perceptron with BatchNorm and Dropout |
| PNNGS | Parallel Neural Network for Genomic Selection (Inception-style) |
| ResGS | Residual Network for Genomic Selection |
| TabAttention | Attention-based model for tabular data |

## Output Results

### Cross-Validation Metrics (cv_fold_metrics.csv)
```
Phenotype,Model,Fold,R2,Pearson_r,Spearman_rho,RMSE,MAE
WheatHead,Ridge,1,0.85,0.92,0.89,0.45,0.32
```

### Holdout Test Metrics (holdout_metrics.csv)
```
Phenotype,Model,R2,Pearson_r,Spearman_rho,RMSE,MAE
WheatHead,PNNGS,0.78,0.88,0.82,0.52,0.38
```

### Saved Models
Trained model artifacts saved as pickle files: `{Phenotype}_Production_Pipeline.pkl`

Contains:
- Selected top K ortholog groups
- Phylogenetic baseline model (Ridge for PGLS correction)
- Feature scaler
- Best-performing functional model (or PyTorch weights)

## Phenotypes Predicted

| Phenotype | Description | Host |
|-----------|-------------|------|
| Phenotype_WheatHead | Virulence on wheat heads | Triticum aestivum |
| Phenotype_WheatStem | Virulence on wheat stems | Triticum aestivum |
| Phenotype_MaizeStem | Virulence on maize stems | Zea mays |
| Phenotype_SoybeanStem | Virulence on soybean stems | Glycine max |

## Computational Requirements

### Recommended Hardware
- **GPU**: NVIDIA A800 80GB or equivalent (for ESM-2 extraction and DL training)
- **RAM**: 64GB+ system memory
- **Storage**: 500GB+ free space
- **CPUs**: 8+ cores recommended

### Software Dependencies
See `environment.yml` for complete package list:
- Python 3.9+
- PyTorch with CUDA support
- ESM-2 transformer model (facebook/esm2_t33_650M_UR50D)
- scikit-learn, pandas, numpy
- HDF5 utilities
- TensorFlow 2.8+ (for ResGS model)
- BioPython (for phylogenetics)

## Methodology Summary

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    Fusarium Virulence Prediction Pipeline               │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐              │
│  │   PAV Matrix │    │   CNV Matrix │    │  ESM-2       │              │
│  │   (Binary)   │    │   (Counts)   │    │  Embeddings  │              │
│  └──────┬───────┘    └──────┬───────┘    └──────┬───────┘              │
│         │                   │                   │                       │
│         └───────────────────┼───────────────────┘                       │
│                             ▼                                            │
│                  ┌─────────────────────┐                                │
│                  │  Feature Fusion     │                                │
│                  │  (Concatenation)    │                                │
│                  └──────────┬──────────┘                                 │
│                             │                                            │
│         ┌───────────────────┼───────────────────┐                       │
│         ▼                                       ▼                       │
│  ┌─────────────┐                         ┌─────────────┐                │
│  │  Random     │                         │   PGLS     │                │
│  │  Forest     │                         │  (Phylo    │                │
│  │  Feature    │                         │  Baseline) │                │
│  │  Selection  │                         └──────┬──────┘                │
│  └──────┬──────┘                                │                       │
│         │                                       │                       │
│         └───────────────────┬───────────────────┘                       │
│                             ▼                                            │
│                  ┌─────────────────────┐                                │
│                  │  Two-Stage Model   │                                │
│                  │  Training          │                                │
│                  └──────────┬──────────┘                                 │
│                             │                                            │
│         ┌───────────────────┼───────────────────┐                       │
│         ▼                   ▼                   ▼                       │
│   ┌──────────┐        ┌──────────┐        ┌──────────┐                  │
│   │  Ridge   │        │   SVR    │        │    DL    │                  │
│   │          │        │          │        │  Models  │                  │
│   └──────────┘        └──────────┘        └──────────┘                  │
│                             │                                            │
│                             ▼                                            │
│                  ┌─────────────────────┐                                │
│                  │   Final Prediction  │                                │
│                  │   Base + Residual   │                                │
│                  └─────────────────────┘                                │
└─────────────────────────────────────────────────────────────────────────┘
```

### Key Steps:

1. **Feature Engineering**:
   - Generate PAV and CNV matrices from ortholog calling (OrthoFinder)
   - Extract ESM-2 embeddings for each protein (1280-dim → PCA reduced to 16-dim)
   - Fuse multi-modal features via concatenation

2. **Feature Selection**:
   - Use Random Forest importance to rank ortholog groups
   - Grid search for optimal K (number of top OGs: 200, 500, 1000)

3. **Phylogenetic Correction**:
   - Apply PGLS regression using Ridge to remove phylogenetic bias
   - Use MDS to embed phylogenetic distances into 10-dimensional space

4. **Model Training**:
   - 10-fold cross-validation for hyperparameter tuning
   - Holdout 5 species for independent testing:
     - Fusarium_aberrans__CBS_119866
     - Fusarium_californicum__CBS_145796
     - Fusarium_languescens__CBS_645.78
     - Fusarium_pseudograminearum__CS3096
     - Fusarium_flocciferum__CBS_821.68

5. **Two-Stage Prediction**:
   - Stage 1: Phylogenetic baseline (Ridge on phylogenetic embeddings)
   - Stage 2: Residual learning with functional models
   - Final prediction = Base + Residual (clipped to ≥0)

## Citation

If you use this pipeline in your research, please cite the original publication:

```
@article{fusarium_virulence_2024,
  title={Multimodal Deep Learning for Fusarium Species Virulence Prediction},
  author={},
  journal={},
  year={2024}
}
```

## License

This project is for research purposes.

## Troubleshooting

### Out of Memory Issues
- Reduce `--batch_size` in ESM-2 extraction
- Reduce `--pca_dim` in benchmarking
- Process species in smaller chunks

### Model Training Issues
- Ensure CUDA is available: `python -c "import torch; print(torch.cuda.is_available())"`
- Check GPU memory with `nvidia-smi`

### Data Format Issues
- Ensure species names match exactly between all input files
- Check for missing values in phenotype data
- Verify ortholog group IDs are consistent across files
