# Fusarium Virulence Prediction: Multimodal Deep Learning Framework

This repository implements a comprehensive pipeline for predicting Fusarium species virulence across multiple host plants using multimodal genomic data and machine learning approaches.

## Overview

Fusarium species are major fungal pathogens causing devastating diseases in agricultural crops worldwide. This project develops a multimodal machine learning framework that integrates:

- **Presence/Absence Variation (PAV)** - Binary gene presence/absence patterns
- **Copy Number Variation (CNV)** - Gene copy number differences
- **Protein Language Model Embeddings (ESM-2)** - Deep sequence representations
- **Phylogenetic Correction** - Evolutionary distance-aware modeling

## Project Structure

```
Fusarium-Virulence-Prediction/
├── code/                           # Analysis scripts
│   ├── 01.prepare_multimodal_data.py    # Data preprocessing
│   ├── 02.extract_esm2_embeddings.py    # ESM-2 feature extraction
│   ├── 03.final_production_benchmark.py # Model training & evaluation
│   ├── phylo_dist.py                     # Phylogenetic distance calculation
│   └── utils/
│       ├── PNNGS.py                      # Parallel neural network for genomic selection
│       └── ResGS.py                      # Residual network for genomic selection
├── data/                           # Input data files
│   ├── metadata.csv                # Phenotype & species information
│   ├── pav_matrix.csv              # PAV matrix (binary)
│   ├── cnv_matrix.csv              # CNV matrix (copy numbers)
│   ├── protein_embeddings.h5       # ESM-2 embeddings
│   ├── og_mapping.tsv              # Ortholog group mapping
│   ├── phylo_dist.csv              # Phylogenetic distance matrix
│   └── 102fusarium_phylogeny.nwk  # Newick phylogeny tree
├── model/                          # Pre-trained model weights
│   └── esm2_t33_650M_UR50D/        # ESM-2 model (650M parameters)
├── results/                        # Output results
│   ├── cv_fold_metrics.csv         # Cross-validation metrics
│   ├── holdout_metrics.csv         # Holdout test metrics
│   └── saved_models/               # Trained model artifacts
└── environment.yml                 # Conda environment specification
```

## Input Data Format

### Phenotype Metadata (metadata.csv)
```csv
Species,Phenotype_WheatHead,Phenotype_WheatStem,Phenotype_MaizeStem,Phenotype_SoybeanStem,Species_Complex
Fusarium_aberrans__CBS_119866,0,0,0,0,FIESC
Fusarium_asiaticum__KCTC_16664,2.76666,1.84,1.77,3.55,FSAMSC
```

### PAV Matrix (pav_matrix.csv)
- Rows: Species names
- Columns: Ortholog Group IDs (OG0000000, OG0000001, ...)
- Values: Binary (0 or 1)

### CNV Matrix (cnv_matrix.csv)
- Rows: Species names
- Columns: Ortholog Group IDs
- Values: Integer copy numbers

### ESM-2 Embeddings (HDF5)
- Structure: `h5_file[species_name][protein_id]` → 1280-dimensional embedding vector

### OG Mapping (og_mapping.tsv)
```tsv
Species    Protein_ID    OG_ID
Fusarium_xxx    ProteinA    OG0000001
```

## Pipeline Description

### Step 1: Data Preparation

Generate multimodal features from OrthoFinder output:

```bash
python code/01.prepare_multimodal_data.py
```

**Input files required:**
- `Orthogroups.GeneCount.tsv` - OrthoFinder gene count results
- `Orthogroups.tsv` - OrthoFinder ortholog assignments
- `102Fusarium_infecting_plants_phenotype.xlsx` - Phenotype data
- `*.faa` files - Protein sequences

**Output:**
- `data/pav_matrix.csv` - Binary presence/absence matrix
- `data/cnv_matrix.csv` - Copy number variation matrix
- `data/og_mapping.tsv` - Protein to ortholog mapping
- `data/metadata.csv` - Processed phenotype data

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
- `--fasta_dir`: Directory containing protein FASTA files
- `--model_path`: Path to ESM-2 model
- `--output_h5`: Output HDF5 file path
- `--num_gpus`: Number of GPUs for parallel processing (default: 4)
- `--batch_size`: Batch size for inference (default: 32)
- `--max_len`: Maximum sequence length (default: 1024)

### Step 3: Phylogenetic Distance Calculation

Convert Newick tree to distance matrix:

```bash
python code/phylo_dist.py
```

This generates `phylo_dist.csv` for phylogenetic correction.

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
- `--pav`: PAV matrix CSV file
- `--cnv`: CNV matrix CSV file
- `--h5_in`: ESM-2 embeddings HDF5 file
- `--og_map`: Ortholog mapping TSV file
- `--metadata`: Species metadata CSV file
- `--phylo_dist`: Phylogenetic distance matrix (optional)
- `--phylo_dim`: Phylogenetic embedding dimensions (default: 10)
- `--outdir`: Output directory
- `--phenotypes`: Target phenotypes (default: all four)
- `--k_grid`: Ortholog group selection grid (default: 200 500 1000)
- `--pca_dim`: PCA dimensions for embeddings (default: 16)
- `--epochs`: Training epochs for deep learning (default: 300)
- `--batch_size`: Training batch size (default: 128)
- `--models`: Models to evaluate

## Supported Models

### Traditional Machine Learning
- **Ridge** - L2-regularized linear regression
- **SVR** - Support Vector Regression with RBF kernel
- **RandomForest** - Ensemble tree-based regressor
- **GBR** - Gradient Boosting Regression

### Deep Learning Architectures
- **MLP** - Multi-layer perceptron with BatchNorm and Dropout
- **PNNGS** - Parallel Neural Network for Genomic Selection
- **ResGS** - Residual Network for Genomic Selection
- **TabAttention** - Tabular data attention model

## Output Results

### Cross-Validation Metrics (cv_fold_metrics.csv)
```csv
Phenotype,Model,Fold,R2,Pearson_r,Spearman_rho,RMSE,MAE
WheatHead,Ridge,1,0.85,0.92,0.89,0.45,0.32
```

### Holdout Test Metrics (holdout_metrics.csv)
```csv
Phenotype,Model,R2,Pearson_r,Spearman_rho,RMSE,MAE
WheatHead,PNNGS,0.78,0.88,0.82,0.52,0.38
```

### Saved Models
Trained model artifacts saved as pickle files:
- `{Phenotype}_Production_Pipeline.pkl`

Contains:
- Selected top K ortholog groups
- Phylogenetic baseline model
- Feature scaler
- Best-performing functional model

## Phenotypes Predicted

| Phenotype | Description |
|-----------|-------------|
| Phenotype_WheatHead | Virulence on wheat heads |
| Phenotype_WheatStem | Virulence on wheat stems |
| Phenotype_MaizeStem | Virulence on maize stems |
| Phenotype_SoybeanStem | Virulence on soybean stems |

## Computational Requirements

### Recommended Hardware
- **GPU**: NVIDIA A800 80GB or equivalent (for ESM-2 extraction and DL training)
- **RAM**: 64GB+ system memory
- **Storage**: 500GB+ free space

### Software Dependencies
See `environment.yml` for complete package list. Key requirements:
- Python 3.9+
- PyTorch with CUDA support
- ESM-2 transformer model
- scikit-learn, pandas, numpy
- HDF5 utilities

## Methodology Summary

1. **Feature Engineering**: 
   - Generate PAV and CNV matrices from ortholog calling
   - Extract ESM-2 embeddings for each protein
   - Fuse multi-modal features via concatenation

2. **Feature Selection**:
   - Use Random Forest importance to rank ortholog groups
   - Grid search for optimal K (number of top OGs)

3. **Phylogenetic Correction**:
   - Apply PGLS regression to remove phylogenetic bias
   - Use MDS to embed phylogenetic distances

4. **Model Training**:
   - 10-fold cross-validation for hyperparameter tuning
   - Holdout species for independent testing

## Citation

If you use this pipeline in your research, please cite the original publication.

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