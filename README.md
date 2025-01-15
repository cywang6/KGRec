# KGRec for Rice Graph Link Prediction

This repository adapts the code from [HKUDS/KGRec](https://github.com/HKUDS/KGRec) to perform link prediction tasks on rice graph datasets. For a detailed explanation of the underlying algorithms, please refer to the KDD'23 paper: **Knowledge Graph Self-Supervised Rationalization for Recommendation**.

---

## Environment Setup

### Prerequisites

Ensure you have access to a PC or cluster with GPU support. To set up the environment, use the following command:

```bash
conda env create -f environment.yml
```

### CUDA Compatibility

Since some packages depend on the CUDA version directly, verify that their versions are compatible with your installed PyTorch version. For our setup, we used CUDA version 12.4, as indicated by running:

```bash
nvidia-smi
```

---

## Data Preparation

### Folder Structure

1. Create a folder named `rice` (or another name of your choice) inside the `data` directory.
2. Download and unzip the original dataset (e.g., `for_predicted_traits_Jan04_2025`) into the `rice` folder.

### Data Processing

To prepare the data for the model:

1. Activate the environment:

   ```bash
   conda activate KGRec
   ```

2. Navigate to the data processing module:

   ```bash
   cd modules/data_process/
   ```

3. Run the data processing script:

   ```bash
   python process_data_v3.py
   ```

This process constructs:

- A background knowledge graph.
- A user-item graph, split into 10 folds, with filenames like `train_i.txt` and `test_i.txt` for each fold.
- A complete user-item graph saved as `train.txt`.

---

## Running the Model

Use the following command to train and evaluate the model:

```bash
python run_kgrec.py \
    --dataset rice \
    --train_file train_0.txt \
    --test_file test_0.txt \
    --rec_coef 1 \
    --mae_coef 1 \
    --cl_coef 1 \
    --node_dropout_rate 0.1
```

### Notes:

- **Loss Coefficients**: We increase the coefficients for the knowledge graph loss (`rec_coef`) and the contrastive loss (`cl_coef`) to enhance phenotype prediction accuracy for white-list genes. Reducing these coefficients can limit the knowledge graph's contribution to the predictions.

---

## Evaluation

### Output

The results of the run will be displayed:

- In the terminal output.
- On [Weights & Biases (WandB)](https://wandb.ai) for experiment tracking.

### Additional Analysis

To generate gene-phenotype scores for white-list genes or identify candidate genes for specific phenotypes, refer to the Jupyter notebook:

```bash
modules/data_process/print_candidate_genes.ipynb
```

---

## Data Processing Details

### Triple File
The `triple.txt` file is processed following the TransE and TransR algorithm conventions.

### Entity2ID
The `entity2id.txt` file is constructed to prioritize gene nodes, meeting the requirements of the KGRec algorithm. It includes all node types except phenotype nodes from `triple.txt`.

### Knowledge Graph
The final knowledge graph (`kg_final.txt`) includes all nodes and edges, excluding phenotype nodes and their edges. Each line represents an edge in the format:

```
entity_id relation_id entity_id
```

### Pheno-Gene Network
Files named `train` or `test` follow the format:

```
pheno_id gene1_id gene2_id ...
```

This format indicates that a phenotype node is connected to the listed gene nodes.

### Additional Files
- `relation2id.txt`: Maps relations to IDs.
- `pheno2id`: Maps phenotype nodes to IDs.

---

## References

- Original Repository: [HKUDS/KGRec](https://github.com/HKUDS/KGRec)
- Paper: [KDD'23 - Knowledge Graph Self-Supervised Rationalization for Recommendation](https://arxiv.org/abs/...)

