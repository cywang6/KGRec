"""
This file is copied from Jinzhe's TransE and LET-ConvE work.
2024.12.26 process data
处理kg数据
Jan 04 2025 更新
"""
import json
import os
import re
import csv
from tqdm import tqdm
import random  # Added
import numpy as np  # Used for np.array_split below

work_name = "rice"  # rice or multi

target_path = "../../data/rice2"
multi_folder_path = "../../data/rice2/for_predicted_traits_Jan04_2025"

entity_set = set()
relation_set = set()
triple_set = set()

# entity relation entity 
csv_name = ["KG_proteomics_Jan06.csv", "KG_transcriptomics_Jan06.csv", "KG_OE_Mut_WOS_PUBMED_RAPDB_traitGene_Jan07.csv", "KG_ppi_Jan04.csv"]
# entity entity relation
another_list = ["KG_gene2transcript_Jan06.csv", "KG__transcript2protein_Jan06.csv", "KG_basic_gene_KeggGO_Dec19_2024.csv", "KG_gene_updownstream.csv", "KG_env_factor_relationGene_Jan04.csv"]

def multi_data():
    """
    多组学数据
    """
    for filename in os.listdir(multi_folder_path):
        if filename.endswith('.csv'):
            file_path = os.path.join(multi_folder_path, filename)
            if filename in another_list:
                with open(file_path, mode='r', encoding='utf-8') as file:
                    reader = csv.reader(file)
                # 逐行读取
                    for row in list(reader)[1:]:
                        head, tail, rela = row[0], row[1], row[2]
                        head_cleaned = re.sub(r"[^\w\s]", "", head.lower()).replace(" ", "_")
                        tail_cleaned = re.sub(r"[^\w\s]", "", tail.lower()).replace(" ", "_")
                        rela_cleaned = re.sub(r"[^\w\s]", "", rela.lower()).replace(" ", "_")
                        # replace \t and \n with "" in head, tail and relation
                        head_cleaned = head_cleaned.replace("\t", "").replace("\n", "")
                        tail_cleaned = tail_cleaned.replace("\t", "").replace("\n", "")
                        rela_cleaned = rela_cleaned.replace("\t", "").replace("\n", "")
                        head_cleaned = head_cleaned.replace(" ", "_")
                        tail_cleaned = tail_cleaned.replace(" ", "_")
                        rela_cleaned = rela_cleaned.replace(" ", "_")
                        if "网站" in head_cleaned or "网站" in tail_cleaned:
                            continue
                        if head_cleaned == "" or tail_cleaned == "" or rela_cleaned == "":
                            continue
                        entity_set.add(head_cleaned)
                        entity_set.add(tail_cleaned)
                        relation_set.add(rela_cleaned)
                        # triple_set.add((head_cleaned, tail_cleaned, rela_cleaned))
                        triple_set.add((head_cleaned, rela_cleaned, tail_cleaned))
            elif filename in csv_name:
                with open(file_path, mode='r', encoding='utf-8') as file:
                    reader = csv.reader(file)
                # 逐行读取
                    for row in list(reader)[1:]:
                        head, tail, rela = row[0], row[2], row[1]
                        # if rela == "described_in_literature":
                        #     continue
                        head_cleaned = re.sub(r"[^\w\s]", "", head.lower()).replace(" ", "_")
                        tail_cleaned = re.sub(r"[^\w\s]", "", tail.lower()).replace(" ", "_")
                        rela_cleaned = re.sub(r"[^\w\s]", "", rela.lower()).replace(" ", "_")
                        # replace \t and \n with "" in head, tail and relation
                        head_cleaned = head_cleaned.replace("\t", "").replace("\n", "")
                        tail_cleaned = tail_cleaned.replace("\t", "").replace("\n", "")
                        rela_cleaned = rela_cleaned.replace("\t", "").replace("\n", "")
                        head_cleaned = head_cleaned.replace(" ", "_")
                        tail_cleaned = tail_cleaned.replace(" ", "_")
                        rela_cleaned = rela_cleaned.replace(" ", "_")
                        if "网站" in head_cleaned or "网站" in tail_cleaned:
                            continue
                        if head_cleaned == "" or tail_cleaned == "" or rela_cleaned == "":
                            continue
                        entity_set.add(head_cleaned)
                        entity_set.add(tail_cleaned)
                        relation_set.add(rela_cleaned)
                        # triple_set.add((head_cleaned, tail_cleaned, rela_cleaned))
                        triple_set.add((head_cleaned, rela_cleaned, tail_cleaned))


def write2txt(file_name, content, type):
    if type == "set":
        with open(file_name, 'w', encoding='utf-8') as file:
            for id, item in enumerate(content, start=0):
                file.write(f"{item}\t{id}\n")
    else:
        with open(file_name, 'w', encoding='utf-8') as file:
            for item in content:
                file.write(f"{item[0]}\t{item[1]}\t{item[2]}\n")  


# Helper: build the final KG file by filtering out "phenotype" relations
def build_kg_final(
    path,
    file_entity2id="entity2id.txt",
    file_relation2id="relation2id.txt",
    file_triple="triple.txt",
    file_out="kg_final.txt",
):
    """
    Build 'kg_final.txt' by reading and transforming entity/relation IDs
    and filtering out any triple whose relation is 'phenotype'.
    """

    entity2id = {}
    relation2id = {}

    # Read entity-to-ID mappings
    with open(os.path.join(path, file_entity2id), "r", encoding="utf-8") as f_ent:
        for line in f_ent:
            entity, eid = line.strip().split("\t")
            entity2id[entity] = eid

    # Read relation-to-ID mappings
    with open(os.path.join(path, file_relation2id), "r", encoding="utf-8") as f_rel:
        for line in f_rel:
            relation, rid = line.strip().split("\t")
            relation2id[relation] = rid

    # Open the output file to write filtered triples
    with open(os.path.join(path, file_out), "w", encoding="utf-8") as f_out:
        # Open the original triples file
        with open(os.path.join(path, file_triple), "r", encoding="utf-8") as f_tri:
            for line in f_tri:
                head, relation, tail = line.strip().split("\t")

                # Assert that all parts of the triple exist in the dictionaries
                assert head in entity2id and tail in entity2id and relation in relation2id

                # Exclude triples that use the 'phenotype' relation
                if relation != "phenotype":
                    # Write the valid triple in ID form to the output
                    f_out.write(
                        f"{entity2id[head]} {relation2id[relation]} {entity2id[tail]}\n"
                    )

def create_kfold(
    folder_path,
    entity2id_file="entity2id.txt",
    triple_file="triple.txt",
    pheno2id_file="pheno2id.txt",
    k=10,
    seed=42,
):
    """
    Create k-fold cross validation splits for pheno-gene mapping.
    For each fold i, produce train_i.txt and test_i.txt.

    Additionally, produce a train.txt that has *all* pheno-gene pairs,
    regardless of the fold.

    train_i.txt, test_i.txt, and train.txt have the format:
        pheno_id gene1_id gene2_id ...

    A single pheno2id.txt is also created with IDs for phenotypes.
    :param k: Number of folds (default = 10)
    :param seed: Random seed for reproducibility
    """

    # 1) Read entity2id.txt into a dictionary
    entity2id_path = os.path.join(folder_path, entity2id_file)
    entity2id = {}
    with open(entity2id_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entity_name, entity_id_str = line.split("\t")
            entity_id = int(entity_id_str)
            entity2id[entity_name] = entity_id

    # 2) Read triple.txt, filtering for (gene, "phenotype", pheno)
    #    We'll store them in a dict: pheno -> set of genes
    triple_path = os.path.join(folder_path, triple_file)
    pheno2genes = {}
    with open(triple_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # triple.txt format: "entity1_name relation_name entity2_name"
            e1_name, relation_name, e2_name = line.split("\t")
            if relation_name != "phenotype":
                continue

            # Caution: assuming all genes start with 'agis_'
            if not e1_name.startswith('agis_'):
                continue

            # Skip if either entity is not in entity2id
            if e1_name not in entity2id or e2_name not in entity2id:
                continue

            if e2_name not in pheno2genes:
                pheno2genes[e2_name] = set()
            pheno2genes[e2_name].add(e1_name)

    # 3) Gather all unique genes
    all_genes = set()
    for pheno, genes in pheno2genes.items():
        all_genes.update(genes)
    all_genes = list(all_genes)

    # 4) Shuffle once for reproducibility
    random.seed(seed)
    random.shuffle(all_genes)

    # 5) Create a pheno2id mapping (0 to num_phenos-1)
    #    We'll just sort phenos by name to have a consistent ordering
    all_phenos = sorted(pheno2genes.keys())
    pheno2id = {pheno_name: idx for idx, pheno_name in enumerate(all_phenos)}

    # 6) Write a single pheno2id.txt
    pheno2id_path = os.path.join(folder_path, pheno2id_file)
    with open(pheno2id_path, "w", encoding="utf-8") as f:
        for pheno_name in all_phenos:
            f.write(f"{pheno_name} {pheno2id[pheno_name]}\n")

    # 7) Split the genes into k folds
    folds = np.array_split(all_genes, k)

    # 8) For each fold i, create train_i.txt and test_i.txt
    for i in range(k):
        test_genes = set(folds[i])
        # Train is everything not in the i-th fold
        train_genes = set()
        for j in range(k):
            if j != i:
                train_genes.update(folds[j])

        # Define output paths for fold i
        train_path = os.path.join(folder_path, f"train_{i}.txt")
        test_path = os.path.join(folder_path, f"test_{i}.txt")

        with open(train_path, "w", encoding="utf-8") as f_train, open(
            test_path, "w", encoding="utf-8"
        ) as f_test:

            # For each phenotype, write the appropriate line to train or test
            for pheno_name in all_phenos:
                pid = pheno2id[pheno_name]
                genes_for_pheno = pheno2genes[pheno_name]

                # Genes in the training fold
                train_genes_for_pheno = genes_for_pheno.intersection(train_genes)
                if train_genes_for_pheno:
                    train_gene_ids = [entity2id[g] for g in sorted(train_genes_for_pheno)]
                    line_str = str(pid) + " " + " ".join(map(str, train_gene_ids))
                    f_train.write(line_str + "\n")

                # Genes in the test fold
                test_genes_for_pheno = genes_for_pheno.intersection(test_genes)
                if test_genes_for_pheno:
                    test_gene_ids = [entity2id[g] for g in sorted(test_genes_for_pheno)]
                    line_str = str(pid) + " " + " ".join(map(str, test_gene_ids))
                    f_test.write(line_str + "\n")

    # 9) Create a train.txt that includes *all* pheno-gene pairs
    train_all_path = os.path.join(folder_path, "train.txt")
    with open(train_all_path, "w", encoding="utf-8") as f_train_all:
        for pheno_name in all_phenos:
            pid = pheno2id[pheno_name]
            genes_for_pheno = pheno2genes[pheno_name]

            if genes_for_pheno:
                all_gene_ids = [entity2id[g] for g in sorted(genes_for_pheno)]
                line_str = str(pid) + " " + " ".join(map(str, all_gene_ids))
                f_train_all.write(line_str + "\n")

    print(f"Successfully created k-fold train/test files (k={k}) and train.txt in {folder_path}")

def get_gene_set(triple_data):
    """
    Identifies which entities are genes by looking for "agis_"
    Returns a set of gene names.
    """
    gene_names = set()
    for (head, relation, tail) in triple_data:
        if head.startswith('agis_'):
            gene_names.add(head)
        if tail.startswith('agis_'):
            gene_names.add(tail)
    return gene_names


def main():
    # 1) Build up entity_set, relation_set, triple_set
    multi_data()

    print("entity_set: ", len(entity_set))
    print("relation_set: ", len(relation_set))
    print("triple_set: ", len(triple_set))

    # 2) Identify genes so that we can give them smaller IDs
    gene_set = get_gene_set(triple_set)
    # We'll reorder entities so that genes come first
    all_entities = list(entity_set)
    genes = sorted(list(gene_set))
    others = sorted([e for e in all_entities if e not in gene_set])
    reordered_entities = genes + others

    # 3) Write out entity2id.txt with genes first
    # (We manually write it in the desired order,
    #  so we won't use write2txt(...) for this file.)
    entity2id_path = os.path.join(target_path, "entity2id.txt")
    with open(entity2id_path, "w", encoding="utf-8") as f_ent:
        for idx, entity in enumerate(reordered_entities):
            f_ent.write(f"{entity}\t{idx}\n")

    # 4) Write out relation2id.txt and triple.txt the usual way
    write2txt(
        os.path.join(target_path, "relation2id.txt"),
        relation_set,
        "set",
    )
    write2txt(
        os.path.join(target_path, "triple.txt"),
        triple_set,
        "list",
    )

    # 5) Build the final KG (kg_final.txt) and run create_kfold
    build_kg_final(target_path, file_out="kg_final.txt")
    create_kfold(target_path, k=10, seed=42)


if __name__ == "__main__":
    main()
