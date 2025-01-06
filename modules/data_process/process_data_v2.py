"""
This file is copied from Jinzhe's TransE and LET-ConvE work.
2024.12.26 process data
处理kg数据
"""
import json
import os
import re
import csv
from tqdm import tqdm
import random  # Added
import numpy as np  # Used for np.array_split below

work_name = "rice" # rice or multi

target_path = "../../data/rice"
multi_folder_path = "../../data/rice/for_predicted_traits_Dec19_2024"

entity_set = set()
relation_set = set()
triple_set = set()

csv_name = ["KG_basic_gene_KeggGO_Dec19_2024.csv", "KG_gene2transcrpit_Dec19.csv", "KG_OE_Mut_WOS_PUBMED_RAPDB_traitGene.csv", "KG_transcrpit2protein_Dec19.csv", ]
another_list = ["KG_proteomics_transcriptomics_Dec07.csv"]

def multi_data():
    """
    多组学数据
    """
    for filename in os.listdir(multi_folder_path):
        if filename.endswith('.csv'):
            if filename == "gene_white_list_Dec19.csv":
                continue
            file_path = os.path.join(multi_folder_path, filename)
            if filename in csv_name:
                with open(file_path, mode='r', encoding='utf-8') as file:
                    reader = csv.reader(file)
                # 逐行读取
                    for row in reader:
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
            elif filename in another_list:
                with open(file_path, mode='r', encoding='utf-8') as file:
                    reader = csv.reader(file)
                # 逐行读取
                    for row in reader:
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


# Added
def build_kg_final(path, file_entity2id='entity2id.txt', file_relation2id="relation2id.txt", 
                   file_triple="triple.txt", file_out="kg_final.txt"):
    """
    Build 'kg_final.txt' by reading and transforming entity/relation IDs 
    and filtering out any triple whose relation is 'phenotype'.
    """
    
    # Initialize dictionaries to map entities and relations to their respective IDs
    entity2id = {}
    relation2id = {}

    # Read the entity-to-ID mappings
    with open(os.path.join(path, file_entity2id), 'r', encoding='utf-8') as f_ent:
        for line in f_ent:
            entity, eid = line.strip().split('\t')
            entity2id[entity] = eid

    # Read the relation-to-ID mappings
    with open(os.path.join(path, file_relation2id), 'r', encoding='utf-8') as f_rel:
        for line in f_rel:
            relation, rid = line.strip().split('\t')
            relation2id[relation] = rid

    # Open the output file to write filtered triples
    with open(os.path.join(path, file_out), 'w', encoding='utf-8') as f_out:
        # Open the original triples file
        with open(os.path.join(path, file_triple), 'r', encoding='utf-8') as f_tri:
            for line in f_tri:
                head, relation, tail = line.strip().split('\t')
                
                # Assert that all parts of the triple exist in the dictionaries
                assert head in entity2id and tail in entity2id and relation in relation2id
                
                # Exclude triples that use the 'phenotype' relation
                if relation != "phenotype":
                    # Write the valid triple in ID form to the output
                    f_out.write(f"{entity2id[head]} {relation2id[relation]} {entity2id[tail]}\n")


def create_kfold(
    folder_path,
    entity2id_file='entity2id.txt',
    triple_file="triple.txt",
    pheno2id_file='pheno2id.txt',
    k=10,
    seed=42
):
    """
    Create k-fold cross validation splits.
    For each fold i, produce train_i.txt and test_i.txt.

    train_i.txt and test_i.txt have the format:
        pheno_id gene1_id gene2_id ...
    We also create a single pheno2id.txt with IDs for phenotypes.
    :param k: Number of folds (default = 10)
    :param seed: Random seed for reproducibility
    """

    # 1) Read entity2id.txt into a dictionary
    entity2id_path = os.path.join(folder_path, entity2id_file)
    entity2id = {}
    with open(entity2id_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entity_name, entity_id_str = line.split('\t')
            entity_id = int(entity_id_str)
            entity2id[entity_name] = entity_id

    # 2) Read triple.txt, filtering for (gene, "phenotype", pheno)
    #    We'll store them in a dict: pheno -> set of genes
    triple_path = os.path.join(folder_path, triple_file)
    pheno2genes = {}
    with open(triple_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # triple.txt format: "entity1_name relation_name entity2_name"
            e1_name, relation_name, e2_name = line.split('\t')
            if relation_name != "phenotype":
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
    with open(pheno2id_path, 'w', encoding='utf-8') as f:
        for pheno_name in all_phenos:
            f.write(f"{pheno_name} {pheno2id[pheno_name]}\n")

    # 7) Split the genes into k folds
    #    np.array_split handles edge cases where len(all_genes) is not
    #    perfectly divisible by k.
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

        with open(train_path, 'w', encoding='utf-8') as f_train, \
             open(test_path, 'w', encoding='utf-8') as f_test:

            # For each phenotype, write the appropriate line to train or test
            for pheno_name in all_phenos:
                pid = pheno2id[pheno_name]
                genes_for_pheno = pheno2genes[pheno_name]

                # Genes that appear in the training set
                train_genes_for_pheno = genes_for_pheno.intersection(train_genes)
                if train_genes_for_pheno:
                    train_gene_ids = [entity2id[g] for g in sorted(train_genes_for_pheno)]
                    line_str = str(pid) + " " + " ".join(map(str, train_gene_ids))
                    f_train.write(line_str + "\n")

                # Genes that appear in the test set
                test_genes_for_pheno = genes_for_pheno.intersection(test_genes)
                if test_genes_for_pheno:
                    test_gene_ids = [entity2id[g] for g in sorted(test_genes_for_pheno)]
                    line_str = str(pid) + " " + " ".join(map(str, test_gene_ids))
                    f_test.write(line_str + "\n")

    print(f"Successfully created k-fold train/test files (k={k}) in {folder_path}")



def main():
    multi_data()
    files_name = ["entity2id.txt", "relation2id.txt", "triple.txt"]
    print("entity_set: ", len(entity_set))
    print("relation_set: ", len(relation_set))
    print("triple_set: ", len(triple_set))
    write2txt(os.path.join(target_path, files_name[0]), entity_set, "set")
    write2txt(os.path.join(target_path, files_name[1]), relation_set, "set")
    write2txt(os.path.join(target_path, files_name[2]), triple_set, "list")
    # Added
    build_kg_final(target_path, file_out="kg_final.txt")
    create_kfold(target_path, k=10, seed=42)

if __name__ == "__main__":
    main()
