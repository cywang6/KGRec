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



def create_train_test(
    folder_path, entity2id_file='entity2id.txt', triple_file="triple.txt", 
    train_file="train.txt", test_file='test.txt', pheno2id_file='pheno2id.txt', 
    test_ratio = 0.1, seed = 42
):
    """
    Create train.txt and test.txt with the format:
       pheno_id gene1_id gene2_id ...
    Also create pheno2id.txt with IDs ranging from 0..(num_phenos-1).
    :param test_ratio: Fraction of genes to put in test set (default = 0.1)
    :param seed: Random seed for reproducibility
    """
    random.seed(seed)
    entity2id_path = os.path.join(folder_path, entity2id_file)
    triple_path = os.path.join(folder_path, triple_file)
    train_path = os.path.join(folder_path, train_file)
    test_path = os.path.join(folder_path, test_file)
    pheno2id_path = os.path.join(folder_path, pheno2id_file)
    
    # 1) Read entity2id.txt into a dictionary
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
            
            # Collect them in a dictionary: pheno -> set of genes
            if e2_name not in pheno2genes:
                pheno2genes[e2_name] = set()
            pheno2genes[e2_name].add(e1_name)
    
    # Gather all unique genes
    all_genes = set()
    for pheno, genes in pheno2genes.items():
        all_genes.update(genes)
    
    # 3) Split genes into train/test sets (90/10 default)
    all_genes = list(all_genes)
    random.shuffle(all_genes)
    split_index = int(len(all_genes) * test_ratio)
    test_genes = set(all_genes[:split_index])
    train_genes = set(all_genes[split_index:])
    
    # 4) Create a pheno2id mapping (0 to num_phenos-1)
    #    We'll just sort phenos by name to have a consistent ordering
    all_phenos = sorted(pheno2genes.keys())
    pheno2id = {pheno_name: idx for idx, pheno_name in enumerate(all_phenos)}
    
    # 5) Write pheno2id.txt
    with open(pheno2id_path, 'w', encoding='utf-8') as f:
        for pheno_name in all_phenos:
            f.write(f"{pheno_name} {pheno2id[pheno_name]}\n")
    
    # 6) Generate train.txt and test.txt
    #    Format for each line: "pheno_id gene1_id gene2_id ..."
    with open(train_path, 'w', encoding='utf-8') as f_train, \
         open(test_path, 'w', encoding='utf-8') as f_test:
        
        for pheno_name in all_phenos:
            # pheno ID
            pid = pheno2id[pheno_name]
            
            # All genes associated with this phenotype
            genes_for_pheno = pheno2genes[pheno_name]
            
            # Separate out genes into train split vs test split
            train_genes_for_pheno = genes_for_pheno.intersection(train_genes)
            test_genes_for_pheno = genes_for_pheno.intersection(test_genes)
            
            # Write train line if there is at least one gene
            if train_genes_for_pheno:
                # Convert gene names to IDs
                train_gene_ids = [entity2id[g] for g in sorted(train_genes_for_pheno)]
                line_str = str(pid) + " " + " ".join(map(str, train_gene_ids))
                f_train.write(line_str + "\n")
            
            # Write test line if there is at least one gene
            if test_genes_for_pheno:
                test_gene_ids = [entity2id[g] for g in sorted(test_genes_for_pheno)]
                line_str = str(pid) + " " + " ".join(map(str, test_gene_ids))
                f_test.write(line_str + "\n")


def main():
    # multi_data()
    # files_name = ["entity2id.txt", "relation2id.txt", "triple.txt"]
    # print(len(entity_set))
    # print(len(relation_set))
    # print(len(triple_set))
    # write2txt(os.path.join(target_path, files_name[0]), entity_set, "set")
    # write2txt(os.path.join(target_path, files_name[1]), relation_set, "set")
    # write2txt(os.path.join(target_path, files_name[2]), triple_set, "list")
    # # Added
    # build_kg_final(target_path, file_out="kg_final.txt")
    create_train_test(target_path, test_ratio=0.1, seed=42)

if __name__ == "__main__":
    main()
