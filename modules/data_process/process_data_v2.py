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


def main():
    multi_data()
    files_name = ["entity2id.txt", "relation2id.txt", "triple.txt"]
    print(len(entity_set))
    print(len(relation_set))
    print(len(triple_set))
    write2txt(os.path.join(target_path, files_name[0]), entity_set, "set")
    write2txt(os.path.join(target_path, files_name[1]), relation_set, "set")
    write2txt(os.path.join(target_path, files_name[2]), triple_set, "list")


if __name__ == "__main__":
    main()
