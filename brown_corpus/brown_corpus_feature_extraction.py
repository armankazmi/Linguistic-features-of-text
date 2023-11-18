import re
import nltk
import stanza
import time
import numpy as np
import pandas as pd
from tqdm import tqdm
from nltk.corpus import brown
from src.lexical_diversity import LexicalDiversity
from src.extract_all_features import Features
from joblib import Parallel, delayed

nlp = stanza.Pipeline(lang="en", processors={"tokenize": "spacy"})


def get_brown_paras():
    # Sub-categories
    fiction = ["adventure", "fiction", "mystery", "romance", "science_fiction"]
    nonfiction = ["government", "hobbies", "learned", "news", "reviews"]
    # fileids
    fiction_ids = [x for y in fiction for x in brown.fileids(categories=y)]
    nonfiction_ids = [x for y in nonfiction for x in brown.fileids(categories=y)]
    para_data = []
    for index, fileid in tqdm(enumerate(fiction_ids + nonfiction_ids)):
        paras = brown.paras(fileids=fileid)
        label = 1 if fileid in fiction_ids else 0
        for j, p in enumerate(paras):
            if len(p) > 4 and len(p) < 7:
                text = " "
                for sent in p:
                    text = text + " ".join(sent)
                text = text.strip()
                temp = {}
                temp["id"] = f"{fileid}_para_{j}"
                temp["text"] = text
                temp["chunked_paras"] = p
                temp["label"] = label
                para_data.append(temp)
    return para_data


## Computing Raw Features
def get_raw_features(data):
    raw_feats = []
    for temp in data:
        sen_len = [len(sen) for sen in temp["chunked_paras"]]
        temp_dict = {}
        temp_dict["id"] = temp["id"]
        temp_dict["label"] = temp["label"]
        temp_dict["avg_sen_len"] = np.mean(sen_len)
        temp_dict["std_sen_len"] = np.std(sen_len)
        raw_feats.append(temp_dict)
    return raw_feats


## Computing Lexical Diversity features
def get_character_diversity_features(data):
    lexical_div_feats = []
    for temp in data:
        text = temp["text"]
        text = text.replace(" ", "")
        ld = LexicalDiversity(text.lower())
        ld_values = ld._lexical_diversity()
        ld_values["id"] = temp["id"]
        ld_values["label"] = temp["label"]
        lexical_div_feats.append(ld_values)
    return lexical_div_feats


## Computing Lexical Density features
def get_lexical_density_features(data_dict):
    feat = Features(nlp_pipeline=nlp, text=data_dict["text"])
    pos_counts_output = feat._pos_counts()
    pos_counts_output["id"] = data_dict["id"]
    pos_counts_output["label"] = data_dict["label"]
    return pos_counts_output


if __name__ == "__main__":
    brown_paras = get_brown_paras()

    # ## Computing Raw features
    # raw_features = get_raw_features(brown_paras)
    # print('Saving Raw features. ')
    # pd.DataFrame(raw_features).to_csv('COLING_Files/data/brown_corpus_raw_features.csv')

    # ## Computing Lexical Features (Time taken ~ 120 sec)
    # diversity_features = get_character_diversity_features(brown_paras)
    # print('Saving Lexical Diversity features. ')
    # pd.DataFrame(diversity_features).to_csv('COLING_Files/data/brown_corpus_charachter_diversity_features.csv')

    ## Computing Lexical Density Features
    ## Parellilizing the computation
    start_time = time.perf_counter()
    lex_den_outputs = Parallel(n_jobs=4, prefer="threads")(
        delayed(get_lexical_density_features)(dic) for dic in brown_paras
    )
    # lex_den_ouputs = [get_lexical_density_features(x) for x in brown_paras]
    finish_time = time.perf_counter()
    print(f"Finished in {finish_time-start_time} seconds")
