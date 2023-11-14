import re
import nltk
import stanza
import numpy as np
import pandas as pd
from tqdm import tqdm
from nltk.corpus import brown
from src.extract_all_features import Features
from joblib import Parallel, delayed


# sub-genres
def _get_lex_data():
    fiction = ["adventure", "fiction", "mystery", "romance", "science_fiction"]
    nonfiction = ["government", "hobbies", "learned", "news", "reviews"]

    # fileids
    fiction_ids = [x for y in fiction for x in brown.fileids(categories=y)]
    nonfiction_ids = [x for y in nonfiction for x in brown.fileids(categories=y)]

    from src.extract_all_features import Features

    lex_den_data = []
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
                temp["label"] = label
                temp["text"] = text
                temp["p"] = p
                lex_den_data.append(temp)
        # print('Finished', index)
    return lex_den_data


nlp = stanza.Pipeline(lang="en", processors={"tokenize": "spacy"})


def _lex_den(t):
    feat = Features(nlp_pipeline=nlp, text=t)
    pos_counts = feat._pos_counts()
    return pos_counts


import multiprocessing
import time


if __name__ == "__main__":
    lex_den_data = _get_lex_data()
    texts = [lex_den_data[0]["text"], lex_den_data[1]["text"], lex_den_data[2]["text"]]

    start_time = time.perf_counter()
    result = Parallel(n_jobs=4, prefer="threads")(delayed(_lex_den)(i) for i in texts)
    finish_time = time.perf_counter()
    print(f"Program finished in {finish_time-start_time} seconds")
    print(result)
