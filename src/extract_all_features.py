import re
import nltk
import stanza
import numpy as np
from src.lexical_diversity import LexicalDiversity
from src.sentence_complexity import SentenceComplexity


class Features:
    def __init__(self, nlp_pipeline, para):
        self.para = para
        self.doc = nlp_pipeline(para)
        self.nlp_pipeline = nlp_pipeline

        # if choice == "all_feats":
        #     output = {}
        #     output["raw_features"] = self._raw_features()
        #     output["lexical_features"] = self._lexical_features()
        #     output["pos_features"] = self._pos_features()
        #     output["syntactic_features"] = self._syntactic_features()

        # if choice == "raw":
        #     return self._raw_features()
        # if choice == "lexical":
        #     return self._lexical_features()
        # if choice == "pos":
        #     return self._pos_features()
        # if choice == "syntactic":
        #     return self._syntactic_features()
        
    @staticmethod
    def _pre_process(text):
        ## Removing punctuations
        text = re.sub(r'[^\w\s]', ' ', text).strip().replace('\n', '')
        text = " ".join(text.split())
        return text
    @staticmethod
    def safe_divide(self, numerator, denominator):
        index = 0 if denominator == 0 else numerator / denominator
        return index
    
    @staticmethod    
    def get_noun_phrases(self, pos):
        count = 0
        half_chunk = ""
        for word, tag in pos:
            if re.match(r"NN.*", tag):
                count += 1
                if count >= 1:
                    half_chunk = half_chunk + word + " "
            else:
                half_chunk = half_chunk + "---"
                count = 0
        half_chunk = re.sub(r"-+", "?", half_chunk).split("?")
        half_chunk = [x.strip() for x in half_chunk if x != ""]
        return len(half_chunk)

    def _extract_features(self, choice):
        output = {}
        if choice == "all_feats":
            output["raw_features"] = self._raw_features()
            output["lexical_features"] = self._lexical_features()
            output["pos_features"] = self._pos_features()
            output["syntactic_features"] = self._syntactic_features()
        else:
            output[choice + "_features"] = getattr(self, "_" + choice + "_features")()
        return output

    def _raw_features(self, sen_list=None):
        print("Computing raw features.")
        if sen_list:
            sen_lens = [len(sen.split()) for sen in sen_list]
        else:
            sen_lens = [
                len([word.text for word in sent.words]) for sent in self.doc.sentences
            ]
        return {"avg_sen_len": np.mean(sen_lens), "std_sen_len": np.std(sen_lens)}

    def _lexical_features(self):
        # Lexical Diversity features
        print("Computing Lexical features: ")
        print("a. Charachter diversity features.")
        ld = LexicalDiversity(self.para.lower())
        ld_measures = ld._lexical_diversity()

        # Lexical Density features
        print("b. Lexical density features.")
        pos_counts = self._pos_counts()
        content_words = sum(pos_counts[key] for key in ["noun", "verb", "adjective", "adverb"])
        function_words = sum(pos_counts[key] for key in ["pronoun", "others"])
        ld_measures["lexical_density"] = self.safe_divide(content_words, function_words)
        return ld_measures

    def _pos_features(self):
        print("Computing POS ratios features.")
        pos_counts = self._pos_counts()
        ratios = {f"{key}/{other}": self.safe_divide(pos_counts[key], pos_counts[other])
                  for key, other in [("adverb", "adjective"), ("adverb", "noun"), ("adverb", "pronoun"),
                                     ("adjective", "verb"), ("adjective", "pronoun"), ("noun", "verb"),
                                     ("noun", "pronoun"), ("verb", "pronoun")]}
        return ratios

    def _pos_counts(self):
        adverb, adjective, pronoun, noun, verb, others = 0, 0, 0, 0, 0, 0
        for sent in self.doc.sentences:
            for word in sent.words:
                if word.upos == "ADJ": adjective += 1
                if word.upos == "ADV": adverb += 1
                if word.upos == "PRON": pronoun += 1
                if word.upos == "NOUN": noun += 1
                if word.upos == "VERB": verb += 1
                else: others += 1
        return {
            "adjective": adjective,
            "adverb": adverb,
            "pronoun": pronoun,
            "noun": noun,
            "verb": verb,
            "others": others,
        }

    def _syntactic_features(self, sen_list=None):
        print("Computing Syntactic features: ")
        sen_list = (
            [
                " ".join([word.text for word in sent.words]).strip()
                for sent in self.doc.sentences
            ]
            if not sen_list
            else sen_list
        )
        ## Sentence Complexity features
        print("a. Sentence Complexity features.")
        sen_comp = SentenceComplexity(self.nlp_pipeline)
        return None

    """ISC, Reference:- https://web.stanford.edu/~bresnan/LabSyntax/szmrecsanyi-syntactic.complexity.pdf"""
    def ISC(self, doc):
        sub, wh_pro, verb_form = 0, 0, 0
        pos_tags = []
        for sent in doc.sentences:
            for word in sent.words:
                if word.upos == "SCONJ":
                    sub = sub + 1
                if word.upos == "VERB":
                    verb_form += 1
                if word.xpos == "WP":
                    wh_pro += 1
                pos_tags.append((word.text, word.xpos))
        noun_phr = self.get_noun_phrases(pos_tags)
        isc_score = 2 * sub + 2 * wh_pro + verb_form + noun_phr
        return isc_score

    