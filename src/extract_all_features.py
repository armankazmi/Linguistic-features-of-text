import re
import nltk
import stanza
import numpy as np
from nltk.tree import ParentedTree
from stanfordcorenlp import StanfordCoreNLP
from src.lexical_diversity import LexicalDiversity


class Features:
    def __init__(
        self,
        para_list,
        nlp_pipeline,
        remove_punctuations=True,
    ):
        self.para_list = para_list
        self.nlp_pipeline = nlp_pipeline
        self.corenlp_resource = "resources/stanford-corenlp-full-2018-02-27"
        # List of stanza tagged document
        self.docs = self.pipeline(nlp_pipeline)
        self.remove_punctuations = remove_punctuations

    def pipeline(self, nlp_pipeline):
        in_docs = [stanza.Document([], text=d) for d in self.para_list]
        docs = nlp_pipeline(in_docs)
        return docs

    @staticmethod
    def _pre_process(text):
        ## Removing punctuations
        text = re.sub(r"[^\w\s]", " ", text).strip().replace("\n", "")
        text = " ".join(text.split())
        return text

    @staticmethod
    def safe_divide(numerator, denominator):
        index = 0 if denominator == 0 else numerator / denominator
        return index

    @staticmethod
    def get_noun_phrases(pos):
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

    @staticmethod
    def initialize_corenlp(corenlp_resource):
        corenlp = StanfordCoreNLP(corenlp_resource)
        return corenlp

    def _extract_features(self, choice):
        if choice == "all":
            output = []
            for para, doc in zip(self.para_list, self.docs):
                temp = self._raw_features(doc)
                temp.update(self._lexical_features(para, doc))
                temp.update(self._pos_features(doc))
                temp.update(self._syntactic_features(doc))
                output.append(temp)
        else:
            if choice == "raw":
                output = [self._raw_features(doc) for doc in self.docs]
            if choice == "lexical":
                output = [
                    self._lexical_features(para, doc)
                    for para, doc in zip(self.para_list, self.docs)
                ]
            if choice == "pos":
                output = [self._pos_features(doc) for doc in self.docs]
            if choice == "syntactic":
                output = [self._syntactic_features(doc) for doc in self.docs]
        return output

    def _raw_features(self, doc):
        sen_lens = [len([word.text for word in sent.words]) for sent in doc.sentences]
        return {"avg_sen_len": np.mean(sen_lens), "std_sen_len": np.std(sen_lens)}

    def _lexical_features(self, para, doc):
        # Lexical Diversity features
        ld = LexicalDiversity(para.lower())
        ld_measures = ld._lexical_diversity()

        # Lexical Density features
        pos_counts = self._pos_counts(doc)
        content_words = sum(
            pos_counts[key] for key in ["noun", "verb", "adjective", "adverb"]
        )
        function_words = sum(pos_counts[key] for key in ["pronoun", "others"])
        ld_measures["content/function"] = self.safe_divide(
            content_words, function_words
        )
        return ld_measures

    def _pos_features(self, doc):
        pos_counts = self._pos_counts(doc)
        ratios = {
            f"{key}/{other}": self.safe_divide(pos_counts[key], pos_counts[other])
            for key, other in [
                ("adverb", "adjective"),
                ("adverb", "noun"),
                ("adverb", "pronoun"),
                ("adjective", "verb"),
                ("adjective", "pronoun"),
                ("noun", "verb"),
                ("noun", "pronoun"),
                ("verb", "pronoun"),
            ]
        }
        return ratios

    def _pos_counts(self, doc):
        adverb, adjective, pronoun, noun, verb, others = 0, 0, 0, 0, 0, 0
        for sent in doc.sentences:
            for word in sent.words:
                if word.upos == "ADJ":
                    adjective += 1
                if word.upos == "ADV":
                    adverb += 1
                if word.upos == "PRON":
                    pronoun += 1
                if word.upos == "NOUN":
                    noun += 1
                if word.upos == "VERB":
                    verb += 1
                else:
                    others += 1
        return {
            "adjective": adjective,
            "adverb": adverb,
            "pronoun": pronoun,
            "noun": noun,
            "verb": verb,
            "others": others,
        }

    def _syntactic_features(self, doc, sentence_list=None):
        sentence_list = (
            [
                " ".join([word.text for word in sent.words]).strip()
                for sent in doc.sentences
            ]
            if not sentence_list
            else sentence_list
        )
        if self.remove_punctuations:
            # Removing punctuations
            pre_processed_sen_list = [
                self._pre_process(sentence) for sentence in sentence_list
            ]
            in_docs = [stanza.Document([], text=s) for s in pre_processed_sen_list]
            pre_processed_sen_docs = self.nlp_pipeline(in_docs)
            # ISC Score
            isc_scores = [
                self._isc(processed_doc.sentences[0])
                for processed_doc in pre_processed_sen_docs
            ]
            # ADD
            add_scores = [
                self._add(processed_doc.sentences[0])
                for processed_doc in pre_processed_sen_docs
            ]
            # Depth of a sentence
            corenlp = self.initialize_corenlp(self.corenlp_resource)
            avg_depth_scores = [
                self._depth(sen, corenlp) for sen in pre_processed_sen_list
            ]

            # Dependency relations & bigrams
            dep_feats = {}
            for processed_doc in pre_processed_sen_docs:
                dep_rel, dep_big = self._dependency_features(processed_doc.sentences[0])
                for key, val in (dep_rel + dep_big).items():
                    if key in dep_feats:
                        dep_feats[key] += val
                    else:
                        dep_feats[key] = val
        """
        else:
            # ISC Score
            isc_scores = [self._isc(sentence) for sentence in doc.sentences]
            # ADD Score
            add_scores = [self._add(sentence) for sentence in doc.sentences]
            # Dependency relations & bigrams
            dep_feats = {}
        """
        output = {}
        output["Mean ISC Score"] = np.mean(isc_scores)
        output["Std ISC Score"] = np.std(isc_scores)
        output["Mean ADD Score"] = np.mean(add_scores)
        output["Std ADD Score"] = np.std(add_scores)
        output["mean para depth"] = np.mean(avg_depth_scores)
        output["Std para depth"] = np.std(avg_depth_scores)
        output["dependency features"] = dep_feats
        return output

    """ISC, Reference:- https://web.stanford.edu/~bresnan/LabSyntax/szmrecsanyi-syntactic.complexity.pdf"""

    def _isc(self, sentence_doc):
        sub, wh_pro, verb_form = 0, 0, 0
        pos_tags = []
        for word in sentence_doc.words:
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

    """Average Dependency Type 
	Reference:- Syntactic Dependency Distance as Sentence Complexity Measure (Masanori Oya)"""

    def _add(self, sentence_doc):
        count = 0
        no = 0
        trips = []
        for word in sentence_doc.words:
            if word.deprel != "root":
                if word.deprel != "punct":
                    ad = np.abs(word.id - word.head)
                    no = no + 1
                    trip = f"{word.deprel}({sentence_doc.words[word.head-1].text}-{sentence_doc.words[word.head-1].id},{word.text}-{word.id})"
                    trips.append(trip)
                    count = count + ad
        if no != 0:
            add_score = count / no
        else:
            add_score = 0
        return add_score

    """Calculating Average Depth of a sentence"""

    def _depth(self, sentence, corenlp):
        ptree = ParentedTree.fromstring(corenlp.parse(sentence))
        depths = []
        a = []
        leaf_nodes = sentence.split()
        for subtree in ptree.subtrees():
            if len(list(subtree)) == 1:
                if list(subtree)[0] in leaf_nodes:
                    a.append(subtree)
        for i in range(len(a)):
            count = 0
            lab = None
            tree = a[i]
            flag = 0
            while flag == 0:
                parent = tree.parent()
                if parent.label() == "ROOT":
                    flag = 1
                if parent.right_sibling():
                    count = count + 1
                tree = parent
            depths.append(count)
        depth_score = np.mean(depths)
        return depth_score

    """Dependency Relations and Bigrams features"""

    def _dependency_features(self, sentence_doc):
        trigrams = []
        relations = []
        for word in sentence_doc.words:
            if word.deprel != "root":
                relations.append(word.deprel)
                if word.id > word.head:
                    position = "before"
                else:
                    position = "after"
                feat = str(
                    (sentence_doc.words[word.head - 1].upos, word.upos, position)
                )
                trigrams.append(feat)
        rel_bi = nltk.FreqDist(relations)
        dep_tri = nltk.FreqDist(trigrams)
        return rel_bi, dep_tri
