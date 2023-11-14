import re
import numpy as np
import stanza
from nltk.tree import ParentedTree
from stanfordcorenlp import StanfordCoreNLP


class SentenceComplexity:
    def __init__(self, nlp_pipeline):
        self.corenlp = StanfordCoreNLP("resources/stanford-corenlp-full-2018-02-27")
        self.nlp_pipeline = nlp_pipeline

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

    """Average Dependency Type 
	Reference:- Syntactic Dependency Distance as Sentence Complexity Measure (Masanori Oya)"""

    def ADD(self, doc):
        count = 0
        no = 0
        trips = []
        for sent in doc.sentences:
            for word in sent.words:
                if word.deprel != "root":
                    if word.deprel != "punct":
                        ad = np.abs(word.id - word.head)
                        no = no + 1
                        trip = f"{word.deprel}({doc.sentences[0].words[word.head-1].text}-{doc.sentences[0].words[word.head-1].id},{word.text}-{word.id})"
                        trips.append(trip)
                        count = count + ad
        if no != 0:
            add_score = count / no
        else:
            add_score = 0
        return add_score

    """
	Depth of sentence 
	"""

    def depth(self, sentence):
        ptree = ParentedTree.fromstring(self.corenlp.parse(sentence))
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
