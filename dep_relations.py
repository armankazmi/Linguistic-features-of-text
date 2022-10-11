import re
import nltk
import stanza

class DependencyRelations:
	def __init__(self):
		self.nlp = stanza.Pipeline(lang='en', processors={'tokenize': 'spacy'})
	
	'''TPS'''
	def get_dep_rel(self, sentence):
		doc = self.nlp(sentence)
		dep = {}
		for sent in doc.sentences:
			for word in sent.words:
				if word.deprel != 'root':
					if word.deprel in dep:
						dep[word.deprel] += 1
					else:
						dep[word.deprel] = 1

		return dep

	#feature - (head pos tag, dependent pos tag, position of head i.e. before the dependent word or after the dependent word)
	def get_dependency_features(self, sentence):
		doc = self.nlp(sentence)
		trigrams = []
		relations = []
		for index,sent in enumerate(doc.sentences):
			for word in sent.words:
				if word.deprel != 'root':
					relations.append(word.deprel)
					if word.id > word.head: 
						position = 'before'
					else:
						position = 'after'
					feat = str((doc.sentences[index].words[word.head-1].upos, word.upos, position))
					trigrams.append(feat)
		rel_bi = nltk.FreqDist(relations)
		dep_tri = nltk.FreqDist(trigrams)
		return rel_bi, dep_tri