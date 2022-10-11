import re
import nltk
import stanza
import numpy as np

from feature_calculation.lexical_div_cal import LexicalDiversity
from feature_calculation.pos_ratios_cal import HighLevelRatios
# from feature_calculation.sentence_complexity import SentenceComplexity
from feature_calculation.dep_relations import DependencyRelations

class Features:

	def __init__(self):
		self.nlp = stanza.Pipeline(lang='en', processors={'tokenize': 'spacy'})

	def safe_divide(self, numerator, denominator):
		if denominator == 0 or denominator == 0.0:
			index = 0
		else:
			index = numerator/denominator
		return index

	def get_all_features(self, sen_list, text=None):
		text = ' '.join(sen_list)
		temp  = {}
		
		sen_len = [len(x.split()) for x in sen_list]
		temp['avg_sen_len'] = np.mean(sen_len)
		temp['std_sen_len'] = np.std(sen_len)
		
		ld_measures = self.get_ld_measure(text)
		temp.update(ld_measures)
		pos_measures = self.get_pos_ratios(text)
		temp.update(pos_measures)
		# sen_com_measures = self.get_sen_comp(sen_list)
		# temp.update(sen_com_measures)
		all_dep_measures = self.get_all_dep_feats(sen_list)
		temp.update(all_dep_measures)
		
		return temp

	def get_ld_measure(self, text):
		ld_text = ''.join(text.lower().split())
		ld = LexicalDiversity(ld_text)
		tokens = ld.get_tokens(char=True)
		ld_data = ld.get_lexical_measures(tokens)
		
		#scaling 
		ld_mean = {'TTR' : 8.11262665e-02, 
					'Root TTR' : 1.47188009e+00, 
					'Log TTR' : 5.64327689e-01, 
					'Maas TTR' : 1.68511737e-01,
					'Msstr' : 3.77196411e-01, 
					'Ma TTR' : 3.77330440e-01, 
					'HDD' : 4.21878311e-01, 
					'MTLD' : 1.43219078e+01,
					'MTLD MA' : 1.42771735e+01, 
					'MTLD MA Bi' : 1.42060573e+01, 
					'VocD' : 6.49950474e+00, 
					'YulesK': 6.02132971e+02
					}
		ld_std = {'TTR' : 3.86666604e-02, 
					'Root TTR' : 3.13960442e-01, 
					'Log TTR' : 4.11986152e-02, 
					'Maas TTR' : 6.58452413e-03,
					'Msstr' : 1.83366045e-02, 
					'Ma TTR' : 1.70646641e-02, 
					'HDD' : 1.96240139e-02, 
					'MTLD' : 8.88665331e-01,
					'MTLD MA' : 7.70135408e-01, 
					'MTLD MA Bi' : 7.66591821e-01, 
					'VocD' : 8.53453334e-01, 
					'YulesK' : 5.04564277e+01
					}
		
		for key, val in ld_data.items():
			ld_data[key] = (val - ld_mean[key])/ld_std[key]
		
		return ld_data
		
	def get_pos_ratios(self, text):

		doc = self.nlp(text)
		adverb, adjective, pronoun, noun, verb, others = 0, 0, 0, 0, 0, 0
		
		for sent in doc.sentences:
			for word in sent.words:
				if word.upos == 'ADJ':
					adjective += 1
				if word.upos == 'ADV':
					adverb += 1
				if word.upos == 'PRON':
					pronoun += 1
				if word.upos == 'NOUN':
					noun += 1
				if word.upos == 'VERB':
					verb += 1
				else:
					others += 1

		content_words = sum([noun, verb, adjective, adverb])
		function_words = sum([pronoun, others])
		
		pos_data = {'adverb/adjective' : self.safe_divide(adverb,adjective), 
					'adverb/noun' : self.safe_divide(adverb,noun), 
					'adverb/pronoun' : self.safe_divide(adverb,pronoun), 
					'adjective/verb' : self.safe_divide(adjective,verb), 
					'adjective/pronoun' : self.safe_divide(adjective,pronoun), 
					'noun/verb' : self.safe_divide(noun,verb), 
					'noun/pronoun' : self.safe_divide(noun,pronoun), 
					'verb/pronoun' : self.safe_divide(verb,pronoun),
					'content/function' : self.safe_divide(content_words, function_words)
					}

		return pos_data

	# def get_sen_comp(self, sen_list, text=None):
	# 	sen_com_data = {}
	# 	sen_com = SentenceComplexity()
	# 	para_depth, isc_scores, add_scores = [], [], []
	# 	for sen in sen_list:
	# 		sen = re.sub(r'[^\w\s]', ' ', sen)
	# 		sen = ' '.join(sen.split())
	# 		# sen = sen.strip().replace('\n', '').replace('  ', '')
	# 		measures = sen_com.get_sen_measures(sen, sen_measure='all')
	# 		para_depth.append(measures['depth_score'])
	# 		isc_scores.append(measures['isc_score'])
	# 		add_scores.append(measures['add_score'])
		
	# 	sen_com_data['mean para depth'] = np.mean(para_depth)
	# 	sen_com_data['Std para depth'] = np.std(para_depth)
	# 	sen_com_data['Mean ISC score'] = np.mean(isc_scores)
	# 	sen_com_data['Std ISC score'] = np.std(isc_scores)
	# 	sen_com_data['Mean ADD'] = np.mean(add_scores)
	# 	sen_com_data['Std Add'] = np.std(add_scores)

	# 	return sen_com_data

	def get_all_dep_feats(self, sen_list, text=None):
		dep_feat_data = {}
		arguments_tags = ['nsubj', 'obj', 'ccomp', 'conj', 'csubj:pass', 'iobj']
		
		for sen in sen_list:
			sen = re.sub(r'[^\w\s]', ' ', sen)
			sen = ' '.join(sen.split())
			rel_bi, dep_tri = self.get_dependency_features(sen)
			adjuncts = sum([rel_bi[x] for x in list(rel_bi.keys()) if x not in arguments_tags])
			arguments = sum([rel_bi[y] for y in list(rel_bi.keys()) if y in arguments_tags])
			dep_feat_data['arguments/adjuncts'] = self.safe_divide(arguments, adjuncts)
			
			for key, val in (rel_bi+dep_tri).items():
				if key in dep_feat_data:
					dep_feat_data[key] += val
				else:
					dep_feat_data[key] = val

		return dep_feat_data

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

