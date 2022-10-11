import stanza

class HighLevelRatios:
    
    def __init__(self):
        self.nlp = stanza.Pipeline(lang='en', processors={'tokenize': 'spacy'})
    
    def safe_divide(self, numerator, denominator):
        if denominator == 0 or denominator == 0.0:
            index = 0
        else:
            index = numerator/denominator
        return index
    
    def feature(self, text):
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
        
        feat = {'adverb/adjective' : self.safe_divide(adverb,adjective), 
                'adverb/noun' : self.safe_divide(adverb,noun), 
                'adverb/pronoun' : self.safe_divide(adverb,pronoun), 
                'adjective/verb' : self.safe_divide(adjective,verb), 
                'adjective/pronoun' : self.safe_divide(adjective,pronoun), 
                'noun/verb' : self.safe_divide(noun,verb), 
                'noun/pronoun' : self.safe_divide(noun,pronoun), 
                'verb/pronoun' : self.safe_divide(verb,pronoun),
                'content/function' : self.safe_divide(content_words, function_words)
               }
        return feat