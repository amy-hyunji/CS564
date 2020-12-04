import pandas as pd
import spacy
import time
from nltk.corpus import words
from nltk.tokenize import word_tokenize

class NLProcV2:
    def __init__(self, path, csv_num):
        self.path = path
        self.csv_num = csv_num
        self.csv = pd.read_csv(f"{self.path}AbstractRetrieval_2019_nlp_{self.csv_num}.csv").dropna()
        self.abstract = self.csv['abstract'].to_numpy()
        self.nlp = spacy.load("en_core_web_sm")
        self.vocab = set(words.words())

    def noun_verb_ratio(self, text): # noun is not just noun , but NNP(고유명사)!
        doc = self.nlp(text)
        nnp = 0
        verb = 0
        for token in doc:
            if token.tag_ == 'NNP':
                nnp += 1
            if token.pos_ == 'VERB':
                verb += 1
        return nnp / (len(doc) or 1), verb / (len(doc) or 1)

    def prof_ratio(self, text): # gets ratio of unknown(=professional) words, which is not inside the nltk vocab.
        tokenized = word_tokenize(text)
        oov = 0
        for tok in tokenized:
            if tok not in self.vocab:
                oov += 1
        return oov / (len(tokenized) or 1)

    def run_and_save(self):
        saved_filename = f"{self.path}AbstractRetrieval_2019_nlp2_{self.csv_num}.csv"
        print(f"{saved_filename} start!")
        start_time = time.time()
        verb, noun, prof = [], [], []
        for text in self.abstract:
            n, v = self.noun_verb_ratio(text)
            noun.append(n)
            verb.append(v)
            prof.append(self.prof_ratio(text))


        self.csv['verb_ratio'] = verb
        self.csv['noun_ratio'] = noun
        self.csv['prof_ratio'] = prof

        self.csv.to_csv(saved_filename)
        end_time = time.time()
        print(f"{saved_filename} SAVED! took {end_time - start_time} seconds.")
        return


csv_data_path = '/Users/user/Desktop/coding/R/project/CS564/processed_data/'
csv_names = [i for i in range(0, 59)]

for num in csv_names:
    ex = NLProcV2(csv_data_path, num)
    ex.run_and_save()
