import pandas as pd
from textblob import TextBlob
import spacy
from spacy.symbols import nsubj, nsubjpass, auxpass
from fastpunct import FastPunct
import time
import numpy as np
from fastpunct import FastPunct



class NLProc:
    def __init__(self, path, csv_num):
        self.path = path
        self.csv_num = csv_num
        self.csv = pd.read_csv(f"{self.path}AbstractRetrieval_2019_{self.csv_num}.csv").dropna()
        self.abstract = self.csv['abstract'].to_numpy()
        self.nlp = spacy.load("en_core_web_sm")
        self.fastpunct = FastPunct('en')

    def average_length(self, text):
        sentences = text.split(". ")
        lengths = np.array([t.count(" ") for t in sentences])
        return (lengths.sum() / len(lengths)) + 1 # we counted spaces, so we should add 1 to convert it to counting words.

    def sentiment(self, text): # Extracts negative-positive, objective-subjective
        polarity, subjectivity = TextBlob(text).sentiment
        return polarity, subjectivity

    def passive_active(self, text): # passive is 0, active is 1
        doc = self.nlp(text)
        # https://github.com/JasonThomasData/NLP_sentence_analysis/blob/master/stanford_NLTK.py#L52
        # above explains auxpass is only way to detect passive voice
        # https://gist.github.com/armsp/30c2c1e19a0f1660944303cf079f831a: setting rules for passive voice
        nsubjpass, auxpass = 0, 0
        for entity in doc:
            if (entity.dep == auxpass) or (entity.dep == nsubjpass):
                return 0
        return 1

    def abstract_ungrammatical(self, abstract):
        sentences = abstract.split(". ")
        grammar = []
        for s in sentences:
            grammar.append(self.ungrammatical(s))

    def ungrammatical(self, text):
        if len(text) > 400:
            text = text[:400]
        print(f"Input: {text}")
        corrected = self.fastpunct.punct([text])[0]
        orig_tokens = text.split(' ')[:-1]
        corrected_tokens = corrected.split(' ')[:-1]
        allow_threshold = (len(orig_tokens) // 10 or 1)
        wrong = 0
        if len(corrected_tokens) != len(orig_tokens):
            wrong += abs(len(corrected_tokens) - len(orig_tokens))
        for idx in range(min(len(corrected_tokens), len(orig_tokens))):
            if corrected_tokens[idx] != orig_tokens[idx]:
                wrong += 1
        if wrong > allow_threshold:
            print("Ungrammatical!")
            return True  # ungrammatical
        print("Grammatical!")
        return False  # grammatical

    def run_and_save(self):
        saved_filename = f"{self.path}AbstractRetrieval_2019_nlp_{self.csv_num}.csv"
        print(f"{saved_filename} start!")
        start_time = time.time()
        av_length, polar, subj, passive = [], [], [], []
        ung = []
        for text in self.abstract:
            ung.append(self.abstract_ungrammatical(text))
            av_length.append(self.average_length(text))
            polarity, subjectivity = self.sentiment(text)
            polar.append(polarity)
            subj.append(subjectivity)
            passive.append(self.passive_active(text))

        self.csv['average_length'] = av_length
        self.csv['polarity'] = polar
        self.csv['subjectivity'] = subj
        self.csv['passive_active'] = passive

        self.csv.to_csv(saved_filename)
        end_time = time.time()
        print(f"{saved_filename} SAVED! took {end_time - start_time} seconds.")
        return


csv_data_path = '/Users/user/Desktop/coding/R/data/'
csv_names = [i for i in range(0, 59)]

for num in csv_names:
    ex = NLProc(csv_data_path, num)
    ex.run_and_save()
