import os
import pickle

import spacy
import numpy as np
import pandas as pd
from textblob import TextBlob
from fastpunct import FastPunct
from spacy.symbols import nsubj, nsubjpass, auxpass

from pybliometrics.scopus import AbstractRetrieval

readFilePath1 = "./"
saveFilePath1 = "./"

nlp = spacy.load("en_core_web_sm")

male = set()
female = set()
gendered = set()

pronoun_i = ['i', 'me', 'my', 'mine', 'myself']
pronoun_we = ['we', 'us', 'our', 'ours', 'ourselves']

with open('male.txt', 'r') as f:
    while True:
        line = f.readline()
        if not line:
            break
        line = line.rstrip('\n').lower()
        male.add(line)
        
with open('female.txt', 'r') as f:
    while True:
        line = f.readline()
        if not line:
            break
        line = line.rstrip('\n').lower()
        female.add(line)
        
with open('gendered.txt', 'r') as f:
    while True:
        line = f.readline()
        if not line:
            break
        line = line.rstrip('\n').lower()
        gendered.add(line)

def average_length(text):
    sentences = text.split(". ")
    lengths = np.array([t.count(" ") for t in sentences])
    return (lengths.sum() / len(lengths)) + 1

def sentiment(text):
    polarity, subjectivity = TextBlob(text).sentiment
    return polarity, subjectivity

def remove_copyright(text):
    if '©' in text[:15]:
        return '.'.join(text.split('.')[1:])
    return text

def get_data(text):
    c1, c2, c3, c4, c5 = 0, 0, 0, 0, 0
    tmp = text.lower()
    for c in '()*+,-./:;<=>?@[]^_`{|}':
        tmp.replace(c, ' ')
    for w in tmp.split(' '):
        if len(w) == 0:
            continue
        if w in male:
            c1 += 1
        if w in female:
            c2 += 1
        if w in gendered:
            c3 += 1
        if w in pronoun_i:
            c4 += 1
        if c5 in pronoun_we:
            c5 += 1
    return c1, c2, c3, c4, c5

data = {
    'title': [],
    'abstract': [],
    'country': [],
    'category': [],
    'year' : [],

    'author_num': [],

    'average_length': [],
    'polarity': [],
    'subjectivity': [],

    'male': [],
    'female': [],
    'i': [],
    'we': [],
    'gender_score': [],
    'gendered_term': []
}

for filename in os.listdir(readFilePath1):
	with open(readFilePath1 + filename, 'rb') as f:
        papers = pickle.load(f)

    for eid in papers:
    	paper = papers[eid]
    	if paper == None:
            continue
        if paper.title == None:
            continue
        if paper.description == None:
            continue
        if paper.affiliation == None:
            continue
        if paper.subject_areas == None:
            continue
        if paper.authors == None:
            continue
        if paper.coverDate == None:
            continue

        aff = paper.affiliation[0].country
        if not aff in ['United States', 'China']:
            continue
        ab = remove_copyright(paper.description)
        year = int(paper.coverDate.split('-')[0])

        if year < 2000 or year > 2019:
        	continue

        tmp['title'].append(paper.title)
        tmp['abstract'].append(ab)
        tmp['category'].append([area.code for area in paper.subject_areas if area.code[:2]=='17'])
        tmp['country'].append(aff)
        tmp['year'].append(year)

        tmp['author_num'].append(len(paper.authors))

        tmp['average_length'].append(average_length(ab))
        polarity, subjectivity = sentiment(ab)
        tmp['polarity'].append(polarity)
        tmp['subjectivity'].append(subjectivity)

        male, female, gendered_term, i, we = get_data(ab)
        tmp['male'].append(male)
        tmp['female'].append(female)
        tmp['i'].append(i)
        tmp['we'].append(we)

        gender_score = 0.0 if (male==0 and female==0) else (male-female)/(male+female)
        tmp['gender_score'].append(gender_score)
        tmp['gendered_term'].append(gendered_term)

df = pd.DataFrame(data)
df.to_csv(saveFilePath1+"R.csv")