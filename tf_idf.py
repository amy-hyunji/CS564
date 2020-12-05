import pandas as pd
import numpy as np
import time
import nltk
import ast
from nltk.corpus import words
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer


class NLProcV2:
    def __init__(self, path, csv_num):
        self.path = path
        self.csv_num = csv_num
        self.csv = pd.read_csv(
            f"{self.path}AbstractRetrieval_2019_nlp_{self.csv_num}.csv"
        ).dropna()
        self.category = self.csv["category"].to_numpy()
        self.category_name = {
            "1700": "General Computer Science",
            "1701": "Computer Science",
            "1702": "Artifical Intelligence",
            "1703": "Computational Theory and Mathematics",
            "1704": "Computer Graphics and Computer-Aided Design",
            "1705": "Computer Networks and Communications",
            "1706": "Computer Science Applications",
            "1707": "Computer Vision and Pattern Recognition",
            "1708": "Hardware and Architecture",
            "1709": "Human-Computer Interaction",
            "1710": "Information Systems",
            "1711": "Signal Processing",
            "1712": "Software",
        }
        self.abstract = self.csv["abstract"].to_numpy()
        self.category_dict = {}
        self.ret_dict = {}

    def tf_idf(self, X, vocab_num=20):
        tfidf_vectorizer = TfidfVectorizer(stop_words="english")
        tfidf_vectorizer.fit(X)
        matrix = np.array(tfidf_vectorizer.transform(X).toarray())
        matrix = np.sum(matrix, axis=0)
        vocab_dict = tfidf_vectorizer.vocabulary_

        idx_list = []
        for i, _vocab in enumerate(vocab_dict.keys()):
            idx_list.append((matrix[vocab_dict[_vocab]], _vocab))
        _sorted = sorted(idx_list, key=lambda x: -x[0])[:vocab_num]
        del idx_list

        ret_list = []
        for _, elem in _sorted:
            ret_list.append(elem)
        return ret_list

    def extract_category(self):
        saved_filename = f"{self.path}AbstractRetrieval_2019_tf_idf_{self.csv_num}.csv"
        print(f"{saved_filename} start!")
        assert len(self.category) == len(self.abstract)
        for category, text in zip(self.category, self.abstract):
            category = ast.literal_eval(category)
            for _category in category:
                _name = self.category_name[_category]
                if _name in self.category_dict.keys():
                    self.category_dict[_name].append(text)
                else:
                    self.category_dict[_name] = [text]

    def run_and_save(self):
        categories = self.category_dict.keys()
        print(f"categories: {categories}")
        print(f"# of category: {len(categories)}")
        for _category in categories:
            top_10 = self.tf_idf(self.category_dict[_category])
            self.ret_dict[_category] = top_10

        df = pd.DataFrame(self.ret_dict)
        df.to_csv("./tf_idf_result.csv")

        return


csv_data_path = "/Users/amy_hyunji/Documents/GitHub/CS564/processed_data/"
csv_names = [i for i in range(0, 59)]
nltk.download('stopwords')

for num in csv_names:
    ex = NLProcV2(csv_data_path, num)
    ex.extract_category()
ex.run_and_save()
print("DONE!!!!!!!")
