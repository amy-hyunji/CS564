import pandas as pd
import pickle

from pybliometrics.scopus import AbstractRetrieval
from pybliometrics.scopus import ScopusSearch

saveFilePath1 = "./"
saveFilePath2 = "./"

s = ScopusSearch("SUBJAREA (comp)  AND  LANGUAGE (english)")
df = pd.DataFrame(s.results)
df.to_csv(saveFilePath1+"ScopusSearch_CS_eng_papers.csv")

n = 1000

for i in range((len(df)+n-1)//n):
	ret = {}
	for eid in list(df['eid'])[n*i:n*i+n]:
		try:
			ab = AbstractRetrieval(eid)
			ret[eid] = ab
		except:
			ret[eid] = None
	with open(saveFilePath2+"AbstractRetrieval_CS_eng_papers_{}.pkl".format(str(i).zfill(3)), "wb") as f:
		pickle.dump(ret, f)