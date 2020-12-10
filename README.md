# CS564 Project: Academic paper writing tone depending on the author's location
This is the repository for storing nlp analyzed sample data and code for our CS564 final project paper.

### Code Structure
```
|__ preprocessed_data/
|       |__ AbstractRetrieval_nlp_{0-58}.csv --> Splitted 2019 scopus dataset csv files that contain average_length, polarity, and subjectivity columns
|       |__ AbstractRetrieval_nlp2_{0-58}.csv --> Splitted 2019 scopus dataset csv files that contain noun_ratio, professional_ratio, and verb ratio in addition to previous dataset
|
|
|__ analysis.R --> R code that is used for average_length, polarity, and subjectivity analysis
|__ female.txt--> List of female terms used for reference when analyzing gendered terms
|__ gendered.txt --> List of gendered terms used for reference when analyzing gendered terms
|__ male.txt --> List of male terms used for reference when analyzing gendered terms
|__ Preprocess_Feature_Extract.py --> Python code that is used for creating columns related to gendered terms
|__ Scopus_crawl.py --> Python code used for crawling scopus dataset
|__ tf_idf.py --> Python code used for TF-IDF analysis on relation to categories
|__ tf_idf_result.csv --> CSV file which contains the output of TF-IDF results
|__ sentiment.py --> Python code used for creating columns related to sentiment(grammatical, passive_active, polarity, average_length, subjectivity)
|__ sentiment_by_continent.py --> Codes for analyzing sentiment features with respect to 5 continents
|__ linguistic.py --> Codes for creating coulmns related to professional_ratio, noun_ratio, and verb_ratio
```