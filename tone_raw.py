import pandas as pd
from textblob import TextBlob
import spacy
from spacy.symbols import nsubj, nsubjpass, auxpass
from fastpunct import FastPunct
import time
import numpy as np
from fastpunct import FastPunct

csv_data_path = '/Users/user/Desktop/coding/R/data/'
csv_names = [i for i in range(50, 59)]


class NLProc:
    def __init__(self, path, csv_num, mode='analyze'):
        self.path = path
        self.csv_num = csv_num
        if mode == 'analyze':
            self.csv = pd.read_csv(f"{self.path}AbstractRetrieval_2019_nlp_{self.csv_num}.csv").dropna()
        else:
            self.csv = pd.read_csv(f"{self.path}AbstractRetrieval_2019_{self.csv_num}.csv").dropna()
        self.abstract = self.csv['abstract'].to_numpy()
        self.nlp = spacy.load("en_core_web_sm")
        self.fastpunct = FastPunct('en')
        import pdb; pdb.set_trace()

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
        import pdb; pdb.set_trace()

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
        import pdb; pdb.set_trace()
        self.csv['average_length'] = av_length
        self.csv['polarity'] = polar
        self.csv['subjectivity'] = subj
        self.csv['passive_active'] = passive

        self.csv.to_csv(saved_filename)
        end_time = time.time()
        print(f"{saved_filename} SAVED! took {end_time - start_time} seconds.")
        return


#for num in csv_names:
#    ex = NLProc(csv_data_path, num)
#    ex.run_and_save()

class Analyzer():
    def __init__(self):
        self.csvs = [pd.read_csv(f"/Users/user/Desktop/coding/R/data/AbstractRetrieval_2019_nlp_{csv_num}.csv").dropna() for csv_num in range(59)]
        self.csv = pd.concat(self.csvs)
        #  df.loc[df['affiliation_country'] == 'China']['polarity'].mean()
        self.countries = set(self.csv['affiliation_country'])
        self.country2continentmapping()
        self.analyze_passive_active_by_country()
        self.analyze_sentiment_by_country()
        self.analyze_avlength_by_country()
        self.analyze_subjt_by_country()
        import pdb; pdb.set_trace()
    def analyze_passive_active_by_country(self):
        print("\nAnalyzing passive_active mean for each country ... ")
        self.dict = {}
        self.list = []
        """
        for country in self.countries:
            val = self.csv.loc[self.csv['affiliation_country'] == country]['passive_active'].mean()
            #print(f"{country}: {val}")
            self.dict[country] = val
            self.list.append([country, val])"""

        for cont in self.cmap.keys():
            val = self.csv.loc[self.csv['affiliation_country'].isin(self.cmap[cont])]['passive_active'].mean()
            #print(f"{country}: {val}")
            self.dict[cont] = val
            self.list.append([cont, val])

        self.list.sort(key=lambda k: k[1])
        for cont, val in self.list:
            print(f"{cont}: {val}")


    def analyze_sentiment_by_country(self):
        print("\nAnalyzing sentiment mean for each country ... ")
        self.dict = {}
        self.list = []
        for cont in self.cmap.keys():
            val = self.csv.loc[self.csv['affiliation_country'].isin(self.cmap[cont])]['polarity'].mean()
            #print(f"{country}: {val}")
            self.dict[cont] = val
            self.list.append([cont, val])

        self.list.sort(key=lambda k: k[1])
        for cont, val in self.list:
            print(f"{cont}: {val}")

    def analyze_avlength_by_country(self):
        print("\nAnalyzing average length mean for each country ... ")
        self.dict = {}
        self.list = []
        for cont in self.cmap.keys():
            val = self.csv.loc[self.csv['affiliation_country'].isin(self.cmap[cont])]['average_length'].mean()
            #print(f"{country}: {val}")
            self.dict[cont] = val
            self.list.append([cont, val])

        self.list.sort(key=lambda k: k[1])
        for cont, val in self.list:
            print(f"{cont}: {val}")

    def analyze_subject_by_country(self):
        print("\nAnalyzing subjectivity mean for each country ... ")
        self.dict = {}
        self.list = []
        for cont in self.cmap.keys():
            val = self.csv.loc[self.csv['affiliation_country'].isin(self.cmap[cont])]['subjectivity'].mean()
            #print(f"{country}: {val}")
            self.dict[cont] = val
            self.list.append([cont, val])

        self.list.sort(key=lambda k: k[1])
        for cont, val in self.list:
            print(f"{cont}: {val}")

    def country2continentmapping(self):
        # by: https://simple.wikipedia.org/wiki/List_of_countries_by_continents
        self.cmap = {'asia': [], 'africa': [], 'europe': [], 'america': [], 'oceania': [], 'etc': []}
        self.revmap = {}
       # africa = ['algeria', 'algiers', 'angola', 'luanda', 'benin', 'porto novo, cotonou', 'botswana', 'gaborone', 'burkina faso', 'ouagadougou', 'burundi', 'gitega', 'cameroon (also spelled cameroun)', 'yaound\xc3\xa9', 'cape verde', 'praia', 'central african republic', 'bangui', 'chad (tchad)', "n'djamena", 'comoros', 'moroni', 'republic of the congo', 'brazzaville', 'democratic republic of the congo (zaire)', 'kinshasa', "c\xc3\xb4te d'ivoire (ivory coast)", 'yamoussoukro', 'djibouti', 'djibouti', 'egypt (misr)', 'cairo', 'equatorial guinea', 'malabo', 'eritrea', 'asmara', 'ethiopia (abyssinia)', 'addis ababa', 'gabon', 'libreville', 'the gambia', 'banjul', 'ghana', 'accra', 'guinea', 'conakry', 'guinea-bissau', 'bissau', 'kenya', 'nairobi', 'lesotho', 'maseru', 'liberia', 'monrovia', 'libya', 'tripoli', 'madagascar', 'antananarivo', 'malawi', 'lilongwe', 'mali', 'bamako', 'mauritania', 'nouakchott', 'mauritius', 'port louis', 'morocco (al maghrib)', 'rabat', 'mozambique', 'maputo', 'namibia', 'windhoek', 'niger', 'niamey', 'nigeria', 'abuja', 'rwanda', 'kigali', 's\xc3\xa3o tom\xc3\xa9 and pr\xc3\xadncipe', 's\xc3\xa3o tom\xc3\xa9', 'senegal', 'dakar', 'seychelles', 'victoria, seychelles', 'sierra leone', 'freetown', 'somalia', 'mogadishu', 'south africa', 'pretoria', 'south sudan', 'juba', 'sudan', 'khartoum', 'swaziland (eswatini)', 'mbabane', 'tanzania', 'dodoma', 'togo', 'lome', 'tunisia', 'tunis', 'uganda', 'kampala', 'western sahara', 'el aai\xc3\xban (disputed)', 'zambia', 'lusaka', 'zimbabwe', 'harare']
        africa = "algeria - algiers angola - luanda benin - porto novo, cotonou botswana - gaborone burkina faso - ouagadougou burundi - gitega cameroon (also spelled cameroun) - yaound\xc3\xa9 cape verde - praia central african republic - bangui chad (tchad) - n'djamena comoros - moroni republic of the congo - brazzaville democratic republic of the congo (zaire) - kinshasa c\xc3\xb4te d'ivoire (ivory coast) - yamoussoukro djibouti - djibouti egypt (misr) - cairo equatorial guinea - malabo eritrea - asmara ethiopia (abyssinia) - addis ababa gabon - libreville the gambia - banjul ghana - accra guinea - conakry guinea-bissau - bissau kenya - nairobi lesotho - maseru liberia - monrovia libya - tripoli madagascar - antananarivo malawi - lilongwe mali - bamako mauritania - nouakchott mauritius - port louis morocco (al maghrib) - rabat mozambique - maputo namibia - windhoek niger - niamey nigeria - abuja rwanda - kigali s\xc3\xa3o tom\xc3\xa9 and pr\xc3\xadncipe - s\xc3\xa3o tom\xc3\xa9 senegal - dakar seychelles - victoria, seychelles sierra leone - freetown somalia - mogadishu south africa - pretoria south sudan - juba sudan - khartoum swaziland (eswatini) - mbabane tanzania - dodoma togo - lome tunisia - tunis uganda - kampala western sahara - el aai\xc3\xban (disputed) zambia - lusaka zimbabwe - harare"
        europe = 'albania (shqip\xc3\xabria) - tirana andorra - andorra la vella austria (\xc3\x96sterreich) - vienna belarus (\xd0\x91\xd0\xb5\xd0\xbb\xd0\xb0\xd1\x80\xd1\x83\xd1\x81\xd1\x8c) - minsk belgium (dutch: belgi\xc3\xab, french: belgique, german: belgien) - brussels bosnia and herzegovina (bosna i hercegovina) - sarajevo bulgaria (\xd0\x91\xd1\x8a\xd0\xbb\xd0\xb3\xd0\xb0\xd1\x80\xd0\xb8\xd1\x8f) - sofia croatia (hrvatska) - zagreb cyprus (\xce\x9a\xcf\x8d\xcf\x80\xcf\x81\xce\xbf\xcf\x82) - nicosia czech republic (\xc4\x8cesko) - prague denmark (danmark) - copenhagen estonia (eesti) - tallinn finland (suomi) - helsinki france - paris georgia - tbilisi germany (deutschland) - berlin greece (\xce\x95\xce\xbb\xce\xbb\xce\xac\xce\xb4\xce\xb1) - athens hungary (magyarorsz\xc3\xa1g) - budapest iceland** (island) - reykjavik republic of ireland (\xc3\x89ire) - dublin italy (italia) - rome kazakhstan - nursultan kosovo** - pristina latvia (latvija) - riga liechtenstein - vaduz lithuania (lietuva) - vilnius luxembourg - luxembourg city north macedonia (\xd0\x9c\xd0\xb0\xd0\xba\xd0\xb5\xd0\xb4\xd0\xbe\xd0\xbd\xd0\xb8\xd1\x98\xd0\xb0) - skopje malta - valletta moldova - chisinau monaco - monte carlo quarter montenegro (crna gora, \xd0\xa6\xd1\x80\xd0\xbd\xd0\xb0 \xd0\x93\xd0\xbe\xd1\x80\xd0\xb0) - podgorica netherlands (nederland) - amsterdam (capital), the hague (government) norway (norge) - oslo poland (polska) - warsaw portugal - lisbon romania - bucharest russia** moscow (europe up to the ural mountains; asia: the rest to vladivostok) san marino - san marino serbia (\xd0\xa1\xd1\x80\xd0\xb1\xd0\xb8\xd1\x98\xd0\xb0) - belgrade slovakia (slovensko) - bratislava slovenia (slovenija) - ljubljana spain (espa\xc3\xb1a) - madrid sweden (sverige) - stockholm switzerland (german: schweiz, french: suisse, italian: svizzera, romansh: svizra) - bern turkey - ankara ukraine (\xd0\xa3\xd0\xba\xd1\x80\xd0\xb0\xd1\x97\xd0\xbd\xd0\xb0) - kyiv or kiev united kingdom - london vatican city** (italian: citt\xc3\xa0 del vaticano, latin: civitas vaticana) - vatican city'
        asia = "afghanistan - kabul armenia - yerevan azerbaijan - baku bahrain - manama bangladesh [1](\xe0\xa6\xac\xe0\xa6\xbe\xe0\xa6\x82\xe0\xa6\xb2\xe0\xa6\xbe\xe0\xa6\xa6\xe0\xa7\x87\xe0\xa6\xb6) - dhaka (\xe0\xa6\xa2\xe0\xa6\xbe\xe0\xa6\x95\xe0\xa6\xbe) bhutan - thimphu brunei - bandar seri begawan cambodia (kampuchea) - phnom penh china - beijing east timor (timor leste) - dili georgia - tbilisi india - new delhi indonesia - jakarta iran - tehran iraq - baghdad israel - jerusalem japan - tokyo jordan (al urdun) - amman kazakhstan - nursultan kuwait - kuwait city kyrgyzstan - bishkek laos - vientiane lebanon (lubnan) - beirut malaysia - kuala lumpur maldives - mal\xc3\xa9 mongolia - ulaanbaatar myanmar (burma) - naypyidaw nepal - kathmandu north korea - pyongyang oman - muscat pakistan - islamabad philippines - manila qatar - doha russia - moscow (russia is a part of asia geographically, but politically it is a part of europe) saudi arabia - riyadh singapore - singapore south korea - seoul sri lanka - sri jayawardenapura kotte (administrative), colombo (commercial) syria - damascus tajikistan - dushanbe thailand (muang thai) - bangkok turkey - ankara turkmenistan - a\xc5\x9fgabat taiwan - taipei united arab emirates - abu dhabi uzbekistan - tashkent vietnam - hanoi yemen - [[sana'a"
        america = "antigua and barbuda - st. john's anguilla - the valley (territory of u.k.) aruba - oranjestad (constituent country of the kingdom of the netherlands) the bahamas - nassau barbados - bridgetown belize - belmopan ( central america) bermuda - hamilton (territory of u.k.) bonaire - part of the netherlands british virgin islands - road town (territory of u.k.) canada - ottawa cayman islands - george town (territory of u.k.) clipperton island - (territory of france) costa rica - san jos\xc3\xa9 ( central america) cuba - havana cura\xc3\xa7ao - willemstad (constituent country of the kingdom of the netherlands) dominica - roseau dominican republic (republica dominicana) - santo domingo el salvador - san salvador ( central america) greenland - nuuk (territory of denmark) grenada - st george's guadeloupe - (territory of france) guatemala - guatemala haiti - port-au-prince honduras - tegucigalpa ( central america) jamaica - kingston martinique - fort-de-france bay (territory of france) mexico - mexico city montserrat - plymouth, brades, little bay (territory of u.k.) navassa island - washington, d.c. (territory of u.s.) nicaragua - managua ( central america) panama (panam\xc3\xa1) - panama city ( central america) puerto rico - san juan (territory of u.s.) saba - the bottom (territory of netherlands) saint barthelemy - gustavia (territory of france) saint kitts and nevis - basseterre saint lucia - castries saint martin - marigot (territory of france) saint pierre and miquelon - saint-pierre (territory of france) saint vincent and the grenadines - kingstown sint eustatius - oranjestad (territory of netherlands) sint maarten - philipsburg (constituent country of the kingdom of the netherlands) trinidad and tobago - port of spain turks and caicos - cockburn town (british overseas territory) united states of america - washington, district of columbia us virgin islands - charlotte amalie (territory of u.s.)Argentina - Buenos Aires\nBolivia - Sucr\xc3\xa9\nBrazil (Brasil) - Bras\xc3\xadlia\nChile - Santiago\nColombia - Bogot\xc3\xa1\nEcuador - Quito\nFalkland Islands - Stanley (territory of U.K.)\nFrench Guiana - Cayenne (territory of France)\nGuyana - Georgetown\nParaguay - Asunci\xc3\xb3n\nPeru - Lima\nSouth Georgia and the South Sandwich Islands - (territory of U.K.)\nSuriname - Paramaribo\nUruguay - Montevideo\nVenezuela - Caracas\n"
        # na = "antigua and barbuda - st. john's anguilla - the valley (territory of u.k.) aruba - oranjestad (constituent country of the kingdom of the netherlands) the bahamas - nassau barbados - bridgetown belize - belmopan ( central america) bermuda - hamilton (territory of u.k.) bonaire - part of the netherlands british virgin islands - road town (territory of u.k.) canada - ottawa cayman islands - george town (territory of u.k.) clipperton island - (territory of france) costa rica - san jos\xc3\xa9 ( central america) cuba - havana cura\xc3\xa7ao - willemstad (constituent country of the kingdom of the netherlands) dominica - roseau dominican republic (republica dominicana) - santo domingo el salvador - san salvador ( central america) greenland - nuuk (territory of denmark) grenada - st george's guadeloupe - (territory of france) guatemala - guatemala haiti - port-au-prince honduras - tegucigalpa ( central america) jamaica - kingston martinique - fort-de-france bay (territory of france) mexico - mexico city montserrat - plymouth, brades, little bay (territory of u.k.) navassa island - washington, d.c. (territory of u.s.) nicaragua - managua ( central america) panama (panam\xc3\xa1) - panama city ( central america) puerto rico - san juan (territory of u.s.) saba - the bottom (territory of netherlands) saint barthelemy - gustavia (territory of france) saint kitts and nevis - basseterre saint lucia - castries saint martin - marigot (territory of france) saint pierre and miquelon - saint-pierre (territory of france) saint vincent and the grenadines - kingstown sint eustatius - oranjestad (territory of netherlands) sint maarten - philipsburg (constituent country of the kingdom of the netherlands) trinidad and tobago - port of spain turks and caicos - cockburn town (british overseas territory) united states of america - washington, district of columbia us virgin islands - charlotte amalie (territory of u.s.)"
        # sa = 'Argentina - Buenos Aires\nBolivia - Sucr\xc3\xa9\nBrazil (Brasil) - Bras\xc3\xadlia\nChile - Santiago\nColombia - Bogot\xc3\xa1\nEcuador - Quito\nFalkland Islands - Stanley (territory of U.K.)\nFrench Guiana - Cayenne (territory of France)\nGuyana - Georgetown\nParaguay - Asunci\xc3\xb3n\nPeru - Lima\nSouth Georgia and the South Sandwich Islands - (territory of U.K.)\nSuriname - Paramaribo\nUruguay - Montevideo\nVenezuela - Caracas\n'
        oceania = "australia - canberra federated states of micronesia - palikir fiji - suva kiribati - south tarawa marshall islands - majuro nauru - no capital; biggest city is yaren new zealand - wellington palau - ngerulmud papua new guinea - port moresby samoa - apia solomon islands - honiara tonga - nuku'alofa tuvalu - funafuti vanuatu - port vila australia flores lombok melanesia new caledonia new guinea sulawesi sumbawa timor"
        for country in self.countries:
            c = country.lower()
            if c in africa:
                self.cmap['africa'].append(country)
                self.revmap[country] = 'africa'
            elif c in europe:
                self.cmap['europe'].append(country)
                self.revmap[country] = 'europe'
            elif c in asia:
                self.cmap['asia'].append(country)
                self.revmap[country] = 'asia'
            elif c in america:
                self.cmap['america'].append(country)
                self.revmap[country] = 'america'
            elif c in oceania:
                self.cmap['oceania'].append(country)
                self.revmap[country] = 'oceania'
            else:
                self.cmap['etc'].append(country)
                self.revmap[country] = 'etc'

aa = Analyzer()