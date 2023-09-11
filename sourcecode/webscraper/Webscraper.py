import requests
import os
from bs4 import BeautifulSoup
import pandas as pd
from transliterate import translit
from webscraper.get_short_text import getShortText
from webscraper.get_middle_text import getMiddleText
from webscraper.get_long_text import getLongText
import codecs


offical_languages_dataframe = pd.read_excel(r'./sourcecode/webscraper/Abbreviations.xlsx')
offical_languages = dict(zip(offical_languages_dataframe['Language'], offical_languages_dataframe['Version 1']))


# --------------------------------- SCRAPING --------------------------------- #
def get_website_text(websiteUrl, textLength):

    if textLength == "short":
        return getShortText(websiteUrl, textLength)
    elif textLength == "middle":
        return getMiddleText(websiteUrl, textLength)
    elif textLength == "long":
        return getLongText(websiteUrl, textLength)


# ------------------------------- START METHOD ------------------------------- #
def start_scraping(filename="webscraper", textLength="null"):
    languageList = []
    sentenceList = []
    languageAbbreviationList = []
    combinationDictionary = {}

    f = codecs.open(os.path.abspath(os.curdir) + '/sourcecode/files/' + textLength + '/' + filename + ".txt", "w", encoding='utf-16')

    for language,language_abbreviation in offical_languages.items():
        url = "https://european-union.europa.eu/institutions-law-budget/institutions-and-bodies/types-institutions-and-bodies_" + language_abbreviation
        
        text = get_website_text(url, textLength)

        # Transliteration for Bulgarian and Greek
        if(language_abbreviation == 'bg' or language_abbreviation == 'el'):
            text = translit(text, language_abbreviation, reversed=True)

        languageList.append(language)
        languageAbbreviationList.append(language_abbreviation)
        sentenceList.append(text)
        combinationDictionary[language] = text

        f.write(url)
        f.write("\n")
        f.write(language)
        f.write("\n")
        f.write(text)
        f.write("\n")
        f.write("---------")
        f.write("\n")
        
    f.close()
    return languageList, languageAbbreviationList, sentenceList, combinationDictionary



