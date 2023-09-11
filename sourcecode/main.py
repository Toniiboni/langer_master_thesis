import webscraper.Webscraper as webscraper
import needleman_wunsch.NeedlemanWunsch as algorithm
import helpers.textprocessing as nlp

import matplotlib.pyplot as plt
import numpy as np
from scipy.cluster import hierarchy
from scipy.spatial.distance import pdist
import os
from datetime import datetime
import pandas as pd
import copy

dendrograms = []
labels = []

def measureTime(message):
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    print(message, current_time)

# ----------------------------------- MAIN ----------------------------------- #
def main():
    print('STARTING ALGORITHM...')
    
    dendrogramData = []
    text_length = "short"
    languages, languageAbb, sentences, combinations = webscraper.start_scraping(textLength=text_length)

    

    # Original
    measureTime("Start Original Dendrogram at ")
    dendrogram_original = algorithm.start_needleman_wunsch(languages, sentences, combinations, 'Combination 1', filename="original", textLength=text_length)
    simpleCreatePlot(dendrogram_original, languages, "original", text_length, 'Combination 1')
    measureTime("End Original Dendrogram at ")
    
    nlp._ABBREVIATION = languageAbb

# ------------------------------- NLP FUNCTIONS ------------------------------ #

    functions_Normalization = [nlp.wordNormalization]
    functions_Punctuation = [nlp.removePunctuation]
    functions_Diacritics = [nlp.removeSpecialCharacter]
    functions_Whitespaces = [nlp.removeWhitespace]



# ------------------------------ SINGLE NLP STEP ----------------------------- #
   
    # Normalization x Original
    normalizedSentences, normalizedCombinations = executeNLPSteps(sentences, combinations, functions_Normalization, languages, "wordNormalization", text_length, 'Combination 2')

    # Remove Punctuation x Original   
    punctuationSentences, puntuationCombinations = executeNLPSteps(sentences, combinations, functions_Punctuation, languages, "removedPunctuation", text_length, 'Combination 3')

    # Remove Diacritics x Original   
    diacriticsSentences, diacriticsCombinations = executeNLPSteps(sentences, combinations, functions_Diacritics, languages, "removedDiacritics", text_length, 'Combination 4')

    # Remove Whitspaces x Original
    whitespacesSentences, whitespacesCombinations = executeNLPSteps(sentences, combinations, functions_Whitespaces, languages, "removedWhitespaces", text_length, 'Combination 5')

# ------------------------------- TWO NLP STEPS ------------------------------ #

    # Normalization x RemovePunctuation
    normalization_punctuation_sentences, normalization_punctuation_combinations = executeNLPSteps(normalizedSentences, normalizedCombinations, functions_Punctuation, languages, "normalization_punctuation", text_length, 'Combination 6')

    # Normalization x RemoveDiacritics
    normalization_diacritics_sentences, normalization_diacritics_combinations = executeNLPSteps(normalizedSentences, normalizedCombinations, functions_Diacritics, languages, "normalization_diacritics", text_length, 'Combination 7')

    # Normalization x RemoveWhitespaces
    executeNLPSteps(normalizedSentences, normalizedCombinations, functions_Whitespaces, languages, "normalization_whitespaces", text_length, 'Combination 8')

#--------------------------------------------------

    # Remove Punctuation x Remove Diacritics
    punctuation_diacritics_sentences, punctuation_diacritics_combinations = executeNLPSteps(punctuationSentences, puntuationCombinations, functions_Diacritics, languages, "punctuation_diacritics", text_length, 'Combination 9')

    # Remove Punctuation x Remove Whitespaces
    executeNLPSteps(punctuationSentences, puntuationCombinations, functions_Whitespaces, languages, "punctuation_whitespaces", text_length, 'Combination 10')

#--------------------------------------------------

    # Remove Diacritics x Remove Whitespaces
    executeNLPSteps(diacriticsSentences, diacriticsCombinations, functions_Whitespaces, languages, "diacritics_whitespaces", text_length, 'Combination 11')


# ------------------------------ THREE NPL STEPS ----------------------------- #

    # Normalization x Remove Punctuation x Remove Diacritics
    executeNLPSteps(normalization_punctuation_sentences, normalization_punctuation_combinations, functions_Diacritics, languages, "normalization_punctuation_diacritics", text_length, 'Combination 12')

    # Normalization x Remove Diacritics x Remove Whitespaces
    normalization_diacritics_whitespaces_sentences, normalization_diacritics_whitespaces_combinations = executeNLPSteps(normalization_diacritics_sentences, normalization_diacritics_combinations, functions_Whitespaces, languages, "normalization_diacritics_whitespaces", text_length, 'Combination 13')

    # Normalization x Remove Punctuation x Remove Whitespaces
    executeNLPSteps(normalization_punctuation_sentences, normalization_punctuation_combinations, functions_Whitespaces, languages, "normalization_punctuation_whitespaces", text_length, 'Combination 14')

    # Remove Punctuation x Remove Diacritics x Remove Whitespaces
    executeNLPSteps(punctuation_diacritics_sentences, punctuation_diacritics_combinations, functions_Whitespaces, languages, "punctuation_diacritics_whitespaces", text_length, 'Combination 15')

# ------------------------------ FOUR NLP STEPS ------------------------------ #

    # Normalization x Remove Diacritcs x Remove Whitespaces x Remove Punctuation
    executeNLPSteps(normalization_diacritics_whitespaces_sentences, normalization_diacritics_whitespaces_combinations, functions_Punctuation, languages, "allmethods", text_length, 'Combination 16')

    #executeDendrogramComparison(text_length)

# ------------------------------- PLOT CREATION ------------------------------ #

def simpleCreatePlot(data, languages, filename, textLength, combination):
    plt.figure(figsize=(14,10))
    plt.title('Dendrogram of language similarity\n('+ textLength.capitalize() + ' text / ' + combination +')')
    plt.xlabel('European languages') 
    plt.ylabel('Similarity')
    dn = hierarchy.dendrogram(data, labels=languages)
    plt.savefig(os.path.abspath(os.curdir) + '/sourcecode/files/' + textLength + '/' + filename +".jpg")


# --------------------------- REPLACE COMBINATIONS --------------------------- #
#Replace the corresponding text from the language with the modified text retrieved from the NLP step.
def replaceCombinations(languages, sentences, combinations):   
    modCombinations = copy.copy(combinations)

    i = 0
    for language in languages:
        modCombinations[language] = sentences[i]
        i = i + 1
    return modCombinations

def executeNLPSteps(sentences, combinations, functions, languages, filename, textLength, combination):
    
    modifiedSentences = functions[0](sentences)
    modifiedCombinations = replaceCombinations(languages, modifiedSentences, combinations)

    measureTime("Start Dendrogram " + filename + " at ")
    dendrogram = algorithm.start_needleman_wunsch(languages, modifiedSentences, modifiedCombinations, combination, filename=filename, textLength=textLength)
    dendrograms.append(dendrogram);
    labels.append(textLength + ' ' + filename)
    simpleCreatePlot(dendrogram, languages, filename,  textLength, combination)
    measureTime("End Dendrogram " + filename + " at ")

    return modifiedSentences,modifiedCombinations


# ----------------------------------- START ---------------------------------- #
if __name__ == '__main__':
    main()

