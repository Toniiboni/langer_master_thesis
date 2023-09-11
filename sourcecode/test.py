import helpers.textprocessing as nlp
import copy
import matplotlib.pyplot as plt
from scipy.cluster import hierarchy
import needleman_wunsch.NeedlemanWunsch as algorithm
import os
from datetime import datetime
from scipy.spatial.distance import pdist, squareform


# ----------------------------------- MAIN ----------------------------------- #
def main():
    print('STARTING ALGORITHM...')
    
    languages = [
    'Deutsch',
    'Englisch',
    'Französisch',
    'Spanisch',
    'Italienisch',
    'Finnisch'
    ]
 
    sentences = [
        'Das ist das Leben',
        'This is life',
        'C\'est la vie',
        'Así es la vida',
        'Questa è vita',
        'Tämä on elämää'
    ]

    combinations = {
        'Deutsch':'Das ist das Leben',
        'Englisch':'This is life',
        'Französisch':'C\'est la vie',
        'Spanisch':'Así es la vida',
        'Italienisch':'Questa è vita',
        'Finnisch':'Tämä on elämää'   
    }

    scoring = algorithm.start_needleman_wunsch(languages, sentences, combinations, 'combo', filename="hello", textLength="short")
    
    textLength = 'short'

    plt.figure(figsize=(14,9.5))
    plt.title("Dendrogram of language similarity\n(" + textLength.capitalize() + ' text / x')
    plt.xlabel('European languages') 
    plt.ylabel('Similarity')
    dn = hierarchy.dendrogram(scoring, labels=languages)
    plt.show()
   
    '''
    # Calculate the pairwise Euclidean distances for the first dendrogram
    distances1 = pdist(scoring)

    # Calculate the pairwise Euclidean distances for the second dendrogram
    distances2 = pdist(scoring)

    # Calculate the cophenetic correlation coefficient
    c, coph_dists = hierarchy.cophenet(linkage, distances1)
    print("Cophenetic correlation coefficient for dendrogram 1:", c)

    print(squareform(hierarchy.cophenet(linkage)))

    results2 = 1 - sp.distance.cdist(matrix1, matrix2, 'cosine')

    print(coph_dists)
    '''
   


# ------------------------------- NLP FUNCTIONS ------------------------------ #

    #functions_Normalization = [nlp.wordNormalization]
    #functions_Punctuation = [nlp.removePunctuation]
    #functions_Diacritics = [nlp.removeSpecialCharacter]
    #functions_Whitespaces = [nlp.removeWhitespace]



# ------------------------------ SINGLE NLP STEP ----------------------------- #
   
    # Normalization x Original
    

    # Remove Punctuation x Original  
    '''
    punctuationSentences, puntuationCombinations = executeNLPSteps(sentences, combinations, functions_Punctuation, languages, "removedPunctuation")
    print("------------------")
    print(punctuationSentences)
    print(puntuationCombinations)
    print("------------------")

    # Remove Diacritics x Original   
    diacriticsSentences, diacriticsCombinations = executeNLPSteps(sentences, combinations, functions_Diacritics, languages, "removedDiacritics")

    # Remove Whitspaces x Original
    whitespacesSentences, whitespacesCombinations = executeNLPSteps(sentences, combinations, functions_Whitespaces, languages, "removedWhitespaces")
    #whitespacesSentences, whitespacesCombinations = executeNLPSteps(sentences, combinations, functions_Whitespaces, languages, "removedWhitespaces")

    print("------------------")
    print(punctuationSentences)
    print(puntuationCombinations)
    print("------------------")



# --------------------------- REPLACE COMBINATIONS --------------------------- #
def replaceCombinations(languages, sentences, combinations):
    """
        Replace the corresponding text from the language with the modified text retrieved from the NLP step.
    """   
    modCombinations = copy.copy(combinations)

    i = 0
    for language in languages:
        modCombinations[language] = sentences[i]
        i = i + 1
    return modCombinations

def executeNLPSteps(sentences, combinations, functions, languages, filename):
    
    modifiedSentences = functions[0](sentences)
    modifiedCombinations = replaceCombinations(languages, modifiedSentences, combinations)

    return modifiedSentences,modifiedCombinations
'''

# ----------------------------------- START ---------------------------------- #
if __name__ == '__main__':
    main()

