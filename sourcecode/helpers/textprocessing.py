import unidecode
import string
from simplemma import text_lemmatizer

_ABBREVIATION = []

def removeWhitespace(sentences):
    modSentences = modifySentence(sentences, 'removeWhitespaces')
    return modSentences

def removePunctuation(sentences):
    modSentences = modifySentence(sentences, 'removePunctation')
    return modSentences

def removeSpecialCharacter(sentences):
    modSentences = modifySentence(sentences, 'removeDiacritics')
    return modSentences

def wordNormalization(sentences):   
    modSentences = modifySentence(sentences, 'normalizeSentence')
    return modSentences
    


#def toLower():

def modifySentence(sentences, operation):
    modSentences = []

    for i in range(len(sentences)):
        if operation == 'removeWhitespaces':
            modifiedSentence = sentences[i].replace(' ', '')
        elif operation == 'removePunctation':
            modifiedSentence = sentences[i].translate(str.maketrans('', '', string.punctuation))
        elif operation == 'removeDiacritics':
            modifiedSentence = unidecode.unidecode(sentences[i])
        elif operation == 'normalizeSentence':
            modifiedSentence = text_lemmatizer(sentences[i], lang=_ABBREVIATION[i])
            modifiedSentence = ' '.join(modifiedSentence)
            print(modifiedSentence)
        else:
            print('Nothing to do!')

        modSentences.append(modifiedSentence)
    return modSentences

    