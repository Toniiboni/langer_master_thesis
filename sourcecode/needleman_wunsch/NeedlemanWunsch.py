import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.cluster import hierarchy
import codecs
from numba import prange, njit

gap_penalty_score = -1
match_score = 1
mismatch_penalty_score = -1

@njit
def match_score(a, b):
    if a == b:
        return match_score
    elif a == '-' or b == '-':
        return gap_penalty_score
    else:
        return mismatch_penalty_score

@njit(parallel=True)
def needleman_wunsch(sequence_one, sequence_two):
    n = len(sequence_one)
    m = len(sequence_two)
    score = np.zeros((m + 1, n + 1))

    for i in prange(0, m + 1):
        score[i][0] = gap_penalty_score * i

    for j in prange(0, n + 1):
        score[0][j] = gap_penalty_score * j

    for k in range(1,m + 1):
        for l in range(1, n + 1):
            match = score[k - 1][l - 1] + match_score(sequence_one[l-1], sequence_two[k-1])
            insert = score[k][l - 1] + gap_penalty_score
            delete = score[k - 1][l] + gap_penalty_score
            score[k][l] = max(match, delete, insert)
    
    alignment_one = ""
    alignment_two = ""

    i = m
    j = n

    while i > 0 and j > 0:
        score_current_cell = score[i][j]
        score_left_cell = score[i-1][j]
        score_upper_cell = score[i][j-1]
        score_diagonal_cell = score[i-1][j-1]
              

        if score_current_cell == score_diagonal_cell + match_score(sequence_one[j-1], sequence_two[i-1]):
            alignment_one += sequence_one[j-1]
            alignment_two += sequence_two[i-1]
            i -= 1
            j -= 1

        elif score_current_cell == score_upper_cell + gap_penalty_score:
            alignment_one += sequence_one[j-1]
            alignment_two += '-'
            j -= 1
        elif score_current_cell == score_left_cell + gap_penalty_score:
            alignment_one += '-'
            alignment_two += sequence_two[i-1]
            i -= 1
    
    while j > 0:
        alignment_one += sequence_one[j-1]
        alignment_two += '-'
        j -= 1
    while i > 0:
        alignment_one += '-'
        alignment_two += sequence_two[i-1]
        i -= 1

    alignment_one = alignment_one[::-1]
    alignment_two = alignment_two[::-1]   
    return (score, alignment_one, alignment_two)

def scoring_matrix(languages, sentences, combinations, filename, textLength):
    scoring_matrix = np.zeros((len(languages),len(languages)))
    i = 0
    f = codecs.open(os.path.abspath(os.curdir) + '/sourcecode/files/' + textLength + '/' + filename + ".txt", "w", encoding='utf-16')
    
  
    for x in languages:
        j = 0
        for y in languages:
            Matrix, alignment1, alignment2 = needleman_wunsch(combinations.get(x), combinations.get(y))
            f.write('------------------------------')
            f.write("\n")
            f.write('Sentence 1: ' + combinations.get(x))
            f.write("\n")
            f.write('Sentence 2: ' + combinations.get(y))
            f.write("\n")
            f.write('Alignment:')
            f.write("\n")
            f.write(alignment1)
            f.write("\n")
            f.write(alignment2)
            f.write("\n")
            f.write('------------------------------')
            f.write("\n")
            scoring_matrix[i][j] = Matrix[-1][-1]
            j = j + 1
        i = i + 1
    f.close()
    return scoring_matrix

@njit(parallel=True)
def get_matrix_max(matrix): 
    matrix_max_value = -np.inf
 
    for i in range(0, len(matrix[0])):
        for j in range(0, len(matrix[0])):
            if(matrix_max_value == -np.inf):
                matrix_max_value = matrix[i][j]
            if(matrix[i][j] >= matrix_max_value):
                matrix_max_value = matrix[i][j]
 
    return matrix_max_value

@njit(parallel=True)
def scoring_distance_matrix(scoring_matrix, languages):
 
    scoring_distance_matrix = np.zeros((len(scoring_matrix[0]), len(scoring_matrix[0])))
    maxR = get_matrix_max(scoring_matrix)

    for i in range(0, len(languages)):
        for j in range(0, len(languages)):
            scoring_distance_matrix[i][j] = abs(scoring_matrix[i][j] - maxR)
 
    return scoring_distance_matrix


def start_needleman_wunsch(languages, sentences, combinations, combination, filename="algorithm", textLength="null"):
    scoring = scoring_matrix(languages, sentences, combinations, filename, textLength)
    scoring_distance_matrix1 = scoring_distance_matrix(scoring, languages)
    
    f =  codecs.open(os.path.abspath(os.curdir) + '/sourcecode/files/' + textLength + '/' + filename + "_scoring_matrix" + ".txt", "w", encoding="utf-16")
    f.write(np.array2string(scoring))
    f.close()

    normalized_matrix = (scoring - np.min(scoring)) / (np.max(scoring) - np.min(scoring))

    plt.figure(figsize = (9,7))
    plt.imshow(normalized_matrix, cmap='hot', interpolation='none')
    plt.colorbar(label='Similarity')

    labels = ['Bulgarian', 'Danish', 'German', 'English', 'Estonian', 'Finnish', 'French', 'Greek', 'Irish', 'Italian', 'Latvian', 'Lithuanian', 'Dutch', 'Polish', 'Portuguese', 'Romanian', 'Swedish', 'Slovak', 'Slovenian', 'Spanish', 'Czech', 'Hungarian']

    ax = plt.gca()
    ax.set_xticks(np.arange(0, 22, 1))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticks(np.arange(0, 22, 1))
    ax.set_yticklabels(labels, rotation=45, rotation_mode="anchor")
    plt.title('Heatmap of language similarity\n('+ textLength.capitalize() + ' text / ' + combination +')')
    plt.savefig(os.path.abspath(os.curdir) + '/sourcecode/files/' + textLength + '/Heatmap/' + filename + '_heatmap.jpg')
    average = hierarchy.linkage(scoring_distance_matrix1, "average")

    return average