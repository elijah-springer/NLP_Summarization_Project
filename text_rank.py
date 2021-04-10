import math
import numpy as np
import networkx


# populate similarity matrix
def similarity_matrix(sentence_vectors: list) -> list:
    matrix = []
    for i in sentence_vectors:
        row = []
        for j in sentence_vectors:
            if i == j:
                row.append(0)
            else:
                row.append(cosine_similarity(i, j))
        matrix.append(row)
    return matrix


# takes in sentence vector
def cosine_similarity(sent_x: list, sent_y: list) -> float:
    dot_product = 0
    for i, j in zip(sent_x, sent_y):
        dot_product += (i * j)

    x_squared_sum = 0
    for x in sent_x:
        x_squared_sum += (x ** 2)
    x_magnitude = math.sqrt(x_squared_sum)

    y_squared_sum = 0
    for y in sent_y:
        y_squared_sum += (y ** 2)
    y_magnitude = math.sqrt(y_squared_sum)

    return dot_product / (x_magnitude * y_magnitude)


# apply page rank to matrix
# adapted from wikipedia: https://en.wikipedia.org/wiki/PageRank
def pagerank(similarity_matrix: list, damping_factor=0.85, num_iterations=100) -> dict:
    similarity_matrix_numpy = np.array([np.array(row) for row in similarity_matrix])
    graph = networkx.from_numpy_array(similarity_matrix_numpy)
    scores = networkx.pagerank(graph)
    return scores


def extract_summary(sentences: list, pagerank_arr: dict) -> list:
    # sort based off the pagerank_arr
    ranked_sentences = sorted(((pagerank_arr[i], s) for i, s in enumerate(sentences)), reverse=True)
    return ranked_sentences


def summarized(sentences: list, sentence_vectors: list, num_sentences: int) -> str:
    ranked = pagerank(similarity_matrix(sentence_vectors))
    return ' '.join([s[1] for s in extract_summary(sentences, ranked)[:num_sentences]])
