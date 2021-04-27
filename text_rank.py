import math
import numpy
import networkx


# populate similarity matrix
def _similarity_matrix(sentence_vectors: list) -> list:
    matrix = []
    for i in sentence_vectors:
        row = []
        for j in sentence_vectors:
            if i == j:
                row.append(0)
            else:
                row.append(_cosine_similarity(i, j))
        matrix.append(row)
    return matrix


# takes in sentence vector
def _cosine_similarity(sent_x: list, sent_y: list) -> float:
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


# networkx and numpy for fast page rank implementation that we adapt for text rank
def _text_rank(similarity_matrix: list, damping_factor=0.85, num_iterations=100) -> dict:
    similarity_matrix_numpy = numpy.array([numpy.array(row) for row in similarity_matrix])
    graph = networkx.from_numpy_array(similarity_matrix_numpy)
    scores = networkx.pagerank(graph, alpha=damping_factor, max_iter=num_iterations)
    return scores


# sort based off the rank
def _extract_summary(sentences: list, ranks: dict) -> list:
    #print(len(sentences), len(ranks))
    return sorted(((ranks[i], s) for i, s in enumerate(sentences)), reverse=True)


# Record indices of sentences from original text
def _summarize_with_index(sentences: list, sentence_vectors: list, num_sentences: int) -> list:
    ranked = _text_rank(_similarity_matrix(sentence_vectors))
    indexed_ranked_lines = []
    for ranked_line in _extract_summary(sentences, ranked)[:num_sentences]:
        for i, line in enumerate(sentences):
            if ranked_line[1] == line:
                indexed_ranked_lines.append((i, line))
    return indexed_ranked_lines


# Returns the sentences in order of page rank
def summarize(sentences: list, sentence_vectors: list, num_sentences: int) -> str:
    ranked = _text_rank(_similarity_matrix(sentence_vectors))
    return ' '.join([s[1] for s in _extract_summary(sentences, ranked)[:num_sentences]])


# Uses ranked sentences with indices from original text to preserve sentence order from the original text
def summarize_preserve_order(sentences: list, sentence_vectors: list, num_sentences: int):
    return ' '.join(sent for index, sent in sorted(_summarize_with_index(sentences, sentence_vectors, num_sentences)))


def summarize_with_context(usable_sents: list, usable_sent_vectors: list,
                           context: list, context_vectors: list, num_sentences: int) -> str:
    ranked = _text_rank(_similarity_matrix(usable_sent_vectors + context_vectors))
    str_builder = ''
    i = 0

    for ranked_line in _extract_summary(usable_sents + context, ranked):
        if i == num_sentences:
            break
        if ranked_line[1] in str_builder:
            continue
        if ranked_line[1] in usable_sents:
            str_builder += ranked_line[1] + ' '
            i += 1

    return str_builder

# separately with context
# together with context
# separate without context
# together without context
