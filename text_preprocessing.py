from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
import numpy as np
import re

# get text from latex file
def get_intro(file_path: str) -> str:
    input_file = open(file_path, 'r')
    lines = input_file.readlines()

    intro_start = -1
    intro_end = -1
    found_intro = False

    for i, line in enumerate(lines):
        if not found_intro and '\\section{Introduction}' in line:
            intro_start = i
            found_intro = True
            continue

        if found_intro and '\\section{' in line:
            intro_end = i
            break

    if intro_start == -1:
        RuntimeError("Could not find introduction. Ensure paper has a introduction section header.")
    intro_end = len(lines) - 1 if intro_end == -1 else intro_end

    return ' '.join(lines[intro_start:intro_end])


def get_conclusion(file_path: str) -> str:
    input_file = open(file_path, 'r')
    lines = input_file.readlines()

    conclusion_start = -1
    conclusion_end = -1
    found_conclusion = False

    for i, line in enumerate(lines):
        if not found_conclusion and '\\section{' in line and 'Conclusion' in line:  # Conclusions sometimes have other words in their titles, check for this separately
            conclusion_start = i
            found_conclusion = True
            continue

        if found_conclusion and '\\section{' in line:
            conclusion_end = i
            break

        if found_conclusion and '\\end{document}' in line:  # sometimes papers don't have any more sections after the conclusion, so look for the end of the doc instead
            conclusion_end = i
            break

    if conclusion_start == -1:
        RuntimeError("Could not find conclusion. Ensure paper has a conclusion section header.")
    conclusion_end = len(lines) - 1 if conclusion_end == -1 else conclusion_end

    conclusion_sents = ' '.join(lines[conclusion_start:conclusion_end])

    return conclusion_sents


def get_body(file_path: str) -> str:
    input_file = open(file_path, 'r')
    lines = input_file.readlines()
    return ' '.join(lines)


# clean unwanted stuff
def clean_section(section: str) -> str:
    lines = section.split('\n')

    # lines = []
    # skip_next = False
    # for i in range(0, len(original_lines) - 1):
    #     if skip_next:
    #         skip_next = False
    #         continue
    #     first_line = original_lines[i]
    #     second_line = original_lines[i + 1]
    #
    #     if len(first_line) == 0 or len(second_line) == 0:
    #         continue
    #
    #     if second_line[0] == ')':
    #         lines.append(first_line + second_line)
    #         skip_next = True
    #     else:
    #         lines.append(first_line)
    #
    # print(lines)

    # Remove latex metadata
    #lines = [re.sub(r'\\.+}', '', line) for line in lines]
    lines = [re.sub(r'\\[a-zA-Z]+\{.*?}', '', line) for line in lines]


    # Remove leading and trailing whitespace including newlines
    lines = [line.lstrip().rstrip() for line in lines]


    # Remove empty and very short lines
    lines = [line for line in lines if len(line) > 1]

    lines = [re.sub(r'\[.+]+?', '', line) for line in lines]

    # Remove table entries and other latex syntax
    lines = [line for line in lines
             if line[0] != '\\' and line[-1] != '\\'
             and line[0] != '[']

    return ' '.join(lines)


# break into sentences
def to_sentences(section: str) -> list:
    sentences = sent_tokenize(section)
    sentences = [s.replace('\\', '') for s in sentences if len(s) > 2]
    sentences_clean = []
    i = 0
    end = len(sentences) - 1
    sent_builder = ''
    while i < end:
        sent_builder += sentences[i]
        if sentences[i + 1][0].islower():
            sent_builder += ' '
            if i == end - 1:
                sentences_clean.append(sent_builder)
                sent_builder = ''
            i += 1
        else:
            sentences_clean.append(sent_builder)
            sent_builder = ''
            i += 1
    return sentences_clean


def clean_sentences(sentences: list) -> list:
    stop_words = set(stopwords.words('english'))
    punctuation = {'.', ',', ';', '(', ')', '!', '?'}

    cleaned = []
    for s in sentences:
        sent = []
        for w in word_tokenize(s):
            if not w.isalpha() or w in stop_words or w in punctuation:
                continue
            sent.append(w)
        cleaned.append(' '.join(sent))

    return cleaned


# convert to vec rep -> list[list[word_vecs]]
def sent_to_vec(sentences: list) -> list:
    word_embeddings = {}
    glove = open('glove/glove.6B.100d.txt', encoding='utf-8')
    for line in glove:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        word_embeddings[word] = coefs
    glove.close()

    sent_vectors = []
    for sent in sentences:
        v = sum([word_embeddings.get(w, np.zeros((100,))) for w in sent.split()]) / float((len(sent.split())))
        sent_vectors.append(v)

    return sent_vectors


def section_to_vecs(section: str) -> (list, list):
    sentences = to_sentences(clean_section(section))
    vectors = [np_vec.tolist() for np_vec in sent_to_vec(clean_sentences(sentences))]
    return sentences, vectors
