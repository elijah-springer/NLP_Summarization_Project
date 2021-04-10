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
        RuntimeError("Could not find introduction")
    intro_end = len(lines) - 1 if intro_end > 0 else intro_end

    return ' '.join(lines[intro_start:intro_end])


# clean unwanted stuff
def clean_section(section: str) -> str:
    lines = section.split('\n')

    # Remove latex metadata
    lines = [re.sub(r'\\.+}', '', line) for line in lines]

    # Remove leading and trailing whitespace including newlines
    lines = [line.lstrip().rstrip() for line in lines]

    # Remove empty and very short lines
    lines = [line for line in lines if len(line) > 1]

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
    while i < end:
        if sentences[i + 1][0].islower():
            sentences_clean.append(sentences[i] + " " + sentences[i + 1])
            i += 2
        else:
            sentences_clean.append(sentences[i])
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
