from text_preprocessing import get_intro, section_to_vecs
from text_rank import summarized

if __name__ == '__main__':
    introduction = get_intro('latex_files/preprocessing_paper.tex')
    sentences, vectors = section_to_vecs(introduction)
    print(summarized(sentences, vectors, 5))
