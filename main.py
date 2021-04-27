from text_preprocessing import get_intro, get_conclusion, section_to_vecs, to_sentences, clean_section, get_body
from summary_scoring import rouge_score
from text_rank import summarize, summarize_preserve_order, summarize_with_context


def get_stats(path: str):
    print('Extracting text from LaTeX files...')
    introduction = get_intro(path)
    conclusion = get_conclusion(path)
    context = get_body(path)

    print('Cleaning and creating sentence vectors from GloVe embeddings...')
    intro_sentences, intro_vectors = section_to_vecs(introduction)
    conclusion_sentences, conclusion_vectors = section_to_vecs(conclusion)
    context_sentences, context_vectors = section_to_vecs(context)

    print('Creating initial summaries w/ and w/o context...')
    intro_summary = summarize_preserve_order(intro_sentences, intro_vectors, 3)
    conclusion_summary = summarize_preserve_order(conclusion_sentences, conclusion_vectors, 3)

    intro_summary_with_context = summarize_with_context(intro_sentences, intro_vectors, context_sentences,
                                                        context_vectors, 3)
    conclusion_summary_with_context = summarize_with_context(conclusion_sentences, conclusion_vectors,
                                                             context_sentences, context_vectors, 3)

    summary_separate_without_context = intro_summary + ' ' + conclusion_summary
    summary_separate_with_context = intro_summary_with_context + ' ' + conclusion_summary_with_context

    print('Creating summaries for combined intro and conclusion...', end='\n\n')

    together = introduction + ' ' + conclusion
    together_sentences, together_vectors = section_to_vecs(together)

    together_summary_without_context = summarize_preserve_order(together_sentences, together_vectors, 6)
    together_summary_with_context = summarize_with_context(together_sentences, together_vectors, context_sentences,
                                                           context_vectors, 6)

    print(summary_separate_without_context, end='\n\n')
    print(summary_separate_with_context, end='\n\n')
    print(together_summary_without_context, end='\n\n')
    print(together_summary_with_context, end='\n\n')


if __name__ == '__main__':
    # Fearn_2021.tex --- https://arxiv.org/pdf/2104.03848.pdf
    input_file_fearn = 'latex_files/Fearn_2021.tex' # works
    input_file_bihani = 'latex_files/bihani.tex' # works
    input_file_ge = 'latex_files/ge.tex' # works
    input_file_pylkkonen = 'latex_files/pylkkonen.tex' # works
    input_file_zaib = 'latex_files/zaib.tex'

    get_stats(input_file_fearn)

    manual_intro_summary_fearn = "Many problems in text analysis use ML methods to perform their task, ranging from classical problems like text classification and topic modeling, to more complex tasks like question answering. We focus on the text classification problem, where the dominant approach to using these non-neural models is to first calculate the number of unique terms in the dataset and encode each instance of the dataset into a bag-of-words (BoW) representation. As preprocessing techniques help contribute to a reduced vocabulary, they should also help alleviate this scaling problem, at least according to folklore. In general, we find that although vocabulary size is highly correlated with testing time, it is not highly correlated with training time or accuracy. Our experiments show that rare word filtering and stopword removal are superior to many other common preprocessing methods, both in terms of their ability to reduce run-time and their potential to increase accuracy. We hope that this study can help both researchers and industry practitioners as they design machine learning pipelines to reach their end-goals."
    manual_intro_summary_bihani = "Dialog-based systems have become increasingly ubiquitous, extending their range of conversational ability from open-ended conversations to task-oriented settings. Recently, neural network approaches have been shown to outperform statistical models in terms of classification accuracy, when classifying intents. We evaluate our approach over multiple membership functions, datasets and fuzzy string similarity mapping techniques, to identify the optimal fuzzy membership generation approach for utterances with differing levels of overlap within the same intent as well as across different intents. Current research on intent classification, including approaches as well as corpora, are limited to a one-dimensional view, where utterances are treated as atomic inputs, with binary memberships within intent classes. This paper proposes a framework towards fuzzy intent classification for unseen multi-intent utterances, without the need for the existence of prior multi-intent utterance data to learn intent memberships. Results reveal that taking the underlying data distribution into account when generating memberships yields more consistent results in mapping and emulating binary memberships."
    manual_intro_summary_ge = "In the process of legal judgment, properly linking each case to its related law articles is a crucial starting point, where any miss or mismatch of linkage might affect further decisions and deteriorate judicial fairness. Nonetheless, a case can contain many facts, accurately connecting all facts to their related law articles is challenging even for highly experienced legal professionals. In recent years, with the open access of big judicial data, applying AI technology to automate the search of relevant law articles has become a hot spot."
    manual_intro_summary_pylkkonen = "Over the recent years, the focus in automatic speech recognition research has shifted from hybrid models to end-to-end (E2E) systems. The benefit of the hybrid models is that they can take advantage of different data sources, especially large amounts of text-only data. End-to-end models, on the other hand, are trained from matched speech and transcriptions, so their exposure to different language content is more limited."
    manual_intro_summary_zaib = "With a large amount of available `big data' and advanced deep learning methods, the objective of designing digital conversation systems as our virtual assistant is no longer a dream. Based on functionality, conversational AI can be categorized into three categories: 1) task-oriented systems, 2) chat-oriented systems, and 3) question answering systems.Thus, we focus more on the question answering systems than the other two dialogue systems as this field has been totally transformed by these pre-trained language models."

    manual_conclusion_summary_fearn = ""
    manual_conclusion_summary_bihani = ""
    manual_conclusion_summary_ge = "However, previous research can only make coarse-grained recommendation instead of pointing out the concrete fact-articles correspondences. We design a pipeline of careful annotation to remove missing, mismatched or incorrect articles in the extracted online legal documents. We hope our study can call for more attention on learning such fine-grained correspondence for a more interpretable and accurate AI system in law."
    manual_conclusion_summary_pylkkonen = "In this paper we have presented a practical algorithm to perform domain adaptation of RNN-transducer E2E model with text-only data. The benefits of the RNN-T adaptation were shown with several evaluation tasks, using an experiment model which was trained from well-known public speech corpora, in the interest of experiment reproducibility. Our evidence of the improvements obtained from adapting the prediction network lead us to conclude that the LM interpretation of the prediction network is not only justified, but also practical."
    manual_conclusion_summary_zaib = "This paper is an effort to investigate the recent trends introduced in language models and their application to the dialogue systems. Whilst these pre-trained models have the potential to address most of the limitations posed by previous methods, nevertheless, there are still some open issues that need to be addressed. Thus, in the end, we have highlighted the challenges pertaining to dialogue systems that demand attention."

    # rs_fearn = rouge_score(intro_summary_fearn, manual_intro_summary_fearn)
    # rs_bihani = rouge_score(intro_summary_bihani, manual_intro_summary_bihani)
    # rs_ge = rouge_score(intro_summary_ge, manual_intro_summary_ge)
    # rs_pylkkonen = rouge_score(intro_summary_pylkkonen, manual_intro_summary_pylkkonen)
    # rs_zaib = rouge_score(intro_summary_zaib, manual_intro_summary_zaib)

    # bleu_score = bleu_score(intro_summary, manual_intro_summary)

    # print("Rouge score:", rs_fearn)
    # print("Bleu score:", bleu_score)

