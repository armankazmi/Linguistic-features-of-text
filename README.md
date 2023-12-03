# Linguistically Motivated Features for Short-Text Genre Classification

## Abstract
This work deploys linguistically motivated features to classify paragraph-level text into fiction and non-fiction genre using a logistic regression model and infers lexical and syntactic properties that help distinguish the two genres. Previous works have focused on classifying document-level text into fiction and non-fiction genres, while in this work, we deal with shorter texts which are closer to real-world applications like sentiment analysis of tweets. For the task of short-text classification on the Brown corpus, a model containing linguistically motivated features confers a substantial accuracy jump over a baseline model consisting of simple POS-ratio features found effective in previous work. The efficacy of the above model containing a linguistically motivated feature set also transfers over to another dataset viz, Baby BNC corpus. Subsequently, we compared the classification accuracy of the logistic regression model with two deep-learning models. A 1D-CNN model gives an increase of 2% accuracy over the logistic regression classifier on both datasets. A BERT-based model gives state-of-the art classification accuracies of 97% on Brown corpus and 98% on Baby BNC corpus. Although, both these deep learning models give better results in terms of classification accuracy, the problem of interpreting these models remains an open question. In contrast, regression model coefficients revealed that fiction texts tend to have more character-level diversity and have lower lexical density (quantified using contentfunction word ratios) compared to non-fiction texts. Moreover, subtle differences in word order exist between the two genres, i.e., in fiction texts Verbs precede Adverbs in contrast to the opposite pattern in non-fiction texts (inter-alia).

## Features
The repository comprises codes to calculate four distinct categories of features, each with various sub-types as outlined in the paper:

1. **Raw Features**
2. **Lexical Features**
3. **POS Features**
4. **Syntactic Features**

## Usage
1. Install the required libraries listed in the `requirements.txt` file.
2. Note: The 'stanfordcorenlp' client requires JAVA 8+ for constituency parsing.
3. Run the inference.py file to calculate features and running the trained models for predictions.

## Folder Structure

- **src**
  - `extract_all_features.py`: Calculates all four categories of features.

- **brown_corpus**
  - `brown_corpus_logistic_regression_analysis.ipynb`: Notebook with classification accuracy using different feature sets on Brown corpus paragraphs (refer to Table 2).
  - `brown_corpus_CNN_Glove_analysis.ipynb`: Notebook with classification accuracy of the 1D CNN model on Brown Corpus (refer to Table 4).
  - `brown_corpus_bert_fine_tune.py`: Fine-tuning codes of BERT on Brown corpus data (refer to Table 4).

- **baby_bnc**
  - `baby_bnc_corpus_logistic_regression.ipynb`: Notebook with classification accuracy on Baby BNC corpus trained on Brown corpus using Logistic Regression model (refer to Table 3).
  - `baby_bnc_corpus_CNN_Glove_analysis.ipynb`: Notebook with classification accuracy of the 1D CNN model on Baby BNC Corpus (refer to Table 3).

- `inference.py`: Example file demonstrating how to calculate features and use the trained model to predict genres for a given text.

# Citation
```bibtex
@inproceedings{kazmi-etal-2022-linguistically,
    title = "Linguistically Motivated Features for Classifying Shorter Text into Fiction and Non-Fiction Genre",
    author = "Kazmi, Arman  and
      Ranjan, Sidharth  and
      Sharma, Arpit  and
      Rajkumar, Rajakrishnan",
    booktitle = "Proceedings of the 29th International Conference on Computational Linguistics",
    month = oct,
    year = "2022",
    address = "Gyeongju, Republic of Korea",
    publisher = "International Committee on Computational Linguistics",
    url = "https://aclanthology.org/2022.coling-1.77",
    pages = "922--937",
    
}
