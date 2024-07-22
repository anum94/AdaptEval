import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from typing import List
import numpy as np
from ds.supported import load_dataset
from collections import Counter
import os
from tqdm import tqdm
class VocabOverlap:
    metric_name = "vocab_overlap"

    def __init__(self) -> None:
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)

    def get_article_vocab(self, text:str):

        tokens = word_tokenize(str(text).lower())

        # Remove special characters from each token
        #tokens = set([re.sub('[^a-zA-Z0-9]+', '', _) for _ in tokens])
        # Remove stopwords
        stop_words = set(stopwords.words('english'))
        special_characters = [",", ".", "(", ")", "'", "’", ";", ":", "!", "@", "#", "$", "%", "^", "&", "*", "-", "_",
                              "+", "=", "[", "]", "}", "{", "<", ">", "/", "~", "``", "``", '\\']
        tokens = [token for token in tokens if (token not in stop_words and token not in special_characters and token is not None)]

        # Remove all numbers
        #tokens = set(val for val in tokens if not val.isdigit())

        return tokens

    def read_vocab(self, domain_vocab_file:str):
        # Read the domain vocabulary
        with open(domain_vocab_file, 'r') as file:
            # Create an empty list to store the lines
            domain_vocab = []
            # Iterate over the lines of the file
            for line in file:
                # Remove the newline character at the end of the line
                line = line.strip()
                # Append the line to the list
                domain_vocab.append(str(line))
        return domain_vocab

    def calculate_vocabulary_overlap(self, text:str, domain_vocab: []):

        # Tokenize the texts
        tokens = word_tokenize(str(str(text).lower()))

        domain_vocab = set(domain_vocab)

        # Remove stopwords ----> optional I guess?
        # stop_words = set(stopwords.words('english'))
        # tokens = tokens - stop_words

        # Calculate precision
        overlap = len(domain_vocab.intersection(tokens))
        # Calculate percentage overlap
        if len(tokens) > 0:
            overlap_percentage = (overlap / len(tokens)) * 100
        else:
            overlap_percentage = 0

        return overlap_percentage

    def compute(self, predictions: List, references:List) -> float:
        # reference here refers to the article, and not the reference summary.
        overlap_percentage = [self.calculate_vocabulary_overlap(predictions[i], references) for i in range(len(predictions))]
        return np.mean(overlap_percentage)

    def compute_domain_corpus(self, path_domain_articles:str, top_k = 1000) -> List:
        domain_vocabulary = []
        domain_vocab_count = Counter()
        articles = []

        if not os.path.exists(path_domain_articles):
            print ("Provided path to domain articles is not valid. Please provide a valid path through the config file.")
            return articles
        # open file and read the content in a list
        with open(path_domain_articles, 'r') as fp:
            for line in fp:
                x = line[:-1]
                articles.append(x)

        # do it in smaller chunks
        chunks = []
        chunk_size = 10000
        for i in range(chunk_size, len(articles), chunk_size):
            chunks.append(i)
        chunks.append(len(articles))
        print (f"Training data is large with {len(articles)}, Calculating Domain Vocab in {len(chunks)} Chunks.")

        for j, chunk in enumerate(chunks):
            print (f"Chunk {j} with samples till {chunk}")
            indices = [idx for idx in range(j*chunk_size, chunk)]
            subset_dtrain = [articles[x] for x in indices]


            for article in subset_dtrain:
                article_vocabulary = self.get_article_vocab(article)
                domain_vocabulary.extend(article_vocabulary)

            domain_vocab_count = domain_vocab_count + Counter(domain_vocabulary)

        domain_vocab_count = domain_vocab_count.most_common(top_k)

        # open file in write mode
        folder = "data/domain_vocabulary/"
        if not os.path.exists(folder):
            os.makedirs(folder)

        file_name = f'{path_domain_articles.split("/")[-1].split("_")[0]}_top{top_k}_vocabulary.txt'
        path = os.path.join(folder,file_name)

        with open(path, 'w') as fp:
            for item in domain_vocab_count:
                # write each item on a new line
                fp.write("%s\n" % str(item[0]))

        print(f"domain vocabulary saved at {path}")
