from collections import Counter

class KieuCorpus:
    """
    A class representing a corpus for processing and analyzing text data.

    Parameters:
    - path (str): The path to the text file containing the corpus.

    Attributes:
    - path (str): The path to the text file.
    - body (str): The content of the corpus.

    Methods:
    - corpus(): Reads and returns the content of the corpus file.
    - lookup_table(): Generates and returns word-to-index and index-to-word lookup tables based on word frequency.
    - sentences(): Tokenizes the corpus into sentences and returns a list of lists, where each inner list represents a sentence.
    - ngram(window_size=3, word_to_idx=None): Generates and returns n-grams (sequences of words or indices) from the sentences in the corpus.

    Usage:
    corpus_instance = KieuCorpus("path/to/corpus.txt")
    corpus_text = corpus_instance.corpus()
    word_to_idx, idx_to_word = corpus_instance.lookup_table()
    sentences_list = corpus_instance.sentences()
    ngram_list = corpus_instance.ngram(window_size=3, word_to_idx=word_to_idx)
    """
    def __init__(self, path):
        self.path = path
        self.body = None
        
    def corpus(self):
        """
        Reads and returns the content of the corpus file.

        Returns:
        str: The content of the corpus.
        """
        if self.body is None:
            f = open(self.path, encoding="utf8")
            self.body = f.read()
            f.close()
        return self.body

    def lookup_table(self):
        """
        Generates and returns word-to-index and index-to-word lookup tables based on word frequency.

        Returns:
        tuple: A tuple containing the word-to-index and index-to-word dictionaries.
        """
        text = self.corpus()
        word_counts = Counter(text.split())
        sorted_vocab = sorted(word_counts, key=word_counts.get, reverse=True)

        word_to_idx = {word: i for i, word in enumerate(sorted_vocab)}
        idx_to_word = {i: word for word, i in word_to_idx.items()}

        return word_to_idx, idx_to_word
    
    def sentences(self):
        """
        Tokenizes the corpus into sentences and returns a list of lists,
        where each inner list represents a sentence.

        Returns:
        list: List of lists, each representing a sentence.
        """
        text = self.corpus()
        sents = text.split("\n")
        sents = [
            sent.split() for sent in sents
        ]
        return sents

    def ngram(self, window_size=3, word_to_idx=None):
        """
        Generates and returns n-grams (sequences of words or indices) from the sentences in the corpus.

        Parameters:
        - window_size (int): The size of the sliding window for creating n-grams (default is 3).
        - word_to_idx (dict): Word-to-index lookup table (optional, used for indexing words in n-grams).

        Returns:
        list: List of n-grams, where each n-gram is represented as a list of words or indices.
        """
        sents = self.sentences()
        ngram = []
        for sent in sents:
            l = len(sent)
            for i in range(window_size, l + 1):
                start = max(0, i - window_size)
                if word_to_idx is None:
                    ngram.append(sent[start:i])
                else:
                    ngram.append([word_to_idx[word] for word in sent[start:i]])
        return ngram
