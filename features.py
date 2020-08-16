from sklearn.base import BaseEstimator
import numpy as np
import pt_core_news_sm
import re

# Contagem de palavras em uma sentença (com expressão regular)
class CountWords(BaseEstimator):
    def __init__(self):
        pass

    def fit(self, X=None, y=None):
        return self

    def transform(self, sentences):
        tokenized_sentences = []
        for sentence in sentences:
            words = len(re.findall(r'[^\s!\?,\(\)\.]+', sentence))
            tokenized_sentences.append(words)
        return np.array(tokenized_sentences).reshape(-1, 1)


# Contagem de adjetivos utilizando spacy
class CountAdjectives(BaseEstimator):
    def __init__(self):
        pass

    def fit(self, X=None, y=None):
        return self

    def transform(self, sentences):
        list_count = []
        for doc in sentences:
            list_count.append(len([token for token in doc if token.pos_ == 'ADJ']))
        return np.array(list_count).reshape(-1, 1)


# Contagem de palavras comparativas utilizando spacy
class CountComparatives(BaseEstimator):
    def __init__(self):
        pass

    def fit(self, X=None, y=None):
        return self

    def transform(self, sentences):
        list_count = []
        for doc in sentences:
            list_count.append(len([token for token in doc if 'KOMP' in token.tag_]))
        return np.array(list_count).reshape(-1, 1)


# Contagem de palavras no superlativo utilizando spacy (rever)
class CountSuperlatives(BaseEstimator):
    def __init__(self):
        pass

    def fit(self, X=None, y=None):
        return self

    def transform(self, sentences):
        list_count = []
        for doc in sentences:
            list_count.append(len([token for token in doc if 'SUP' in token.tag_]))
        return np.array(list_count).reshape(-1, 1)


# Contagem de advérbios utilizando spacy
class CountAdverbs(BaseEstimator):
    def __init__(self):
        pass

    def fit(self, X=None, y=None):
        return self

    def transform(self, sentences):
        list_count = []
        for doc in sentences:
            list_count.append(len([token for token in doc if token.pos_ == 'ADV']))
        return np.array(list_count).reshape(-1, 1)


# Contagem de nomes utilizando spacy
class CountNouns(BaseEstimator):
    def __init__(self):
        pass

    def fit(self, X=None, y=None):
        return self

    def transform(self, sentences):
        list_count = []
        for doc in sentences:
            list_count.append(len([token for token in doc if token.pos_ == 'NOUN' or token.pos_ == 'PROPN']))
        return np.array(list_count).reshape(-1, 1)


# Contagem de adjectival modifiers utilizando spacy
class CountAmod(BaseEstimator):
    def __init__(self):
        pass

    def fit(self, X=None, y=None):
        return self

    def transform(self, sentences):
        list_count = []
        for doc in sentences:
            list_count.append(len([token for token in doc if token.dep_ == 'amod']))
        return np.array(list_count).reshape(-1, 1)


# Grau de polaridade com valores do iFeel
class DegreePolarity(BaseEstimator):

    def __init__(self, ifeel_data, score_method='SENTIMENT140'):
        self.ifeel_data = ifeel_data
        self.score_method = score_method

    def fit(self, X=None, y=None):
        return self

    def transform(self, sentences):
        filtered_data = self.ifeel_data.loc[self.ifeel_data.index.isin(sentences.index)]
        return filtered_data[self.score_method].to_numpy().reshape(-1, 1)


# Grau de subjetividade com valores do iFeel
class DegreeSubjectivity(BaseEstimator):

    def __init__(self, ifeel_data, score_method='EMOTICONDS'):
        self.ifeel_data = ifeel_data
        self.score_method = score_method

    def fit(self, X=None, y=None):
        return self

    def transform(self, sentences):
        filtered_data = self.ifeel_data.loc[self.ifeel_data.index.isin(sentences.index)]
        return np.array([abs(score) for score in filtered_data[self.score_method]]).reshape(-1, 1)


# tree
class SpacyTransformer(BaseEstimator):

    def __init__(self):
        self.nlp = pt_core_news_sm.load()

    def fit(self, X=None, y=None):
        return self

    def transform(self, sentences):
        return list(self.nlp.pipe(sentences))