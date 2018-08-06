from sklearn.preprocessing import LabelBinarizer, FunctionTransformer
from sklearn.pipeline import make_union, make_pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.externals import joblib
import settings

class CustomBinarizer(BaseEstimator, TransformerMixin):
    def __init__(self, binarizer):
        self.transformer_ = LabelBinarizer() if binarizer==None else binarizer

    def fit(self, X, y=None):
        return self.transformer_.fit(X)

    def transform(self, X):
        return self.transformer_.transform(X)

    def fit_transform(self, X, y=None):
        return self.transformer_.fit_transform(X)

def get_titles(X):
    return X['title']

def get_users(X):
    return X['user']

def get_union_transformer(binarizer=None, tfidf=None):

    vec = make_union(*[
        make_pipeline(FunctionTransformer(get_users, validate=False), CustomBinarizer(binarizer)),
        make_pipeline(FunctionTransformer(get_titles, validate=False), TfidfVectorizer(sublinear_tf=True, min_df=5, norm='l2', ngram_range=(1, 2)) if tfidf==None else tfidf)
    ])

    return vec


def load(pkl_path):
    return joblib.load(pkl_path)


def dump(model, pkl_path):
    joblib.dump(model, pkl_path)


def load_transformer():
    binarizer = joblib.load(settings.BINARIZER_PATH)
    tfidf = joblib.load(settings.TFIDF_PATH)
    return get_union_transformer(binarizer, tfidf)


def dump_transformer(vec):
    joblib.dump(vec.transformer_list[0][1].steps[1][1], settings.BINARIZER_PATH)
    joblib.dump(vec.transformer_list[1][1].steps[1][1], settings.TFIDF_PATH)
