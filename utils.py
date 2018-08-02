import re
import pymorphy2
import pandas as pd
from pymorphy2.tokenizers import simple_word_tokenize
from pymorphy2.shapes import is_punctuation
from stop_words import get_stop_words
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.multioutput import MultiOutputClassifier

morph = pymorphy2.MorphAnalyzer()
stop_words = get_stop_words('russian')

def replace_by_re(s, pattern, repl):
    return re.sub(pattern, repl, s)

def lemmatize(s):
    l = [morph.parse(w)[0].normal_form for w in simple_word_tokenize(s) if not is_punctuation(w)]
    l = [w for w in l if w not in stop_words]
    return " ".join(l)

def preprocess(s):
    s = re.sub('1[CСcс]', "1С", s)
    s = re.sub('FW:\s+', "", s)
    s = re.sub('RE:\s+', "", s)
    s = re.sub('(\d{2}[\.\\\/]){2}(\d{2})+', "дата", s)
    s = lemmatize(s)
    return s

def drop_below(df, name, quantile):
    grouped = df[name].value_counts()
    grouped = grouped[grouped >= grouped.quantile(q=quantile)]
    return df[df[name].isin(grouped.index.tolist())]

def train(file_data):
    df = pd.read_csv(file_data, sep=',')
    df = df.dropna()

    df['specialist'].replace(['Руководители службы Service Desk', 'Шкурупий Денис'], 'Манюхин Андрей', inplace=True)
    df['specialist'].replace('Кухарчук Дмитрий', 'Сергиевич Юрий', inplace=True)
    df['specialist'].replace('Талаева Вера', 'Кропис Юлия', inplace=True)
    df['specialist'].replace('Быков Вадим', 'Полегошко Андрей', inplace=True)

    df = drop_below(df, 'specialist', 0.25)
    df = drop_below(df, 'analytics1', 0.50)
    df = drop_below(df, 'analytics2', 0.85)
    df = drop_below(df, 'analytics3', 0.85)
    df = drop_below(df, 'category', 0.75)

    df['title'] = df['title'].apply(lambda x: preprocess(x))

    tfidf = TfidfVectorizer(sublinear_tf=True, min_df=5, norm='l2', ngram_range=(1, 2))
    features = tfidf.fit_transform(df.title).toarray()
    labels = df[['specialist', 'category', 'analytics1', 'analytics2', 'analytics3']]

    model = MultinomialNB()

    multi_target_nb = MultiOutputClassifier(model, n_jobs=-1)
    multi_target_nb.fit(features, labels)

    return tfidf, model
