import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

from typing import List
from preprocess import get_X_samples

class StaticFeatureAdder(BaseEstimator, TransformerMixin):
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        df = X.copy()
        df["month"] = df.index.month
        df["week"] = df.index.isocalendar().week
        df["day_of_week"] = df.index.day_of_week
        df["hr"] = df.index.hour
        df["is_weekend"] = np.where(df["day_of_week"] > 4, 1, 0)
        return df


class CyclicalTransformer(BaseEstimator, TransformerMixin):

    def __init__(self, features:List[str]):
        self.features = features

    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        df = X.copy()
        for feature in self.features:
            n = df[feature].max()-df[feature].min()+1
            df[feature+"_cos"] = np.cos(df[feature]*2*np.pi/n)
            df[feature+"_sin"] = np.sin(df[feature]*2*np.pi/n)
        return df


class LagsAdder(BaseEstimator, TransformerMixin):
    
    def __init__(self, freqs):
        self.freqs = freqs
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        df = X.copy()
        for freq in self.freqs:
            df[f"consumption_lag_{freq}"] = df["consumption"].shift(freq=freq)
        return df
    

class WindowFeatureAdder(BaseEstimator, TransformerMixin):
    
    def __init__(self, window, aggs):
        self.window = window
        self.aggs = aggs
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        df = X.copy()
        window = self.window
        for agg in self.aggs:
            df[f"consumption_{agg}_{window}H"]=df.join(df['consumption'].rolling(window=window).agg([agg]).shift(periods=1), how="left")[agg]
        return df


class RowRemover(BaseEstimator, TransformerMixin):
    
    def __init__(self, n_rows):
        self.n_rows = n_rows
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        df = X.drop(X.index[:self.n_rows])
        return df
    

class ColumnRemover(BaseEstimator, TransformerMixin):
    
    def __init__(self, columns=['month', 'week', 'day_of_week', 'hr' ]):
        self.columns = columns
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        return X.drop(columns=self.columns)
    

class X_y_splitter(BaseEstimator, TransformerMixin):
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        X_train, y_train = X.drop(columns=['consumption']), X.consumption #.values.astype('float32')
        return X_train, y_train
    

class X_y_sampler(BaseEstimator, TransformerMixin):
    
    def fit(self, X, y):
        return self
    
    def transform(self, X, y):
        X_samples = get_X_samples(X)
        y_samples =  y.values[23:]
        return X_samples, y_samples