from typing import List, Optional, Union
from exceptiongroup import catch
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, is_classifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
import os

CLASSIFIERS = [
    DecisionTreeClassifier(),
    RandomForestClassifier(),
    XGBClassifier(),
]

DATA_PATH = "../data"

HISTORY_PATH = "../history"

class ModelSuplier():
    def __init__(self, classifiers: Optional[List[Union[BaseEstimator, ClassifierMixin]]] = None):
        self.classifiers = classifiers if classifiers is not None else CLASSIFIERS

        for clf in self.classifiers:
            try:
                assert is_classifier, f"{type(clf)} is not classifier instance"
            except AssertionError as e:
                raise TypeError(e.args)

        self.transformer = self._create_transformer()

    def _create_transformer(self):
        num_pipeline = Pipeline(steps=[
            ('impute', SimpleImputer()),
            ('scale', MinMaxScaler())
        ])

        cat_pipeline = Pipeline(steps=[
            ('impute', SimpleImputer(strategy="constant", fill_value="missing")),
            ('one-hot', OneHotEncoder(handle_unknown='ignore'))
        ])


        col_trans = ColumnTransformer([
            ('num_pipeline', num_pipeline, make_column_selector(dtype_include = np.number)),
            ('cat_pipeline', cat_pipeline, make_column_selector(dtype_include = np.object_))
        ])

        return col_trans
    
    def _create_pipe(self, clf = Union[BaseEstimator, ClassifierMixin]):
        return (type(clf), Pipeline([("transformer", self.transformer), ("model", clf)]))
    
    @property
    def pipelines(self):
        return [self._create_pipe(clf) for clf in self.classifiers]
    

class DataLoader():
    def __init__(self, data_path: str = None):
        data_path_checked = type(data_path) is str and os.path.isdir(data_path)
        if not data_path_checked: print("Using default path")
        self.path = data_path if data_path_checked else DATA_PATH

    def load(self):
        data = []
        for file in os.listdir(self.path):
            data.append(pd.read_csv(os.path.join(self.path,file)))
        return data
    
    def transform_to_X_y(self, data: pd.DataFrame):
        """
        Last column of the data should be target.
        """
        return data.iloc[:, :-1], LabelEncoder().fit_transform(data.iloc[:,-1])
    
    @property
    def transformed_data(self):
        return [self.transform_to_X_y(data) for data in self.load()]
