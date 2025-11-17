import logging
from pathlib import Path

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.ensemble import GradientBoostingClassifier
import joblib

from etl import load_raw, clean_data
from feature_engineer import crear_materializado_30d

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Train:

    def __init__(self, df, target_column, model, test_size=0.2):
        self.df = df
        self.model = model
        self.test_size = test_size
          
        self.target_column = "materializado_30d"

        self.features = [
            "Causa_DS",
            "AgenteGenerador_DS",
            "Severidad_NUM",
            "Año",
            "valor_acumulado"
        ]
                
        
        # Solo categóricas dentro de tus features
        self.categorical_features = df[self.features].select_dtypes(
            include=['object', 'category']
        ).columns.tolist()

    def train_test_split(self):
        X = self.df[self.features]
        y = self.df[self.target_column]
        return train_test_split(X, y, test_size=self.test_size, random_state=42)

    def create_preprocessor(self):
        preprocessor = ColumnTransformer(
            transformers=[
                ('cat', OneHotEncoder(handle_unknown='ignore'), self.categorical_features)
            ],
            remainder='passthrough'
        )
        return preprocessor

    def create_pipeline_train(self):
        pipeline = Pipeline(steps=[
            ('preprocessor', self.create_preprocessor()),
            ('classifier', self.model)
        ])
        return pipeline

    def train(self):
        X_train, X_test, y_train, y_test = self.train_test_split()
        pipeline = self.create_pipeline_train()
        pipeline.fit(X_train, y_train)
        return pipeline, X_train, X_test, y_train, y_test
    