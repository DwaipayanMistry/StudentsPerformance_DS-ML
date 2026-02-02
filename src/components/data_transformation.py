import sys
from dataclasses import dataclass
import os

import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.exception import CustomException
from src.logger import logging

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path= os.path.join("artifacts", "preprocessor.plk")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config= DataTransformationConfig()

    def get_data_transformation_obj(self):
        """
        The function is responsible for data transformation.
        """
        try:
            numerical_features=['reading score', 'writing score']
            categorical_features= ['gender', 'race/ethnicity', 'parental level of education', 'lunch', 'test preparation course']

            numerical_pipeline= Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy='median')),
                    ("scalar",StandardScaler() )
                ]
            )

            categorical_pipeline=Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("one_hot_enc", OneHotEncoder()),
                ]
            )

            logging.info(f'categorical Features: {categorical_features}')
            logging.info(f'Numerical Feature: {numerical_features}')

            preprocessor= ColumnTransformer(
            [
                ("Numerical Pipeline", numerical_pipeline, numerical_features),
                ("Categorical Pipeline", categorical_pipeline, categorical_features)
            ]
        )

            return preprocessor
        
        except  Exception as e:
            raise CustomException(e,sys)