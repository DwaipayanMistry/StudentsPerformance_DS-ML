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
        
    
    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df= pd.read_csv(train_path)
            test_df= pd.read_csv(test_path)
            logging.info("Read train & test data completed")

            logging.info("Obtaining processing object")
            preprocessing_obj= self.get_data_transformation_obj()

            target_column_name="math_score"
            numerical_features=['reading score', 'writing score']

            #  Train dataset
            input_feature_train_df= train_df.drop(column=['math score'], axis=1)
            target_feature_train_df= train_df['math score']

            # Test dataset
            input_feature_test_df= test_df.drop(columns=['math score'], axis=1)
            target_feature_test_df= test_df['math score']

            logging.info("Applying preprocessing object on training dataframe and testing dataframe")

            input_feature_train_df= preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_df= preprocessing_obj.transform(input_feature_test_df)

            train_arr= np.c_[input_feature_train_df, np.array(target_feature_train_df)]
            test__arr= np.c_[input_feature_test_df, np.array(target_feature_test_df)]

            return(
                train_arr,
                test__arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )

        except Exception as e:
            raise CustomException(e, sys)