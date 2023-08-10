import pandas as pd
import os
import torch
from autogluon.tabular import TabularDataset, TabularPredictor
import numpy as np

import sys
from tqdm import tqdm
from sklearn.metrics import accuracy_score, matthews_corrcoef, roc_auc_score, precision_score, recall_score, f1_score
import mlflow
import mlflow.pyfunc
from mlflow import log_metric, log_param, log_artifact
from scipy.stats import pearsonr

from autogluon.common.features.types import R_INT,R_FLOAT,R_OBJECT,R_CATEGORY,S_TEXT_AS_CATEGORY 

from autogluon.features.generators import CategoryFeatureGenerator, AsTypeFeatureGenerator, BulkFeatureGenerator, DropUniqueFeatureGenerator, FillNaFeatureGenerator, PipelineFeatureGenerator, OneHotEncoderFeatureGenerator,IdentityFeatureGenerator

from .feature_generator import one_hot_Generator

class AutogluonModel(mlflow.pyfunc.PythonModel):

    def __init__(self, predictor):
        self.predictor = predictor
        
    # def load_context(self, context):
    #     print("Loading context")
    #     # self.predictor = TabularPredictor.load(context.artifacts.get("predictor_path"))
    #     self.predictor = context.artifacts.get("predictor_path")

    def predict(self, model_input):
        return self.predictor.predict(model_input)
    
    def evaluate(self, model_input):
        return self.predictor.evaluate(model_input)
    
    def leaderboard(self, model_input):
        return self.predictor.leaderboard(model_input)




if __name__ == "__main__":
    
    alpha = float(sys.argv[1]) if len(sys.argv) > 1 else 0.5
    quality = sys.argv[2] 
    print(" !!!!!quality:", quality)
    print("!!!!alpha",alpha)


    train_data = TabularDataset('data/train.csv')
    test_data = TabularDataset('data/test.csv')

    # train_data = train_data.iloc[:500]
    # test_data = test_data[:100]
    concatenated_df  = pd.concat([train_data,test_data], axis=0)

    train_feature_generator = PipelineFeatureGenerator(
        generators=[
            [# count_charge_Generator(),
            # net_charge_Generator(),
            one_hot_Generator(verbosity=3,features_in=['seq'],seq_type = "protein"),
            IdentityFeatureGenerator(infer_features_in_args=dict(
            valid_raw_types=[R_INT, R_FLOAT])),
            ],
            
        ],
        verbosity=3,
        post_drop_duplicates=False,
        post_generators=[IdentityFeatureGenerator()]
    )
    one_hot_all_data = train_feature_generator.fit_transform(X=concatenated_df)

    one_hot_train_data = one_hot_all_data[:len(train_data)]
    one_hot_test_data = one_hot_all_data[len(train_data):]

    one_hot_valid_data1 = one_hot_train_data[one_hot_train_data["fold"] ==0.0]
    one_hot_train_data1 = one_hot_train_data[one_hot_train_data["fold"] !=0.0]

    one_hot_train_data1 = one_hot_train_data1.drop(["fold"],axis=1)
    one_hot_valid_data1 = one_hot_valid_data1.drop(["fold"],axis=1)
    one_hot_test_data = one_hot_test_data.drop(["fold"],axis=1)
    print(one_hot_train_data1.shape)
    print(one_hot_valid_data1.shape)
    print(one_hot_test_data.shape)
    print(one_hot_train_data1)
    print(one_hot_test_data)

    with mlflow.start_run() as run:
        predictor = TabularPredictor(label='solubility',eval_metric="precision")
        if quality == "medium_quality":
            print("!!!!!medium quality!!!!!!")
            predictor.fit(train_data=one_hot_train_data1, tuning_data=one_hot_valid_data1, feature_generator=None)
        elif quality == "best_quality":
            print("!!!!!best quality!!!!!!")
            predictor.fit(train_data=one_hot_train_data1, feature_generator=None, presets=quality)
            
        test_eval = predictor.evaluate(one_hot_test_data)
        print("test eval:",test_eval)
        valid_eval = predictor.evaluate(one_hot_valid_data1)
        print("valid eval:",valid_eval)
        
        mlflow.log_metric("test_precision", test_eval["precision"])
        mlflow.log_metric("test_auc", test_eval["roc_auc"])
        mlflow.log_metric("test_balanced_acc", test_eval["accuracy"])
        mlflow.log_metric("test_balanced_acc", test_eval["balanced_accuracy"])
        mlflow.log_metric("test_mcc", test_eval["mcc"])

        mlflow.end_run()
        
        

