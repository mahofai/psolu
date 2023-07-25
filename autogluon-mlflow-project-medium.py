import pandas as pd
import os
import torch
from autogluon.tabular import TabularDataset, TabularPredictor
import numpy as np

from autogluon.common.features.types import R_INT,R_FLOAT,R_OBJECT,R_CATEGORY,S_TEXT_AS_CATEGORY 

from autogluon.features.generators import CategoryFeatureGenerator, AsTypeFeatureGenerator, BulkFeatureGenerator, DropUniqueFeatureGenerator, FillNaFeatureGenerator, PipelineFeatureGenerator, OneHotEncoderFeatureGenerator,IdentityFeatureGenerator
# import sys
# sys.path.append('/user/mahaohui/autoML/autogluon_examples')
from feature_generator import count_charge_Generator, net_charge_Generator, one_hot_Generator

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

# custom_hyperparameters = {'NN_TORCH': {}}
# hyperparameters=custom_hyperparameters, 

# predictor = TabularPredictor(label='solubility',eval_metric="precision")
# predictor = TabularPredictor(label='solubility',eval_metric="roc_auc")
# predictor = TabularPredictor(label='solubility',eval_metric="balanced_accuracy")
predictor = TabularPredictor(label='solubility',eval_metric="precision")


# tuning_data=one_hot_valid_data1,
predictor.fit(train_data=one_hot_train_data1, tuning_data=one_hot_valid_data1,  feature_generator=None,presets="medium_quality")
# predictor.fit(train_data=one_hot_train_data1, tuning_data=one_hot_valid_data1,  feature_generator=None, presets='medium_quality')
valid_eva = predictor.evaluate(one_hot_valid_data1, silent=True)
print("valid_eva:",valid_eva)

test_eva = predictor.evaluate(one_hot_test_data, silent=True)
print("test_eva:",test_eva)


