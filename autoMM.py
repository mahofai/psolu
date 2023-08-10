import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from autogluon.tabular import TabularDataset, TabularPredictor
from autogluon.core.utils.loaders import load_pd
import pandas as pd

from autogluon.common.features.types import R_INT,R_FLOAT,R_OBJECT,R_CATEGORY,S_TEXT_AS_CATEGORY 

from autogluon.features.generators import CategoryFeatureGenerator, AsTypeFeatureGenerator, BulkFeatureGenerator, DropUniqueFeatureGenerator, FillNaFeatureGenerator, PipelineFeatureGenerator, OneHotEncoderFeatureGenerator,IdentityFeatureGenerator
import sys
from sklearn.model_selection import ParameterGrid
from ray import tune

import argparse
import re
from ray.tune.search.basic_variant import BasicVariantGenerator
from autogluon.multimodal import MultiModalPredictor
from autogluon.common import space
from sklearn.model_selection import train_test_split


#TODO common train 
#TODO HPO
#TODO grid search HPO 
#TODO cross validation

parser = argparse.ArgumentParser(description='download parser')
parser.add_argument('--train_data', type=str, help='path to train data csv')
parser.add_argument('--test_data', type=str, help='path to train data csv')
parser.add_argument('--test_n_fold', type=int, help='choose nth fold as validation set',default = 0)
parser.add_argument('--searcher', type=str, help='grid/bayes/random', default = "")
parser.add_argument('--num_trials', type=int, help='HPO trials number', default = 2)
parser.add_argument('--check_point_name', type=str, help='huggingface_checkpoint', default = "facebook/esm2_t6_8M_UR50D")

parser.add_argument('--lr', type=float, nargs='+', help='learning rate', default = [0.01,0.1])
parser.add_argument('--lr_decay', type=float, nargs='+', help='learning rate decay', default = [0.02,0.2])
parser.add_argument('--weight_decay', type=float, nargs='+', help='weight decay', default = [0.03,0.3])
parser.add_argument('--batch_size', type=float, nargs='+', help='batch size', default = [32,64])
parser.add_argument('--optim_type', type=str, nargs='+', help='adam/adamw/sgd', default = ["adam"])
parser.add_argument('--max_epochs', type=int,  help='max traning epoch', default = 2)

parser.add_argument('--metric', type=str,  help='evaluation metric', default = "roc_auc")

args = parser.parse_args()

def check_sequence_type(sequence):
    print("sequence:", sequence)
    # Pattern to match DNA sequence
    dna_pattern = r"^[ACGTN]+$"

    # Pattern to match protein sequence
    protein_pattern = r"^[ACDEFGHIKLMNPQRSTVWYX]{11,}$"

    # Match the sequence against the patterns
    if re.match(dna_pattern, sequence):
        return "DNA"
    elif re.match(protein_pattern, sequence):
        return "Protein"
    else:
        return "Unknown"

# 检查序列列的类型
def find_sequence_columns(df):
    # 存储符合条件的列名
    sequence_columns = []
    # 迭代遍历数据框的列
    for column in df.columns:
        if df[column].dtype == 'object':
            column_type = df[column].head(50).apply(check_sequence_type).unique()
        # 如果集合中只有一个唯一类型，并且该类型为 "DNA" 或 "Protein"，则将该列添加到结果列表中
            if len(column_type) == 1 and ("DNA" in column_type or "Protein" in column_type):
                sequence_columns.append(column)

    return sequence_columns

if __name__ == "__main__" : 
    
    train_data = pd.read_csv(args.train_data)
    test_data = pd.read_csv(args.test_data)
    train_data = train_data[:500]
    test_data = test_data[:200]
    
    if args.test_n_fold != -1:
        valid_data = train_data[train_data["fold"] == args.test_n_fold]
        train_data = train_data[train_data["fold"] != args.test_n_fold]
        print("args.test_n_fold:",args.test_n_fold)

        train_data = train_data.drop(["fold"],axis=1)
        valid_data = valid_data.drop(["fold"],axis=1)
        # test_data = test_data.drop(["sid"],axis=1)
    else:
        train_data, valid_data = train_test_split(train_data, test_size=0.2, random_state=42)

    seqs_columns = find_sequence_columns(train_data)
    print("seqs columns:", seqs_columns)
    column_types = {}
    for seqs_column in seqs_columns:
        column_types[seqs_column] = "text"
    print("column_types:",column_types)
    
    #grid search
    
    print("!!!!!learning rate:",args.lr)

    custom_hyperparameters={
        "optimization.learning_rate": tune.uniform(args.lr[0], args.lr[-1]),
        "optimization.lr_decay":tune.uniform(args.lr_decay[0], args.lr_decay[-1]),
        "optimization.weight_decay": tune.uniform(args.lr_decay[0], args.lr_decay[-1]),
        "env.batch_size": tune.choice(args.batch_size),
        "optimization.optim_type": tune.choice(args.optim_type),
        'model.hf_text.checkpoint_name': args.check_point_name,
        'optimization.max_epochs': args.max_epochs,
        "env.num_gpus": 1,
    }
    

    grid_paras= {
    "optimization.learning_rate" : args.lr,
    "optimization.lr_decay" : args.lr_decay,
    "optimization.weight_decay" : args.lr_decay,
    "env.batch_size": args.batch_size,
    "optimization.optim_type": args.optim_type,
    }
    
    num_trails = args.num_trials
    hyperparameter_tune_kwargs = {}
    
    if args.searcher == "grid":
        points=[i for i in ParameterGrid(grid_paras)]
        print("points:",points, "len of points:", len(points))
        searcher = BasicVariantGenerator(constant_grid_search=True, points_to_evaluate = points)
        hyperparameter_tune_kwargs["searcher"] = searcher
        hyperparameter_tune_kwargs["scheduler"] = "ASHA"
        hyperparameter_tune_kwargs["num_trials"] = len(points)

    elif args.searcher == "bayes":
        hyperparameter_tune_kwargs["searcher"] = "bayes"
        hyperparameter_tune_kwargs["scheduler"] = "ASHA"
        hyperparameter_tune_kwargs["num_trials"] = args.num_trials

    elif args.searcher == "random":
        hyperparameter_tune_kwargs["searcher"] = "random"
        hyperparameter_tune_kwargs["scheduler"] = "ASHA"
        hyperparameter_tune_kwargs["num_trials"] = args.num_trials
    else:
        print("no searcher. skip hpo")


    print("hyperparameters:",custom_hyperparameters)

    predictor = MultiModalPredictor(label='solubility',eval_metric = args.metric,)
    predictor.fit(train_data = train_data, tuning_data =valid_data,  column_types = column_types,
                hyperparameters=custom_hyperparameters,
                hyperparameter_tune_kwargs = hyperparameter_tune_kwargs
                )
    print("finish")

# python autoMM.py  --train_data /user/mahaohui/autoML/autogluon/autogluon_examples/soluprot/data/train.csv   --test_data /user/mahaohui/autoML/autogluon/autogluon_examples/soluprot/data/test.csv  --test_n_fold 1 
