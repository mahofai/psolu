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

import mlflow
import mlflow.pyfunc
from mlflow import log_metric, log_param, log_artifact, log_text

import argparse
import re
from ray.tune.search.basic_variant import BasicVariantGenerator
from autogluon.multimodal import MultiModalPredictor
from autogluon.common import space
from sklearn.model_selection import train_test_split

#TODO cross validation
 
parser = argparse.ArgumentParser(description='argument parser')
# complusory settings
parser.add_argument('--target_column', type=str, help='prediction target column', default = "solubility")
parser.add_argument('--metric', type=str,  help='evaluation metric', default = "roc_auc")
parser.add_argument('--train_data', type=str, help='path to train data csv', default = "./data/soluprot/train.csv")
parser.add_argument('--test_data', type=str, help='path to train data csv', default = "./data/soluprot/test.csv")


# HPO settings
parser.add_argument('--mode', type=str, help='HPO bayes preset', choices = ["medium_quality", "best_quality","manual"], default = "manual")
parser.add_argument('--test_n_fold', type=int, help='choose nth fold as validation set',default = 0)
parser.add_argument('--searcher', type=str, help='grid/bayes/random', default = "")
parser.add_argument('--num_trials', type=int, help='HPO trials number', default = 2)
parser.add_argument('--check_point_name', type=str, help='huggingface_checkpoint', default = "facebook/esm2_t6_8M_UR50D")
parser.add_argument('--max_epochs', type=int,  help='max traning epoch', default = 2)

# parameters settings
parser.add_argument('--lr', type=lambda x: [float(i) for i in x.split()], default = [1e-6,0.1])
parser.add_argument('--lr_decay', type=lambda x: [float(i) for i in x.split()], help='learning rate decay', default = [2e-6,0.2])
parser.add_argument('--weight_decay', type=lambda x: [float(i) for i in x.split()], help='weight decay', default = [3e-6,0.3])
parser.add_argument('--batch_size', type=lambda x: [int(i) for i in x.split()], help='batch size', default = [32])
parser.add_argument('--optim_type', type=lambda x: [str(i) for i in x.split()], help='adam/adamw/sgd', default = ["adam"])
parser.add_argument('--lr_schedule', type=lambda x: [str(i) for i in x.split()], help='cosine_decay/polynomial_decay/linear_decay', default = ["linear_decay"])

args = parser.parse_args()

class AutogluonModel(mlflow.pyfunc.PythonModel):

    def __init__(self, predictor):
        self.predictor = predictor
        
    # def load_context(self, context):
    #     print("Loading context")
    #     # self.predictor = TabularPredictor.load(context.artifacts.get("predictor_path"))
    #     self.predictor = context.artifacts.get("predictor_path")
    
    def predict(self, model_input):
        return self.predictor.predict(model_input)
    
    def evaluate(self, model_input, metrics):
        return self.predictor.evaluate(model_input,metrics=metrics)
    
    def leaderboard(self, model_input):
        return self.predictor.leaderboard(model_input)

def check_sequence_type(sequence):

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

def find_sequence_columns(df):
    sequence_columns = []
    for column in df.columns:
        if df[column].dtype == 'object':
            column_type = df[column].head(50).apply(check_sequence_type).unique()
            if len(column_type) == 1 and ("DNA" in column_type or "Protein" in column_type):
                sequence_columns.append(column)

    return sequence_columns

if __name__ == "__main__" : 
    
    train_data = pd.read_csv(args.train_data)
    test_data = pd.read_csv(args.test_data)
    train_data = train_data[:200]
    test_data = test_data[:200]
    print("!!!args.lr:",args.lr)
        
    if args.test_n_fold != -1:
        valid_data = train_data[train_data["fold"] == args.test_n_fold]
        train_data = train_data[train_data["fold"] != args.test_n_fold]
        print("args.test_n_fold:",args.test_n_fold)

        train_data = train_data.drop(["fold"],axis=1)
        valid_data = valid_data.drop(["fold"],axis=1)

    else:
        train_data, valid_data = train_test_split(train_data, test_size=0.2, random_state=42)
        
    print("train_data:",train_data)
    print("valid_data:",valid_data)
    seqs_columns = find_sequence_columns(train_data)
    
    print("seqs columns:", seqs_columns)
    column_types = {}
    for seqs_column in seqs_columns:
        column_types[seqs_column] = "text"
    print("column_types:",column_types)
    
    custom_hyperparameters={
        "optimization.learning_rate": tune.uniform(args.lr[0], args.lr[-1]),
        "optimization.lr_decay":tune.uniform(args.lr_decay[0], args.lr_decay[-1]),
        "optimization.weight_decay": tune.uniform(args.lr_decay[0], args.lr_decay[-1]),
        "env.batch_size": tune.choice(args.batch_size),
        "optimization.optim_type": tune.choice(args.optim_type),
        'model.hf_text.checkpoint_name': args.check_point_name,
        'optimization.max_epochs': args.max_epochs,
        "optimization.lr_schedule":tune.choice(args.lr_schedule),
        "env.num_gpus": 1,
    }

    grid_paras= {
    "optimization.learning_rate" : args.lr,
    "optimization.lr_decay" : args.lr_decay,
    "optimization.weight_decay" : args.lr_decay,
    "env.batch_size": args.batch_size,
    "optimization.optim_type": args.optim_type,
    "optimization.lr_schedule":args.lr_schedule
    }
    
    num_trails = args.num_trials
    hyperparameter_tune_kwargs = {}
    
    if args.mode == "manual":
        print("manual !!!")
        if args.searcher == "grid":
            print("grid search !!!")
            points=[i for i in ParameterGrid(grid_paras)]
            searcher = BasicVariantGenerator(constant_grid_search=True, points_to_evaluate = points)
            hyperparameter_tune_kwargs["searcher"] = searcher
            hyperparameter_tune_kwargs["scheduler"] = "ASHA"
            hyperparameter_tune_kwargs["num_trials"] = len(points)

        elif args.searcher == "bayes":
            print("bayes search !!!")
            hyperparameter_tune_kwargs["searcher"] = "bayes"
            hyperparameter_tune_kwargs["scheduler"] = "ASHA"
            hyperparameter_tune_kwargs["num_trials"] = args.num_trials

        elif args.searcher == "random":
            print("random search !!!")
            hyperparameter_tune_kwargs["searcher"] = "random"
            hyperparameter_tune_kwargs["scheduler"] = "ASHA"
            hyperparameter_tune_kwargs["num_trials"] = args.num_trials
        else:
            print("no searcher. skip hpo")
            custom_hyperparameters={
                        'model.hf_text.checkpoint_name': args.check_point_name,
                        'optimization.max_epochs': args.max_epochs,
                        "env.num_gpus": 1,
                    }
    else:
        print("HPO preset !!!")
        custom_hyperparameters={
            "optimization.learning_rate": tune.uniform(1e-5, 0.1),
            "env.batch_size": tune.choice([16,32,64,128,256,512,1024,2048]),
            "optimization.optim_type": tune.choice(["adam"]),
            'model.hf_text.checkpoint_name': args.check_point_name,
            'optimization.max_epochs': 2,
            "env.num_gpus": 1,
        }
        hyperparameter_tune_kwargs["searcher"] = "bayes"
        hyperparameter_tune_kwargs["scheduler"] = "ASHA"
        if args.mode == "medium_quality":
            hyperparameter_tune_kwargs["num_trials"] = 2    
            print("medium quality!!!")
        elif  args.mode == "best_quality":

            custom_hyperparameters["optimization.lr_decay"] = tune.uniform(1e-5, 0.1)
            custom_hyperparameters["optimization.weight_decay"] = tune.uniform(1e-5, 0.1)
            hyperparameter_tune_kwargs["num_trials"] = 4
            print("best quality!!!")
        else:
            print("please choose medium or best quality!!!")

    with mlflow.start_run() as run:
        predictor = MultiModalPredictor(label=args.target_column,eval_metric = args.metric)
        predictor.set_verbosity(4)
        predictor.fit(train_data = train_data, tuning_data =valid_data,
                    column_types = column_types,
                    hyperparameters=custom_hyperparameters,
                    hyperparameter_tune_kwargs = hyperparameter_tune_kwargs
                    )
        
        model = AutogluonModel(predictor)
        
        model_info = mlflow.pyfunc.log_model(
            artifact_path='1ag-model', python_model=model
        )
        print("model_info.model_uri:",model_info.model_uri)

        model = mlflow.pyfunc.load_model(model_uri=model_info.model_uri).unwrap_python_model()
        
        eval_metrics = []
        print("model.predictor.problem_type!!!!:",model.predictor.problem_type)
        if model.predictor.problem_type == "binary" or model.predictor.problem_type == "multiclass":

            eval_metrics=["balanced_accuracy","precision","mcc","f1","recall"]
        elif model.predictor.problem_type == "regression":
            eval_metrics = ["mae","rmse"]

        if args.metric not in eval_metrics:
            eval_metrics.append(args.metric)

        test_metrics = model.evaluate(test_data, metrics=eval_metrics)

        print("test eval!!!!:",test_metrics)

        
        valid_metrics = model.evaluate(valid_data, metrics=eval_metrics) 
        print("valid eval:",valid_metrics)
        
        for k,v in valid_metrics.items():
            log_metric('valid_'+k, v)
        for k,v in test_metrics.items():
            log_metric('test_'+k, v)
        
        # mlflow.log_metric("test_precision", test_metrics["precision"]) # type: ignore
        # mlflow.log_metric("test_auc", test_metrics["roc_auc"]) # type: ignore
        # mlflow.log_metric("test_balanced_acc", test_metrics["accuracy"]) # type: ignore
        # mlflow.log_metric("test_balanced_acc", test_metrics["balanced_accuracy"]) # type: ignore
        # mlflow.log_metric("test_mcc", test_metrics["mcc"]) # type: ignore

# python autoMM.py  --train_data /user/mahaohui/autoML/autogluon/autogluon_examples/soluprot/data/train.csv   --test_data /user/mahaohui/autoML/autogluon/autogluon_examples/soluprot/data/test.csv  --test_n_fold 1 



