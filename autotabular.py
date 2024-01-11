import os
import tqdm
import numpy as np
import json
from mlflow.models import ModelSignature
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from autogluon.tabular import TabularDataset, TabularPredictor
from autogluon.core.utils.loaders import load_pd
import pandas as pd

from autogluon.common.features.types import R_INT,R_FLOAT,R_OBJECT,R_CATEGORY,S_TEXT_AS_CATEGORY 

from autogluon.features.generators import CategoryFeatureGenerator, AsTypeFeatureGenerator, BulkFeatureGenerator, DropUniqueFeatureGenerator, FillNaFeatureGenerator, PipelineFeatureGenerator, OneHotEncoderFeatureGenerator,IdentityFeatureGenerator
from feature_generator import count_charge_Generator, net_charge_Generator, one_hot_Generator
import sys
from sklearn.model_selection import ParameterGrid
from ray import tune


import mlflow
import mlflow.pyfunc
from mlflow import log_metric, log_param, log_artifact, log_text
from sklearn.model_selection import train_test_split
import argparse
import re
from ray.tune.search.basic_variant import BasicVariantGenerator
from autogluon.multimodal import MultiModalPredictor
from autogluon.common import space
from sklearn.model_selection import train_test_split

from zipfile import ZipFile
import os
from propythia.dna.descriptors import DNADescriptor
from propythia.protein.descriptors import ProteinDescritors
from propythia.dna.calculate_features import calculate_and_normalize
from propythia.dna.sequence import ReadDNA

model_mapping = {
    'random_forrest': 'RF',
    'extra_tree': 'XT',
    "XGBoost": 'XGB',  # Add other models as needed
    # Add more mappings as needed
}
 
parser = argparse.ArgumentParser(description='argument parser')

# complusory settings
parser.add_argument('--target_column', type=str, help='prediction target column', default = "solubility")
parser.add_argument('--metric', type=str,  help='evaluation metric', default = "precision")
# parser.add_argument('--train_data', type=str, help='path to train data csv', default = "data/dna_activity_train.csv")
# parser.add_argument('--valid_data', type=str, help='path to valid data csv', default = "data/dna_activity_valid.csv")
# parser.add_argument('--test_data', type=str, help='path to test data csv', default = "data/soluprot_test.csv")
parser.add_argument('--data', type=str, help='path to test data csv', default = "data/filtered_soluprot.csv")
# HPO settings
parser.add_argument('--mode', type=str, help='HPO bayes preset', choices = ["medium_quality", "best_quality","manual"], default = "manual")
parser.add_argument('--searcher', type=str, help='grid/bayes/random', default = "bayes")
parser.add_argument('--num_trials', type=int, help='HPO trials number', default = 3)
parser.add_argument('--checkpoint_name', type=str, help='huggingface_checkpoint',default="model/esm2_8m")
parser.add_argument('--max_epochs', type=int,  help='max traning epoch', default = 3)

# huggingface model parameters settings
parser.add_argument('--lr', type=lambda x: [float(i) for i in x.split(",")], default = [1e-6,1e-3,0.1])
parser.add_argument('--lr_decay', type=lambda x: [float(i) for i in x.split(",")], help='learning rate decay', default = [2e-6])
parser.add_argument('--weight_decay', type=lambda x: [float(i) for i in x.split(",")], help='weight decay', default = [3e-6])
parser.add_argument('--batch_size', type=lambda x: [int(i) for i in x.split(",")], help='batch size', default = [32])
parser.add_argument('--optim_type', type=lambda x: [str(i) for i in x.split(",")], help='adam/adamw/sgd', default = ["adam"])
parser.add_argument('--lr_schedule', type=lambda x: [str(i) for i in x.split(",")], help='cosine_decay/polynomial_decay/linear_decay', default = ["linear_decay"])



# tabualr settings
parser.add_argument('--tabular', type=int, help='tabular predictor', default = 1)
parser.add_argument('--auto_features', type=int, help='generate some default features by sequences', default = 1)
# parser.add_argument('--tabular_model', type=str, help='tabular predictor model', default = "XGB")
parser.add_argument('--tabular_model', 
                    choices=model_mapping.values(), 
                    type=lambda s: model_mapping.get(s, s), 
                    help='tabular predictor model (e.g., random forest, XGB)',
                    default="random_forrest")

# RF/XT HPO settings
parser.add_argument('--n_estimators', type=lambda x: [int(i) for i in x.split(",")], help='The number of trees in the forest.', default = [100,500])
parser.add_argument('--criterion', choices=["gini", "entropy", "log_loss"], help='The function to measure the quality of a split', default = "gini")
parser.add_argument('--max_depth', type=lambda x: [int(i) for i in x.split(",")], help='The maximum depth of the tree', default = [100,10])
parser.add_argument('--min_samples_split', type=lambda x: [int(i) for i in x.split(",")], help='The minimum number of samples required to split an internal node', default = [2,5])
parser.add_argument('--min_samples_leaf', type=lambda x: [int(i) for i in x.split(",")], help='The minimum number of samples required to be at a leaf node', default = [1,10])

#XGB/CAT/LGB
parser.add_argument('--subsample', type=lambda x: [int(i) for i in x.split(",")], help='subsample.', default = [0.6,0.8])
parser.add_argument('--reg_lambda', type=lambda x: [float(i) for i in x.split(",")], help='reg_lambda.', default = [0.0000001,0.0001])

#KNN
parser.add_argument('--power', type=int, help='1:manhattan_distance , 2:euclidean_distance', default = 2)
parser.add_argument('--n_neighbors', type=lambda x: [int(i) for i in x.split(",")], help='subsample.', default = [0])
parser.add_argument('--leaf_size', type=lambda x: [int(i) for i in x.split(",")], help='subsample.', default = [0])
#LR
parser.add_argument('--tol', type=lambda x: [float(i) for i in x.split(",")], help='1:manhattan_distance , 2:euclidean_distance', default = [0.0001,0.001])
parser.add_argument('--C', type=lambda x: [float(i) for i in x.split(",")], help='1:manhattan_distance , 2:euclidean_distance', default = [1.0])
args = parser.parse_args()

class autogluonPyModel(mlflow.pyfunc.PythonModel):

    # def __init__(self, predictor, ps_featurize, signature):
    #     self.predictor = predictor
    #     self.ps_featurize = ps_featurize
    #     self.class autogluonPyModel(mlflow.pyfunc.PythonModel):

    def __init__(self, predictor, signature=""):

        self.predictor = predictor
        # self.ps_featurize = ps_featurize
        # self.signature = signature
        # self.input_names, self.output_names = signature.inputs.input_names(), signature.outputs.input_names()
        print("PWD:",os.getcwd())
        print("self.predictor.path:",self.predictor.path)
        os.system(f"cp -r ../ml/model/code/{self.predictor.path} .")
        print("ls:",os.system("ls"))
        
    def evaluate(self,  model_input, metrics=[]):
        if  len(metrics) > 1:
            return self.predictor.evaluate(model_input,metrics=metrics)
        else:
            return self.predictor.evaluate(model_input)
    
    def predict(self, context, model_input):
        print("context:", context)
        '''
        context:
            a instance of PythonModelContext
            PythonModelContext对象由save_model() 和log_model()持久化方法隐式创建, 使用这些方法的artifacts参数指定的内容
        model_input:
            if request from flask, it will be a dataframe format
            model_input: [pandas.DataFrame, numpy.ndarray, scipy.sparse.(csc.csc_matrix | csr.csr_matrix),
            
        return 
            -> [numpy.ndarray | pandas.(Series | DataFrame) | List]
        '''

        print("model_input:", model_input)
        outputs = self.predictor.predict(model_input)
        print("outputs:",outputs)
        return outputs

    # def featurize_mlflow(self, row):
    #     seq = row['seq']
    #     data_dict = {}
    #     # data = self.ps_featurize.featurize(seq)
    #     data_dict['input'] = seq
    #     return data_dict

inp = json.dumps([{'name': 'seq','type':'string'}])
oup = json.dumps([{'name': 'score','type':'double'}])
signature = ModelSignature.from_dict({'inputs': inp, 'outputs': oup})

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
    dna_columns = []
    protein_columns = []
    for column in df.columns:
        if df[column].dtype == 'object':
            column_type = df[column].head(50).apply(check_sequence_type).unique()
            if len(column_type) == 1 and ("DNA" in column_type):
                dna_columns.append(column)
            elif  len(column_type) == 1 and ("Protein" in column_type):
                protein_columns.append(column)
    sequence_columns = {"dna_columns":dna_columns, "protein_columns":protein_columns}
    return sequence_columns


def split_data(df):
    if "split" in df.columns and "test" in df["split"].values :
        train_data = df[df["split"]=="train"]
        valid_data = df[df["split"]=="valid"]
        test_data = df[df["split"]=="test"]
    else:
        train_data,valid_data = train_test_split(df, test_size=0.2)
        test_data,valid_data = train_test_split(df, test_size=0.5)
    return train_data, valid_data, test_data
    
def add_protein_features(df, seq_column):
    descriptors_df = ProteinDescritors(dataset= df , col= seq_column)
    output_dataframe = descriptors_df.dataset
    processed_dataframe = pd.DataFrame()
    processed_dataframe = descriptors_df.get_all_physicochemical()
    output_dataframe = processed_dataframe.merge(output_dataframe, on=seq_column, how='inner') 
    filtered_columns = [col for col in output_dataframe.columns if "_y" not in col]
    output_dataframe = output_dataframe[filtered_columns]
    mapping_function = lambda col: col.replace("_x", "")

    # Use the rename method to apply the mapping function to all columns
    output_dataframe.rename(columns=mapping_function, inplace=True)
    
    # processed_df = descriptors_df.get_all_aac()
    # df = df.merge(processed_df, on=seq_column, how='inner') 
    # descriptors_df = ProteinDescritors(dataset= df , col= seq_column)
    # processed_df = descriptors_df.get_all_correlation()
    # df = df.merge(processed_df, on=seq_column, how='inner')
    # df = df.loc[:, ~df.columns.duplicated()]
    return output_dataframe

def add_dna_features(df, seq_column):
    df = df[~df[seq_column].str.contains('N')]
    features=['nucleic_acid_composition','dinucleotide_composition','gc_content','DAC']
    df = df.rename(columns={seq_column: "sequence"})
    print(df)
    fps_x = pd.DataFrame()
    fps_x, fps_y = calculate_and_normalize(data=df, descriptor_list=features)
    print(fps_x)
    # descriptors_df = ProteinDescritors(dataset= df , col= seqs_columns[0])
    df = df.rename(columns={"sequence": seq_column})
    output_dataframe = pd.concat([df, fps_x], axis=1) # type: ignore
    output_dataframe = output_dataframe.loc[:, ~output_dataframe.columns.duplicated()]
    return output_dataframe
     
def auto_features(df):
    
        sequence_columns = find_sequence_columns(df)
        if len(sequence_columns["protein_columns"]) > 1 or len(sequence_columns["dna_columns"]) > 1:
            print("autofeature can only deal with one protein/dna column !!!")
            exit()
        for prot_column in sequence_columns["protein_columns"]:
            df = df[~df[prot_column].str.contains('X')]
            print("!!protein_column:",prot_column)
            df = add_protein_features(df, prot_column)
            print(df)
            
        for dna_column in sequence_columns["dna_columns"]:
            print("!!dna_column:",dna_column)
            df = df[~df[dna_column].str.contains('N')]
            df = add_dna_features(df,dna_column)
            print(df)
        if 'solubility_x' in df.columns:
            df = df.rename(columns={'solubility_x': 'solubility'})
            
        df = df.dropna()
        return df

def get_best_model(predictor,testing_data):
    leaderboard_hpo = predictor.leaderboard(testing_data, silent=True)
    print("leade_board:", leaderboard_hpo)
    best_model_name = leaderboard_hpo[leaderboard_hpo['stack_level'] == 1]['model'].iloc[0]
    if best_model_name == "WeightedEnsemble_L2":
        best_model_name = leaderboard_hpo[leaderboard_hpo['stack_level'] == 1]['model'].iloc[1]
    
    predictor_info = predictor.info()
    best_model_info = predictor_info['model_info'][best_model_name]

    print(f'Best Model Hyperparameters ({best_model_name}):')
    print(best_model_info['hyperparameters'])
    print("best_model_info:",best_model_info)
    predictor.delete_models(models_to_keep=best_model_name,dry_run=False)
    
    return predictor

from autogluon.features.generators import PipelineFeatureGenerator, CategoryFeatureGenerator, IdentityFeatureGenerator, AutoMLPipelineFeatureGenerator

if __name__ == "__main__" : 
    current_file_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_file_dir)
    
    df = pd.read_csv(f'{parent_dir}/{args.data}')
    
    # auto feature engineering
    if args.auto_features and args.tabular:
        df = auto_features(df)
        np.set_printoptions(threshold=np.inf) # type: ignore
        print("auto featrues.columns:",df.columns[:20])
        print("auto featrues:",df)
    
    train_data, valid_data, test_data = split_data(df)
    #keep unique or auplicated values
    mypipeline = AutoMLPipelineFeatureGenerator(post_drop_duplicates=False,pre_drop_useless=False)
    
        
    print("train_data:",train_data)
    print("valid_data:",valid_data)
    print("test_data:",test_data)

    grid_paras= {
    "optimization.learning_rate" : args.lr,
    "optimization.lr_decay" : args.lr_decay,
    "optimization.weight_decay" : args.weight_decay,
    "env.batch_size": args.batch_size,
    "optimization.optim_type": args.optim_type,
    "optimization.lr_schedule":args.lr_schedule
    }
    
    rf_grid_paras= {
    "n_estimators" : args.n_estimators,
    "max_depth" : args.max_depth,
    "min_samples_leaf" : args.min_samples_leaf,
    }
    
    custom_hyperparameters={
        "optimization.learning_rate": tune.uniform(args.lr[0], args.lr[-1]),
        "optimization.lr_decay":tune.uniform(args.lr_decay[0], args.lr_decay[-1]),
        "optimization.weight_decay": tune.uniform(args.lr_decay[0], args.lr_decay[-1]),
        "env.batch_size": tune.choice(args.batch_size),
        "optimization.optim_type": tune.choice(args.optim_type),
        'model.hf_text.checkpoint_name': f'{parent_dir}/{args.checkpoint_name}',
        'optimization.max_epochs': args.max_epochs,
        "optimization.lr_schedule":tune.choice(args.lr_schedule),
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
            custom_hyperparameters={
                "optimization.learning_rate": tune.uniform(args.lr[0], args.lr[-1]),
                "optimization.lr_decay":tune.uniform(args.lr_decay[0], args.lr_decay[-1]),
                "optimization.weight_decay": tune.uniform(args.lr_decay[0], args.lr_decay[-1]),
                "env.batch_size": tune.choice(args.batch_size),
                "optimization.optim_type": tune.choice(args.optim_type),
                'model.hf_text.checkpoint_name': f'{parent_dir}/{args.checkpoint_name}',
                'optimization.max_epochs': args.max_epochs,
                "optimization.lr_schedule":tune.choice(args.lr_schedule),
            }
        elif args.searcher == "random":
            print("random search !!!")
            hyperparameter_tune_kwargs["searcher"] = "random"
            hyperparameter_tune_kwargs["scheduler"] = "ASHA"
            hyperparameter_tune_kwargs["num_trials"] = args.num_trials
            hyperparameter_tune_kwargs["metric"] = args.num_trials
        else:
            print("no searcher. skip hpo")
            custom_hyperparameters={
                        'model.hf_text.checkpoint_name': f'{parent_dir}/{args.checkpoint_name}',
                        'optimization.max_epochs': args.max_epochs,
                        "optimization.learning_rate" : args.lr[0],
                        "optimization.lr_decay" : args.lr_decay[0],
                        "optimization.weight_decay" : args.lr_decay[0],
                        "env.batch_size": args.batch_size[0],
                        "optimization.optim_type": args.optim_type[0],
                        "optimization.lr_schedule":args.lr_schedule[0]
                    }
    else:
        print("HPO preset !!!")
        custom_hyperparameters={
            "optimization.learning_rate": tune.uniform(1e-5, 0.1),
            "env.batch_size": tune.choice([16,32,64,128,256,512,1024,2048]),
            "optimization.optim_type": tune.choice(["adam"]),
            'model.hf_text.checkpoint_name': f'{parent_dir}/{args.checkpoint_name}',
            'optimization.max_epochs': args.max_epochs,
        }
        hyperparameter_tune_kwargs["searcher"] = "bayes"
        hyperparameter_tune_kwargs["scheduler"] = "ASHA"
        if args.mode == "medium_quality":
            hyperparameter_tune_kwargs["num_trials"] = 5
            print("medium quality!!!")
        elif  args.mode == "best_quality":
            custom_hyperparameters["optimization.lr_decay"] = tune.uniform(1e-5, 0.1)
            custom_hyperparameters["optimization.weight_decay"] = tune.uniform(1e-5, 0.1)
            hyperparameter_tune_kwargs["num_trials"] = 50
            print("best quality!!!")
        else:
            print("please choose medium or best quality!!!")
            
            
    if args.tabular!=0:
        options = {}
        if args.tabular_model == "RF" or args.tabular_model == "XT":
            options = {
                'n_estimators': space.Int(args.n_estimators[0], args.n_estimators[-1], default= (args.n_estimators[0] + args.n_estimators[-1])/2),
                'criterion': args.criterion,
                'max_depth': space.Int(args.max_depth[0], args.max_depth[-1], default=int((args.max_depth[0] + args.max_depth[-1])/2)),
                'min_samples_leaf': space.Int(args.min_samples_leaf[0], args.min_samples_leaf[-1], default=int((args.min_samples_leaf[0] + args.min_samples_leaf[-1])/2)),
            }
        elif args.tabular_model == "GBM" or   args.tabular_model == "XGB":
            options = {
                'learning_rate': space.Real(args.lr[0], args.lr[-1], default=float((args.lr[0] + args.lr[-1])/2)),
                # 'max_depth': space.Int(args.max_depth[0], args.max_depth[-1], default=int((args.max_depth[0] + args.max_depth[-1])/2)),
                'subsample': space.Real(args.subsample[0], args.subsample[-1], default=float((args.subsample[0] + args.subsample[-1])/2)),
                'reg_lambda': space.Real(args.reg_lambda[0], args.reg_lambda[-1], default=float((args.reg_lambda[0] + args.reg_lambda[-1])/2)),
            }
        elif args.tabular_model == "CAT":
            options = {
                'learning_rate': space.Real(args.lr[0], args.lr[-1], default=float((args.lr[0] + args.lr[-1])/2)),
                # 'max_depth': space.Int(args.max_depth[0], args.max_depth[-1], default=int((args.max_depth[0] + args.max_depth[-1])/2)),
                'subsample': space.Real(args.subsample[0], args.subsample[-1], default=float((args.subsample[0] + args.subsample[-1])/2)),
                'l2_leaf_reg': space.Real(args.reg_lambda[0], args.reg_lambda[-1], default=float((args.reg_lambda[0] + args.reg_lambda[-1])/2)),
            }
        elif  args.tabular_model == "KNN":
            options = {
                'leaf_size': space.Int(args.max_depth[0], args.max_depth[-1], default=int((args.max_depth[0] + args.max_depth[-1])/2)),
                'p': space.Categorical(i for i in args.power),
                'n_neighbors': space.Int(args.max_depth[0], args.max_depth[-1], default=int((args.max_depth[0] + args.max_depth[-1])/2)),
            }
        elif args.tabular_model == "LR":
            options = {
                'C': space.Real(args.lr[0], args.lr[-1], default=float((args.lr[0] + args.lr[-1])/2)),
                'tol': space.Real(args.lr[0], args.lr[-1], default=float((args.lr[0] + args.lr[-1])/2)),
            }
        if args.tabular:
            custom_hyperparameters = {f'{args.tabular_model}': options}
            hyperparameter_tune_kwargs["scheduler"] = 'local'
            hyperparameter_tune_kwargs["searcher"] = args.searcher
            hyperparameter_tune_kwargs["num_trials"] = args.num_trials
            if args.searcher == "grid":
                points=[i for i in ParameterGrid(rf_grid_paras)]
                print("grid points:",points)
                custom_hyperparameters = {f'{args.tabular_model}': points}
                hyperparameter_tune_kwargs = {}
                # searcher = BasicVariantGenerator(constant_grid_search=True, points_to_evaluate = points)
                # hyperparameter_tune_kwargs["searcher"] = "random"
            
            
            

        # hyperparameter_tune_kwargs = {  # HPO is not performed unless hyperparameter_tune_kwargs is specified
        #     'num_trials': args.num_trials,
        #     'scheduler' : 'local',
        #     'searcher': args.searcher,
        # }  # Refer to TabularPredictor.fit docstring for all valid values

    with mlflow.start_run() as run:
        current_dir = os.getcwd()
        tabular_save_path = "automl_tabular_model"
        automm_save_path = "automl_automm_model"
        save_path = "model"
        if args.tabular:
            save_path = tabular_save_path
            print("feature engineering processing!!!")
            predictor = TabularPredictor(
                label=args.target_column,
                eval_metric=args.metric,
                path=save_path,
            )
            if args.mode != "manual" :
                print("!!!default parameter")
                predictor.fit(train_data = train_data, tuning_data=valid_data, presets=args.mode, feature_generator=mypipeline,)
            elif args.searcher :
                predictor.fit(
                    train_data = train_data, 
                    tuning_data=valid_data, 
                    feature_generator=mypipeline,
                    hyperparameters=custom_hyperparameters,
                    hyperparameter_tune_kwargs=hyperparameter_tune_kwargs,
                    )
                predictor = get_best_model(predictor,testing_data=test_data)
            else:
                predictor.fit(train_data = train_data, tuning_data=valid_data, feature_generator=mypipeline,)

        else:
            save_path = tabular_save_path
            seqs_columns = find_sequence_columns(train_data)
            print("seqs columns:", seqs_columns)
            column_types = {}
            for seqs_column in seqs_columns:
                column_types[seqs_column] = "text"
            print("column_types:",column_types)
            predictor = MultiModalPredictor(label=args.target_column,eval_metric = args.metric, path=save_path)
            predictor.set_verbosity(4)
            predictor.fit(train_data = train_data, 
                        tuning_data =valid_data,
                        column_types = column_types,
                        hyperparameters=custom_hyperparameters,
                        hyperparameter_tune_kwargs = hyperparameter_tune_kwargs
                        )

        # model = AutogluonModel(predictor)
        model=autogluonPyModel(predictor = predictor)
        
        # os.system("mv ./code/automl_model ./")
        model_info = mlflow.pyfunc.log_model(
            artifact_path='model', python_model=model,
            registered_model_name="model",
            input_example=train_data.iloc[[-1]],
            code_path=[f'{current_dir}/{save_path}']
        )
        # model = mlflow.pyfunc.load_model(model_uri=model_info.model_uri).unwrap_python_model()

        eval_metrics = []
        print("model.predictor.problem_type!!!!:",model.predictor.problem_type)
        if model.predictor.problem_type == "binary":
            eval_metrics=["accuracy","balanced_accuracy","precision","mcc","f1","precision","log_loss","roc_auc"]
        elif model.predictor.problem_type == "regression":
            eval_metrics = ["mae","rmse","r2","mse"]
        elif model.predictor.problem_type == "multiclass":
            eval_metrics=["accuracy","balanced_accuracy","mcc","f1","recall", "log_loss"]

        if args.metric not in eval_metrics:
            eval_metrics.append(args.metric)
            
        # test_data = pd.read_csv("/user/mahaohui/autoML/git/psolu/test.csv")
        # if args.tabular:
        #     valid_metrics = model.evaluate(test_data) 
        #     print("valid eval:",valid_metrics)
        # else:
        #     valid_metrics = model.evaluate(model_input=test_data, metrics=eval_metrics) 
        #     print("valid eval:",valid_metrics)
        
        # for k,v in valid_metrics.items():
        #     log_metric('valid_'+k, v)



