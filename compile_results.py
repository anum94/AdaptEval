import os
import json
import pandas as pd
from statistics import geometric_mean
import time
from metrics.rouge import Rouge
from metrics.bertscore import BertScore
from metrics.vocab_overlap import VocabOverlap
from typing import List
from config import config
# read the exisitng prediction


def compute_geometric_mean(metrics_dict):
    rouge1 = metrics_dict["rouge1"]
    rouge2 = metrics_dict["rouge2"]
    rougeL = metrics_dict["rougeL"]

    average_rouge = (rouge1+rouge2+rougeL) / 3

    return average_rouge, rouge1, rouge2, rougeL


def compute_da_pairs(runs: List):
    df = pd.DataFrame(columns=['model', 'ds', 'task', 'run_id', 'f_name'])
    for i, run in enumerate(runs):
        model = run.split('/')[1].split('_')[2]
        ds = run.split('/')[1].split('_')[4]
        task = run.split('/')[1].split('_')[5]
        run_id = run.split('/')[1].split('_')[0]
        df.loc[i] = [model,ds,task,run_id,run]


def compute_automatic_evaluation(root_folder: str,
                                 out_file : str,
                                 selected_metrics: List = ['rouge', 'bertscore', 'vocab_overlap'],
                                 dict_domain_articles= None,
                                 dict_domain_vocab = None
                                 ):
    rouge = Rouge()
    bertscore = BertScore()
    vocaboverlap = VocabOverlap()

    all_files = os.listdir(root_folder)
    csv_files = list(filter(lambda f: f.endswith('.csv'), all_files))
    runs = [os.path.join(root_folder, folder_name) for folder_name in csv_files]

    for run in runs:

        try:
            eval_scores = dict()
            print (run)
            df = pd.read_csv(run)
            predictions = df['prediction']
            references = df['reference']

            ds = run.split('/')[-1].split('_')[4]

            if 'rouge' in selected_metrics:
                print(f"calculating rouge for {run}")
                eval_scores['rouge'] = rouge.compute(predictions=predictions, references=references)

            if 'bertscore' in selected_metrics:
                print(f"calculating bertscore for {run}")
                eval_scores['bertscore'] = bertscore.compute(predictions=predictions, references=references)

            if 'vocab_overlap' in selected_metrics:
                print (os.getcwd())
                print("Calculating vocbulary overlap of summary to the domain vocabulary.")

                # read the domain vocabulary
                ds = run.split('/')[-1].split('_')[4]
                path = dict_domain_vocab[ds]
                if not os.path.exists(path):
                    print("Provided domain vocabulary path doesn't exist")
                    print ("Computing domain vocabulary using the provided path")
                    top_k = 10000
                    vocaboverlap.compute_domain_corpus(path_domain_articles = dict_domain_articles[ds],
                                                       top_k=top_k)

                domain_vocab = vocaboverlap.read_vocab(path)
                score = vocaboverlap.compute(predictions=predictions, references=domain_vocab)
                eval_scores['vocab_overlap'] = {'vocab_overlap_score': score}

            # push the eval scores of each run into wandb
            write_scores(eval_scores, run, out_file)
        except Exception as error:
            print(f"failed to compile results for {run} with the following error. \n {error}")


def write_scores(eval_scores: dict, run:str, out_file:str):

    model = ''
    ds = ''
    task = ''
    run_id = ''
    model_family = ''

    # Prepare config
    try:
        model = run.split('/')[-1].split('_')[2]
        ds = run.split('/')[-1].split('_')[4]
        task = run.split('/')[-1].split('_')[5]
        run_id = run.split('/')[-1].split('_')[0]
        model_family = model if len(model.split('-')) == 1 else ''.join(model.split('-')[:2])
    except:
        print (f"Some configs were not dervied from the filename {run}")

    df = pd.DataFrame()
    for score_name, score in eval_scores.items():
        df_score = pd.DataFrame(score, index=[0])
        df = pd.concat([df,df_score], axis = 1)

    df["g_ROUGE"] =  geometric_mean([df["rouge1"],df["rouge2"],df["rougeL"]])
    df.insert(0, "model", [model])
    df.insert(1, "dataset", [ds])
    df.insert(2, "task", [task])
    df.insert(3, "run_id", [run_id])
    df.insert(3, "model_family", [model_family])

    if os.path.exists(out_file):
        df_old = pd.read_csv(out_file)
        df = pd.concat([df, df_old], axis = 0)

    df.to_csv(out_file, index=False)
    print(out_file)


def main():


    print (config.exec_kwargs)
    input_dir = config.exec_kwargs.get('input_directory')
    selected_metrics = config.exec_kwargs.get('metrics')
    dict_domain_articles = config.exec_kwargs.get('domain_articles')
    dict_domain_vocab = config.exec_kwargs.get('domain_vocab_path')
    timestr = time.strftime("%Y%m%d-%H%M%S")
    out_file = f"eval_scores_{timestr}.csv"

    compute_automatic_evaluation(input_dir, selected_metrics = selected_metrics,
                                 dict_domain_articles= dict_domain_articles,
                                 dict_domain_vocab = dict_domain_vocab,
                                 out_file = out_file)


if __name__ == "__main__":
    main()
