import pandas as pd
from nltk.stem import SnowballStemmer
from termcolor import colored
import random
import os
import torch
from openai import OpenAI
from typing import List
from transformers import GenerationConfig, AutoModelForCausalLM, AutoTokenizer
from torch.utils.data import DataLoader
from datasets.dataset_dict import DatasetDict
from datasets import Dataset
from tqdm import tqdm
import os
import json
import openai
from ds.supported import load_dataset
from metrics.rouge import Rouge
import openai
import tiktoken
from tqdm import tqdm
import nltk
from nltk import pos_tag, word_tokenize

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

# Globals
DATA_DIR = "data/"
DATA_FILENAME = f"pubmed_eval_samples123.csv"
openai_api_key = ""


import os
import requests
import pandas as pd
from openai import OpenAI

def prompt(user_prompt,system_prompt):

    temperature = 0
    model = "gpt-4"
    model= "gpt-3.5-turbo-0613"

    #tok = tiktoken.encoding_for_model(model)


    client = OpenAI(
        # This is the default and can be omitted
        api_key=openai_api_key,
    )
    response = client.chat.completions.create(
                messages=[
                    {
                        "role": "user",
                        "content": user_prompt,
                    },
                    {
                        "role": "system",
                        "content": system_prompt,
                    }
                ],
                model="gpt-3.5-turbo",
                max_tokens=5,
            )
    reply = response.choices[0].message.content
    return reply


# load data
df = pd.read_csv(f"{DATA_DIR}/{DATA_FILENAME}")
print(df.columns)
print("# of samples: ", len(df))

# Evaluation prompt template based on G-Eval
EVALUATION_SYSTEM_PROMPT_TEMPLATE = """
You will be given one summary written for an article. Your task is to rate the summary on one metric.
Please make sure you read and understand these instructions very carefully. 
Please keep this document open while reviewing, and refer to it as needed.

Evaluation Criteria:

{criteria}

Evaluation Steps:

{steps}
"""
EVALUATION_USER_PROMPT_TEMPLATE = """

Example:

Source Text:

{document}

Summary:

{summary}

Evaluation Form:

Please provide your response in two parts. First the score as a numeric value followed by an explanation for the score. Please limit your response to 30 words


- {metric_name}
"""

# Metric 1: Coherence

COHERENCE_SCORE_CRITERIA = """
Coherence(1-5) - the collective quality of all sentences. \
We align this dimension with the DUC quality question of structure and coherence \
whereby "the summary should be well-structured and well-organized. \
The summary should not just be a heap of related information, but should build from sentence to a\
coherent body of information about a topic."
"""

COHERENCE_SCORE_STEPS = """
1. Read the article carefully and identify the main topic and key points.
2. Read the summary and compare it to the article. Check if the summary covers the main topic and key points of the article,
and if it presents them in a clear and logical order.
3. Assign a score for coherence on a scale of 1 to 5, where 1 is the lowest and 5 is the highest based on the Evaluation Criteria.
"""

# Metric 2: Fluency

FLUENCY_SCORE_CRITERIA = """
Fluency(1-3): the quality of the summary in terms of grammar, spelling, punctuation, word choice, and sentence structure.
1: Poor. The summary has many errors that make it hard to understand or sound unnatural.
2: Fair. The summary has some errors that affect the clarity or smoothness of the text, but the main points are still comprehensible.
3: Good. The summary has few or no errors and is easy to read and follow.
"""

FLUENCY_SCORE_STEPS = """
Read the summary and evaluate its fluency based on the given criteria. Assign a fluency score from 1 to 3.
"""

# Metric 2: Domain-Adaptation

DOMAIN_ADAPTATION_SCORE_CRITERIA = """
Domain Adaptation(1-5) - the degree to which the summary adheres to the doamin specific language. \
A good summary employs domain-specific terminology and conveys the sense that model comprehends and encapsulates domain-specific knowledge.\
It resembles the content that would authored by a domain expert. \
Annotators were also asked to penalize summaries that didn't adhere to domain-specific knowledge, and rather used simple words.
"""

DOMAIN_ADAPTATION_SCORE_STEPS = """
1. Read the article carefully and understand the domain it belongs to.
2. Read the summary and check if it contains domain-specific terminologies and concepts, and if it is able to concisely summaries the domain specific concept in the article. 
3. Assign a score for domain adaptation based on the Evaluation Criteria.
"""


def get_geval_score(
        criteria: str, steps: str, document: str, summary: str, metric_name: str
):
    system_prompt = EVALUATION_SYSTEM_PROMPT_TEMPLATE.format(
        criteria=criteria,
        steps=steps
    )

    user_prompt = EVALUATION_USER_PROMPT_TEMPLATE.format(
        metric_name=metric_name,
        document=document,
        summary=summary,
    )

    response = prompt(
        user_prompt=user_prompt,
        system_prompt=system_prompt,
    )
    return response


evaluation_metrics = {
    "Domain_Adaptation": (DOMAIN_ADAPTATION_SCORE_CRITERIA, DOMAIN_ADAPTATION_SCORE_STEPS),
    "Coherence": (COHERENCE_SCORE_CRITERIA, COHERENCE_SCORE_STEPS),
    "Fluency": (FLUENCY_SCORE_CRITERIA, FLUENCY_SCORE_STEPS),
}

models_to_evaluate = ['meta-llama-Llama-2-7b-chat-hf_2-SHOT', 'meta-llama-Llama-2-7b-hf-mtc-pubmed_0-SHOT',
                      'meta-llama-Llama-2-70b-chat-hf_2-SHOT', 'pegasusx_Finetuned']

eval_column_names = [f"{eval_metrics}_{model}" for model in models_to_evaluate for eval_metrics in
                     evaluation_metrics.keys()]
eval_column_names.insert(0, 'sample_id')

df_geval = pd.DataFrame(columns=eval_column_names)

for i, sample in df.iterrows():
    eval_scores = []
    eval_scores.append(sample['Unnamed: 0'])
    for model in models_to_evaluate:
        print(model)

        # time.sleep(5)

        article = sample.article
        summary = sample[model]

        for eval_type, (criteria, steps) in evaluation_metrics.items():
            print(eval_type)
            result = get_geval_score(criteria, steps, article, summary, eval_type)
            eval_scores.append(result)
    df_geval.loc[len(df_geval)] = eval_scores


df_geval.to_excel("pubmed_GPT4_Evaluation_final.xlsx")
print (f"Evaluation Score stored at {df_geval}")


