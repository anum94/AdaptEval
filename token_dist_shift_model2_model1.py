import pandas as pd
import random
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import scipy
import nltk
from nltk import pos_tag, word_tokenize
import numpy
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
from config import config
import time

def prepare_context(df, row_idx: int = None, idx = None):
    if not row_idx:
        row_idx = random.randint(0, len(df))
    sample = df.iloc[row_idx]

    pred = sample['prediction']
    prompt = sample['prompt_0']

    if not idx:
        idx = [random.randint(0, len(pred.split())) for i in range(10)]

    pred_idx = [' '.join(map(str, pred.split()[:i])) for i in idx]
    model_input = [f"{prompt} SUMMARY: {p}" for p in pred_idx]
    return model_input, idx


def get_indices_by_pos(text, pos_tag_to_find):
        tokens = word_tokenize(text)
        tagged_words = pos_tag(tokens)
        indices = [i - 1 for i, (word, pos) in enumerate(tagged_words) if pos.startswith(pos_tag_to_find)]
        return indices


def find_indices_of_nouns_verbs_adjectives(text):
    noun_indices = get_indices_by_pos(text, 'N')
    verb_indices = get_indices_by_pos(text, 'V')
    adjective_indices = get_indices_by_pos(text, 'J')

    return noun_indices + adjective_indices + verb_indices


def get_token_idx(df,  domain_vocab_path: str, num_items_to_select=None, random_samples=False, use_all_tokens=False, use_domain_words=False, row_idx = 0):
    # define the tokens for which the distribution should be derived
    predictions_after = df['prediction'].tolist()
    pred = predictions_after[row_idx]
    idx = []

    # Or if you want to test across every nth samples of the text
    if random_samples:
        summary_words = len(pred.split())
        idx = [i for i in range(0, summary_words, 7)]

    elif use_all_tokens:
        summary_words = len(pred.split())
        idx = [i for i in range(0, summary_words)]
    # compute only for the words that are part of the domain vocabulary
    elif use_domain_words:
        # read the domain vocabulary

        if not os.path.exists(domain_vocab_path):
            print("Provided domain vocabulary path doesn't exist")
        else:
            with open(domain_vocab_path, 'r') as file:
                # Create an empty list to store the lines
                domain_vocab = []
                # Iterate over the lines of the file
                for line in file:
                    # Remove the newline character at the end of the line
                    line = line.strip()
                    # Append the line to the list
                    domain_vocab.append(str(line))
            summary_words = pred.split()
            idx = [i for i, word in enumerate(summary_words) if word in domain_vocab]

    else:
        # selecting N random words from the sample for evaluating the distribution shift
        idx = find_indices_of_nouns_verbs_adjectives(pred)

    if num_items_to_select:
        if len(idx) > num_items_to_select:
            idx = idx[:num_items_to_select]
    return idx


def get_token_distribution(inputs, model_hf_key: str, top_k=5, max_len=2048, truncation=True, verbose=False,):
    top_k_values_all = []
    predicted_tokens_all = []
    probability_distribution_all = []

    # Loading the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_hf_key)
    tokenizer.pad_token = tokenizer.eos_token  # Most LLMs don't have a pad token by default
    # load the model

    model = AutoModelForCausalLM.from_pretrained(
    model_hf_key,
    torch_dtype=torch.bfloat16,
    load_in_8bit=True,
    trust_remote_code=True,
    device_map="auto",)

    for input in tqdm(inputs):
        # Tokenize input text
        input_ids = tokenizer.encode(input, return_tensors="pt", max_length=max_len, truncation=truncation)

        # Generate probabilities for the next token
        with torch.no_grad():
            outputs = model(input_ids)
            logits = outputs.logits[:, -1, :]  # Take the logits for the last token
            probabilities = torch.nn.functional.softmax(logits, dim=-1)
            probability_distribution_all.append(probabilities)

        # Get the top-k predicted tokens and their probabilities

        top_k_values, top_k_indices = torch.topk(probabilities, k=top_k)

        # Convert indices back to tokens
        predicted_tokens = tokenizer.convert_ids_to_tokens(top_k_indices[0].tolist())

        top_k_values_all.append(top_k_values)
        predicted_tokens_all.append(predicted_tokens)
    return top_k_values_all, predicted_tokens_all, probability_distribution_all


def token_dist_model1_model2(model_1, model_2, domain_vocab_path: str, file_path = '', num_items_to_select=None, random_samples=False,
                             use_all_tokens=False, use_domain_words=False,row_idx = 0 ):

    df_after = pd.read_csv(file_path)

    # Get samples across which tokens should be predicted
    idx = get_token_idx(df_after, num_items_to_select=num_items_to_select, random_samples=random_samples,
                        use_all_tokens=use_all_tokens, use_domain_words=use_domain_words, domain_vocab_path = domain_vocab_path, row_idx = row_idx)

    # prepare input for the model
    inputs, token_idx = prepare_context(df=df_after, row_idx=row_idx, idx=idx)

    # Set the device to GPU if available
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Getting the token distribution
    top_k_values_before, predicted_tokens_before, probability_distribution_all_before = get_token_distribution(inputs,
                                                                                                               model_hf_key=model_1,
                                                                                                               top_k=50,
                                                                                                               max_len=4096,
                                                                                                               truncation=True,
                                                                                                               verbose=False,
                                                                                                               )

    # Getting the token distribution
    top_k_values_after, predicted_tokens_after, probability_distribution_all_after = get_token_distribution(inputs,
                                                                                                           model_hf_key=model_2,
                                                                                                           top_k=50,
                                                                                                           max_len=4096,
                                                                                                           truncation=True,
                                                                                                           verbose=False,
                                                                                                           )

    # Automatic Evaluation for Token Distribution Shift
    eval_scores = dict()

    # Calculate KL Divergence
    kl_divergence = []
    frequently_shift_tokens = []
    # compute for all tokens of the sample
    for prob_dist_bef, prob_dist_aft in zip(probability_distribution_all_before, probability_distribution_all_after):
        kl_div = scipy.special.kl_div(prob_dist_bef, prob_dist_aft)
        kl_div = numpy.sum(kl_div.numpy())
        # print("KL Divergence: ", kl_div)
        kl_divergence.append(kl_div)

    eval_scores['kl_divergence_mean'] = [numpy.average(kl_divergence)]

    token_shift = []
    base_rank = []
    base_prob = []

    unshifted = 0
    marginal_shift = 1
    shifted = 2
    for predicted_token_before, top_k_value_before, predicted_token_after, top_k_value_after in zip(predicted_tokens_before, top_k_values_before, predicted_tokens_after, top_k_values_after):

        top_k_value_before = top_k_value_before[0]
        top_k_value_after = top_k_value_after[0]

        # Calculate token shift rate
        if predicted_token_before[0] == predicted_token_after[0]:
            # print(predicted_token_before[0], predicted_token_after[0])
            # print("Unshifted")
            token_shift.append(unshifted)
        elif predicted_token_after[0] in predicted_token_before[:3]:
            # print(predicted_token_after[0], predicted_token_before[:3])
            # print("marginal shift")
            token_shift.append(marginal_shift)
            frequently_shift_tokens.append(predicted_token_after[0])
        else:
            # print(predicted_token_after[:5], predicted_token_before[:5])
            token_shift.append(shifted)
            # print("shifted")

        # Base rank of token
        try:
            rank = predicted_token_before.index(predicted_token_after[0])
        except:
            rank = -1
        base_rank.append(rank)

        # Base Probability of token
        if rank == -1:
            base_prob.append(0)
        else:
            base_prob.append(top_k_value_before[rank].item())

    eval_scores['token_shift_rate'] = [token_shift.count(shifted) / len(token_shift)]

    eval_scores['base_rank_mean'] = [numpy.average(base_rank)]

    eval_scores['base_prob_mean'] = [numpy.average(base_prob)]

    eval_scores['freq_shifted_tokens'] = [frequently_shift_tokens]


    return eval_scores


def write_scores(eval_scores: dict, row_idx:int):


    model = f"{model_1}-{model_2}"


    df = pd.DataFrame()

    for key, value in eval_scores.items():
        df.insert(0, key, [value])


    df.insert(0, "model", [model])
    df.insert(1, "sample_id", [row_idx])

    if os.path.exists(out_file):
        df_old = pd.read_csv(out_file)
        df = pd.concat([df, df_old], axis = 0)

    df.to_csv(out_file, index=False)
    print(f"Sample {row_idx} writen to {out_file}")


timestr = time.strftime("%Y%m%d-%H%M%S")
out_file = f"eval_scores_{timestr}.csv"

ds = config.exec_kwargs.get('tds-model2-model1').get('ds')
samples_to_compute = config.exec_kwargs.get('tds-model2-model1').get('num_samples')

model_1 = config.exec_kwargs.get('tds-model2-model1').get('model_1')
model_2 = config.exec_kwargs.get('tds-model2-model1').get('model_2')
pred_fname = config.exec_kwargs.get('tds-model2-model1').get('path_model2_pred')
domain_vocab_path = config.exec_kwargs.get('tds-model2-model1').get('domain_vocab_path')


df = pd.read_csv(pred_fname)
num_samples = len(pred_fname)

s = list(range(num_samples))
random.shuffle(s)
samples = s[-samples_to_compute:]


for i in range(len(samples)):
    row_idx = i
    print(ds, row_idx)

    eval_scores = token_dist_model1_model2(model_1=model_1, model_2=model_2, file_path=pred_fname,  use_domain_words=True, domain_vocab_path = domain_vocab_path, row_idx = row_idx)
    # use_all_tokens=True,  num_items_to_select=3, random_samples = False,)num_items_to_select=5, random_samples = True)
    write_scores(eval_scores, row_idx= row_idx)
