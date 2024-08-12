# AdaptEval - Framework for evaluation of domain adaptation on a text summarization tasks


## Usage 
---
<br>


### 1. Using config.json
 See `config.json`  for execution configuration 
```bash
python compile_results.py --config <path_to_json> 
```

```bash
python token_dist_shift_few_shot_zero_shot.py --config <path_to_json> 
```
```bash
python token_dist_shift_model2_model1.py --config <path_to_json> 
```
```
python gpt4-eval.py 
```
## Notes

For Reference-based Evaluation
- `compile_results.py` is intended to compute ROUGE, BERTscore, and domain vocabulary overlap on the provided runs in the input directory through the config.json

For Model Comparisons:
- `token_dist_shift_few_shot_zero_shot.py` is intended to do Token Distribution Shift analysis for the same model with a zero-shot vs two-shot model.
- `token_dist_shift_model2_model1.py` is intended to do Token Distribution Shift analysis between two given model. Where Model_2 is the bigger/finetuned model and Model_2 is the smaller 

Please provide the parameters in the config file. A sample config file is attached.
For running these two scripts, the csv should contain a column prompt_0 that contains the model prompt in the following form
_`You are an expert at summarization.
Proceed to summarize the following text.
TEXT: {article}
SUMMARY: 
{summary}
Proceed to summarize the following text.
TEXT: {article}
SUMMARY: 
{summary}
…
TEXT: {article}
SUMMARY:`_


For Evaluation using GPT-4L
- `run_chagpt_eval.ipynb` is intended to run summarization eval using the OpenAI API

### 1.Configuration

Defined in config.py, this script just reads all the configuration from the config.json and passes it on to the resepective evaluation scripts.

### 2. Metrics
Currently supported metrics:
- BERTscore
- Domain Vocabulary overlap
- ROUGE
- Token Dist Shift
- Evaluation using GPT4
### 3. Others

- All .csv files in the input directory should have two columns "reference" and "prediction". 
- The csv files should follow the name convention `{random-name}_MODEL_{model-name}_TASK_{ds}_{task}`. This naming convention helps the later group and compare the runs. 
- Most importantly, it allows the correct mapping of predictions to their respective domain vocabulary when calculating Domain Vocabulary Overlap.
- During the first run, the domain vocabulary is computed using the domain articles. For later runs, the previously computed domain vocabulary is used.
- Please provide your huggingface token in .env 
