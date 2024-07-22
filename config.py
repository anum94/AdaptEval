import argparse
import time
import os
from json import load
from typing import Sequence

from dotenv import load_dotenv
import torch

import wandb


class Config:
    tokenizer_parallelism_key: str = "TOKENIZERS_PARALLELISM"
    hf_token_key: str = "HF_API_TOKEN"
    hf_user_key: str = "HF_USER"
    hf_hub_cache_key: str = "TRANSFORMERS_CACHE"
    hf_ds_cache_key: str = "HF_DATASETS_CACHE"
    container_prefix: str = "/mnt/container/"
    ckpt_dir_path: str = os.path.join(container_prefix, "ckpt")
    hf_hub_cache_path: str = os.path.join(container_prefix, ".cache/huggingface/hub")
    hf_ds_cache_path: str = os.path.join(
        container_prefix, ".cache/huggingface/datasets"
    )

    exec_args: dict = {}
    exec_kwargs: dict = {}
    exec_string: str = ""
    exec_timestamp: str = ""

    method: str = ""
    working_dir: str = ""
    ckpt: str = ""

    hf_token: str = ""
    hf_user: str = ""


    def __init__(self) -> None:

        self.parse_args(self.configure_parser())
        self.configure_env()

        self.exec_string = f""
        self.exec_timestamp = time.strftime("%d%m%Y%H%M%S", time.localtime())

        self.working_dir = os.path.dirname(os.path.abspath(__file__))


    def configure_parser(self) -> dict:
        parser = argparse.ArgumentParser()
        parser.add_argument("--config", type=str, help="configuration file path")
        parser.add_argument("--input_dir", type=str, help="Directory where the .csv files are stored")
        parser.add_argument("--metrics", nargs="+", help="list metrics to evaluate")
        return vars(parser.parse_args())

    def parse_args(self, args: dict) -> None:
        if args["config"]:
            with open(args["config"], "r") as fp:
                json = load(fp)
                self.exec_args = json["exec_args"]
                self.exec_kwargs = json["exec_kwargs"]


    def configure_env(self) -> str:
        load_dotenv()

        # distributed setup
        # set the device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # hf setup
        token = os.environ.get(self.hf_token_key)
        assert token and token != "<token>", "HuggingFace API token is not defined"
        user = os.environ.get(self.hf_user_key)
        assert user and user != "<user>", "HuggingFace user is not defined"
        self.hf_token, self.hf_user = token, user

        os.environ[self.hf_ds_cache_key] = self.hf_ds_cache_path
        os.environ[self.hf_hub_cache_key] = self.hf_hub_cache_path

    def log_path(self) -> str:
        return f"{self.working_dir}/results/{self.exec_string}-{self.exec_timestamp}"

    def ckpt_log_path(self) -> str:
        return f"{self.ckpt_dir_path}/{self.exec_string}-{self.exec_timestamp}"

    def prev_log_path(self) -> str:
        # remove the last /checkpoint-1600 from:
        # /dss/dsshome1/0F/ge58hep2/transformer-research/results/longt5-pubmed-4096-10052023004612/checkpoint-1600
        return self.ckpt.rsplit("/", 1)[0]


config = Config()