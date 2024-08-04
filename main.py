import os
import argparse
import json
import sys
import numpy as np
import torch
import random
import logging

from train import LSTMLMTrainer, VAETrainer
from generate import LSTMLMGenerator, VAEGenerator

parser = argparse.ArgumentParser(description="ChemRxnAIGen")

parser.add_argument("--config", default=None, help='Configuration file path')
parser.add_argument('--save_dir', type=str,
                    help="Save directory")
parser.add_argument("--main_dir", type=str, default="Liu_Kheyer_Retrosynthesis_Data",
                    help="Path to the data directory containing the dataset")
parser.add_argument("--train_path", type=str, default="/train/train_targets_ids_200.data",
                    help="Path to the data directory containing the train dataset")
parser.add_argument("--val_path", type=str, default="validation/validation_targets_ids_200.data",
                    help="Path to the data directory containing the val dataset")
parser.add_argument("--gene_path", type=str, default="/train/v2.4/gene.data",
                    help="Path to the data directory containing the generated dataset")
parser.add_argument("--load_path", type=str, default="",
                    help="Path to the trained models")
parser.add_argument("--log_path", type=str, default="",
                    help="Path to the log file")
parser.add_argument("--model", type=str, default="LSTM", choices=["LSTM", "VAE"],
                    help="model to train ")
parser.add_argument('--n_gen_samples', type=int, default=39579, metavar='N',
                    help='number of samples generated per time (default: 39579)')
parser.add_argument("--mode", type=str, default="train", choices=["train", "generate"],
                    help="mode for model to train or to generate new reaction ")
parser.add_argument('--vocab_size', type=int, default=56, metavar='N',
                    help='vocabulary size (default: 56)')
parser.add_argument('--batch_size', type=int, default=64, metavar='N',
                    help='batch size (default: 64)')
parser.add_argument('--seq_len', type=int, default=200,
                    help='Generator sequence length (default: 200)')
parser.add_argument('--no_cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--epochs', type=int, default=200,
                    help='number of training epochs')
parser.add_argument('--seed', type=int, default=42,
                    help='random seed')

# LSTM
parser.add_argument('--LSTM_lr', type=float, default=1e-3, metavar='LR',
                    help='learning rate of generator optimizer (default: 1e-3)')
parser.add_argument('--LSTM_embed_dim', type=int, default=64,
                    help='Generator embedding dimension (default: 64)')
parser.add_argument('--LSTM_hidden_dim', type=int, default=256,
                    help='Generator hidden dimension (default: 256)')
parser.add_argument('--LSTM_num_layers', type=int, default=4,
                    help='Number of generator layers (default: 4)')
parser.add_argument('--LSTM_dropout_prob', type=float, default=0.5,
                    help='Generator dropout probability (default: 0.5)')

# VAE
parser.add_argument('--momentum', type=float, default=0.0, help='sgd momentum')
parser.add_argument('--warm_up', type=int, default=10,
                    help='number of annealing epochs')
parser.add_argument('--kl_start', type=float, default=1.0,
                    help='starting KL weight')
parser.add_argument('--aggressive', type=int, default=0,
                    help='apply aggressive training when nonzero, reduce to vanilla VAE when aggressive is 0')
parser.add_argument('--LSTM_encoder_hidden_dim', type=int, default=1024,
                    help='encoder hidden size')
parser.add_argument('--LSTM_decoder_hidden_dim', type=int, default=1024,
                    help='decoder hidden size')
parser.add_argument('--LSTM_decoder_dropout_in', type=float,
                    default=0.5, help='decoder input dropout')
parser.add_argument('--LSTM_decoder_dropout_out', type=float,
                    default=0.5, help='decoder output dropout')
parser.add_argument("--clip_grad", type=int, default=5)
parser.add_argument("--decay_epoch", type=int, default=2)
parser.add_argument("--lr_decay", type=float, default=0.5)
parser.add_argument("--max_decay", type=int, default=5)
parser.add_argument("--VAE_n_training_samples", type=int, default=1)
parser.add_argument("--VAE_LSTM_embed_dim", type=int, default=512)
parser.add_argument("--VAE_latent_dim", type=int, default=32)


# Parse the arguments
config = parser.parse_args()
if config.config is not None:
    with open(config.config, 'r', encoding='utf-8') as file:
        config = json.load(file)
        config = {param: value for _, params in config.items()
                  for param, value in params.items()}
else:
    logging.info(
        "Arguments are required when configuration file is not provided")
    parser.print_help()
    sys.exit(1)


# set up logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[logging.FileHandler(config["log_path"]),
                              logging.StreamHandler()])

config["cuda"] = not config["no_cuda"] and torch.cuda.is_available()

logging.info(json.dumps(config, indent=4))

# fix seeds
torch.manual_seed(config["seed"])
np.random.seed(config["seed"])
random.seed(config["seed"])

if config["cuda"]:
    logging.info("Using Cuda")
    torch.cuda.manual_seed(config["seed"])
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

if config["mode"] == "train":
    if config["model"] == "LSTM":
        logging.info("training LSTM")
        trainer = LSTMLMTrainer(config=config)
    elif config["model"] == "VAE":
        logging.info("training VAE")
        trainer = VAETrainer(config=config)
    trainer.train()

elif config["mode"] == "generate":
    if config["model"] == "LSTM":
        logging.info("generating with LSTM")
        generator = LSTMLMGenerator(config=config)
    elif config["model"] == "VAE":
        logging.info("generating with VAE")
        generator = VAEGenerator(config=config)
    generator.generate_samples()
