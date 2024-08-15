import torch
import torch.nn as nn
from models.LSTM.LSTM_LM import LSTM_LM
from models.VAE.VAE import VAE
from models.VAE.LSTM_Decoder import LSTMDecoder
from models.VAE.LSTM_Encoder import LSTMEncoder
import sentencepiece as spm
import logging
import os
import random
import numpy as np


class LSTMLMGenerator:
    def __init__(self, config):
        self.config = config
        spm.SentencePieceTrainer.train(
            "--input=Liu_Kheyer_Retrosynthesis_Data/vocab2.txt --model_prefix=m  --user_defined_symbols=[BOS],[EOS],[PAD],. --vocab_size=56 --bos_id=-1 --eos_id=-1")
        self.tokenizer = spm.SentencePieceProcessor()
        self.device = torch.device("cuda" if self.config["cuda"] else "cpu")
        self.tokenizer.load('m.model')
        self.PAD_TOKEN = self.tokenizer.encode_as_ids("[PAD]")[1]
        self.BOS_TOKEN = self.tokenizer.encode_as_ids("[BOS]")[1]
        self.EOS_TOKEN = self.tokenizer.encode_as_ids("[EOS]")[1]
        self.generator = LSTM_LM(vocab_size=config['vocab_size'], embedding_dim=config['LSTM_embed_dim'],
                                 hidden_dim=config['LSTM_hidden_dim'], num_layers=config['LSTM_num_layers'],
                                 use_cuda=config['cuda'], dropout_prob=config['LSTM_dropout_prob'],
                                 BOS_TOKEN=self.BOS_TOKEN, EOS_TOKEN=self.EOS_TOKEN).to(self.device)
        self.generator.load_state_dict(torch.load(self.config["load_path"]))
        self.generated_path = os.path.join(config["gene_path"])

    def generate_samples(self, seed=42):
        if self.config["cuda"]:
            rng = torch.cuda.manual_seed(seed)
        else:
            rng = torch.manual_seed(seed)
        self.generator.eval()
        samples = []
        for _ in range(int(self.config["n_gen_samples"] / self.config["batch_size"])):
            sample = self.generator.sample(
                self.config["batch_size"], self.config["seq_len"], generator=rng).cpu().tolist()
            samples.extend(sample)
        with open(self.generated_path, 'w', encoding="utf-8") as fout:
            lines_to_write = [
                ' '.join(map(str, sample)) + '\n' for sample in samples]
            fout.writelines(lines_to_write)


class VAEGenerator:
    def __init__(self, config):
        self.config = config
        self.device = torch.device("cuda" if self.config["cuda"] else "cpu")
        model_init = uniform_initializer(0.01)
        emb_init = uniform_initializer(0.1)
        spm.SentencePieceTrainer.train(
            "--input=Liu_Kheyer_Retrosynthesis_Data/vocab2.txt --model_prefix=m  --user_defined_symbols=[BOS],[EOS],[PAD],. --vocab_size=56 --bos_id=-1 --eos_id=-1")
        self.tokenizer = spm.SentencePieceProcessor()
        self.tokenizer.load('m.model')
        self.PAD_TOKEN = self.tokenizer.encode_as_ids("[PAD]")[1]
        self.BOS_TOKEN = self.tokenizer.encode_as_ids("[BOS]")[1]
        self.EOS_TOKEN = self.tokenizer.encode_as_ids("[EOS]")[1]
        self.encoder = LSTMEncoder(
            vocab_size=self.config["vocab_size"],
            model_init=model_init, emb_init=emb_init, embed_dim=self.config["VAE_LSTM_embed_dim"],
            hidden_dim=self.config["LSTM_encoder_hidden_dim"],
            latent_dim=self.config["VAE_latent_dim"], use_cuda=self.config["cuda"])
        self.decoder = LSTMDecoder(
            model_init=model_init, emb_init=emb_init,
            BOS_token=self.BOS_TOKEN, EOS_token=self.EOS_TOKEN, embed_dim=self.config[
                "VAE_LSTM_embed_dim"], hidden_dim=self.config["LSTM_decoder_hidden_dim"],
            latent_dim=self.config["VAE_latent_dim"], use_cuda=self.config["cuda"],
            seq_len=self.config["seq_len"], vocab_size=self.config["vocab_size"],
            dropout_in=self.config["LSTM_decoder_dropout_in"],
            dropout_out=self.config["LSTM_decoder_dropout_out"])
        self.vae = VAE(encoder=self.encoder, decoder=self.decoder,
                       latent_dim=self.config["VAE_latent_dim"],
                       use_cuda=self.config["cuda"]).to(self.device)
        self.vae.load_state_dict(torch.load(self.config["load_path"]))
        self.generated_path = os.path.join(config["gene_path"])

    def generate_samples(self, seed=42):
        self.vae.eval()
        logging.info('begin decoding..................................')
        if self.config["cuda"]:
            rng = torch.cuda.manual_seed(seed)
        else:
            rng = torch.manual_seed(seed)
        with torch.no_grad():
            self.vae.sample_from_prior(self.config["n_gen_samples"],
                                       "sample",
                                       self.generated_path, generator=rng)


class uniform_initializer(object):
    def __init__(self, stdv):
        self.stdv = stdv

    def __call__(self, tensor):
        nn.init.uniform_(tensor, -self.stdv, self.stdv)


class xavier_normal_initializer(object):
    def __call__(self, tensor):
        nn.init.xavier_normal_(tensor)
