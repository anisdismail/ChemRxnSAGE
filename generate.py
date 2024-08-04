import torch
from models.LSTM.LSTM_LM import LSTM_LM
from models.VAE.VAE import VAE
from models.VAE.LSTM_Decoder import LSTMDecoder
from models.VAE.LSTM_Encoder import LSTMEncoder
import sentencepiece as spm


class LSTMLMGenerator:
    def __init__(self, config):
        self.config = config
        self.generator = LSTM_LM(config['vocab_size'], config['LSTM_embed_dim'],
                                 config['LSTM_hidden_dim'], config['LSTM_num_layers'],
                                 config['cuda'], config['LSTM_dropout_prob'])

        self.generator.load_state_dict(torch.load(config["save_path"]))

    def generate_samples(self):
        self.generator.eval()
        samples = []
        for _ in range(int(self.config["n_gen_samples"] / self.config["batch_size"])):
            sample = self.generator.sample(
                self.config["batch_size"], self.config["seq_len"]).cpu().data.numpy().tolist()
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
        self.vocab = {self.tokenizer.id_to_piece(
            i): i for i in range(self.tokenizer.get_piece_size())}

        self.encoder = LSTMEncoder(
            self.config, config["vocab_size"], model_init, emb_init)
        self.decoder = LSTMDecoder(
            self.config, self.vocab, model_init, emb_init)
        self.vae = VAE(self.encoder, self.decoder, self.config).to(self.device)
        self.vae.load_state_dict(torch.load(config["save_path"]))

    def generate_samples(self):
        self.vae.eval()
        print('begin decoding..................................')
        with torch.no_grad():
            self.vae.sample_from_prior(self.config["n_gen_samples"],
                                       "sample",
                                       self.generated_path)


class uniform_initializer(object):
    def __init__(self, stdv):
        self.stdv = stdv

    def __call__(self, tensor):
        nn.init.uniform_(tensor, -self.stdv, self.stdv)


class xavier_normal_initializer(object):
    def __call__(self, tensor):
        nn.init.xavier_normal_(tensor)
