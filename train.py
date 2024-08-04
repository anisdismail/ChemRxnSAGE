import os
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

from models.LSTM.LSTM_LM import LSTM_LM
from models.VAE.VAE import VAE
from models.VAE.LSTM_Decoder import LSTMDecoder
from models.VAE.LSTM_Encoder import LSTMEncoder
from dataloader import DataIterator
from eval import generate_metrics_evaluation
import sentencepiece as spm


class LSTMLMTrainer:
    def __init__(self, config):
        self.config = config
        self.train_path = os.path.join(
            config["data_dir"], config["train_path"])
        self.val_path = os.path.join(
            config["data_dir"], config["val_path"])
        self.generated_path = os.path.join(
            config["data_dir"], config["gene_path"])

        with open(os.path.join(config["data_dir"], "train", "centroids_200.data"), "r", encoding='utf-8') as f:
            self.centroids = np.loadtxt(f)
        with open(os.path.join(config["data_dir"], "train", "centroids_strings_200.data"), "r", encoding='utf-8') as f:
            self.centroids_strings = np.loadtxt(f)

        spm.SentencePieceTrainer.train(
            "--input=Liu_Kheyer_Retrosynthesis_Data/vocab2.txt --model_prefix=m  --user_defined_symbols=[BOS],[EOS],[PAD],. --vocab_size=56 --bos_id=-1 --eos_id=-1")
        self.tokenizer = spm.SentencePieceProcessor()
        self.tokenizer.load('m.model')
        self.PAD_TOKEN = self.tokenizer.encode_as_ids("[PAD]")[1]
        self.BOS_TOKEN = self.tokenizer.encode_as_ids("[BOS]")[1]
        self.EOS_TOKEN = self.tokenizer.encode_as_ids("[EOS]")[1]
        self.train_iter = DataIterator(
            data_file=self.train_path, batch_size=config["batch_size"], PAD_TOKEN=self.PAD_TOKEN)
        self.eval_iter = DataIterator(
            data_file=self.val_path, batch_size=config["batch_size"], PAD_TOKEN=self.PAD_TOKEN)

        # Set models, criteria, optimizers
        self.generator = LSTM_LM(vocab_size=config['vocab_size'], embedding_dim=config['LSTM_embed_dim'],
                                 hidden_dim=config['LSTM_hidden_dim'], num_layers=config['LSTM_num_layers'],
                                 use_cuda=config['cuda'], dropout_prob=config['LSTM_dropout_prob'],
                                 BOS_TOKEN=self.BOS_TOKEN, EOS_TOKEN=self.EOS_TOKEN)

        self.nll_loss = nn.NLLLoss()
        self.device = torch.device("cuda" if self.config["cuda"] else "cpu")
        self.generator = self.generator.to(self.device)
        self.nll_loss = self.nll_loss.to(self.device)

        self.gen_optimizer = optim.Adam(
            params=self.generator.parameters(), lr=config["LSTM_lr"])

    def train(self):
        print('#####################################################')
        print('Start training generator with MLE...')
        print('#####################################################\n')

        for i in range(0, self.config["epochs"]):
            train_loss = self.train_mle()
            val_loss = self.eval_nll(self.eval_iter)
            print("generating...")
            self.generate_samples()
            print("evaluating...")
            jsd, avg_similarity, avg_str_similarity, valid, filter0, filter2, filter4, filter5, df, rxn_pred, sims, gen_fingerprints = generate_metrics_evaluation(
                self.generated_path, self.centroids, self.centroids_strings, self.tokenizer, self.config)
            print(
                f"Epoch {i}, Train Loss: {train_loss:.5f}, Val Loss: {val_loss:.5f}")
            print(
                f"JSD: {jsd:.5f}, Similarity: {avg_similarity:.5f}, String Similarity: {avg_str_similarity:.5f}, Validity: {valid:.5f}\n")
            torch.save(self.generator.state_dict(), os.path.join(
                self.config["save_path"], f"lstm_epoch{i}_loss{val_loss:.4f}.pt"))

        print('#####################################################\n\n')

    def train_mle(self):
        """
        Train generator with MLE
        """
        self.generator.train()
        total_loss = 0.
        for data, target in self.train_iter:
            data, target = data.to(self.device), target.to(self.device)
            target = target.contiguous().view(-1)
            output = self.generator(data)
            loss = self.nll_loss(output, target)
            total_loss += loss.item()
            self.gen_optimizer.zero_grad()
            loss.backward()
            self.gen_optimizer.step()
        self.train_iter.reset()
        avg_loss = total_loss / len(self.train_iter)
        return avg_loss

    def eval_nll(self, data_iter):
        """
        Evaluate generator with NLL
        """
        total_loss = 0.
        self.generator.eval()
        with torch.no_grad():
            for data, target in data_iter:
                data, target = data.to(self.device), target.to(self.device)
                target = target.contiguous().view(-1)
                pred = self.generator(data)
                loss = self.nll_loss(pred, target)
                total_loss += loss.item()
        avg_loss = total_loss / len(data_iter)
        print('val loss:', avg_loss)
        data_iter.reset()
        return avg_loss

    def generate_samples(self):
        self.generator.eval()
        samples = []
        for _ in range(int(self.config["n_gen_samples"] / self.config["batch_size"])):
            sample = self.generator.sample(batch_size=self.config["batch_size"],
                                           seq_len=self.config["seq_len"]).cpu().tolist()
            samples.extend(sample)
        with open(self.generated_path, 'w', encoding="utf-8") as fout:
            lines_to_write = [
                ' '.join(map(str, sample)) + '\n' for sample in samples]
            fout.writelines(lines_to_write)


class VAETrainer:
    def __init__(self, config):
        self.config = config
        self.train_path = os.path.join(
            config["data_dir"], config["train_path"])
        self.val_path = os.path.join(
            config["data_dir"], config["val_path"])
        self.generated_path = os.path.join(
            config["data_dir"], config["gene_path"])

        with open(os.path.join(config["data_dir"], "train", "centroids_200.data"), "r", encoding='utf-8') as f:
            self.centroids = np.loadtxt(f)
        with open(os.path.join(config["data_dir"], "train", "centroids_strings_200.data"), "r", encoding='utf-8') as f:
            self.centroids_strings = np.loadtxt(f)

        spm.SentencePieceTrainer.train(
            "--input=Liu_Kheyer_Retrosynthesis_Data/vocab2.txt --model_prefix=m  --user_defined_symbols=[BOS],[EOS],[PAD],. --vocab_size=56 --bos_id=-1 --eos_id=-1")
        self.tokenizer = spm.SentencePieceProcessor()
        self.tokenizer.load('m.model')
        self.PAD_TOKEN = self.tokenizer.encode_as_ids("[PAD]")[1]
        self.BOS_TOKEN = self.tokenizer.encode_as_ids("[BOS]")[1]
        self.EOS_TOKEN = self.tokenizer.encode_as_ids("[EOS]")[1]
        self.train_iter = DataIterator(
            data_file=self.train_path, batch_size=config["batch_size"], PAD_TOKEN=self.PAD_TOKEN)
        self.eval_iter = DataIterator(
            data_file=self.val_path, batch_size=config["batch_size"], PAD_TOKEN=self.PAD_TOKEN)
        self.device = torch.device("cuda" if self.config["cuda"] else "cpu")
        model_init = uniform_initializer(0.01)
        emb_init = uniform_initializer(0.1)

        self.encoder = LSTMEncoder(
            config=self.config, vocab_size=config["vocab_size"],
            model_init=model_init, emb_init=emb_init)
        self.decoder = LSTMDecoder(
            config=self.config, model_init=model_init, emb_init=emb_init,
            BOS_token=self.BOS_TOKEN, EOS_token=self.EOS_TOKEN)
        self.vae = VAE(encoder=self.encoder, decoder=self.decoder,
                       config=self.config).to(self.device)

        self.enc_optimizer = optim.SGD(self.vae.encoder.parameters(),
                                       lr=1.0, momentum=self.config["momentum"])
        self.dec_optimizer = optim.SGD(self.vae.decoder.parameters(),
                                       lr=1.0, momentum=self.config["momentum"])

    def eval_nll(self,  data_iter):
        self.vae.eval()
        with torch.no_grad():
            report_kl_loss, report_rec_loss = 0, 0
            report_num_words, report_num_sents = 0, 0

            for data, target in data_iter:
                data, target = data.to(self.device), target.to(self.device)
                batch_size, sent_len = data.size()
                report_num_sents += batch_size
                report_num_words += (sent_len - 1) * batch_size

                # not predict start symbol
                report_num_words += (sent_len - 1) * batch_size
                report_num_sents += batch_size

                loss, loss_rc, loss_kl = self.vae.loss(
                    src=data, kl_weight=1.0, nsamples=self.config["VAE_n_training_samples"])

                assert (not loss_rc.requires_grad)

                loss_rc = loss_rc.sum()
                loss_kl = loss_kl.sum()
                report_rec_loss += loss_rc.item()
                report_kl_loss += loss_kl.item()

            test_loss = (report_rec_loss + report_kl_loss) / report_num_sents

            nll = (report_kl_loss + report_rec_loss) / report_num_sents
            kl = report_kl_loss / report_num_sents
            ppl = np.exp(nll * report_num_sents / report_num_words)
            print(
                f'VAL --- avg_loss: {test_loss:.4f}, kl: {report_kl_loss / report_num_sents:.4f}, recon: {report_rec_loss / report_num_sents:.4f}, nll: {nll:.4f}, ppl: {ppl:.4f}')
            data_iter.reset()
            return test_loss, nll, kl, ppl

    def train(self):
        # Initialize training state
        opt_dict = {"not_improved": 0, "lr": 1.0, "best_loss": 1e4}
        best_metrics = {"loss": 1e4, "kl": 0, "nll": 0, "ppl": 0}
        decay_cnt, pre_mi = 0, 0
        kl_weight = self.config["kl_start"]
        anneal_rate = (1.0 - self.config["kl_start"]) / (self.config["warm_up"] *
                                                         (self.train_iter.get_data_num() / self.config["batch_size"]))

        print("Starting Training.............")

        for epoch in range(self.config["epochs"]):
            self.vae.train()
            report_metrics = {"kl_loss": 0, "rec_loss": 0,
                              "num_words": 0, "num_sents": 0}

            for data, target in self.train_iter:
                data, target = data.to(self.device), target.to(self.device)
                batch_size, sent_len = data.size()
                report_metrics["num_sents"] += batch_size
                report_metrics["num_words"] += (sent_len - 1) * batch_size
                kl_weight = min(1.0, kl_weight + anneal_rate)

                # Burn-in phase for aggressive training
                if self.config["aggressive"]:
                    self.perform_aggressive_training(data)

                # Normal training step
                loss, loss_rc, loss_kl = self.compute_loss(
                    data=data, kl_weight=kl_weight)
                self.optimize_loss(loss)
                self.update_report_metrics(
                    report_metrics=report_metrics, loss_rc=loss_rc, loss_kl=loss_kl)

                # Monitor mutual information (MI) during aggressive training
                if self.config["aggressive"]:
                    self.monitor_mutual_information(pre_mi)

            # Report and log training progress
            train_loss = (
                report_metrics["rec_loss"] + report_metrics["kl_loss"]) / report_metrics["num_sents"]
            print(f'kl weight {kl_weight:.4f}')
            print(
                f'epoch: {epoch}, avg_loss: {train_loss:.4f}, kl: {report_metrics["kl_loss"] / report_metrics["num_sents"]:.4f}, recon: {report_metrics["rec_loss"] / report_metrics["num_sents"]:.4f}')

            # Evaluate on validation set
            eval_metrics = self.evaluate()

            # Check for improvement and adjust learning rate if necessary
            self.check_improvement(
                eval_metrics=eval_metrics, opt_dict=opt_dict,
                best_metrics=best_metrics, decay_cnt=decay_cnt, epoch=epoch)

            # Generate samples and evaluate them
            self.generate_samples()
            jsd, avg_similarity, avg_str_similarity, valid, filter0, filter2, filter4, filter5, df, rxn_pred, sims, gen_fingerprints = generate_metrics_evaluation(
                self.generated_path, self.centroids, self.centroids_strings, self.tokenizer, self.config)
            print(
                f"JSD: {jsd:.5f}, Similarity: {avg_similarity:.5f}, String Similarity: {avg_str_similarity:.5f}, Validity: {valid:.5f}\n")

            if decay_cnt == self.config["max_decay"]:
                break

            self.train_iter.reset()
            torch.save(self.vae.state_dict(), os.path.join(
                self.config["save_path"], f"vae_epoch{epoch}_aggressive{self.config['aggressive']}_loss{eval_metrics["loss"]:.4f}.pt"))

    def perform_aggressive_training(self, data):
        sub_iter = 1
        burn_pre_loss = 1e4
        burn_cur_loss = 0
        batch_data_enc = data

        while sub_iter < 100:
            self.enc_optimizer.zero_grad()
            self.dec_optimizer.zero_grad()

            burn_cur_loss += self.burn_in_step(batch_data_enc)

            if sub_iter % 15 == 0:
                burn_cur_loss /= (self.train_iter.get_data_num() -
                                  1) * batch_data_enc.size(0)
                if burn_pre_loss - burn_cur_loss < 0:
                    break
                burn_pre_loss = burn_cur_loss
                burn_cur_loss = 0

            batch_data_enc, _ = self.train_iter.sample()
            batch_data_enc = batch_data_enc.to(self.device)
            sub_iter += 1

    def burn_in_step(self, batch_data_enc):
        loss, _, _ = self.vae.loss(
            batch_data_enc, kl_weight=self.config["kl_start"], nsamples=self.config["VAE_n_training_samples"])
        burn_loss = loss.sum().item()
        loss = loss.mean(dim=-1)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.vae.parameters(), self.config["clip_grad"])
        self.enc_optimizer.step()

        return burn_loss

    def compute_loss(self, data, kl_weight):
        loss, loss_rc, loss_kl = self.vae.loss(
            src=data, kl_weight=kl_weight, nsamples=self.config["VAE_n_training_samples"])
        loss = loss.mean(dim=-1)
        return loss, loss_rc, loss_kl

    def optimize_loss(self, loss):
        self.enc_optimizer.zero_grad()
        self.dec_optimizer.zero_grad()

        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.vae.parameters(), self.config["clip_grad"])

        if not self.config["aggressive"]:
            self.enc_optimizer.step()

        self.dec_optimizer.step()

    def update_report_metrics(self, report_metrics, loss_rc, loss_kl):
        report_metrics["rec_loss"] += loss_rc.sum().item()
        report_metrics["kl_loss"] += loss_kl.sum().item()

    def monitor_mutual_information(self, pre_mi):
        self.vae.eval()
        cur_mi = self.vae.calc_mi(data_loader=self.eval_iter)
        self.vae.train()
        print(f"pre mi: {pre_mi:.4f}. cur mi: {cur_mi:.4f}")
        if cur_mi - pre_mi < 0:
            self.config["aggressive"] = False
            print("STOP BURNING")
        pre_mi = cur_mi

    def evaluate(self):
        self.vae.eval()
        with torch.no_grad():
            mi = self.vae.calc_mi(data_loader=self.eval_iter)
            au, _ = self.vae.calc_au(data_loader=self.eval_iter)
            loss, nll, kl, ppl = self.eval_nll(self.eval_iter)

        print(f'mi: {mi:.4f} au: {au}')
        return {"loss": loss, "nll": nll, "kl": kl, "ppl": ppl}

    def check_improvement(self, eval_metrics, opt_dict, best_metrics, decay_cnt, epoch):
        if eval_metrics["loss"] < best_metrics["loss"]:
            best_metrics.update(eval_metrics)
            print(
                f'update best loss: {best_metrics["loss"]:.4f}, best_nll: {best_metrics["nll"]:.4f}, best_kl: {best_metrics["kl"]:.4f}, best_ppl: {best_metrics["ppl"]:.4f}')

        if eval_metrics["loss"] > opt_dict["best_loss"]:
            opt_dict["not_improved"] += 1
            if opt_dict["not_improved"] >= self.config["decay_epoch"] and epoch >= 15:
                opt_dict.update(
                    {"best_loss": eval_metrics["loss"], "not_improved": 0, "lr": opt_dict["lr"] * self.config.lr_decay})
                decay_cnt += 1
                self.enc_optimizer = optim.SGD(self.vae.encoder.parameters(
                ), lr=opt_dict["lr"], momentum=self.config["momentum"])
                self.dec_optimizer = optim.SGD(self.vae.decoder.parameters(
                ), lr=opt_dict["lr"], momentum=self.config["momentum"])

                print(f'new lr: {opt_dict["lr"]}, new decay: {decay_cnt}')
        else:
            opt_dict.update(
                {"best_loss": eval_metrics["loss"], "not_improved": 0})

    def generate_samples(self):
        self.vae.eval()
        print('begin decoding..................................')
        with torch.no_grad():
            self.vae.sample_from_prior(nsamples=self.config["n_gen_samples"],
                                       strategy="sample",
                                       fname=self.generated_path)


class uniform_initializer(object):
    def __init__(self, stdv):
        self.stdv = stdv

    def __call__(self, tensor):
        nn.init.uniform_(tensor, -self.stdv, self.stdv)


class xavier_normal_initializer(object):
    def __call__(self, tensor):
        nn.init.xavier_normal_(tensor)
