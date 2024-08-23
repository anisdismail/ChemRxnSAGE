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
from eval import Evaluator
import sentencepiece as spm
import logging


class LSTMLMTrainer:
    def __init__(self, config):
        self.config = config
        self.train_path = os.path.join(config["train_path"])
        self.val_path = os.path.join(config["val_path"])
        self.generated_path = os.path.join(config["gene_path"])

        with open(os.path.join(config["main_dir"], "train", "centroids_200.data"), "r", encoding='utf-8') as f:
            self.centroids = np.loadtxt(f)
        with open(os.path.join(config["main_dir"], "train", "centroids_strings_200.data"), "r", encoding='utf-8') as f:
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
        self.evaluator = Evaluator(config)
        if self.config["load_path"]:
            self.generator.load_state_dict(
                torch.load(self.config["load_path"]))
        self.nll_loss = nn.NLLLoss()
        self.device = torch.device("cuda" if self.config["cuda"] else "cpu")
        self.generator = self.generator.to(self.device)
        self.nll_loss = self.nll_loss.to(self.device)

        self.gen_optimizer = optim.Adam(
            params=self.generator.parameters(), lr=config["LSTM_lr"])

    def train(self):
        logging.info('#####################################################')
        logging.info('Start training generator with MLE...')
        logging.info('#####################################################\n')

        for i in range(0, self.config["epochs"]):
            train_loss = self.train_mle()
            val_loss = self.eval_nll(self.eval_iter)
            logging.info(
                f"Epoch {i}, Train Loss: {train_loss:.5f}, Val Loss: {val_loss:.5f}")
            self.generate_and_evaluate()
            torch.save(self.generator.state_dict(), os.path.join(
                self.config["save_path"], f"lstm_epoch{i}_loss{val_loss:.4f}.pt"))

        logging.info(
            '#####################################################\n\n')

    def generate_and_evaluate(self):
        # Initialize metrics dictionary
        metrics = {
            "JSS": [],
            "Similarity": [],
            "String Similarity": [],
            "Validity": [],
            "Exact Matches Percentage": [],
            "Duplicates Percentage": [],
            "Average Inter Similarity": [],
            "Overall Validity": [],
            "Vendi Score": [],
            "Vendi Score (q=0.1)": [],
            "Vendi Score (q=inf)": [],
            "Avg Vendi Score Per Class": []
        }

        seeds = [42, 0, 250, 1000, 350]

        for seed in seeds:
            self.generate_samples(seed=seed)
            self.evaluator.generate_metrics_evaluation(self.generated_path)
            metrics["JSS"].append(self.evaluator.results["jss"])
            metrics["Similarity"].append(
                self.evaluator.results["avg_similarity"])
            metrics["String Similarity"].append(
                self.evaluator.results["avg_str_similarity"])
            metrics["Validity"].append(self.evaluator.results["valid"])
            metrics["Exact Matches Percentage"].append(
                self.evaluator.results["exact_perc"])
            metrics["Duplicates Percentage"].append(
                self.evaluator.results["duplicates_perc"])
            metrics["Average Inter Similarity"].append(
                self.evaluator.results["average_inter_similarity"])
            metrics["Overall Validity"].append(
                self.evaluator.results["validated"])
            metrics["Vendi Score"].append(
                self.evaluator.results["vendi_score_k"])
            metrics["Vendi Score (q=0.1)"].append(
                self.evaluator.results["vendi_score_k_small"])
            metrics["Vendi Score (q=inf)"].append(
                self.evaluator.results["vendi_score_k_inf"])
            metrics["Avg Vendi Score Per Class"].append(
                self.evaluator.results["avg_vs_score_per_class"])

            # Print the results for the current seed
            logging.info(f"""Seed {seed}: JSS={self.evaluator.results['jss']:.4f}, Sim={self.evaluator.results['avg_similarity']:.4f}, StrSim={self.evaluator.results['avg_str_similarity']:.4f}, Val={self.evaluator.results['valid']:.4f}, ExactMatchesPerc={self.evaluator.results['exact_perc']:.4f},
                DuplicatesPerc={self.evaluator.results['duplicates_perc']:.4f}, AvgInterSim={self.evaluator.results['average_inter_similarity']:.4f}, OverallVal={self.evaluator.results['validated']:.4f},
                VS={self.evaluator.results['vendi_score_k']:.4f}, VS(q=0.1)={self.evaluator.results['vendi_score_k_small']:.4f}, VD(q=inf)={self.evaluator.results['vendi_score_k_inf']:.4f}, AvgVSPerClass={self.evaluator.results['avg_vs_score_per_class']:.4f}""")

        # Print summary statistics
        logging.info("\nSummary Statistics:")
        logging.info(
            f"{'Metric':<25} {'Avg':<8} {'Std':<8} {'Min':<8} {'Max':<8}")

        for metric, values in metrics.items():
            logging.info(
                f"{metric:<25} {np.mean(values):<8.4f} {np.std(values):<8.4f} {np.min(values):<8.4f} {np.max(values):<8.4f}")

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
        logging.info(f'val loss: {avg_loss: .4f}')
        data_iter.reset()
        return avg_loss

    def generate_samples(self, seed=42):
        self.generator.eval()
        samples = []
        if self.config["cuda"]:
            rng = torch.cuda.manual_seed(seed)
        else:
            rng = torch.manual_seed(seed)
        for _ in range(int(self.config["n_gen_samples"] / self.config["batch_size"])):
            sample = self.generator.sample(batch_size=self.config["batch_size"],
                                           seq_len=self.config["seq_len"], generator=rng).cpu().tolist()
            samples.extend(sample)
        with open(self.generated_path, 'w', encoding="utf-8") as fout:
            lines_to_write = [
                ' '.join(map(str, sample)) + '\n' for sample in samples]
            fout.writelines(lines_to_write)


class VAETrainer:
    def __init__(self, config):
        self.config = config
        self.train_path = os.path.join(self.config["train_path"])
        self.val_path = os.path.join(self.config["val_path"])
        self.generated_path = os.path.join(self.config["gene_path"])

        with open(os.path.join(self.config["main_dir"], "train", "centroids_200.data"), "r", encoding='utf-8') as f:
            self.centroids = np.loadtxt(f)
        with open(os.path.join(self.config["main_dir"], "train", "centroids_strings_200.data"), "r", encoding='utf-8') as f:
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
        if self.config["load_path"]:
            self.vae.load_state_dict(torch.load(self.config["load_path"]))
        self.enc_optimizer = optim.SGD(self.vae.encoder.parameters(),
                                       lr=1.0, momentum=self.config["momentum"])
        self.dec_optimizer = optim.SGD(self.vae.decoder.parameters(),
                                       lr=1.0, momentum=self.config["momentum"])
        self.evaluator = Evaluator(config)

    def eval_nll(self,  data_iter):
        self.vae.eval()
        with torch.no_grad():
            report_kl_loss, report_rec_loss = 0, 0
            report_num_words, report_num_sents = 0, 0

            for data, target in data_iter:
                data, target = data.to(self.device), target.to(self.device)
                batch_size, sent_len = data.size()

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
            logging.info(
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

        logging.info("Starting Training.............")

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
                    pre_mi = self.monitor_mutual_information(pre_mi)

            # Report and log training progress
            train_loss = (
                report_metrics["rec_loss"] + report_metrics["kl_loss"]) / report_metrics["num_sents"]
            logging.info(f'kl weight {kl_weight:.4f}')
            logging.info(
                f'epoch: {epoch}, avg_loss: {train_loss:.4f}, kl: {report_metrics["kl_loss"] / report_metrics["num_sents"]:.4f}, recon: {report_metrics["rec_loss"] / report_metrics["num_sents"]:.4f}')

            # Evaluate on validation set
            eval_metrics = self.evaluate()

            # Check for improvement and adjust learning rate if necessary
            self.check_improvement(
                eval_metrics=eval_metrics, opt_dict=opt_dict,
                best_metrics=best_metrics, decay_cnt=decay_cnt, epoch=epoch)

            # Generate samples and evaluate them
            self.generate_and_evaluate()

            if decay_cnt == self.config["max_decay"]:
                break

            self.train_iter.reset()
            torch.save(self.vae.state_dict(), os.path.join(
                self.config["save_path"], f"vae_epoch{epoch}_kl{self.config['kl_start']}_warm_up{self.config['warm_up']}_aggressive{self.config['aggressive']}_loss{eval_metrics['loss']:.4f}.pt"))

    def generate_and_evaluate(self):
        # Initialize metrics dictionary
        metrics = {
            "JSS": [],
            "Similarity": [],
            "String Similarity": [],
            "Validity": [],
            "Exact Matches Percentage": [],
            "Duplicates Percentage": [],
            "Average Inter Similarity": [],
            "Overall Validity": [],
            "Vendi Score": [],
            "Vendi Score (q=0.1)": [],
            "Vendi Score (q=inf)": [],
            "Avg Vendi Score Per Class": []
        }

        seeds = [42, 0, 250, 1000, 350]

        for seed in seeds:
            self.generate_samples(seed=seed)
            self.evaluator.generate_metrics_evaluation(self.generated_path)
            metrics["JSS"].append(self.evaluator.results["jss"])
            metrics["Similarity"].append(
                self.evaluator.results["avg_similarity"])
            metrics["String Similarity"].append(
                self.evaluator.results["avg_str_similarity"])
            metrics["Validity"].append(self.evaluator.results["valid"])
            metrics["Exact Matches Percentage"].append(
                self.evaluator.results["exact_perc"])
            metrics["Duplicates Percentage"].append(
                self.evaluator.results["duplicates_perc"])
            metrics["Average Inter Similarity"].append(
                self.evaluator.results["average_inter_similarity"])
            metrics["Overall Validity"].append(
                self.evaluator.results["validated"])
            metrics["Vendi Score"].append(
                self.evaluator.results["vendi_score_k"])
            metrics["Vendi Score (q=0.1)"].append(
                self.evaluator.results["vendi_score_k_small"])
            metrics["Vendi Score (q=inf)"].append(
                self.evaluator.results["vendi_score_k_inf"])
            metrics["Avg Vendi Score Per Class"].append(
                self.evaluator.results["avg_vs_score_per_class"])

            # Print the results for the current seed
            logging.info(f"""Seed {seed}: JSS={self.evaluator.results['jss']:.4f}, Sim={self.evaluator.results['avg_similarity']:.4f}, StrSim={self.evaluator.results['avg_str_similarity']:.4f}, Val={self.evaluator.results['valid']:.4f}, ExactMatchesPerc={self.evaluator.results['exact_perc']:.4f}, 
                DuplicatesPerc={self.evaluator.results['duplicates_perc']:.4f}, AvgInterSim={self.evaluator.results['average_inter_similarity']:.4f}, OverallVal={self.evaluator.results['validated']:.4f}, 
                VS={self.evaluator.results['vendi_score_k']:.4f}, VS(q=0.1)={self.evaluator.results['vendi_score_k_small']:.4f}, VD(q=inf)={self.evaluator.results['vendi_score_k_inf']:.4f}, AvgVSPerClass={self.evaluator.results['avg_vs_score_per_class']:.4f}""")

        # Print summary statistics
        logging.info("\nSummary Statistics:")
        logging.info(
            f"{'Metric':<25} {'Avg':<8} {'Std':<8} {'Min':<8} {'Max':<8}")

        for metric, values in metrics.items():
            logging.info(
                f"{metric:<25} {np.mean(values):<8.4f} {np.std(values):<8.4f} {np.min(values):<8.4f} {np.max(values):<8.4f}")

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
        cur_mi = self.vae.calc_mi_oneshot(data_loader=self.eval_iter)
        self.vae.train()
        logging.info(f"pre mi: {pre_mi:.4f}. cur mi: {cur_mi:.4f}")
        if cur_mi - pre_mi < 0:
            self.config["aggressive"] = False
            logging.info("STOP BURNING")
        return cur_mi

    def evaluate(self):
        self.vae.eval()
        with torch.no_grad():
            mi = self.vae.calc_mi_oneshot(data_loader=self.eval_iter)
            au, _ = self.vae.calc_au(data_loader=self.eval_iter)
            loss, nll, kl, ppl = self.eval_nll(self.eval_iter)

        logging.info(f'mi: {mi:.4f} au: {au}')
        return {"loss": loss, "nll": nll, "kl": kl, "ppl": ppl}

    def check_improvement(self, eval_metrics, opt_dict, best_metrics, decay_cnt, epoch):
        if eval_metrics["loss"] < best_metrics["loss"]:
            best_metrics.update(eval_metrics)
            logging.info(
                f'update best loss: {best_metrics["loss"]:.4f}, best_nll: {best_metrics["nll"]:.4f}, best_kl: {best_metrics["kl"]:.4f}, best_ppl: {best_metrics["ppl"]:.4f}')

        if eval_metrics["loss"] > opt_dict["best_loss"]:
            opt_dict["not_improved"] += 1
            if opt_dict["not_improved"] >= self.config["decay_epoch"] and epoch >= 15:
                opt_dict.update(
                    {"best_loss": eval_metrics["loss"], "not_improved": 0, "lr": opt_dict["lr"] * self.config["lr_decay"]})
                decay_cnt += 1
                self.enc_optimizer = optim.SGD(self.vae.encoder.parameters(
                ), lr=opt_dict["lr"], momentum=self.config["momentum"])
                self.dec_optimizer = optim.SGD(self.vae.decoder.parameters(
                ), lr=opt_dict["lr"], momentum=self.config["momentum"])

                logging.info(
                    f'new lr: {opt_dict["lr"]}, new decay: {decay_cnt}')
        else:
            opt_dict.update(
                {"best_loss": eval_metrics["loss"], "not_improved": 0})

    def generate_samples(self, seed=42):
        self.vae.eval()
        if self.config["cuda"]:
            rng = torch.cuda.manual_seed(seed)
        else:
            rng = torch.manual_seed(seed)
        with torch.no_grad():
            self.vae.sample_from_prior(self.config["n_gen_samples"],
                                       "sample",
                                       self.generated_path, generator=rng)

    def cyclical_annealing(T, M, step, R=0.4, max_kl_weight=1):
        """
        Implementing: <https://arxiv.org/abs/1903.10145>
        T = Total steps 
        M = Number of cycles 
        R = Proportion used to increase beta
        t = Global step 
        """
        period = (T/M)  # N_iters/N_cycles
        # Itteration_number/(Global Period)
        internal_period = (step) % (period)
        tau = internal_period/period
        if tau > R:
            tau = max_kl_weight
        else:
            tau = min(max_kl_weight, tau/R)  # Linear function
        return tau


class uniform_initializer(object):
    def __init__(self, stdv):
        self.stdv = stdv

    def __call__(self, tensor):
        nn.init.uniform_(tensor, -self.stdv, self.stdv)


class xavier_normal_initializer(object):
    def __call__(self, tensor):
        nn.init.xavier_normal_(tensor)
