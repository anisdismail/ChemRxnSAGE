import logging
import os

import numpy as np
import sentencepiece as spm
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.nn.functional as F

from dataloader import DataIterator
from eval import Evaluator
from models.LSTM.LSTM_LM import LSTM_LM
from models.Transformer.transformer import Transformer
from models.VAE.LSTM_Decoder import LSTMDecoder
from models.VAE.LSTM_Encoder import LSTMEncoder
from models.VAE.VAE import VAE


class LSTMLMTrainer:
    def __init__(self, config):
        self.config = config
        self.train_path = os.path.join(config["train_path"])
        self.val_path = os.path.join(config["val_path"])
        self.generated_path = os.path.join(config["gene_path"])

        with open(
            os.path.join(config["main_dir"], "train", "centroids_200.data"),
            "r",
            encoding="utf-8",
        ) as f:
            self.centroids = np.loadtxt(f)
        with open(
            os.path.join(config["main_dir"], "train", "centroids_strings_200.data"),
            "r",
            encoding="utf-8",
        ) as f:
            self.centroids_strings = np.loadtxt(f)

        spm.SentencePieceTrainer.train(
            "--input=Liu_Kheyer_Retrosynthesis_Data/vocab2.txt --model_prefix=m  --user_defined_symbols=[BOS],[EOS],[PAD],. --vocab_size=56 --bos_id=-1 --eos_id=-1"
        )
        self.tokenizer = spm.SentencePieceProcessor()
        self.tokenizer.load("m.model")
        self.PAD_TOKEN = self.tokenizer.encode_as_ids("[PAD]")[1]
        self.BOS_TOKEN = self.tokenizer.encode_as_ids("[BOS]")[1]
        self.EOS_TOKEN = self.tokenizer.encode_as_ids("[EOS]")[1]
        self.train_iter = DataIterator(
            data_file=self.train_path,
            batch_size=config["batch_size"],
            PAD_TOKEN=self.PAD_TOKEN,
        )
        self.eval_iter = DataIterator(
            data_file=self.val_path,
            batch_size=config["batch_size"],
            PAD_TOKEN=self.PAD_TOKEN,
        )

        # Set models, criteria, optimizers
        self.generator = LSTM_LM(
            vocab_size=config["vocab_size"],
            embedding_dim=config["LSTM_embed_dim"],
            hidden_dim=config["LSTM_hidden_dim"],
            num_layers=config["LSTM_num_layers"],
            use_cuda=config["cuda"],
            dropout_prob=config["LSTM_dropout_prob"],
            BOS_TOKEN=self.BOS_TOKEN,
            EOS_TOKEN=self.EOS_TOKEN,
            PAD_TOKEN=self.PAD_TOKEN,
        )
        self.evaluator = Evaluator(config)
        if self.config["load_path"]:
            self.generator.load_state_dict(torch.load(self.config["load_path"]))
        self.nll_loss = nn.NLLLoss(ignore_index=self.PAD_TOKEN, reduction="sum")
        self.device = torch.device("cuda" if self.config["cuda"] else "cpu")
        self.generator = self.generator.to(self.device)
        self.nll_loss = self.nll_loss.to(self.device)

        self.gen_optimizer = optim.Adam(
            params=self.generator.parameters(), lr=config["LSTM_lr"]
        )

    def train(self):
        logging.info("#####################################################")
        logging.info("Start training generator with MLE...")
        logging.info("#####################################################\n")

        for i in range(0, self.config["epochs"]):
            train_loss = self.train_mle()
            val_loss = self.eval_nll(self.eval_iter)
            logging.info(
                f"Epoch {i}, Train Loss: {train_loss:.5f}, Val Loss: {val_loss:.5f}"
            )
            self.generate_and_evaluate()
            torch.save(
                self.generator.state_dict(),
                os.path.join(
                    self.config["save_path"], f"lstm_epoch{i}_loss{val_loss:.4f}.pt"
                ),
            )

        logging.info("#####################################################\n\n")

    def generate_and_evaluate(self):
        # Initialize metrics dictionary
        metrics = {
            "JSS": [],
            "Similarity": [],
            "String Similarity": [],
            "Validity": [],
            "Novelty Percentage": [],
            "Unique Percentage": [],
            "Average Inter Dissimilarity": [],
            "Overall Validity": [],
            "Normalized Vendi Score": [],
            "Normalized Vendi Score (q=0.1)": [],
            "Normalized Vendi Score (q=inf)": [],
            "Avg Normalized Vendi Score Per Class": [],
            "Filter 1": [],
            "Filter 2": [],
            "Filter 3": [],
            "Filter 4": [],
        }

        seeds = [42, 0, 250, 1000, 350]

        for seed in seeds:
            self.generate_samples(seed=seed)
            self.evaluator.generate_metrics_evaluation(self.generated_path)
            metrics["JSS"].append(self.evaluator.results["jss"])
            metrics["Similarity"].append(self.evaluator.results["avg_similarity"])
            metrics["String Similarity"].append(
                self.evaluator.results["avg_str_similarity"]
            )
            metrics["Validity"].append(self.evaluator.results["valid"])
            metrics["Novelty Percentage"].append(self.evaluator.results["novelty_perc"])
            metrics["Unique Percentage"].append(self.evaluator.results["unique_perc"])
            metrics["Average Inter Dissimilarity"].append(
                self.evaluator.results["average_inter_dissimilarity"]
            )
            metrics["Overall Validity"].append(self.evaluator.results["validated"])
            metrics["Normalized Vendi Score"].append(
                self.evaluator.results["vendi_score_k"]
            )
            metrics["Normalized Vendi Score (q=0.1)"].append(
                self.evaluator.results["vendi_score_k_small"]
            )
            metrics["Normalized Vendi Score (q=inf)"].append(
                self.evaluator.results["vendi_score_k_inf"]
            )
            metrics["Avg Normalized Vendi Score Per Class"].append(
                self.evaluator.results["avg_vs_score_per_class"]
            )
            metrics["Filter 1"].append(self.evaluator.results["filter1"])
            metrics["Filter 2"].append(self.evaluator.results["filter2"])
            metrics["Filter 3"].append(self.evaluator.results["filter3"])
            metrics["Filter 4"].append(self.evaluator.results["filter4"])

            # Print the results for the current seed
            logging.info(f"""Seed {seed}: JSS={self.evaluator.results["jss"]:.4f}, Sim={self.evaluator.results["avg_similarity"]:.4f}, StrSim={self.evaluator.results["avg_str_similarity"]:.4f}, Val={self.evaluator.results["valid"]:.4f}, NoveltyPerc={self.evaluator.results["novelty_perc"]:.4f}, 
            UniquePerc={self.evaluator.results["unique_perc"]:.4f}, IntDiv={self.evaluator.results["average_inter_dissimilarity"]:.4f}, OverallVal={self.evaluator.results["validated"]:.4f}, 
            NVS={self.evaluator.results["vendi_score_k"]:.4f}, NVS(q=0.1)={self.evaluator.results["vendi_score_k_small"]:.4f}, NVS(q=inf)={self.evaluator.results["vendi_score_k_inf"]:.4f}, AvgNVSPerClass={self.evaluator.results["avg_vs_score_per_class"]:.4f}""")

        # Print summary statistics
        logging.info("\nSummary Statistics:")
        logging.info(f"{'Metric':<25} {'Avg':<8} {'Std':<8} {'Min':<8} {'Max':<8}")

        for metric, values in metrics.items():
            logging.info(
                f"{metric:<25} {np.mean(values):<8.4f} {np.std(values):<8.4f} {np.min(values):<8.4f} {np.max(values):<8.4f}"
            )

    def train_mle(self):
        """
        Train generator with MLE
        """
        self.generator.train()
        total_loss = 0.0
        for data, target in self.train_iter:
            data, target = data.to(self.device), target.to(self.device)
            target = target.contiguous().view(-1)
            output = self.generator(data)
            loss = self.nll_loss(output, target)
            non_pad_mask = target != self.PAD_TOKEN
            num_valid_tokens = non_pad_mask.sum()
            loss = loss / num_valid_tokens
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
        total_loss = 0.0
        self.generator.eval()
        with torch.no_grad():
            for data, target in data_iter:
                data, target = data.to(self.device), target.to(self.device)
                target = target.contiguous().view(-1)
                pred = self.generator(data)
                loss = self.nll_loss(pred, target)
                non_pad_mask = target != self.PAD_TOKEN
                num_valid_tokens = non_pad_mask.sum()
                loss = loss / num_valid_tokens
                total_loss += loss.item()
        avg_loss = total_loss / len(data_iter)
        logging.info(f"val loss: {avg_loss: .4f}")
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
            sample = (
                self.generator.sample(
                    batch_size=self.config["batch_size"],
                    seq_len=self.config["seq_len"],
                    generator=rng,
                )
                .cpu()
                .tolist()
            )
            samples.extend(sample)
        with open(self.generated_path, "w", encoding="utf-8") as fout:
            lines_to_write = [" ".join(map(str, sample)) + "\n" for sample in samples]
            fout.writelines(lines_to_write)


class VAETrainer:
    def __init__(self, config):
        self.config = config
        self.train_path = os.path.join(self.config["train_path"])
        self.val_path = os.path.join(self.config["val_path"])
        self.generated_path = os.path.join(self.config["gene_path"])

        with open(
            os.path.join(self.config["main_dir"], "train", "centroids_200.data"),
            "r",
            encoding="utf-8",
        ) as f:
            self.centroids = np.loadtxt(f)
        with open(
            os.path.join(
                self.config["main_dir"], "train", "centroids_strings_200.data"
            ),
            "r",
            encoding="utf-8",
        ) as f:
            self.centroids_strings = np.loadtxt(f)

        spm.SentencePieceTrainer.train(
            "--input=Liu_Kheyer_Retrosynthesis_Data/vocab2.txt --model_prefix=m  --user_defined_symbols=[BOS],[EOS],[PAD],. --vocab_size=56 --bos_id=-1 --eos_id=-1"
        )
        self.tokenizer = spm.SentencePieceProcessor()
        self.tokenizer.load("m.model")
        self.PAD_TOKEN = self.tokenizer.encode_as_ids("[PAD]")[1]
        self.BOS_TOKEN = self.tokenizer.encode_as_ids("[BOS]")[1]
        self.EOS_TOKEN = self.tokenizer.encode_as_ids("[EOS]")[1]
        self.train_iter = DataIterator(
            data_file=self.train_path,
            batch_size=config["batch_size"],
            PAD_TOKEN=self.PAD_TOKEN,
        )
        self.eval_iter = DataIterator(
            data_file=self.val_path,
            batch_size=config["batch_size"],
            PAD_TOKEN=self.PAD_TOKEN,
        )
        self.device = torch.device("cuda" if self.config["cuda"] else "cpu")
        model_init = uniform_initializer(0.01)
        emb_init = uniform_initializer(0.1)

        self.encoder = LSTMEncoder(
            vocab_size=self.config["vocab_size"],
            model_init=model_init,
            emb_init=emb_init,
            embed_dim=self.config["VAE_LSTM_embed_dim"],
            hidden_dim=self.config["LSTM_encoder_hidden_dim"],
            latent_dim=self.config["VAE_latent_dim"],
            use_cuda=self.config["cuda"],
        )
        self.decoder = LSTMDecoder(
            model_init=model_init,
            emb_init=emb_init,
            BOS_token=self.BOS_TOKEN,
            EOS_token=self.EOS_TOKEN,
            PAD_token=self.PAD_TOKEN,
            embed_dim=self.config["VAE_LSTM_embed_dim"],
            hidden_dim=self.config["LSTM_decoder_hidden_dim"],
            latent_dim=self.config["VAE_latent_dim"],
            use_cuda=self.config["cuda"],
            seq_len=self.config["seq_len"],
            vocab_size=self.config["vocab_size"],
            dropout_in=self.config["LSTM_decoder_dropout_in"],
            dropout_out=self.config["LSTM_decoder_dropout_out"],
        )
        self.vae = VAE(
            encoder=self.encoder,
            decoder=self.decoder,
            latent_dim=self.config["VAE_latent_dim"],
            use_cuda=self.config["cuda"],
        ).to(self.device)
        if self.config["load_path"]:
            self.vae.load_state_dict(torch.load(self.config["load_path"]))

        self.enc_optimizer = optim.SGD(
            self.vae.encoder.parameters(),
            lr=self.config["lr"],
            momentum=self.config["momentum"],
        )
        self.dec_optimizer = optim.SGD(
            self.vae.decoder.parameters(),
            lr=self.config["lr"],
            momentum=self.config["momentum"],
        )

        self.enc_scheduler = lr_scheduler.ReduceLROnPlateau(
            self.enc_optimizer,
            mode="min",
            factor=self.config["lr_decay"],
            patience=self.config["decay_epoch"],
            verbose=True,
        )
        self.dec_scheduler = lr_scheduler.ReduceLROnPlateau(
            self.dec_optimizer,
            mode="min",
            factor=self.config["lr_decay"],
            patience=self.config["decay_epoch"],
            verbose=True,
        )
        self.evaluator = Evaluator(config)

    def eval_nll(self, data_iter):
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
                    src=data,
                    kl_weight=1.0,
                    nsamples=self.config["VAE_n_training_samples"],
                )

                assert not loss_rc.requires_grad

                loss_rc = loss_rc.sum()
                loss_kl = loss_kl.sum()
                report_rec_loss += loss_rc.item()
                report_kl_loss += loss_kl.item()

            test_loss = (report_rec_loss + report_kl_loss) / report_num_sents

            nll = (report_kl_loss + report_rec_loss) / report_num_sents
            kl = report_kl_loss / report_num_sents
            ppl = np.exp(nll * report_num_sents / report_num_words)
            logging.info(
                f"VAL --- avg_loss: {test_loss:.4f}, kl: {report_kl_loss / report_num_sents:.4f}, recon: {report_rec_loss / report_num_sents:.4f}, nll: {nll:.4f}, ppl: {ppl:.4f}"
            )
            data_iter.reset()
            return test_loss, nll, kl, ppl

    def train(self):
        # Initialize training state
        best_metrics = {"loss": float("inf"), "kl": 0, "nll": 0, "ppl": 0}
        pre_mi = 0

        # KL Annealing Setup
        total_batches = self.train_iter.get_data_num() / self.config["batch_size"]
        total_updates = self.config["warm_up"] * total_batches
        kl_weight = self.config["kl_start"]
        anneal_rate = (1.0 - kl_weight) / total_updates if total_updates > 0 else 0.0
        update_count = 0

        logging.info("Starting Training.............")

        for epoch in range(self.config["epochs"]):
            self.vae.train()
            report_metrics = {
                "kl_loss": 0,
                "rec_loss": 0,
                "num_words": 0,
                "num_sents": 0,
            }

            for data, target in self.train_iter:
                data, target = data.to(self.device), target.to(self.device)
                batch_size, sent_len = data.size()
                report_metrics["num_sents"] += batch_size
                report_metrics["num_words"] += (sent_len - 1) * batch_size

                # Update KL weight
                if update_count < total_updates:
                    kl_weight = min(1.0, kl_weight + anneal_rate)
                    update_count += 1

                # Burn-in phase for aggressive training
                if self.config["aggressive"]:
                    self.perform_aggressive_training(data)

                # Normal training step
                loss, loss_rc, loss_kl = self.compute_loss(
                    data=data, kl_weight=kl_weight
                )
                self.optimize_loss(loss)
                self.update_report_metrics(
                    report_metrics=report_metrics, loss_rc=loss_rc, loss_kl=loss_kl
                )

                # Monitor mutual information (MI) during aggressive training
                if self.config["aggressive"]:
                    pre_mi = self.monitor_mutual_information(pre_mi)

            # Report and log training progress
            train_loss = (
                report_metrics["rec_loss"] + report_metrics["kl_loss"]
            ) / report_metrics["num_sents"]
            logging.info(f"kl weight {kl_weight:.4f}")
            logging.info(
                f"epoch: {epoch}, avg_loss: {train_loss:.4f}, kl: {report_metrics['kl_loss'] / report_metrics['num_sents']:.4f}, recon: {report_metrics['rec_loss'] / report_metrics['num_sents']:.4f}"
            )

            # Evaluate on validation set
            eval_metrics = self.evaluate()
            if eval_metrics["loss"] < best_metrics["loss"]:
                best_metrics.update(eval_metrics)
                logging.info(
                    f"update best loss: {best_metrics['loss']:.4f}, best_nll: {best_metrics['nll']:.4f}, best_kl: {best_metrics['kl']:.4f}, best_ppl: {best_metrics['ppl']:.4f}"
                )
            self.enc_scheduler.step(eval_metrics["loss"])
            self.dec_scheduler.step(eval_metrics["loss"])

            # Generate samples and evaluate them
            self.generate_and_evaluate()

            self.train_iter.reset()
            torch.save(
                self.vae.state_dict(),
                os.path.join(
                    self.config["save_path"],
                    f"vae_epoch{epoch}_kl{self.config['kl_start']}_warm_up{self.config['warm_up']}_aggressive{self.config['aggressive']}_loss{eval_metrics['loss']:.4f}.pt",
                ),
            )

    def generate_and_evaluate(self):
        # Initialize metrics dictionary
        metrics = {
            "JSS": [],
            "Similarity": [],
            "String Similarity": [],
            "Validity": [],
            "Novelty Percentage": [],
            "Unique Percentage": [],
            "Average Inter Dissimilarity": [],
            "Overall Validity": [],
            "Normalized Vendi Score": [],
            "Normalized Vendi Score (q=0.1)": [],
            "Normalized Vendi Score (q=inf)": [],
            "Avg Normalized Vendi Score Per Class": [],
            "Filter 1": [],
            "Filter 2": [],
            "Filter 3": [],
            "Filter 4": [],
        }

        seeds = [42, 0, 250, 1000, 350]

        for seed in seeds:
            self.generate_samples(seed=seed)
            self.evaluator.generate_metrics_evaluation(self.generated_path)
            metrics["JSS"].append(self.evaluator.results["jss"])
            metrics["Similarity"].append(self.evaluator.results["avg_similarity"])
            metrics["String Similarity"].append(
                self.evaluator.results["avg_str_similarity"]
            )
            metrics["Validity"].append(self.evaluator.results["valid"])
            metrics["Novelty Percentage"].append(self.evaluator.results["novelty_perc"])
            metrics["Unique Percentage"].append(self.evaluator.results["unique_perc"])
            metrics["Average Inter Dissimilarity"].append(
                self.evaluator.results["average_inter_dissimilarity"]
            )
            metrics["Overall Validity"].append(self.evaluator.results["validated"])
            metrics["Normalized Vendi Score"].append(
                self.evaluator.results["vendi_score_k"]
            )
            metrics["Normalized Vendi Score (q=0.1)"].append(
                self.evaluator.results["vendi_score_k_small"]
            )
            metrics["Normalized Vendi Score (q=inf)"].append(
                self.evaluator.results["vendi_score_k_inf"]
            )
            metrics["Avg Normalized Vendi Score Per Class"].append(
                self.evaluator.results["avg_vs_score_per_class"]
            )
            metrics["Filter 1"].append(self.evaluator.results["filter1"])
            metrics["Filter 2"].append(self.evaluator.results["filter2"])
            metrics["Filter 3"].append(self.evaluator.results["filter3"])
            metrics["Filter 4"].append(self.evaluator.results["filter4"])

            # Print the results for the current seed
            logging.info(f"""Seed {seed}: JSS={self.evaluator.results["jss"]:.4f}, Sim={self.evaluator.results["avg_similarity"]:.4f}, StrSim={self.evaluator.results["avg_str_similarity"]:.4f}, Val={self.evaluator.results["valid"]:.4f}, NoveltyPerc={self.evaluator.results["novelty_perc"]:.4f}, 
            UniquePerc={self.evaluator.results["unique_perc"]:.4f}, IntDiv={self.evaluator.results["average_inter_dissimilarity"]:.4f}, OverallVal={self.evaluator.results["validated"]:.4f}, 
            NVS={self.evaluator.results["vendi_score_k"]:.4f}, NVS(q=0.1)={self.evaluator.results["vendi_score_k_small"]:.4f}, NVS(q=inf)={self.evaluator.results["vendi_score_k_inf"]:.4f}, AvgNVSPerClass={self.evaluator.results["avg_vs_score_per_class"]:.4f}""")

        # Print summary statistics
        logging.info("\nSummary Statistics:")
        logging.info(f"{'Metric':<25} {'Avg':<8} {'Std':<8} {'Min':<8} {'Max':<8}")

        for metric, values in metrics.items():
            logging.info(
                f"{metric:<25} {np.mean(values):<8.4f} {np.std(values):<8.4f} {np.min(values):<8.4f} {np.max(values):<8.4f}"
            )

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
                burn_cur_loss /= (
                    self.train_iter.get_data_num() - 1
                ) * batch_data_enc.size(0)
                if burn_pre_loss - burn_cur_loss < 0:
                    break
                burn_pre_loss = burn_cur_loss
                burn_cur_loss = 0

            batch_data_enc, _ = self.train_iter.sample()
            batch_data_enc = batch_data_enc.to(self.device)
            sub_iter += 1

    def burn_in_step(self, batch_data_enc):
        loss, _, _ = self.vae.loss(
            batch_data_enc,
            kl_weight=self.config["kl_start"],
            nsamples=self.config["VAE_n_training_samples"],
        )
        burn_loss = loss.sum().item()
        loss = loss.mean(dim=-1)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.vae.parameters(), self.config["clip_grad"])
        self.enc_optimizer.step()

        return burn_loss

    def compute_loss(self, data, kl_weight):
        loss, loss_rc, loss_kl = self.vae.loss(
            src=data,
            kl_weight=kl_weight,
            nsamples=self.config["VAE_n_training_samples"],
        )
        loss = loss.mean(dim=-1)
        return loss, loss_rc, loss_kl

    def optimize_loss(self, loss):
        self.enc_optimizer.zero_grad()
        self.dec_optimizer.zero_grad()

        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.vae.parameters(), self.config["clip_grad"])

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

        logging.info(f"mi: {mi:.4f} au: {au}")
        return {"loss": loss, "nll": nll, "kl": kl, "ppl": ppl}

    def check_improvement(self, eval_metrics, best_metrics):
        if eval_metrics["loss"] < best_metrics["loss"]:
            best_metrics.update(eval_metrics)
            logging.info(
                f"update best loss: {best_metrics['loss']:.4f}, best_nll: {best_metrics['nll']:.4f}, best_kl: {best_metrics['kl']:.4f}, best_ppl: {best_metrics['ppl']:.4f}"
            )

    def generate_samples(self, seed=42):
        self.vae.eval()
        if self.config["cuda"]:
            rng = torch.cuda.manual_seed(seed)
        else:
            rng = torch.manual_seed(seed)
        with torch.no_grad():
            self.vae.sample_from_prior(
                self.config["n_gen_samples"],
                "sample",
                self.generated_path,
                generator=rng,
            )

    def cyclical_annealing(T, M, step, R=0.4, max_kl_weight=1):
        """
        Implementing: <https://arxiv.org/abs/1903.10145>
        T = Total steps
        M = Number of cycles
        R = Proportion used to increase beta
        t = Global step
        """
        period = T / M  # N_iters/N_cycles
        # Itteration_number/(Global Period)
        internal_period = (step) % (period)
        tau = internal_period / period
        if tau > R:
            tau = max_kl_weight
        else:
            tau = min(max_kl_weight, tau / R)  # Linear function
        return tau


class uniform_initializer(object):
    def __init__(self, stdv):
        self.stdv = stdv

    def __call__(self, tensor):
        nn.init.uniform_(tensor, -self.stdv, self.stdv)


class xavier_normal_initializer(object):
    def __call__(self, tensor):
        nn.init.xavier_normal_(tensor)


class TransformerTrainer:
    def __init__(self, config):
        self.config = config

        # Setup paths
        self.train_path = config["train_path"]
        self.val_path = config["val_path"]
        self.generated_path = config["gene_path"]
        self.max_seq_len = config["seq_len"]
        self.device = torch.device("cuda" if config["cuda"] else "cpu")
        self.warmup_steps = config.get("warmup_steps", 4000)

        # Load tokenizer
        spm.SentencePieceTrainer.train(
            "--input=Liu_Kheyer_Retrosynthesis_Data/vocab2.txt --model_prefix=m "
            "--user_defined_symbols=[BOS],[EOS],[PAD],. --vocab_size=56 --bos_id=-1 --eos_id=-1"
        )
        self.tokenizer = spm.SentencePieceProcessor()
        self.tokenizer.load("m.model")
        self.PAD_TOKEN = self.tokenizer.encode_as_ids("[PAD]")[1]
        self.BOS_TOKEN = self.tokenizer.encode_as_ids("[BOS]")[1]
        self.EOS_TOKEN = self.tokenizer.encode_as_ids("[EOS]")[1]

        # Data
        self.train_iter = DataIterator(
            self.train_path, config["batch_size"], self.PAD_TOKEN
        )
        self.eval_iter = DataIterator(
            self.val_path, config["batch_size"], self.PAD_TOKEN
        )

        # Model
        self.model = Transformer(
            ntoken=config["vocab_size"],
            ninp=config["d_model"],
            nhead=config["nhead"],
            nhid=config["dim_feedforward"],
            nlayers=config["num_layers"],
            max_len=config["seq_len"],
            dropout=config["dropout"],
        ).to(self.device)

        # Load pretrained model if specified
        if config.get("load_path"):
            self.model.load_state_dict(
                torch.load(config["load_path"], map_location=self.device)
            )

        # Criterion and optimizer
        self.criterion = nn.CrossEntropyLoss(ignore_index=self.PAD_TOKEN, reduction="sum",label_smoothing=config["label_smoothing"]).to(
            self.device
        )
        self.optimizer = optim.Adam(
            self.model.parameters(), lr=config["lr"], betas=(0.9, 0.98), eps=1e-9
        )
        self.scheduler = optim.lr_scheduler.LambdaLR(
            self.optimizer,
            lr_lambda=lambda step: (self.config["d_model"] ** -0.5)
            * min((step + 1) ** -0.5, (step + 1) * self.warmup_steps**-1.5),
        )

        self.evaluator = Evaluator(config)

    def train(self):
        logging.info("### Starting Transformer training ###")
        for epoch in range(1, self.config["epochs"] + 1):
            train_loss = self.train_epoch()
            val_loss = self.evaluate()

            logging.info(
                f"Epoch {epoch} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}"
            )
            self.generate_and_evaluate()

            torch.save(
                self.model.state_dict(),
                os.path.join(
                    self.config["save_path"],
                    f"transformer_epoch{epoch}_loss{val_loss:.4f}.pt",
                ),
            )

    def train_epoch(self):
        self.model.train()
        total_loss = 0.0

        for src, tgt in self.train_iter:
            src, tgt = src.to(self.device), tgt.to(self.device)
            self.optimizer.zero_grad()

            output = self.model(src)
            output = F.log_softmax(output, dim=-1)
            loss = self.criterion(output.view(-1, output.size(-1)), tgt.view(-1))

            loss.backward()
            self.optimizer.step()
            self.scheduler.step()

            non_pad_mask = tgt.view(-1) != self.PAD_TOKEN
            num_valid_tokens = non_pad_mask.sum()
            loss = loss / num_valid_tokens
            total_loss += loss.item()

        self.train_iter.reset()
        return total_loss / len(self.train_iter)

    def evaluate(self):
        self.model.eval()
        total_loss = 0.0

        with torch.no_grad():
            for src, tgt in self.eval_iter:
                src, tgt = src.to(self.device), tgt.to(self.device)
                output = self.model(src)
                output = F.log_softmax(output, dim=-1)
                loss = self.criterion(output.view(-1, output.size(-1)), tgt.view(-1))
                non_pad_mask = tgt.view(-1) != self.PAD_TOKEN
                num_valid_tokens = non_pad_mask.sum()
                loss = loss / num_valid_tokens
                total_loss += loss.item()

        self.eval_iter.reset()
        return total_loss / len(self.eval_iter)

    def generate_and_evaluate(self):
        # Initialize metrics dictionary
        metrics = {
            "JSS": [],
            "Similarity": [],
            "String Similarity": [],
            "Validity": [],
            "Novelty Percentage": [],
            "Unique Percentage": [],
            "Average Inter Dissimilarity": [],
            "Overall Validity": [],
            "Normalized Vendi Score": [],
            "Normalized Vendi Score (q=0.1)": [],
            "Normalized Vendi Score (q=inf)": [],
            "Avg Normalized Vendi Score Per Class": [],
            "Filter 1": [],
            "Filter 2": [],
            "Filter 3": [],
            "Filter 4": [],
        }

        seeds = [42, 0, 250, 1000, 350]

        for seed in seeds:
            self.generate_samples(seed=seed)
            self.evaluator.generate_metrics_evaluation(self.generated_path)
            metrics["JSS"].append(self.evaluator.results["jss"])
            metrics["Similarity"].append(self.evaluator.results["avg_similarity"])
            metrics["String Similarity"].append(
                self.evaluator.results["avg_str_similarity"]
            )
            metrics["Validity"].append(self.evaluator.results["valid"])
            metrics["Novelty Percentage"].append(self.evaluator.results["novelty_perc"])
            metrics["Unique Percentage"].append(self.evaluator.results["unique_perc"])
            metrics["Average Inter Dissimilarity"].append(
                self.evaluator.results["average_inter_dissimilarity"]
            )
            metrics["Overall Validity"].append(self.evaluator.results["validated"])
            metrics["Normalized Vendi Score"].append(
                self.evaluator.results["vendi_score_k"]
            )
            metrics["Normalized Vendi Score (q=0.1)"].append(
                self.evaluator.results["vendi_score_k_small"]
            )
            metrics["Normalized Vendi Score (q=inf)"].append(
                self.evaluator.results["vendi_score_k_inf"]
            )
            metrics["Avg Normalized Vendi Score Per Class"].append(
                self.evaluator.results["avg_vs_score_per_class"]
            )
            metrics["Filter 1"].append(self.evaluator.results["filter1"])
            metrics["Filter 2"].append(self.evaluator.results["filter2"])
            metrics["Filter 3"].append(self.evaluator.results["filter3"])
            metrics["Filter 4"].append(self.evaluator.results["filter4"])

            # Print the results for the current seed
            logging.info(f"""Seed {seed}: JSS={self.evaluator.results["jss"]:.4f}, Sim={self.evaluator.results["avg_similarity"]:.4f}, StrSim={self.evaluator.results["avg_str_similarity"]:.4f}, Val={self.evaluator.results["valid"]:.4f}, NoveltyPerc={self.evaluator.results["novelty_perc"]:.4f}, 
            UniquePerc={self.evaluator.results["unique_perc"]:.4f}, IntDiv={self.evaluator.results["average_inter_dissimilarity"]:.4f}, OverallVal={self.evaluator.results["validated"]:.4f}, 
            NVS={self.evaluator.results["vendi_score_k"]:.4f}, NVS(q=0.1)={self.evaluator.results["vendi_score_k_small"]:.4f}, NVS(q=inf)={self.evaluator.results["vendi_score_k_inf"]:.4f}, AvgNVSPerClass={self.evaluator.results["avg_vs_score_per_class"]:.4f}""")

        # Print summary statistics
        logging.info("\nSummary Statistics:")
        logging.info(f"{'Metric':<25} {'Avg':<8} {'Std':<8} {'Min':<8} {'Max':<8}")

        for metric, values in metrics.items():
            logging.info(
                f"{metric:<25} {np.mean(values):<8.4f} {np.std(values):<8.4f} {np.min(values):<8.4f} {np.max(values):<8.4f}"
            )

    def generate_samples(self, seed):
        if self.config["cuda"]:
            rng = torch.cuda.manual_seed(seed)
        else:
            rng = torch.manual_seed(seed)
        self.model.eval()
        samples = []

        with torch.no_grad():
            for _ in range(self.config["n_gen_samples"]):
                input_seq = torch.tensor([[self.BOS_TOKEN]], dtype=torch.long).to(
                    self.device
                )
                generated = [self.BOS_TOKEN]

                for _ in range(1, self.max_seq_len):
                    output = self.model(input_seq, False)
                    output = F.log_softmax(output, dim=-1)
                    logits = (
                        output[0, -1]
                        .div(self.config.get("temperature", 1.0))
                        .exp()
                        .cpu()
                    )

                    next_token = torch.multinomial(
                        logits, num_samples=1, generator=rng
                    ).item()

                    generated.append(next_token)
                    if next_token == self.EOS_TOKEN:
                        generated.extend(
                            [self.PAD_TOKEN] * (self.max_seq_len - len(generated))
                        )
                        break

                    input_seq = torch.cat(
                        [
                            input_seq,
                            torch.tensor([[next_token]], dtype=torch.long).to(
                                self.device
                            ),
                        ],
                        dim=1,
                    )

                if len(generated) < self.max_seq_len:
                    generated.extend(
                        [self.PAD_TOKEN] * (self.max_seq_len - len(generated))
                    )

                samples.append(generated)

        with open(self.generated_path, "w", encoding="utf-8") as fout:
            fout.writelines([" ".join(map(str, sample)) + "\n" for sample in samples])
