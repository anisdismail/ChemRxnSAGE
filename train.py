import os
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn

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

        # Set models, criteria, optimizers
        self.generator = LSTM_LM(config['vocab_size'], config['g_embed_dim'],
                                 config['g_hidden_dim'], config['g_num_layers'], config['cuda'], config['g_dropout_prob'])

        self.nll_loss = nn.NLLLoss()
        if config["cuda"]:
            self.generator = self.generator.cuda()
            self.nll_loss = self.nll_loss.cuda()
            cudnn.benchmark = True
        self.gen_optimizer = optim.Adam(
            params=self.generator.parameters(), lr=config["gen_lr"])

        spm.SentencePieceTrainer.train(
            "--input=Liu_Kheyer_Retrosynthesis_Data/vocab2.txt --model_prefix=m  --user_defined_symbols=[BOS],[EOS],[PAD],. --vocab_size=56 --bos_id=-1 --eos_id=-1")
        self.tokenizer = spm.SentencePieceProcessor()
        self.tokenizer.load('m.model')
        self.train_iter = DataIterator(
            self.train_path, batch_size=config["batch_size"], PAD_TOKEN=self.tokenizer.encode_as_ids("[PAD]")[1])
        self.eval_iter = DataIterator(
            self.val_path, batch_size=config["batch_size"], PAD_TOKEN=self.tokenizer.encode_as_ids("[PAD]")[1])

    def train(self):
        print('#####################################################')
        print('Start pre-training generator with MLE...')
        print('#####################################################\n')

        for i in range(0, self.config["epochs"]):
            print(f"Step {i}")
            train_loss = self.train_mle()
            val_loss = self.eval_nll(self.eval_iter)
            self.generate_samples()
            jsd, avg_similarity, avg_str_similarity, valid, filter0, filter2, filter4, filter5, df, rxn_pred, sims, gen_fingerprints = generate_metrics_evaluation(
                self.generated_path, self.centroids, self.centroids_strings, self.tokenizer, self.config)

            # Print the values directly
            print(f"Train Loss: {train_loss:.5f}, Val Loss: {val_loss:.5f}")
            print(
                f"JSD: {jsd:.5f}, Similarity: {avg_similarity:.5f}, String Similarity: {avg_str_similarity:.5f}, Validity: {valid:.5f}\n")

        print('#####################################################\n\n')

    def train_mle(self):
        """
        Train generator with MLE
        """
        self.generator.train()
        total_loss = 0.
        for data, target in self.train_iter:
            if self.config["cuda"]:
                data, target = data.cuda(), target.cuda()
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
                if self.config["cuda"]:
                    data, target = data.cuda(), target.cuda()
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
        for _ in range(int(self.config["n_samples"] / self.config["batch_size"])):
            sample = self.generator.sample(
                self.config["batch_size"], self.config["seq_len"]).cpu().data.numpy().tolist()
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
        self.train_iter = DataIterator(
            self.train_path, batch_size=config["batch_size"], PAD_TOKEN=self.tokenizer.encode_as_ids("[PAD]")[1])
        self.eval_iter = DataIterator(
            self.val_path, batch_size=config["batch_size"], PAD_TOKEN=self.tokenizer.encode_as_ids("[PAD]")[1])
        self.config = config
        self.device = torch.device("cuda" if self.config["cuda"] else "cpu")
        model_init = uniform_initializer(0.01)
        emb_init = uniform_initializer(0.1)

        self.encoder = LSTMEncoder(
            self.config, config["vocab_size"], model_init, emb_init)
        self.decoder = LSTMDecoder(
            self.config, model_init, emb_init, self.tokenizer.encode_as_ids("[BOS]")[1], self.tokenizer.encode_as_ids("[EOS]")[1])
        self.vae = VAE(self.encoder, self.decoder, self.config).to(self.device)

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
                if self.config["cuda"]:
                    data, target = data.cuda(), target.cuda()
                batch_size, sent_len = data.size()
                report_num_sents += batch_size
                report_num_words += (sent_len - 1) * batch_size

                # not predict start symbol
                report_num_words += (sent_len - 1) * batch_size
                report_num_sents += batch_size

                loss, loss_rc, loss_kl = self.vae.loss(
                    data, 1.0, nsamples=self.config["n_training_samples"])

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
        opt_dict = {"not_improved": 0, "lr": 1., "best_loss": 1e4}
        best_loss = 1e4
        best_kl, best_nll, best_ppl, decay_cnt = 0, 0, 0, 0
        kl_weight = self.config["kl_start"]
        anneal_rate = (1.0 - self.config["kl_start"]) / (self.config["warm_up"] *
                                                         (self.train_iter.get_data_num() / self.config["batch_size"]))

        print("Starting Training.............")
        for epoch in range(self.config["epochs"]):
            self.vae.train()
            report_kl_loss = report_rec_loss = 0
            report_num_words = report_num_sents = 0
            for data, target in self.train_iter:
                if self.config["cuda"]:
                    data, target = data.cuda(), target.cuda()
                batch_size, sent_len = data.size()
                report_num_sents += batch_size
                report_num_words += (sent_len - 1) * batch_size
                kl_weight = min(1.0, kl_weight + anneal_rate)

                """
                sub_iter = 1
                burn_pre_loss = 1e4
                burn_cur_loss = 0
                burn_num_words = 0
                
                while self.config.aggressive and sub_iter < 100:

                    self.enc_optimizer.zero_grad()
                    self.dec_optimizer.zero_grad()
                    burn_batch_size, burn_sents_len = data.size()
                    burn_num_words += (burn_sents_len - 1) * burn_batch_size
                    loss, loss_rc, loss_kl = self.vae.loss(
                        data, target,
                        kl_weight, nsamples=self.config.nsamples)
                    burn_cur_loss += loss.sum().item()
                    loss = loss.mean(dim=-1)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        self.vae.parameters(), self.config.clip_grad)

                    self.enc_optimizer.step()
                    id_ = np.random.random_integers(
                        0, self.train_iter.get_data_num() - 1)

                    data, target = self.train_iter[id_]

                    if sub_iter % 15 == 0:
                        burn_cur_loss /= burn_num_words
                        if burn_pre_loss - burn_cur_loss < 0:
                            break
                        burn_pre_loss = burn_cur_loss
                        burn_cur_loss = burn_num_words = 0

                    sub_iter += 1
                """
                self.enc_optimizer.zero_grad()
                self.dec_optimizer.zero_grad()

                loss, loss_rc, loss_kl = self.vae.loss(
                    data, kl_weight, nsamples=self.config["n_training_samples"])
                loss = loss.mean(dim=-1)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.vae.parameters(), self.config["clip_grad"])
                report_rec_loss += loss_rc.sum().item()
                report_kl_loss += loss_kl.sum().item()

                """if not self.config["aggressive"]:
                    self.enc_optimizer.step()
                """
                self.enc_optimizer.step()
                self.dec_optimizer.step()

            train_loss = (report_rec_loss + report_kl_loss) / \
                report_num_sents

            print(f'epoch: {epoch}, avg_loss: {train_loss:.4f}, kl: {report_kl_loss / report_num_sents:.4f}, recon: {report_rec_loss / report_num_sents:.4f}')
            report_rec_loss, report_kl_loss, report_num_words, report_num_sents = 0, 0, 0, 0
            """
                TODO: add the mi to work on aggressive training
                if aggressive_flag and (iter_ % self.train)) == 0:
                    vae.eval()
                    cur_mi = calc_mi(vae, val_data_batch)
                    vae.train()
                    print("pre mi:%.4f. cur mi:%.4f" % (pre_mi, cur_mi))
                    if cur_mi - pre_mi < 0:
                        aggressive_flag = False
                        print("STOP BURNING")

                    pre_mi = cur_mi
            """

            print(f'kl weight { kl_weight:.4f}')
            loss, nll, kl, ppl = self.eval_nll(self.eval_iter)

            if loss < best_loss:
                best_loss, best_nll, best_kl, best_ppl = loss, nll, kl, ppl
                print(
                    f'update best loss: {best_loss:.4f}, best_nll: {best_nll:.4f}, best_kl: {best_kl:.4f}, best_pll: {best_ppl:.4f}')
                # torch.save(self.vae.state_dict(), self.config["save_path"])

            if loss > opt_dict["best_loss"]:
                opt_dict["not_improved"] += 1
                if opt_dict["not_improved"] >= self.config["decay_epoch"] and epoch >= 15:
                    opt_dict.update(
                        {"best_loss": loss, "not_improved": 0, "lr": opt_dict["lr"] * self.config.lr_decay})
                    decay_cnt += 1
                    # self.vae.load_state_dict(
                    #    torch.load(self.config["save_path"]))
                    print(f'new lr: {opt_dict["lr"]}, new decay: {decay_cnt}')

                    self.enc_optimizer = optim.SGD(self.vae.encoder.parameters(
                    ), lr=opt_dict["lr"], momentum=self.config["momentum"])
                    self.dec_optimizer = optim.SGD(self.vae.decoder.parameters(
                    ), lr=opt_dict["lr"], momentum=self.config["momentum"])
            else:
                opt_dict.update(
                    {"best_loss": loss, "not_improved": 0})

            if decay_cnt == self.config["max_decay"]:
                break
            self.train_iter.reset()

            self.generate_samples()
            jsd, avg_similarity, avg_str_similarity, valid, filter0, filter2, filter4, filter5, df, rxn_pred, sims, gen_fingerprints = generate_metrics_evaluation(
                self.generated_path, self.centroids, self.centroids_strings, self.tokenizer, self.config)
            print(
                f"JSD: {jsd:.5f}, Similarity: {avg_similarity:.5f}, String Similarity: {avg_str_similarity:.5f}, Validity: {valid:.5f}\n")

    def generate_samples(self):
        self.vae.eval()
        print('begin decoding..................................')
        with torch.no_grad():
            self.vae.sample_from_prior(self.config["n_samples"],
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
