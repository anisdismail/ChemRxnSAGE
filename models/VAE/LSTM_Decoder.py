import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence

import numpy as np

# TODO: provide reference where I brought the og implementation


class LSTMDecoder(nn.Module):
    """LSTM decoder with constant-length batching"""

    def __init__(self, args, model_init, emb_init, BOS_token):
        super().__init__()
        self.ni = args["ni"]
        self.nh = args["dec_nh"]
        self.nz = args["nz"]
        self.bos_token = BOS_token
        self.device = torch.device("cuda" if args["cuda"] else "cpu")

        # no padding when setting padding_idx to -1
        self.embed = nn.Embedding(
            args['vocab_size'], args["ni"], padding_idx=-1)

        self.dropout_in = nn.Dropout(args["dec_dropout_in"])
        self.dropout_out = nn.Dropout(args["dec_dropout_out"])

        # for initializing hidden state and cell
        self.trans_linear = nn.Linear(args["nz"], args["dec_nh"], bias=False)

        # concatenate z with input
        self.lstm = nn.LSTM(input_size=args["ni"] + args["nz"],
                            hidden_size=args["dec_nh"],
                            num_layers=1,
                            batch_first=True)

        # prediction layer
        self.pred_linear = nn.Linear(
            args["dec_nh"], args['vocab_size'], bias=False)

        vocab_mask = torch.ones(args['vocab_size'])
        # vocab_mask[vocab['[PAD]']] = 0
        self.loss = nn.CrossEntropyLoss(weight=vocab_mask, reduction='none')

        self.reset_parameters(model_init, emb_init)

    def reset_parameters(self, model_init, emb_init):
        for param in self.parameters():
            model_init(param)
        emb_init(self.embed.weight)

    def decode(self, input, z):
        """
        Args:
            input: (batch_size, seq_len)
            z: (batch_size, n_sample, nz)
        """

        # not predicting start symbol
        # sents_len -= 1

        batch_size, n_sample, _ = z.size()
        seq_len = input.size(1)

        # (batch_size, seq_len, ni)
        word_embed = self.embed(input)
        word_embed = self.dropout_in(word_embed)
        if n_sample == 1:
            z_ = z.expand(batch_size, seq_len, self.nz)

        else:
            word_embed = word_embed.unsqueeze(1).expand(batch_size, n_sample, seq_len, self.ni) \
                                   .contiguous()

            # (batch_size * n_sample, seq_len, ni)
            word_embed = word_embed.view(
                batch_size * n_sample, seq_len, self.ni)

            z_ = z.unsqueeze(2).expand(batch_size, n_sample,
                                       seq_len, self.nz).contiguous()
            z_ = z_.view(batch_size * n_sample, seq_len, self.nz)

        # (batch_size * n_sample, seq_len, ni + nz)
        word_embed = torch.cat((word_embed, z_), -1)

        z = z.view(batch_size * n_sample, self.nz)
        c_init = self.trans_linear(z).unsqueeze(0)
        h_init = torch.tanh(c_init)
        # h_init = self.trans_linear(z).unsqueeze(0)
        # c_init = h_init.new_zeros(h_init.size())
        output, _ = self.lstm(word_embed, (h_init, c_init))

        output = self.dropout_out(output)

        # (batch_size * n_sample, seq_len, vocab_size)
        output_logits = self.pred_linear(output)

        return output_logits

    def sample_decode(self, z):
        """sampling decoding from z
        Args:
            z: (batch_size, nz)
        Returns: List1
            List1: the decoded word sentence list
        """

        batch_size = z.size(0)
        decoded_batch = [[] for _ in range(batch_size)]

        # (batch_size, 1, nz)
        c_init = self.trans_linear(z).unsqueeze(0)
        h_init = torch.tanh(c_init)

        decoder_hidden = (h_init, c_init)
        decoder_input = torch.tensor(
            [self.bos_token] * batch_size, dtype=torch.long, device=self.device).unsqueeze(1)
        # end_symbol = torch.tensor([0] * batch_size, dtype=torch.long, device=self.device)

        mask = torch.ones((batch_size), dtype=torch.uint8, device=self.device)
        # TODO: investigate this number of maximum characters
        length_c = 1
        while length_c <= 201:

            # (batch_size, 1, ni) --> (batch_size, 1, ni+nz)
            word_embed = self.embed(decoder_input)
            word_embed = torch.cat((word_embed, z.unsqueeze(1)), dim=-1)

            output, decoder_hidden = self.lstm(word_embed, decoder_hidden)

            # (batch_size, 1, vocab_size) --> (batch_size, vocab_size)
            decoder_output = self.pred_linear(output)
            output_logits = decoder_output.squeeze(1)

            # (batch_size)
            sample_prob = F.softmax(output_logits, dim=1)
            sample_index = torch.multinomial(
                sample_prob, num_samples=1).squeeze(1)

            decoder_input = sample_index.unsqueeze(1)
            length_c += 1

            for i in range(batch_size):
             #       if mask[i].item():
                decoded_batch[i].append(sample_index[i].item())

            # mask = torch.mul((sample_index != end_symbol), mask)

        return decoded_batch

    def reconstruct_error(self, x, z):
        """Cross Entropy in the language case
        Args:
            x: (batch_size, seq_len)
            z: (batch_size, n_sample, nz)
        Returns:
            loss: (batch_size, n_sample). Loss
            across different sentence and z
        """

        # TODO: understand what is this doing
        # remove end symbol
        src = x[:, :-1]

        # remove start symbol
        tgt = x[:, 1:]

        batch_size, seq_len = src.size()
        n_sample = z.size(1)

        # (batch_size * n_sample, seq_len, vocab_size)
        output_logits = self.decode(src, z)

        if n_sample == 1:
            tgt = tgt.contiguous().view(-1)
        else:
            # (batch_size * n_sample * seq_len)
            tgt = tgt.unsqueeze(1).expand(batch_size, n_sample, seq_len) \
                .contiguous().view(-1)

        # (batch_size * n_sample * seq_len)
        loss = self.loss(output_logits.view(-1, output_logits.size(2)),
                         tgt)

        # (batch_size, n_sample)
        return loss.view(batch_size, n_sample, -1).sum(-1)

    def log_probability(self, x, z):
        """Cross Entropy in the language case
        Args:
            x: (batch_size, seq_len)
            z: (batch_size, n_sample, nz)
        Returns:
            log_p: (batch_size, n_sample).
                log_p(x|z) across different x and z
        """

        return -self.reconstruct_error(x, z)

# TODO: delete it if needed


class VarLSTMDecoder(LSTMDecoder):
    """LSTM decoder with variable-length batching"""

    def __init__(self, args, vocab, model_init, emb_init):
        super(VarLSTMDecoder, self).__init__(args, vocab, model_init, emb_init)

        self.embed = nn.Embedding(
            len(vocab), args.ni, padding_idx=vocab['<pad>'])
        vocab_mask = torch.ones(len(vocab))
        vocab_mask[vocab['<pad>']] = 0
        self.loss = nn.CrossEntropyLoss(weight=vocab_mask, reduction='none')

        self.reset_parameters(model_init, emb_init)

    def decode(self, input, z):
        """
        Args:
            input: tuple which contains x and sents_len
                    x: (batch_size, seq_len)
                    sents_len: long tensor of sentence lengths
            z: (batch_size, n_sample, nz)
        """

        input, sents_len = input

        # not predicting start symbol
        sents_len = sents_len - 1

        batch_size, n_sample, _ = z.size()
        seq_len = input.size(1)

        # (batch_size, seq_len, ni)
        word_embed = self.embed(input)
        word_embed = self.dropout_in(word_embed)

        if n_sample == 1:
            z_ = z.expand(batch_size, seq_len, self.nz)

        else:
            word_embed = word_embed.unsqueeze(1).expand(batch_size, n_sample, seq_len, self.ni) \
                                   .contiguous()

            # (batch_size * n_sample, seq_len, ni)
            word_embed = word_embed.view(
                batch_size * n_sample, seq_len, self.ni)

            z_ = z.unsqueeze(2).expand(batch_size, n_sample,
                                       seq_len, self.nz).contiguous()
            z_ = z_.view(batch_size * n_sample, seq_len, self.nz)

        # (batch_size * n_sample, seq_len, ni + nz)
        word_embed = torch.cat((word_embed, z_), -1)

        sents_len = sents_len.unsqueeze(1).expand(
            batch_size, n_sample).contiguous().view(-1)
        packed_embed = pack_padded_sequence(
            word_embed, sents_len.tolist(), batch_first=True)

        z = z.view(batch_size * n_sample, self.nz)
        # h_init = self.trans_linear(z).unsqueeze(0)
        # c_init = h_init.new_zeros(h_init.size())
        c_init = self.trans_linear(z).unsqueeze(0)
        h_init = torch.tanh(c_init)
        output, _ = self.lstm(packed_embed, (h_init, c_init))
        output, _ = pad_packed_sequence(output, batch_first=True)

        output = self.dropout_out(output)

        # (batch_size * n_sample, seq_len, vocab_size)
        output_logits = self.pred_linear(output)

        return output_logits

    def reconstruct_error_vae(self, x, z):
        """Cross Entropy in the language case
        Args:
            x: tuple which contains x_ and sents_len
                    x_: (batch_size, seq_len)
                    sents_len: long tensor of sentence lengths
            z: (batch_size, n_sample, nz)
        Returns:
            loss: (batch_size, n_sample). Loss
            across different sentence and z
        """

        x, sents_len = x

        # remove end symbol
        src = x[:, :-1]

        # remove start symbol
        tgt = x[:, 1:]

        batch_size, seq_len = src.size()
        n_sample = z.size(1)

        # (batch_size * n_sample, seq_len, vocab_size)
        output_logits = self.decode((src, sents_len), z)

        if n_sample == 1:
            tgt = tgt.contiguous().view(-1)
        else:
            # (batch_size * n_sample * seq_len)
            tgt = tgt.unsqueeze(1).expand(batch_size, n_sample, seq_len) \
                .contiguous().view(-1)

        # (batch_size * n_sample * seq_len)
        loss = self.loss(output_logits.view(-1, output_logits.size(2)),
                         tgt)

        # (batch_size, n_sample)
        return loss.view(batch_size, n_sample, -1).sum(-1)

    def log_probability(self, x, z):
        """Cross Entropy in the language case
        Args:
            x: (batch_size, seq_len)
            z: (batch_size, n_sample, nz)
        Returns:
            log_p: (batch_size, n_sample).
                log_p(x|z) across different x and z
        """

        return -self.reconstruct_error(x, z)
