import torch
import torch.nn as nn
import torch.nn.functional as F

"""
Code adapted from PyTorch implementation of 
"Lagging Inference Networks and Posterior Collapse
in Variational Autoencoders" (ICLR 2019),
https://github.com/jxhe/vae-lagging-encoder
"""


class LSTMDecoder(nn.Module):
    """LSTM decoder with constant-length batching"""

    def __init__(self, config, model_init, emb_init, BOS_token, EOS_token):
        super().__init__()
        self.ni = config["VAE_LSTM_embed_dim"]
        self.nh = config["LSTM_decoder_hidden_dim"]
        self.nz = config["VAE_latent_dim"]
        self.seq_len = config["seq_len"]
        self.bos_token = BOS_token
        self.eos_token = EOS_token
        self.device = torch.device("cuda" if config["cuda"] else "cpu")

        # no padding when setting padding_idx to -1
        self.embed = nn.Embedding(
            config['vocab_size'], config["VAE_LSTM_embed_dim"], padding_idx=-1)

        self.dropout_in = nn.Dropout(config["LSTM_decoder_dropout_in"])
        self.dropout_out = nn.Dropout(config["LSTM_decoder_dropout_out"])

        # for initializing hidden state and cell
        self.trans_linear = nn.Linear(
            config["VAE_latent_dim"], config["LSTM_decoder_hidden_dim"], bias=False)

        # concatenate z with input
        self.lstm = nn.LSTM(input_size=config["VAE_LSTM_embed_dim"] + config["VAE_latent_dim"],
                            hidden_size=config["LSTM_decoder_hidden_dim"],
                            num_layers=1,
                            batch_first=True)

        # prediction layer
        self.pred_linear = nn.Linear(
            config["LSTM_decoder_hidden_dim"], config['vocab_size'], bias=False)

        vocab_mask = torch.ones(config['vocab_size'])
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

        # Initialize the decoded_batch list to store outputs
        decoded_batch = [[] for _ in range(batch_size)]

        # Precompute c_init and h_init
        c_init = self.trans_linear(z).unsqueeze(0)
        h_init = torch.tanh(c_init)
        decoder_hidden = (h_init, c_init)

        # Initialize decoder input
        decoder_input = torch.full(
            (batch_size, 1), self.bos_token, dtype=torch.long, device=self.device)

        # Precompute z for concatenation to avoid repeating this inside the loop
        z_unsqueezed = z.unsqueeze(1)

        # Initialize the length counter
        length_c = 1

        # Tensor to store the final output sequences
        output_sequences = torch.zeros(
            (batch_size, self.seq_len+1), dtype=torch.long, device=self.device)

        # Tensor to track if the decoding should continue for each batch item
        active_mask = torch.ones(
            batch_size, dtype=torch.bool, device=self.device)
        while length_c <= self.seq_len+1 and active_mask.any():

            # Embedding and concatenation
            word_embed = self.embed(decoder_input)
            word_embed = torch.cat((word_embed, z_unsqueezed), dim=-1)

            # LSTM and linear layers
            output, decoder_hidden = self.lstm(word_embed, decoder_hidden)
            decoder_output = self.pred_linear(output).squeeze(1)

            # Sampling
            sample_prob = F.softmax(decoder_output, dim=1)
            sample_index = torch.multinomial(
                sample_prob, num_samples=1).squeeze(1)

            # Store outputs
            output_sequences[:, length_c - 1] = sample_index * active_mask

            # Update the input for the next step
            decoder_input = sample_index.unsqueeze(1)

            # Update active_mask to stop decoding if end token is reached
            # Assuming eos_token is the end-of-sequence token
            active_mask &= sample_index != self.eos_token

            # Update length counter
            length_c += 1

        # Convert the output_sequences tensor into a list of lists
        decoded_batch = output_sequences.tolist()

        decoded_batch = [
            [token for token in sequence] for sequence in decoded_batch
        ]

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
