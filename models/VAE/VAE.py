import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence

import numpy as np


class VAE(nn.Module):
    """VAE with normal prior"""

    def __init__(self, encoder, decoder, args):
        super(VAE, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

        self.args = args

        self.nz = args.nz

        loc = torch.zeros(self.nz, device=args.device)
        scale = torch.ones(self.nz, device=args.device)

        self.prior = torch.distributions.normal.Normal(loc, scale)

    def encode(self, x, nsamples=1):
        """
        Returns: Tensor1, Tensor2
            Tensor1: the tensor latent z with shape [batch, nsamples, nz]
            Tensor2: the tenor of KL for each x with shape [batch]
        """
        return self.encoder.encode(x, nsamples)

    def encode_stats(self, x):
        """
        Returns: Tensor1, Tensor2
            Tensor1: the mean of latent z with shape [batch, nz]
            Tensor2: the logvar of latent z with shape [batch, nz]
        """

        return self.encoder(x)

    def decode(self, z, strategy, K=5):
        """generate samples from z given strategy

        Args:
            z: [batch, nsamples, nz]
            strategy: "sample"
            K: the beam width parameter

        Returns: List1
            List1: a list of decoded word sequence
        """

        if strategy == "sample":
            return self.decoder.sample_decode(z)
        else:
            raise ValueError("the decoding strategy is not supported")

    def reconstruct(self, x, decoding_strategy="greedy", K=5):
        """reconstruct from input x

        Args:
            x: (batch, *)
            decoding_strategy: "sample"
            K: the beam width parameter (if applicable)

        Returns: List1
            List1: a list of decoded word sequence
        """
        z = self.sample_from_inference(x).squeeze(1)

        return self.decode(z, decoding_strategy, K)

    def loss(self, src, target, kl_weight, nsamples=1):
        """
        Args:
            x: if the data is constant-length, x is the data tensor with
                shape (batch, *). Otherwise x is a tuple that contains
                the data tensor and length list

        Returns: Tensor1, Tensor2, Tensor3
            Tensor1: total loss [batch]
            Tensor2: reconstruction loss shape [batch]
            Tensor3: KL loss shape [batch]
        """

        z, KL = self.encode(src, nsamples)

        # (batch)
        reconstruct_err = self.decoder.reconstruct_error(
            src, z).mean(dim=1)

        return reconstruct_err + kl_weight * KL, reconstruct_err, KL

    def KL(self, x):
        _, KL = self.encode(x, 1)

        return KL

    def sample_from_prior(self, nsamples):
        """sampling from prior distribution

        Returns: Tensor
            Tensor: samples from prior with shape (nsamples, nz)
        """
        return self.prior.sample((nsamples,))

    def sample_from_inference(self, x, nsamples=1):
        """perform sampling from inference net
        Returns: Tensor
            Tensor: samples from infernece nets with
                shape (batch_size, nsamples, nz)
        """
        z, _ = self.encoder.sample(x, nsamples)

        return z
