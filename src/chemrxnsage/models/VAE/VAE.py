import torch
import torch.nn as nn
import math
import numpy as np
import random

"""
Code adapted from PyTorch implementation of
"Lagging Inference Networks and Posterior Collapse
in Variational Autoencoders" (ICLR 2019),
https://github.com/jxhe/vae-lagging-encoder
"""


def log_sum_exp(value, dim=None, keepdim=False):
    """Numerically stable implementation of the operation
    value.exp().sum(dim, keepdim).log()
    """
    if dim is not None:
        m, _ = torch.max(value, dim=dim, keepdim=True)
        value0 = value - m
        if keepdim is False:
            m = m.squeeze(dim)
        return m + torch.log(torch.sum(torch.exp(value0), dim=dim, keepdim=keepdim))
    else:
        m = torch.max(value)
        sum_exp = torch.sum(torch.exp(value - m))
        return m + torch.log(sum_exp)


class VAE(nn.Module):
    """VAE with normal prior"""

    def __init__(self, encoder, decoder, latent_dim, use_cuda):
        super(VAE, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

        self.nz = latent_dim
        self.device = 'cuda' if use_cuda else 'cpu'
        loc = torch.zeros(self.nz, device=self.device)
        scale = torch.ones(self.nz, device=self.device)
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

    def reconstruct(self, x, decoding_strategy="sample", K=5):
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

    def loss(self, src, kl_weight, nsamples=1):
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

    def sample_from_prior(self, nsamples, strategy, fname, generator):
        """sampling from prior distribution

        Returns: Tensor
            Tensor: samples from prior with shape (nsamples, nz)
        """
        # z = self.prior.sample((nsamples,))
        loc = torch.zeros(self.nz, device=self.device)
        scale = torch.ones(self.nz, device=self.device)
        with torch.no_grad():
            z = torch.normal(loc.expand((nsamples, self.nz)),
                             scale.expand((nsamples, self.nz)), generator=generator)
        with open(fname, 'w', encoding="utf-8") as fout:
            decoded_batch = self.decode(z, strategy)
            lines_to_write = [
                ' '.join(map(str, sample)) + '\n' for sample in decoded_batch]
            fout.writelines(lines_to_write)

    def sample_from_inference(self, x, nsamples=1):
        """perform sampling from inference net
        Returns: Tensor
            Tensor: samples from infernece nets with
                shape (batch_size, nsamples, nz)
        """
        z, _ = self.encoder.sample(x, nsamples)

        return z

    def calc_au(self, data_loader, delta=0.01):
        """compute the number of active units
        """
        cnt = 0
        for batch_data, _ in data_loader:
            batch_data = batch_data.to(self.device)
            mean, _ = self.encode_stats(batch_data)
            if cnt == 0:
                means_sum = mean.sum(dim=0, keepdim=True)
            else:
                means_sum = means_sum + mean.sum(dim=0, keepdim=True)
            cnt += mean.size(0)

        # (1, nz)
        mean_mean = means_sum / cnt
        data_loader.reset()

        cnt = 0
        for batch_data, _ in data_loader:
            batch_data = batch_data.to(self.device)
            mean, _ = self.encode_stats(batch_data)
            if cnt == 0:
                var_sum = ((mean - mean_mean) ** 2).sum(dim=0)
            else:
                var_sum = var_sum + ((mean - mean_mean) ** 2).sum(dim=0)
            cnt += mean.size(0)

        data_loader.reset()

        # (nz)
        au_var = var_sum / (cnt - 1)
        return (au_var >= delta).sum().item(), au_var

    def calc_mi(self, data_loader):
        mi = 0
        num_examples = 0
        for batch_data, _ in data_loader:
            batch_data = batch_data.to(self.device)
            batch_size = batch_data.size(0)
            num_examples += batch_size
            mutual_info = self.calc_mi_q(batch_data)
            mi += mutual_info * batch_size

        data_loader.reset()
        return mi / num_examples

    def calc_mi_q(self, x):
        """Approximate the mutual information between x and z
        I(x, z) = E_xE_{q(z|x)}log(q(z|x)) - E_xE_{q(z|x)}log(q(z))

        Returns: Float

        """

        # [x_batch, nz]
        mu, logvar = self.encode_stats(x)

        x_batch, nz = mu.size()

        # E_{q(z|x)}log(q(z|x)) = -0.5*nz*log(2*\pi) - 0.5*(1+logvar).sum(-1)
        neg_entropy = (-0.5 * nz * math.log(2 * math.pi) -
                       0.5 * (1 + logvar).sum(-1)).mean()

        # [z_batch, 1, nz]
        z_samples = self.encoder.reparameterize(mu, logvar, 1)

        # [1, x_batch, nz]
        mu, logvar = mu.unsqueeze(0), logvar.unsqueeze(0)
        var = logvar.exp()

        # (z_batch, x_batch, nz)
        dev = z_samples - mu

        # (z_batch, x_batch)
        log_density = -0.5 * ((dev ** 2) / var).sum(dim=-1) - \
            0.5 * (nz * math.log(2 * math.pi) + logvar.sum(-1))

        # log q(z): aggregate posterior
        # [z_batch]
        log_qz = log_sum_exp(log_density, dim=1) - math.log(x_batch)

        return (neg_entropy - log_qz.mean(-1)).item()

    def calc_mi_oneshot(self, data_loader):
        mus = []
        logvars = []
        z_samples_list = []

        for batch_data, _ in data_loader:
            batch_data = batch_data.to(self.device)

            # Encode statistics
            mu, logvar = self.encode_stats(batch_data)

            # Collect mu and logvar
            mus.append(mu)
            logvars.append(logvar)

            # Reparameterization trick to sample z
            z_samples = self.encoder.reparameterize(mu, logvar, 1)
            z_samples_list.append(z_samples)

        # Concatenate all collected data
        mus = torch.cat(mus, dim=0)
        logvars = torch.cat(logvars, dim=0)
        # Concatenate along batch dimension
        z_samples = torch.cat(z_samples_list, dim=0)

        # Entire dataset size and latent space size
        x_batch, nz = mus.size()

        # E_{q(z|x)}log(q(z|x))
        neg_entropy = (-0.5 * nz * math.log(2 * math.pi) -
                       0.5 * (1 + logvars).sum(-1)).mean()

        # Reshape for broadcasting
        mus = mus.unsqueeze(0)
        logvars = logvars.unsqueeze(0)
        var = logvars.exp()

        # Calculate the deviation
        dev = z_samples - mus

        # Log density calculation
        log_density = -0.5 * ((dev ** 2) / var).sum(dim=-1) - \
            0.5 * (nz * math.log(2 * math.pi) + logvars.sum(-1))

        # log q(z): aggregate posterior
        log_qz = log_sum_exp(log_density, dim=1) - math.log(x_batch)

        # Mutual information calculation
        mi = (neg_entropy - log_qz.mean(-1)).item()

        data_loader.reset()
        return max(mi, 0)
