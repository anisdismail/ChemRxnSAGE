
<!-- PROJECT SHIELDS -->
[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![MIT Liscence][license-shield]][license-url]

<!-- PROJECT LOGO -->
<br />
<p align="center">
 <!-- <img src="" alt="logo" align="center"> -->
  <h3 align="center">ChemRxnSAGE: A Novel Framework for De Novo Chemical Reaction Generation and Evaluation</h3>

  <p align="center">
    <br />
    <br />
    <a href="https://github.com/anisdismail/ChemRxnSAGE/issues">Report Bug</a>
    ·
    <a href="https://github.com/anisdismail/ChemRxnSAGE/pulls">Add Feature</a>
  </p>
</p>
<p><b>Please note that the repo and documentation are still under active development.</b></p>

<!-- TABLE OF CONTENTS -->
<details open="open">
  <summary><h2 style="display: inline-block">Table of Contents</h2></summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#contributing">Contributing</a></li>
   <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
  </ol>
</details>



<!-- ABOUT THE PROJECT -->
## About The Project

![alt text](https://github.com/anisdismail/ChemRxnSAGE/blob/main/ChemRxnSAGE.png)

The generation and evaluation of chemical reactions remain challenging, with limited comprehensive studies addressing these issues. We introduce the **Chem**ical Reaction (**Rxn**)
**S**ystematic **A**ssessment of **G**eneration and **E**valuation (**ChemRxnSAGE**) framework, an **adaptable end-to-end approach** for evaluating the **quality**, **validity**, and **diversity** of **machine-generated
chemical reactions**. Combining automated validity filters with quality metrics and expert insights, ChemRxnSAGE systematically eliminates invalid reactions. We test its robustness
using generative models, including Recurrent Neural Networks and Variational Autoencoders, followed by validation using a chemical ”Turing test” with domain experts. Additionally,
we assess reaction feasibility through thermodynamic analysis and compare the generated reactions against existing literature to ensure relevance and novelty. By combining
computational tools with expert-driven metrics, ChemRxnSAGE offers a comprehensive and extendable solution that advances the state of chemical reaction generation and evaluation.

## Overview
- notebook_demo.ipynb:
- chem_reaction_classification.ipynb:
- data_preprocessing.ipynb:
- data_visualization.ipynb:
- evaluating_survey.ipynb: 

<!-- GETTING STARTED -->
## Getting Started

To get a local copy up and running, follow the steps below.

### Prerequisites

It is recommended to create a new virtual environment with [conda](https://www.anaconda.com/).

### Installation

1. Clone the repo
   ```sh
   git clone https://github.com/anisdismail/ChemRxnSAGE
   ```
2. Change to the project repository
   ```sh
   cd ChemRxnSAGE
   ```
3. Run the following command to create a conda environment named chemgen_env with the necessary packages. Please note that it might take a few minutes for the environment to be created
   ```sh
   conda env create --file environment.yaml
   ```
4. Activate the environment
   ```sh
   conda activate chemgen_env
   ```
5. Download the dataset to your local directory:

The training and evaluation datasets, trained model weights for all variations, and the generated results presented in the paper are available [here](https://drive.google.com/drive/folders/1-314k6YASzLk5l3bQ2PEsRxmQ4PMlI_t?usp=sharing).


<!-- USAGE EXAMPLES -->
## Usage

### Running in command line
To run the framework in the command line, run the following command with a configuration file as input
   ```sh
   python main.py --config config.json
   ```
There are two modes: **train** for training a model with the provided hyperparameters and **generate** is for sampling the trained latent space of this model and evaluating the generated reactions. 
The configuration file template is as follows:
   ```json
{
    "general": {
        "save_path": "<ADD_PATH>",
        "main_dir": "<ADD_DIRECTORY>",
        "train_path": "Liu_Kheyer_Retrosynthesis_Data/train/train_targets_ids_200.data",
        "val_path": "Liu_Kheyer_Retrosynthesis_Data/validation/validation_targets_ids_200.data",
        "data_ref_path": "train_ref_dataset.csv",
        "gene_path": "<ADD_PATH>/gene.data",
        "log_path": "<ADD_DIRECTORY>/train.log",
        "load_path": "<ADD_PATH>",
        "model": "LSTM",
        "mode": "generate",
        "seed": 42,
        "n_gen_samples": 39579,
        "cuda": true,
        "vocab_size": 56,
        "batch_size": 256,
        "seq_len": 200,
        "epochs": 200
    },
    "LSTM": {
        "LSTM_lr": 0.001,
        "LSTM_embed_dim": 64,
        "LSTM_hidden_dim": 256,
        "LSTM_num_layers": 1,
        "LSTM_dropout_prob": 0.5
    },
    "VAE": {
        "VAE_LSTM_embed_dim": 512,
        "VAE_latent_dim": 32,
        "momentum": 0.0,
        "warm_up": 10,
        "kl_start": 0.1,
        "aggressive": false,
        "LSTM_encoder_hidden_dim": 1024,
        "LSTM_decoder_hidden_dim": 1024,
        "LSTM_decoder_dropout_in": 0.5,
        "LSTM_decoder_dropout_out": 0.5,
        "clip_grad": 5.0,
        "decay_epoch": 2,
        "lr_decay": 0.5,
        "max_decay": 5,
        "VAE_n_training_samples": 1,
        "cyclical_annealing": false,
        "number_of_cycles": 20
    }
}
   ```
The configuration parameters have the following functionality
---
## `general` Section

| Key | Type | Description |
|---|---|---|
| `save_path` | `str` | Path to save model weights. |
| `main_dir` | `str` | Root directory containing datasets. |
| `train_path` | `str` | Path to the training dataset. |
| `val_path` | `str` | Path to the validation dataset. |
| `data_ref_path` | `str` | Reference Dataset CSV (e.g., mapping target IDs). |
| `gene_path` | `str` | Path to the generated data output. |
| `log_path` | `str` | Training log file path. |
| `load_path` | `str` | Pretrained model path (used in `generate` mode). |
| `model` | `str` | Type of model: `"LSTM"` or `"VAE"`. |
| `mode` | `str` | Mode of operation: `"train"` or `"generate"`. |
| `seed` | `int` | Random seed for reproducibility. |
| `n_gen_samples` | `int` | Number of samples to generate. |
| `cuda` | `bool` | Enable GPU acceleration. |
| `vocab_size` | `int` | Size of vocabulary used in tokenization. |
| `batch_size` | `int` | Batch size for training. |
| `seq_len` | `int` | Maximum input sequence length. |
| `epochs` | `int` | Number of training epochs. |

---

## `VAE` Section

| Key | Type | Description |
|---|---|---|
| `VAE_LSTM_embed_dim` | `int` | Embedding dimension for LSTM in VAE. |
| `VAE_latent_dim` | `int` | Size of latent space. |
| `momentum` | `float` | Momentum parameter for SGD. |
| `warm_up` | `int` | Epochs to linearly increase KL divergence weight. |
| `kl_start` | `float` | Initial weight for KL divergence. |
| `aggressive` | `bool` | Use aggressive training (true/false). |
| `LSTM_encoder_hidden_dim` | `int` | Hidden dimension for the encoder LSTM. |
| `LSTM_decoder_hidden_dim` | `int` | Hidden dimension for the decoder LSTM. |
| `LSTM_decoder_dropout_in` | `float` | Dropout rate for decoder input. |
| `LSTM_decoder_dropout_out` | `float` | Dropout rate for decoder output. |
| `clip_grad` | `float` | Gradient clipping value. |
| `decay_epoch` | `int` | Number of epochs after which to start learning rate decay. |
| `lr_decay` | `float` | Learning rate decay factor. |
| `max_decay` | `int` | Maximum number of decay steps. |
| `VAE_n_training_samples` | `int` | Number of training samples per epoch. |
| `cyclical_annealing` | `bool` | Enable cyclical annealing for KL divergence. |
| `number_of_cycles` | `int` | Number of cycles in cyclical annealing schedule. |

---

## `LSTM` Section

| Key | Type | Description |
|---|---|---|
| `LSTM_lr` | `float` | Learning rate for the LSTM model. |
| `LSTM_embed_dim` | `int` | Dimension of embedding layer. |
| `LSTM_hidden_dim` | `int` | Hidden layer size for LSTM. |
| `LSTM_num_layers` | `int` | Number of LSTM layers. |
| `LSTM_dropout_prob` | `float` | Dropout rate applied in LSTM layers. |

---
### Running in a Jupyter Notebook Environment
You can also run the code in a Jupyter notebook. Details about training and evaluation in a Jupyter Notebook can be found in the notebook provided [here](demo_notebook.ipynb).  

<!-- CONTRIBUTING -->
## Contributing

Any contributions you make are **greatly appreciated**.

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

<!--LICENSE -->
## License

Distributed under the GPL-3 License. See `LICENSE` for more information.


<!-- CONTACT -->
## Contact

[Anis Ismail](https://linkedin.com/in/anisdimail) - anis[dot]ismail[at]kuleuven[dot]be







<!-- MARKDOWN LINKS & IMAGES -->
[contributors-shield]: https://img.shields.io/github/contributors/anisdismail/ChemRxnSAGE.svg?style=for-the-badge
[contributors-url]: https://github.com/anisdismail/ChemRxnSAGE/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/anisdismail/ChemRxnSAGE.svg?style=for-the-badge
[forks-url]: https://github.com/anisdismail/ChemRxnSAGE/network/members
[stars-shield]: https://img.shields.io/github/stars/anisdismail/ChemRxnSAGE.svg?style=for-the-badge
[stars-url]: https://github.com/anisdismail/ChemRxnSAGE/stargazers
[issues-shield]: https://img.shields.io/github/issues/anisdismail/ChemRxnSAGE.svg?style=for-the-badge
[issues-url]: https://github.com/anisdismail/ChemRxnSAGE/issues
[license-shield]: https://img.shields.io/badge/license-GPL--3.0--only-green?style=for-the-badge
[license-url]: https://github.com/anisdismail/ChemRxnSAGE/LICENSE
