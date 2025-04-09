
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
<p><b>Please note that the repo and documnentation is still under active development.</b></p>

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
      <li><a href="#practicalities">Practicalities</a></li>
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
computational tools with expert-driven metrics, ChemRxnSAGE offers a comprehensive and exntendable solution that advances the state of chemical reaction generation and evaluation.

<!-- GETTING STARTED -->
## Getting Started

To get a local copy up and running follow steps below.

### Prerequisites

It is recommended to create a new virtual enviroment with [conda](https://www.anaconda.com/).

### Installation

1. Clone the repo
   ```sh
   git clone https://github.com/anisdismail/ChemRxnSAGE
   ```
2. Change to the project repositry
   ```sh
   cd ChemRxnSAGE
   ```
3. Run the following command to create a conda enviroment named chemgen_env with the necessary packages. Please note that it might take a few minutes for the environment to be created
   ```sh
   conda env create --file environment.yaml
   ```
4. Activate the environment
   ```sh
   conda activate chemgen_env
   ```

<!-- USAGE EXAMPLES -->
## Usage

### Running in command line
To run the algorithm in command line, run the following command with a configuration file as input
   ```sh
   python main.py --config config.json
   ```
The configuration file template is as follows
   ```json
{
    
}
   ```
The configuration parameters have the following functionality
```sh

```
### Running in a Jupyter Notebook Environment
You can also run the code in jupyter notebook. Details about setting up your own workflow in jupyter notebook can be found in the notebooks provided. Check the demo notebook for general usage.  

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
