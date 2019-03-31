# Computing Nonvacuous Generalization Bounds for Deep (Stochastic) Neural Networks with Many More Parameters than Training Data
This is an implementation of the PAC-Bayes generalization bound optimization for stochastic neural networks, as described in the article "[Computing Nonvacuous Generalization Bounds for Deep (Stochastic) Neural Networks with Many More Parameters than Training Data](https://arxiv.org/pdf/1703.11008.pdf)" by Dziugaite and Roy, published in *Uncertainty in AI* (2017).

## Requirements
- Python 3.5
- Numpy 1.14.5
- Tensorflow 1.10.1
- Keras 2.2.2

## Instructions
Running the code involves 2 steps:
1. SGD optimization, which saves initial and final network weights to `snn/experiments`  
2. PAC-Bayes optimization, which first loads the weights saved in the previous step, and then optimizes the PAC-Bayes bound over the weights and variances.

### SGD Optimization
To run SGD on a fully connected neural network consisting of a hidden layer with 600 neurons for 20 epochs on binary MNIST, execute the following command:

`python3.5 snn/experiments/run_sgd.py fc --layers 600 --sgd_epochs 20 --binary`

The code will throw a `FileExistsError` if a checkpoint already exists. To overwrite an existing checkpoint, use the following command: 

`python3.5 snn/experiments/run_sgd.py fc --layers 600 --sgd_epochs 20 --overwrite --binary`

### PAC-Bayes Optimization

The following command can be used to run the PAC-Bayes optimization for 1000 epochs on the saved checkpoint. 

`python3.5 snn/experiments/run_pacb.py fc --layers 600 --sgd_epochs 20 --pacb_epochs 1000 --lr 0.001 --drop_lr 250 --lr_factor 0.1 --binary`

The learning rate starts at 0.001 and is dropped to 0.0001 after 250 epochs.

## Acknowledgments

The development of this code was initiated by [Gintare Karolina Dziugaite](https://gkdz.org) and [Daniel M. Roy](http://danroy.org), while they were visiting the Simons Institute for the Theory of Computing at U.C. Berkeley. During this course of this research project, GKD was supported by an EPSRC studentship; DMR was supported by an NSERC Discovery Grant, Connaught Award, and U.S. Air Force Office of Scientific Research grant #FA9550-15-1-0074.

Waseem Gharbieh (Element AI) and Gabriel Arpino (University of Toronto) contributed to improving and testing the code, and helped produce this code release.

## BiBTeX

    @inproceedings{DR17,
            title = {Computing Nonvacuous Generalization Bounds for Deep (Stochastic) Neural Networks with Many More Parameters than Training Data},
           author = {Gintare Karolina Dziugaite and Daniel M. Roy},
             year = {2017},
        booktitle = {Proceedings of the 33rd Annual Conference on Uncertainty in Artificial Intelligence (UAI)},
    archivePrefix = {arXiv},
           eprint = {1703.11008},
    }
