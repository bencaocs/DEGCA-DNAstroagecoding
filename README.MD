# GCNSA：DNA storage encoding with graph convolutional Neural network and Self-attention

## Installation

```bash
git@github.com:caobencs/DEGCA-DNAstroagecoding.git
```

## Requirements
* tensorflow (>0.12)
* networkx
## recommend
* networkx==2.2
* scipy==1.3.0
* setuptools==59.6.0
* numpy==1.18.5
* tensorflow-gpu==1.13.1

more detail in requirements.yaml 
please use 
```bash
conda env create -f environments.yaml 
```

Since model training is time consuming, we provide a trained model that you can try for the first time directly by running dnacoding.py with data.
```bash
python DNAcoding.py
```
You can also train your own model after configuring the full environment
```bash
cd DEGCA
python train.py
```

If you're interested in preprocessing DNA sequences into graphs, see the image below.
![image](https://user-images.githubusercontent.com/87692512/189596982-09558b96-7f29-4f39-9804-45e174f753c0.png)

This is a tensorflow implementation of solving the DNA storage coding problem using graph convolutional networks and self-attention. The graph convolutional networks implementation is based on [GCN](https://github.com/tkipf/gcn) (MIT License).

