# GCNSA：DNA storage encoding with graph convolutional Neural network and Self-attention

## Installation

```bash
git@github.com:caobencs/DEGCA-DNAstroagecoding.git
```

## Requirements
* tensorflow (>0.12)
* networkx
* networkx==2.2
* scipy==1.1.0
* setuptools==40.6.3
* numpy==1.15.4
* tensorflow==1.15.4
* 


Since model training is time consuming, we provide a trained model that you can try for the first time directly by running dnacoding.py with data.
```bash
python dnacoding.py
```
You can also train your own model after configuring the full environment
```bash
cd DEGCA
python train.py
```

If you're interested in preprocessing DNA sequences into graphs, see the image below.
![image](https://user-images.githubusercontent.com/87692512/189596982-09558b96-7f29-4f39-9804-45e174f753c0.png)

This is a tensorflow implementation of solving the DNA storage coding problem using graph convolutional networks and self-attention. The graph convolutional networks implementation is based on [GCN](https://github.com/tkipf/gcn) (MIT License).

