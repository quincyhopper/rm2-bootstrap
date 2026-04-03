# Comparing two classification models with bootstrapping
In this project, I train and evaluate the performance of two models for sentiment classification. The data consists of 36,515 Amazon reviews. Reviews are labelled 'positive' or 'negative'. Model 1 is a logistic regression and Model 2 uses the RoBERTa-large checkpoint with a small classification head (one fully connected linear layer, no activations). I use boostrapping to compare the performance of these models on the test set. Results showed that...

## Project structure
```
.
├── data/                    
│   ├── Compiled_Reviews.txt # Not included by default
├── boostrapping.py          # Boostrapping logic
├── data.py                  # Logic for making the datasets
├── jobscript.slurm          # Jobscript for running on cluster
├── model.py                 # Logreg and transformer model architecture
├── train.py                 # Training and evaluation logic
├── run.py                   # Main logic 
```

## Setup and installation
This project uses uv for dependency management. If you don't have uv installed, install it via:
```
curl -LsSf https://astral.sh/uv/install.sh | sh
```
1. **Clone this repo.**
```
git clone https://github.com/quincyhopper/rm2-bootstrap.git
cd rm2-bootstrap
```

2. **Install dependencies**
```
uv sync
```

## Run
To train and evaluate the two models on the cluster, run:
```
sbatch jobscript.slurm
```
Or locally:
```
uv run run.py
```

## Methods
No preprocessing was applied to the texts. The data was split 80/10/10% using seed 42. Both models used the AdamW optimiser (`lr=1e-4, weight_decay=1e-4`) to minimise the cross-entropy loss function. 

### Model 1: Logistic regression
To prepare the data for the linear regression model, each text was tokenised using the nltk library. A vocab-index mapping was then extracted from the unique tokens in the training set. Using this mapping, all texts were multi-hot encoded such that each one was represented by a tensor $T$ of shape ($1$, $V$) where $V$ is the vocab size. Each element of this tensor $T_i$ was set to $1$ if the text contained the token $V_i$. Finally, for each batch, the tensors were stacked into a tensor $\mathbf{X}$ of shape ($B$, $V$), where $B=256$. 

The linear layer effectively consists of a matrix $\mathbf{W}$ of shape ($V$, $2$) and a bias vector $\mathbf{b}$ of shape ($2$). As such, the output of this layer is

$$
\text{Logits} = \mathbf{X} \mathbf{W} + \mathbf{b}
$$

(Note: this is technically multi-class logistic regression but mathematically it is the same is normal linear regression.)

### Model 2: RoBERTa-large + logistic regression
I used the RoBERTa-large checkpoint to precompute embeddings for each review. Then, a logistic regression was trained on these embeddings. This logistic regression model differs from the first only in terms of its input size - that is, each batch has the shape ($B$, $1024$).

## Results