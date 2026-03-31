# Mini-GPT from Scratch

Transformer decoder-only (type GPT) implemented from scratch in PyTorch. 

Built as part of my engineering project at CentraleSupélec, heavily inspired by [Andrej Karpathy's "Let's build GPT"](https://www.youtube.com/watch?v=kCc8FmEb1nY).


## Usage

```bash
# download a dataset (e.g. Shakespeare)
wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt

# train with default config
python model.py

# experiment with architecture
python model.py --n_heads 8 --n_embd 128 --n_layer 6
python model.py --n_heads 1 --n_embd 64     # single head vs multi-head
python model.py --block_size 256             # longer context window
```


## Experiments

The model is intentionally small (~0.5M-5M params) so you can train on CPU/laptop and iterate fast. Things to try:

| Experiment | Command | What to observe |
|---|---|---|
| 1 head vs 4 heads | `--n_heads 1` vs `--n_heads 4` | Does multi-head attention help? |
| Small vs large embeddings | `--n_embd 32` vs `--n_embd 256` | Quality vs training speed tradeoff |
| Shallow vs deep | `--n_layer 2` vs `--n_layer 8` | When does depth stop helping? |
| Short vs long context | `--block_size 64` vs `--block_size 256` | Impact on long-range coherence |

## Stack

- Python 3.10+
- PyTorch

