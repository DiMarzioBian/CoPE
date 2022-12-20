# CoPE

This is an unofficial PyTorch implementation for CIKM'21 paper [CoPE: Modeling Continuous Propagation and Evolution on Interaction Graph](https://dl.acm.org/doi/abs/10.1145/3459637.3482419).

The official PyTorch implementation by paper author is [here](https://github.com/FDUDSDE/CoPE).

## Difference from official repo
This project is re-written in my tuning framework from the paper's authors code.
- The only major difference is that I clone and detach the users' and items' embeddings during getting initial dynamic states.
- Same as the paper's author's code, meta-learning is not adopted. I've tried both but the changes on result is relatively small.
- Fast adaptation is not adopted.


## Usage

```shell
python main.py 
```

## Benchmarks

|          Model           | Garden |        | Video |       | Game  |       | ML100K |       | ML1M  |       | Yoochoosebuy |        |
|:------------------------:|:------:|:------:|:-----:|:-----:|:-----:|:-----:|:------:|:-----:|:-----:|:-----:|:------------:|:------:|
|                          |  mrr   | hr@10  |  mrr  | hr@10 |  mrr  | hr@10 |  mrr   | hr@10 |  mrr  | hr@10 |     mrr      | hr@10  | 
|     CoPE* (official)     | 0.081  | 0.192  | 0.048 | 0.088 | 0.026 | 0.047 | 0.038  | 0.081 | 0.025 | 0.049 |    0.0113    | 0.0191 |
|       CoPE* (mine)       | 0.0820 | 0.1913 |       |       |       |       |        |       |       |       |              |        |
| CoPE* + jump_loss (mine) | 0.0853 | 0.1905 |       |       |       |       |        |       |       |       |              |        |


## Result

# Garden
```shell
python main.py --cuda 0 --data garden --lr 1e-3 --weight_decay 5e-3 --alpha_jump 0
```

|  data  |  mrr   | hr@10  | alpha_jump |
|:------:|:------:|:------:|:----------:|
| garden |        |        |    1e-2    |
| garden |        |        |    1e-3    |
| garden | 0.0842 | 0.1837 |    1e-4    | 
| garden | 0.0853 | 0.1905 |    1e-5    |
| garden | 0.0820 | 0.1913 |     0      |


# Video


| data  | mrr | hr@10 | alpha_jump | lr  | weight_decay |
|:-----:|:---:|:-----:|:----------:|:---:|:------------:|
| video |     |       |            |     |              |
| video |     |       |            |     |              |
| video |     |       |            |     |              |


# Game


| data | mrr | hr@10 | alpha_jump | lr  | weight_decay |
|:----:|:---:|:-----:|:----------:|:---:|:------------:|
| game |     |       |            |     |              |
| game |     |       |            |     |              |
| game |     |       |            |     |              |


# ML100K


| data | mrr | hr@10 | alpha_jump | lr  | weight_decay |
|:----:|:---:|:-----:|:----------:|:---:|:------------:|
|  ml  |     |       |            |     |              |
|  ml  |     |       |            |     |              |
|  ml  |     |       |            |     |              |


# Yoo


| data | mrr | hr@10 | alpha_jump | lr  | weight_decay |
|:----:|:---:|:-----:|:----------:|:---:|:------------:|
| yoo  |     |       |            |     |              |
| yoo  |     |       |            |     |              |
| yoo  |     |       |            |     |              |