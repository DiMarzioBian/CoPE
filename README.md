# CoPE

This is an unofficial PyTorch implementation for CIKM'21 paper [CoPE: Modeling Continuous Propagation and Evolution on Interaction Graph](https://dl.acm.org/doi/abs/10.1145/3459637.3482419).

The official PyTorch implementation by paper author is [here](https://github.com/FDUDSDE/CoPE).

## Difference from official repo
This project is re-written in my tuning framework from the paper's authors code.
- The only major difference is that I clone and detach the users' and items' embeddings during getting initial dynamic states.
- Same as the paper's author's code, meta-learning is not adopted. I've tried w/ and w/o but the difference on result is relatively small.
- Fast adaptation is not adopted which requires training on validation and testing time.


## Usage

```shell
python main.py 
```

## Benchmarks

|          Model          | Garden |        | Video  |        |  Game  |        | ML100K |        |  ML1M  |        | Yoochoosebuy |        |
|:-----------------------:|:------:|:------:|:------:|:------:|:------:|:------:|:------:|:------:|:------:|:------:|:------------:|:------:|
|                         |  mrr   | hr@10  |  mrr   | hr@10  |  mrr   | hr@10  |  mrr   | hr@10  |  mrr   | hr@10  |     mrr      | hr@10  | 
|    CoPE* (official)     | 0.081  | 0.192  | 0.048  | 0.088  | 0.026  | 0.047  | 0.038  | 0.081  | 0.025  | 0.049  |    0.0113    | 0.0191 |
|       CoPE* (new)       | 0.0820 | 0.1913 | 0.0465 | 0.0951 | 0.0307 | 0.0585 | 0.0424 | 0.0882 | 0.0253 | 0.0508 |    0.0462    | 0.0957 |
| CoPE* + jump_loss (new) | 0.0853 | 0.1905 | 0.0468 | 0.0959 | 0.0316 | 0.0588 | 0.0423 | 0.0908 | 0.0255 | 0.0519 |    0.0473    | 0.0980 |


## Result

# Garden
```shell
python main.py --cuda 0 --data garden --lr 1e-3 --l2 5e-3 --alpha_jump 0
```
|  data  |  mrr   | hr@10  | alpha_jump |
|:------:|:------:|:------:|:----------:|
| garden | 0.0782 | 0.1935 |    1e-2    |
| garden | 0.0812 | 0.1935 |    1e-3    |
| garden | 0.0842 | 0.1837 |    1e-4    | 
| garden | 0.0853 | 0.1905 |    1e-5    |
| garden | 0.0820 | 0.1913 |     0      |


# Video
```shell
python main.py --cuda 0 --data video --lr 1e-3 --l2 5e-3 --alpha_jump 0
```
| data  |  mrr   | hr@10  | alpha_jump |  lr  |  l2  |
|:-----:|:------:|:------:|:----------:|:----:|:----:|
| video | 0.0341 | 0.0751 |    1e-2    | 1e-3 | 2e-2 | 
| video | 0.0348 | 0.0805 |    1e-3    | 1e-3 | 2e-2 | 
| video | 0.0445 | 0.0959 |    1e-4    | 1e-3 | 2e-2 | 
| video | 0.0468 | 0.0959 | ***1e-5*** | 1e-3 | 2e-2 |
| video | 0.0465 | 0.0951 |     0      | 1e-3 | 2e-2 |


# Game
```shell
python main.py --cuda 0 --data game --lr 1e-3 --l2 5e-3 --alpha_jump 0
```
| data |  mrr   | hr@10  | alpha_jump |  lr  |  l2  |
|:----:|:------:|:------:|:----------:|:----:|:----:|
| game | 0.0285 | 0.0554 |    1e-2    | 1e-3 | 1e-3 |
| game | 0.0288 | 0.0560 |    1e-3    | 1e-3 | 1e-3 |
| game | 0.0297 | 0.0572 |    1e-4    | 1e-3 | 1e-3 |
| game | 0.0316 | 0.0588 | ***1e-5*** | 1e-3 | 1e-3 |
| game | 0.0307 | 0.0585 |     0      | 1e-3 | 1e-3 |


# ML100K
```shell
python main.py --cuda 0 --data ml --lr 5e-4 --l2 1e-3 --alpha_jump 0
```
| data |  mrr   | hr@10  | mrr_loss | hr@10_loss | alpha_jump |  lr  |  l2  |
|:----:|:------:|:------:|:--------:|:----------:|:----------:|:----:|:----:|
|  ml  | 0.0444 | 0.0957 |          |            |    1e-2    | 5e-4 | 1e-3 |
|  ml  | 0.0456 | 0.1008 |  0.0421  |   0.0858   |    1e-3    | 5e-4 | 1e-3 |
|  ml  | 0.0467 | 0.1001 |  0.0412  |   0.0880   |    1e-4    | 5e-4 | 1e-3 |
|  ml  | 0.0463 | 0.0977 |  0.0423  |   0.0908   |    1e-5    | 5e-4 | 1e-3 |
|  ml  | 0.0440 | 0.0921 |  0.0424  |   0.0882   |     0      | 5e-4 | 1e-3 |


# ML1M
```shell
python main.py --cuda 0 --data mlm --lr 5e-4 --l2 1e-3 --alpha_jump 1e-5
```
| data |  mrr   | hr@10  | alpha_jump |  lr  |  l2  |
|:----:|:------:|:------:|:----------:|:----:|:----:|
|  ml  |        |        |    1e-2    | 5e-4 | 1e-3 |
|  ml  | 139-1  |        |    1e-3    | 5e-4 | 1e-3 |
|  ml  | 201-1  |        |    1e-4    | 5e-4 | 1e-3 |
|  ml  | 0.0255 | 0.0519 |    1e-5    | 5e-4 | 1e-3 |
|  ml  | 0.0253 | 0.0508 |     0      | 5e-4 | 1e-3 |


# Yoo
```bash
python main.py --cuda 0 --data yoo --n_batch_load 20 --lr 1e-5 --lr_step 1 --lr_gamma 0.1 --l2 2 --alpha_jump 0
```
| data |  mrr   | hr@10  | alpha_jump |  lr  | l2  |
|:----:|:------:|:------:|:----------:|:----:|:---:|
| yoo  | 0.0473 | 0.0959 |    1e-2    | 1e-5 |  2  |
| yoo  | 0.0473 | 0.0980 |    1e-3    | 1e-5 |  2  |
| yoo  | 0.0467 | 0.0966 |    1e-4    | 1e-5 |  2  |
| yoo  | 0.0465 | 0.0964 |    1e-5    | 1e-5 |  2  |
| yoo  | 0.0462 | 0.0957 |     0      | 1e-5 |  2  |