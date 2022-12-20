# CoPE

This is an unofficial PyTorch implementation for CIKM'21 paper [CoPE: Modeling Continuous Propagation and Evolution on Interaction Graph](https://dl.acm.org/doi/abs/10.1145/3459637.3482419).

The official PyTorch implementation by paper author is [here](https://github.com/FDUDSDE/CoPE).


## Usage

```shell
python main.py 
```

## Benchmarks

|                   | Garden |       | Video |       | Game  |       | ML100K |       | ML1M  |       | Yoochoosebuy |        |
|:-----------------:|:------:|:-----:|:-----:|:-----:|:-----:|:-----:|:------:|:-----:|:-----:|:-----:|:------------:|:------:|
|                   |  mrr   | hr@10 |  mrr  | hr@10 |  mrr  | hr@10 |  mrr   | hr@10 |  mrr  | hr@10 |     mrr      | hr@10  | 
| CoPE* (Official)  | 0.081  | 0.192 | 0.048 | 0.088 | 0.026 | 0.047 | 0.038  | 0.081 | 0.025 | 0.049 |    0.0113    | 0.0191 |
|       CoPE*       | 0.081  | 0.192 | 0.048 | 0.088 | 0.026 | 0.047 | 0.038  | 0.081 | 0.025 | 0.049 |    0.0113    | 0.0191 |
| CoPE* + jump_loss |        |       |       |       |       |       |        |       |       |       |              |        |


## Result

# Garden


|  data  | mrr | hr@10 | alpha_jump | lr  | weight_decay |
|:------:|:---:|:-----:|:----------:|:---:|:------------:|
| garden |     |       |            |     |              |
| garden |     |       |            |     |              |
| garden |     |       |            |     |              |


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