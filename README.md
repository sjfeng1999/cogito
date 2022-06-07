# Cogito ( A simple header-only cuda library (easy version of cub & cutlass))

![](https://img.shields.io/github/workflow/status/sjfeng1999/cogito/release)


## Common

1. LD/ST 提高单条指令中操作的数据量
2. 类似cooperative_group概念的特定group的LD/ST操作

## Algo  

|   General           |       BLAS            |       DNN             |
|:-------------------:|:---------------------:|:---------------------:|
| Elementwise         |     axpy(todo)        |  sigmoid              |
| Reduce              |     gemm              |  softmax              |
| scan(todo)          |     gemv(todo)        | conv2d (todo)         |

