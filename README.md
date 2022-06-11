# Cogito ( A simple header-only cuda library (easy version of cub & cutlass))

![](https://img.shields.io/github/workflow/status/sjfeng1999/cogito/release)


## Common

1. LD/ST 提高单条指令中操作的数据量
2. threadGroup概念的LD/ST操作
3. device函数生成特定的SASS指令

## Throughput

| BLAS           |    Size              |    cogito         |       cublas      |
|:--------------:|:--------------------:|:-----------------:|:-----------------:|
| Sgemm          | 2048 * 2048 * 2048   |    2036.19GB/s    |    2699.73GB/s    |

