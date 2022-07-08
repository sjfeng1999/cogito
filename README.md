# Cogito ( A simple header-only cuda library (easy version of cub & cutlass))

![](https://img.shields.io/github/workflow/status/sjfeng1999/cogito/release)

## Common

1. 向量化LD/ST操作
2. Reg Bankconflict消除

## Throughput

| Algo           |    Size              |    cogito      |    cublas      |   cudnn     |
|:--------------:|:--------------------:|:--------------:|:--------------:|:-----------:|
| Sgemm          | 2048 * 2048 * 2048   |  2057.19GB/s   |  2560.73GB/s   |      -      |
| Sigmoid        |   512 * 8192         |  -             |       -        |      -      |

## Roadmap

- [ ] 添加int32/8*4，float16等的支持
- [ ] 实现dnn::conv
- [ ] 实现splitK Gemm
- [ ] 实现tensorcore Gemm 
- [ ] 添加一些性能相关的static_assert
