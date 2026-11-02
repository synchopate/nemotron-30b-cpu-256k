| model                          |       size |     params | backend    | threads | n_ubatch | type_k | type_v |  fa |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | ------: | -------: | -----: | -----: | --: | --------------: | -------------------: |
| nemotron_h_moe 31B.A3.5B Q4_0  |  17.59 GiB |    31.58 B | BLAS,MTL   |      10 |     1024 |   q8_0 |   q8_0 |   1 |          pp2048 |        776.64 ± 7.29 |
| nemotron_h_moe 31B.A3.5B Q4_0  |  17.59 GiB |    31.58 B | BLAS,MTL   |      10 |     1024 |   q8_0 |   q8_0 |   1 |            tg64 |         56.86 ± 0.07 |
| nemotron_h_moe 31B.A3.5B Q4_0  |  17.59 GiB |    31.58 B | BLAS,MTL   |      10 |     1024 |   q8_0 |   q8_0 |   1 |  pp2048 @ d8192 |        698.47 ± 0.14 |
| nemotron_h_moe 31B.A3.5B Q4_0  |  17.59 GiB |    31.58 B | BLAS,MTL   |      10 |     1024 |   q8_0 |   q8_0 |   1 |    tg64 @ d8192 |         54.40 ± 0.02 |
| nemotron_h_moe 31B.A3.5B Q4_0  |  17.59 GiB |    31.58 B | BLAS,MTL   |      10 |     1024 |   q8_0 |   q8_0 |   1 | pp2048 @ d32768 |        524.07 ± 0.72 |
| nemotron_h_moe 31B.A3.5B Q4_0  |  17.59 GiB |    31.58 B | BLAS,MTL   |      10 |     1024 |   q8_0 |   q8_0 |   1 |   tg64 @ d32768 |         48.99 ± 0.25 |

build: 5266f24da (10809)
