| model                          |       size |     params | backend    |  n_cpu_moe | threads | n_ubatch | type_k | type_v |  fa |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | ---------: | ------: | -------: | -----: | -----: | --: | --------------: | -------------------: |
| nemotron_h_moe 31B.A3.5B Q4_0  |  17.59 GiB |    31.58 B | BLAS,MTL   |          0 |      10 |      512 |   q8_0 |   q8_0 |   1 |          pp2048 |        696.44 ± 1.08 |
| nemotron_h_moe 31B.A3.5B Q4_0  |  17.59 GiB |    31.58 B | BLAS,MTL   |          0 |      10 |      512 |   q8_0 |   q8_0 |   1 |           tg128 |         57.01 ± 0.03 |
| nemotron_h_moe 31B.A3.5B Q4_0  |  17.59 GiB |    31.58 B | BLAS,MTL   |          0 |      10 |     2048 |   q8_0 |   q8_0 |   1 |          pp2048 |        828.54 ± 0.64 |
| nemotron_h_moe 31B.A3.5B Q4_0  |  17.59 GiB |    31.58 B | BLAS,MTL   |          0 |      10 |     2048 |   q8_0 |   q8_0 |   1 |           tg128 |         57.20 ± 0.00 |
| nemotron_h_moe 31B.A3.5B Q4_0  |  17.59 GiB |    31.58 B | BLAS,MTL   |          4 |      10 |      512 |   q8_0 |   q8_0 |   1 |          pp2048 |       602.71 ± 31.39 |
| nemotron_h_moe 31B.A3.5B Q4_0  |  17.59 GiB |    31.58 B | BLAS,MTL   |          4 |      10 |      512 |   q8_0 |   q8_0 |   1 |           tg128 |         58.56 ± 0.04 |
| nemotron_h_moe 31B.A3.5B Q4_0  |  17.59 GiB |    31.58 B | BLAS,MTL   |          4 |      10 |     2048 |   q8_0 |   q8_0 |   1 |          pp2048 |       706.69 ± 32.40 |
| nemotron_h_moe 31B.A3.5B Q4_0  |  17.59 GiB |    31.58 B | BLAS,MTL   |          4 |      10 |     2048 |   q8_0 |   q8_0 |   1 |           tg128 |         57.33 ± 1.60 |
| nemotron_h_moe 31B.A3.5B Q4_0  |  17.59 GiB |    31.58 B | BLAS,MTL   |          7 |      10 |      512 |   q8_0 |   q8_0 |   1 |          pp2048 |       324.60 ± 18.89 |
| nemotron_h_moe 31B.A3.5B Q4_0  |  17.59 GiB |    31.58 B | BLAS,MTL   |          7 |      10 |      512 |   q8_0 |   q8_0 |   1 |           tg128 |        34.37 ± 11.61 |
| nemotron_h_moe 31B.A3.5B Q4_0  |  17.59 GiB |    31.58 B | BLAS,MTL   |          7 |      10 |     2048 |   q8_0 |   q8_0 |   1 |          pp2048 |       250.37 ± 10.59 |
| nemotron_h_moe 31B.A3.5B Q4_0  |  17.59 GiB |    31.58 B | BLAS,MTL   |          7 |      10 |     2048 |   q8_0 |   q8_0 |   1 |           tg128 |         51.10 ± 9.46 |

build: 5266f24da (10809)
