# magic

Welcome to the magic autoencoder.



## Experiment 1
Experiments 1* are predominantly development experiments.
Experiments 2 and later will be actual experiments.

## Experiment 2

### Randomly uniformly guessing in range
| Experiment | Min | Max | Avg. squared loss |
| --- | :---: |:---: | -----:|
| 2a | 0 | 10000 | 1.9e+7 |
| 2b | 500 | 500 | 3.95e+6 |
| 2c | 350 | 2000 | 2.50e+6 |
| 2d | 0 | 0 | 5.8e+6 |

## Siamese architecture experiments

### Dense siamese networks
| Experiment | Num FC layers after dot product | Squared loss* |
| --- | :---: | ---: |
| 6b | 0 | 3.39e-1 |
| 6c | 1 | 1.43e+6 |
| 6d | 2 | 7.84e+6 |


`* - the loss displayed here is the average loss for the last 50
steps of a 10-epoch training session`

## Sampling tests

### 7d
This is without any specific metric space learning; however, we get
a really cool image if we sample the latent space.

![Sampling the latent space](https://raw.githubusercontent.com/ag8/magic/master/7d/latent_space_2d_sampling.png)

## Metricization experiments

| Experiment | Encoder neurons | Latent dimensions | Metric distance | mse (50 epochs) |
| --- | :---: | --- |--- | ---: |
| 8c | 500, 500, 500, 500 | 20 | cosine | 7.70e+6|
| 8d | 500, 500, 500, 500 | 200 | cosine | 7.35e+6|
| 8e | 500, 500, 500, 500 | 2000 | cosine | 7.54e+6|

Note: the more latent dimensions, the slower the reconstruction
seems to be learned

\*\*\*wow ok I just realized that these three experiments were all run
on the tiny dataset :(\*\*\*

### 9a

Distance is `1 over the overlap area`; that is, shapes that are close
together in latent space overlap really well, while shapes that are far
don't overlap well at all. This seems a lot more useful from a sampling
perspective (like if I have a drug in latent space, I can just sample
around it to get ligands that have very negative binding energies)

