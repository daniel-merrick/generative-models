
## Overview
This page is mostly based on [this excellent resource](https://arxiv.org/pdf/2208.11970).

One approach of generative modeling, termed "likelihood-based", is to learn a model to
maximize the likelihood p(x) of all observed x.

We can consider a VAE as something that models the joint distribution of x and z. 

To maximize p(x) is impossible because it's intractible (see resource), so we maximize the Evidence Lower BOund (ELBO) of $p_\theta(x)$.

## ELBO Derivation
Starting with the log likelihood and using the fact that $\int q_\phi(z|x)dz = 1$:

```math
\log p_\theta(x) = \log p_\theta(x) \int q_\phi(z|x)dz
```

This is equal to the expectation:

```math
= \mathbb{E}_{q_\phi(z|x)}[\log p_\theta(x)]
```

We can rewrite using the conditional probability rule:

```math
= \mathbb{E}_{q_\phi(z|x)}\left[\log \frac{p_\theta(x,z)}{p_\theta(z|x)}\right]
```

Multiply and divide by the variational distribution:

```math
= \mathbb{E}_{q_\phi(z|x)}\left[\log \frac{p_\theta(x,z)}{p_\theta(z|x)} \cdot \frac{q_\phi(z|x)}{q_\phi(z|x)}\right]
```

Which gives us:

```math
= \mathbb{E}_{q_\phi(z|x)}\left[\log \frac{p_\theta(x,z)}{q_\phi(z|x)}\right] + D_{KL}(p_\theta(z|x) \Vert q_\phi(z|x))
```

This form shows how the log likelihood decomposes into the ELBO plus the KL divergence between the true posterior and the approximate posterior.

## Deriving the VAE Objective from ELBO

Starting from our derived ELBO:

```math
\mathbb{E}_{q_\phi(z|x)}\left[\log \frac{p_\theta(x,z)}{q_\phi(z|x)}\right]
```

We can expand the joint distribution using the chain rule:

```math
\mathbb{E}_{q_\phi(z|x)}\left[\log \frac{p_\theta(x|z)p(z)}{q_\phi(z|x)}\right]
```

Using the properties of logarithms:

```math
\mathbb{E}_{q_\phi(z|x)}\left[\log p_\theta(x|z) + \log p(z) - \log q_\phi(z|x)\right]
```

We can separate this into two expectations:

```math
\mathbb{E}_{q_\phi(z|x)}[\log p_\theta(x|z)] + \mathbb{E}_{q_\phi(z|x)}[\log p(z) - \log q_\phi(z|x)]
```

The second term is the negative KL divergence:

```math
= \mathbb{E}_{q_\phi(z|x)}[\log p_\theta(x|z)] - D_{KL}(q_\phi(z|x)\Vert p(z))
```
Where:
- $q_\phi(z|x)$ is the encoder
- $p_\theta(x|z)$ is the decoder
- $p(z)$ is the prior distribution, a zero mean unit norm gaussian
- $D_{KL}$ is the Kullback-Leibler divergence

The ELBO consists of two terms:
1. A reconstruction term $E_{q_\phi(z|x)}[\log p_\theta(x|z)]$ that measures how well we can reconstruct the input
2. A regularization term $D_{KL}(q_\phi(z|x) || p(z))$ that ensures our learned latent distribution stays close to the prior (a zero mean, unit gaussian)

## Reconstruction Term
The reconstruction term in the ELBO can be directly related to MSE when we make certain assumptions about the decoder's output distribution. 

For a Gaussian decoder with fixed variance $\sigma^2$, the negative log-likelihood (NLL) is proportional to MSE:

$$
-\log p_\theta(x|z) \propto \frac{1}{2\sigma^2}||x - \hat{x}||^2 + C
$$

Where:
- $x$ is the input data
- $\hat{x}$ is the reconstructed output
- $||x - \hat{x}||^2$ is the squared L2 norm (MSE)
- $C$ is a constant term

Therefore, minimizing the negative reconstruction term:

```math
-\mathbb{E}_{q_\phi(z|x)}[\log p_\theta(x|z)]
```
is equivalent to minimizing MSE when:
1. The decoder models a Gaussian distribution
2. The variance $\sigma^2$ is fixed
3. We ignore constant terms

This is why many VAE implementations simply use MSE as their reconstruction loss, though this is technically only correct under these specific assumptions about the decoder's output distribution.

## Independent Gaussian Assumption in VAEs

For an image with dimensions H×W×C, the decoder models:

$$p_\theta(x|z) = \prod_{i=1}^{H} \prod_{j=1}^{W} \prod_{c=1}^{C} \mathcal{N}(x_{i,j,c}; \mu_{i,j,c}, \sigma^2_{i,j,c})$$

This means:
1. Each pixel is its own Gaussian distribution
2. Pixels are assumed to be independent of each other
3. Color channels are also independent

### Limitations of this Assumption
- Pixels in real images are highly correlated with their neighbors
- This independence assumption leads to blurry reconstructions
- It's why VAEs often struggle with sharp details compared to GANs
