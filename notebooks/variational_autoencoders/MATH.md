## Variational Autoencoder (VAE) and ELBO

The ELBO provides a tractable lower bound on the log-likelihood of the data and is maximized during VAE training:

```math
\text{ELBO}(\theta,\phi)=\mathbb{E}_{q_\phi(z|x)}[\log p_\theta(x|z)]-D_{KL}(q_\phi(z|x)\Vert p(z))
```
Where:
- $q_\phi(z|x)$ is the encoder
- $p_\theta(x|z)$ is the decoder
- $p(z)$ is the prior distribution (usually $\mathcal{N}(0,I)$)
- $D_{KL}$ is the Kullback-Leibler divergence

The ELBO consists of two terms:
1. A reconstruction term $\mathbb{E}_{q_\phi(z|x)} [\log p_\theta(x|z)]$ that measures how well we can reconstruct the input
2. A regularization term $D_{KL}(q_\phi(z|x) || p(z))$ that ensures our learned latent distribution stays close to the prior (a zero mean, unit gaussian)

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
