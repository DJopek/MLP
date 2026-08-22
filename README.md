
The network is

$$
    a^{(0,s)} \rightarrow a^{(1,s)} \rightarrow \dots \rightarrow a^{(L,s)} \rightarrow a^{(L+1,s)} ,
$$

with $$L$$ hidden layers, where

- $$a^{(0,s)}$$ is the input of sample $$s$$,
- $$a^{(L+1,s)}$$ is the output.

For each layer

$$
    a^{(K,s)}_\mu
    := \sigma \left(z^{(K,s)}_\mu\right)
    \equiv \sigma \left(
        w^{(K)}_{\mu\rho}\ a^{(K-1,s)}_\rho
        + b^{(K)}_\mu 
    \right),
$$

where

$$
    w^{(K)}_{\mu\rho}
    := \left(W^{(K)}\right)_{\mu\rho};
    \ K=1,\dots,L+1.
$$

We define

$$
    \eta^{(L+1,s)}_\mu
    := \frac{2}{d} \left(
        a^{(L+1,s)}_\mu 
        - \hat a^{(L+1,s)}_\mu
    \right) \sigma' \left(z^{(L+1,s)}_\mu\right),
$$

where $$d$$ is the number of training samples and
$$\hat a^{(L+1,s)}$$ is the GT.

We define recursive relation

$$
    \eta^{(K,s)}_\mu
    := \eta^{(K+1,s)}_\rho w^{(K+1)}_{\rho\mu} \sigma' \left(z^{(K,s)}_\mu\right); 
    \ K=L,\dots,1.
$$

The gradient with respect to the weights of layer $$K$$ is

$$
    \nabla_{W^{(K)}} C
    \equiv
    \frac{\partial C}{\partial W^{(K)}} 
    = \sum_{s=0}^{d-1} \eta^{(K,s)}_\mu a^{(K-1,s)}_\lambda e_\mu\otimes e_\lambda.
$$

In components

$$
    \left(\frac{\partial C}{\partial W^{(K)}}\right)_{\mu\lambda}
    = \sum_{s=0}^{d-1} \eta^{(K,s)}_\mu a^{(K-1,s)}_\lambda.
$$

The gradient with respect to the biases is

$$
    \nabla_{b^{(K)}} C
    \equiv
    \frac{\partial C}{\partial b^{(K)}}
    = \sum_{s=0}^{d-1}
    \eta^{(K,s)}_\mu e_\mu.
$$

In components

$$
    \left(\frac{\partial C}{\partial b^{(K)}}\right)_\mu
    = \sum_{s=0}^{d-1}
    \eta^{(K,s)}_\mu.
$$
