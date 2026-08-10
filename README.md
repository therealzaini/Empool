# Empool

This GitHub repository contains the source code of the *Empool* architecture
defined in our [paper](https://arxiv.org) and used for the experiments and
benchmarking presented therein.

## Overview of the Architecture

Our model is grounded in discrete Morse theory and leverages Forman's
collapsing to reduce graph size and computational resources while preserving
the model's predictive performance.

The model follows a Graph U-Net-like architecture, consisting of an
encoder-bottleneck-decoder pipeline.

The graph collapsing mechanism is governed by **Morse vectors**
$f^{(\ell)}$, which are dynamically computed and learned at each encoder
level through a small MLP conditioned on the feature representations
$\bm{H}^{(\ell)}$:

$$
f^{(\ell)}
=
f_{\mathrm{init}}^{(\ell)}
+
g_{\theta_\ell}\left(\bm{H}^{(\ell)}\right),
$$

where $f_{\mathrm{init}}^{(\ell)}$ is generated through a stochastic
initialization process:

$$
f_{\mathrm{init}}^{(\ell)}
=
\frac{\mathbf{x}_T}
{\left\|\mathbf{x}_T\right\|_2},
$$

with

$$
\mathbf{x}_T
=
\left(\bm{I}_n-\alpha\bm{L}\right)^T
\left(\mathbf{x}-\hat{\mathbf{x}}\bm{1}\right),
$$

where $\mathbf{x}$ is a random vector sampled from
$\mathcal{N}(\bm{0},\bm{I}_n)$, $T\in\mathbb{N}\setminus\{0\}$, and

$$
\alpha\in\left(0,\frac{1}{2d_{\max}}\right).
$$

The initialization process provides the model with a structured starting
point for learning the Morse function while avoiding the computational cost
of an explicit eigendecomposition of the graph Laplacian.

### General Forward Pass Pipeline

```mermaid
flowchart TD
    input["$$\text{Input } G_0, X_0$$"]
    en["Encoder"]
    en_out["$$H_{\text{bottleneck}}$$"]
    gc["Graph classifier"]
    nc["Node classifier"]
    ec["Edge classifier"]
    fcat["Feature concatenation"]
    dec["Decoder"]
    dec_out["$$H_{\text{output}}$$"]

    input --> en --> en_out
    en_out -->|"Graph classification"| gc
    en_out -->|"Node / Edge classification"| dec
    dec --> dec_out
    dec_out -->|"Node classification"| nc
    dec_out -->|"Edge classification"| fcat
    fcat --> ec
