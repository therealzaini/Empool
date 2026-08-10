# Empool

This GitHub repository contains the source code of the *Empool* architecture
defined in our [paper](https://arxiv.org) and used for the experiments and
benchmarking presented therein.


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
```