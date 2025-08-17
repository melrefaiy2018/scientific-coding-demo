# Model Scoring Demonstration

This repository demonstrates how to compare simulated fluorescence data against experimental measurements using a custom scoring function.

---

## Mathematical definitions (GitHub render‑friendly)

### Cap-normalization

Each error is normalized and capped at 1.0:

```math
E_{i,\text{norm}} = \min\!\left( \frac{E_i}{E_{i,\max}}, 1.0 \right)
```

with fixed caps:

```math
E_{\text{time},\max} = 0.05, \quad
E_{\text{wl},\max} = 0.05, \quad
E_{\text{area},\max} = 0.20, \quad
E_{\tau,\max} = 0.10
```

### Final score

The combined score (lower is better):

```math
S = \tfrac{1}{2}\Big( w_{\text{time}}E_{\text{time,norm}}
+ w_{\text{wl}}E_{\text{wl,norm}}
+ w_{\text{area}}E_{\text{area,norm}}
+ w_{\tau}E_{\tau,\text{norm}} \Big)
+ \tfrac{1}{2}\sqrt{E_{\text{time,norm}} E_{\text{wl,norm}}}, \quad
w_i = 0.25
```

### Hard penalty

If either primary error is too large, the score is strongly penalized:

```math
\text{If } E_{\text{time,norm}} > 0.9 \text{ or } E_{\text{wl,norm}} > 0.9,
\quad S \leftarrow 10 S
```
