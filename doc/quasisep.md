---
title: "Using quasiseparable matrices for scalable \\\\ Gaussian Process regression"
exports:
  - format: tex
    template: "../template"
  - format: pdf
    template: "../template"
authors:
  - name: Daniel Foreman-Mackey
    affiliations:
      - Center for Computational Astrophysics, Flatiron Institute
bibliography: quasisep.bib
math:
  '\mv': '\boldsymbol{#1}'
  '\mm': '\boldsymbol{#1}'
  '\T': '^\mathrm{T}'
---

+++ {"part": "acknowledgments"}

The authors would like to thank the Astronomical Data Group at the Center for Computational Astrophysics for listening to every iteration of this project and for providing great feedback every step of the way.

+++

+++ {"part": "abstract"}

This note presents some examples of how a class of matrices called "quasi-separable" appear naturally in many Gaussian Process (GP) regression models, and how this structure can be exploited to develop inference algorithms with linear scaling.
This class of models encapsulates many scalable GP methods that have been previously studied in the astrophysics literature, including _celerite_ [@Foreman-Mackey2017; @Gordon2020], _S+LEAF_ [@Delisle2020], and linear-Gaussian state-space models [@Kelly2014; @Jordan2021].
However, working within this framework also permits some generalizations that are useful for the analysis of current and upcoming astronomical datasets.
Alongside this note, these algorithms are implemented as part of the open source library [_tinygp_](https://github.com/dfm/tinygp).

+++

# Introduction

The quasiseparable kernels built in to `tinygp` are all designed to be used with one-dimensional data (see {ref}`api-kernels-quasisep`), but one of the key selling points of the `tinygp` implementation over other similar projects (e.g. [celerite](https://celerite.readthedocs.io), [celerite2](https://celerite2.readthedocs.io), [S+LEAF](https://obswww.unige.ch/~delisle/spleaf/doc/)), is that it has a model building interface that is more expressive and flexible.
In this tutorial, we present some examples of the kinds of extensions that are possible within this framework.
This will be one of the most technical `tinygp` tutorials, and the implementation details are likely to change in future versions; you have been warned!

```{figure} figures/demo.pdf
:name: demo
:width: 90%
:align: center

There exist a range of definitions for _quasiseparable matrices_ in the literature, so to be explicit, let's select the one that we will consider in all that follows.
```

# Quasiseparable matrices

There exist a range of definitions for _quasiseparable matrices_ in the literature, so to be explicit, let's select the one that we will consider in all that follows.
The most suitable definition for our purposes is nearly identical to the one used by @Eidelman1999, with some small modifications that will become clear as we go.

Let's start by considering an $N \times N$ _square quasiseparable matrix_ $\mm{M}$ with lower quasiseparable order $m_l$ and upper quasiseparable order $m_u$.
In this note, we represent this matrix $\mm{M}$ as:[^definition]

[^definition]: Comparing this definition to the one from @Eidelman1999, you may notice that we have swapped the labels of $\mv{g}_j$ and $\mv{h}_i$, and that we've added an explicit transpose to $\mm{B}_k\T$. These changes simplify the notation and implementation for symmetric matrices where, with our definition, $\mv{g} = \mv{p}$, $\mv{h} = \mv{q}$, and $\mm{B} = \mm{A}$.
\begin{equation}\label{eq:square-qsm-def}
  M_{ij} = \left \{ \begin{array}{ll}
    d_i\quad,                                                                   & \mbox{if }\, i = j \\
    \mv{p}_i\T\,\left ( \prod_{k=i-1}^{j+1} \mm{A}_k \right )\,\mv{q}_j\quad,   & \mbox{if }\, i > j \\
    \mv{h}_i\T\,\left ( \prod_{k=i+1}^{j-1} \mm{B}_k\T \right )\,\mv{g}_j\quad, & \mbox{if }\, i < j \\
  \end{array}\right .
\end{equation}
where

- $i$ and $j$ both range from $1$ to $N$,
- $d_i$ is a scalar,
- $\mv{p}_i$ and $\mv{q}_j$ are both vectors with $m_l$ elements,
- $\mm{A}_k$ is an $m_l \times m_l$ matrix,
- $\mv{g}_j$ and $\mv{h}_i$ are both vectors with $m_u$ elements, and
- $\mm{B}_k$ is an $m_u \times m_u$ matrix.

In [](#eq:square-qsm-def), the product notation is a little sloppy so, to be more explicit, this is how the products expand:
\begin{align}
  \prod_{k=i-1}^{j+1} \mm{A}_k &\equiv \mm{A}_{i-1}\,\mm{A}_{i-2}\cdots\mm{A}_{j+2}\,\mm{A}_{j+1}\quad\mbox{, and} \nonumber\\
  \prod_{k=i+1}^{j-1} \mm{B}_k\T &\equiv \mm{B}_{i+1}\T\,\mm{B}_{i+2}\T\cdots\mm{B}_{j-2}\T\,\mm{B}_{j-1}\T \quad.
\end{align}
When building intuition, it can often be useful to see a concrete example.
So, for example, in our notation, the general $4 \times 4$ quasiseparable matrix is represented as:
\begin{equation}
  \mm{M} = \left(\begin{array}{cccc}
      d_1                                & \mv{h}_1\T\mv{g}_2         & \mv{h}_1\T\mm{B}_2\T\mv{g}_3 & \mv{h}_1\T\mm{B}_2\T\mm{B}_3\T\mv{g}_4 \\
      \mv{p}_2\T\mv{q}_1                 & d_2                        & \mv{h}_2\T\mv{g}_3           & \mv{h}_2\T\mm{B}_3\T\mv{g}_4           \\
      \mv{p}_3\T\mm{A}_2\mv{q}_1         & \mv{p}_3\T \mv{q}_2        & d_3                          & \mv{h}_3\T\mv{g}_4                     \\
      \mv{p}_4\T\mm{A}_3\mm{A}_2\mv{q}_1 & \mv{p}_4\T\mm{A}_3\mv{q}_2 & \mv{p}_4\T\mv{q}_3           & d_4                                    \\
    \end{array}\right)
\end{equation}

# Quasiseparable linear algebra

The real reason why we're interested in quasiseparable matrices is that all the usual linear algebraic operations (matrix--vector products, factorization, inversion, solves, and others) can be computed using algorithms that scale linearly with the size of the matrix.
This is crucial for Gaussian process applications.

**Matrix--vector product:** Let's start by considering the strict lower quasiseparable matrix $\mm{L}$ with elements given by the following:
\begin{equation}
  L_{ij} = \left \{ \begin{array}{ll}
    \mv{p}_i\T\,\left ( \prod_{k=i-1}^{j+1} \mm{A}_k \right )\,\mv{q}_j\quad, & \mbox{if }\, i > j \\
    0\quad,                                                                   & \mbox{otherwise}   \\
  \end{array}\right .
\end{equation}
