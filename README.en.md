# Common Model IR (`.cm`) — Language Specification

## Purpose

`.cm` (Common Model IR) is a specification for describing

> **the “world structure” of papers, equations, and models
> in a canonical form that can be read consistently by humans and AI.**

This is **not a programming language**.  
It is **not** a replacement for Stan, PyTorch, JuMP, FEniCS, or similar tools.

`.cm` is an **intermediate representation (IR)** that directly captures:

> **the model section of a paper**
> (generative process, constraints, equations, and objectives)
> *as-is*, without embedding implementation details.

Implementation choices, numerical solvers, and algorithmic strategies are
**explicitly out of scope** for Common Model IR.

---

## Core Idea

A `.cm` document always consists of the following **seven blocks**:

```

model
index
given
unknown
define
rules
want

````

Using only these blocks, `.cm` can represent:

- Statistical models
- Mathematical optimization problems
- Ordinary / partial differential equations
- Physical laws
- Neural networks
- Physics-Informed Neural Networks (PINNs)

**within a single, unified abstract structure.**

---

## Semantics of Each Block

### `model`

The model name: paper title, section name, or theory name.

```yaml
model LinearRegression:
model SIR:
model MaxwellFDTD:
model GeneralRelativity:
````

This is an identifier only and carries no semantics.

---

### `index`

Defines the **meaning** of indices.

```yaml
index:
  i : sample
  t : time
  x : spacetime
  d : feature
  l : layer
```

* No numeric ranges or sizes are specified
* Only semantic meaning is written
* Indices serve as units of quantification for `for`

---

### `given`

Quantities provided externally.

Typical examples include:

* Observed data
* Constants
* Known fields
* Boundary / initial conditions
* Model structure (e.g., number of layers)

```yaml
given:
  x@train[i,d] : real
  y@train[i]   : real
  beta         : real
  N            : real
```

Annotations such as `@train` / `@pred` indicate
**observed vs. unobserved quantities**.
They are optional but recommended as semantic hints.

---

### `unknown`

Quantities that are **solved for or estimated** in this world.

* Parameters
* State variables
* Fields
* Design variables

```yaml
unknown:
  w[d]
  sigma
  g[x,a,b]
```

#### Principles

* Keep `unknown` **minimal**
* Do not include intermediate or derived quantities
* Only include quantities that are *free to be chosen*

---

### `define`

Quantities **derived from `given` and `unknown`**.

Examples include:

* Intermediate tensors
* Attention, Ricci tensor
* Softmax
* Matrix inverse
* Functions defined by equations

```yaml
define:
  mu[i] = sum_d x[i,d] * w[d] + b
  g_inv[x,a,b] = inverse(g[x,a,b])
```

#### Semantics of `define`

* `=` is allowed
* `define` introduces **definitions**, not constraints
* It is a macro-like expansion layer for mathematical structure

Crucially, `define` is also the place to **explicitly expose ambiguity**.

Many papers reference quantities that are:

* Implicit
* Defined only in code or appendices
* Subject to multiple reasonable interpretations

Such ambiguity must not be hidden.

##### Explicit Representation of Underspecification

```yaml
define:
  a(x) = ?                 # unknown / unspecified
  b(x) = ?{d(x), e(x)}     # multiple plausible candidates
  c(x) = ?{f(x)}           # implicit but dominant candidate
```

* `?` denotes an undefined quantity
* `?{...}` enumerates plausible interpretations

These are **not constraints**.
They preserve uncertainty without forcing implementation choices.

##### Granularity of `define`

The level of expansion is intentionally unspecified.

The goal is to:

* Preserve the paper’s model structure
* Avoid black boxes
* Improve readability of `rules`
* **Reveal where the model is not fully closed**

Numerical solvers, optimization algorithms,
and implementation-specific details
should either be omitted or contained within `define`.

---

### `rules`

The **laws, constraints, equations, and likelihoods**
governing the world.

Assignments are **not allowed** here.

Permitted relations:

* Equality constraints: `==`
* Inequalities: `< <= > >=`
* Probabilistic relations: `~`

```yaml
rules:
  sigma > 0
  for i:
    y[i] ~ Normal(mu[i], sigma)
```

All of the following belong in `rules`:

* ODEs / PDEs
* Physical laws
* Constraints
* Likelihoods
* Optimization constraints

---

### `for` (Quantification)

`for` represents **universal quantification (∀)**.

```yaml
for t:
  ∂S[t]/∂t == ...
```

Meaning:

> For all `t`, this relation holds.

* Not a loop
* No execution order
* Does not imply time-stepping or algorithms

---

### Mathematical Operators (Whitelist)

Allowed primitives:

* Arithmetic: `+ - * / ^`
* `sum`
* `exp`
* `sqrt`
* `max`
* `min`
* `∂`
* `∇`

Other functions must be expanded explicitly in `define`.

---

### `want`

Specifies **what is requested from this world**.

Only three verbs are allowed:

```yaml
want:
  fit ...
  predict ...
  optimize ...
```

#### Meanings

* `fit`
  Estimate `unknown` using observed `given`

* `predict`
  Generate unknowns or derived quantities following `rules`

* `optimize`
  Optimize an objective function

---

## Summary of the Semantics

A `.cm` document expresses:

> **a set of constraints (`rules`) imposed on unknowns and derived quantities,
> and a declaration (`want`) of how that constrained system is to be used.**

Statistical models, PDEs, optimization problems,
neural networks, and physical theories
all conform to this structure.

---

## Converting a Paper into `.cm`

1. Identify indices → `index`
2. List externally given quantities → `given`
3. Identify quantities to be solved → `unknown`
4. Write intermediate definitions → `define`
5. Write equations / laws / likelihoods → `rules`
6. Specify the objective → `want`

---

## Papers That Cannot Be Written in `.cm`

If a paper cannot be expressed in `.cm`,
it likely means that:

> **the model structure is underspecified
> and lacks a reproducible definition.**

Such papers require careful interpretation.

---

## What Is Common Model IR?

Common Model IR is

> **a shared specification for the “model layer”
> in science, physics, statistics, and machine learning.**

Implementation belongs to individual tools.
Common Model IR records **meaning and structure only**.
