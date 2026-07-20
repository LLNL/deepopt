# CLI reference

DeepOpt exposes two commands:

- `deepopt learn` trains a surrogate and writes a checkpoint.
- `deepopt optimize` loads a checkpoint and proposes new candidate inputs.

Training data files must be NumPy `.npz` files with keys `X` and `y`. `X` is an `N x d` input array and `y` is an `N` or `N x 1` objective array. Bounds are JSON encoded as one `[lower, upper]` pair per input dimension.

## Train a model

```bash
bounds='[[0, 1], [0, 1], [0, 1]]'
deepopt learn -i sims.npz -o learner_GP.ckpt -b "$bounds" -m GP
```

Important options:

| Option | Meaning |
| ------ | ------- |
| `-i`, `--infile` | Training data `.npz` file containing `X` and `y`. |
| `-o`, `--outfile` | Checkpoint path to write. |
| `-b`, `--bounds` | JSON list of `[lower, upper]` bounds in original input units. |
| `-m`, `--model-type` | Surrogate type: `GP`, `delUQ`, or `nnEnsemble`. |
| `-c`, `--config-file` | Optional YAML/JSON config file. |
| `-r`, `--random-seed` | Random seed for training and candidate generation. |
| `-k`, `--k-folds` | Number of folds used by delUQ training. |
| `-d`, `--device` | `auto`, `cpu`, `gpu`, or `cuda`. |
| `--multi-fidelity` | Treat the last input column as an integer fidelity index. |

## Propose candidates

Modern DeepOpt checkpoints are self-describing, so `optimize` can reload the training data, bounds, model type, configuration, and scaler state from the checkpoint:

```bash
deepopt optimize -l learner_GP.ckpt -o suggested_inputs.npy -a EI
```

Important options:

| Option | Meaning |
| ------ | ------- |
| `-l`, `--learner-file` | Checkpoint produced by `deepopt learn`. |
| `-o`, `--outfile` | NumPy `.npy` file to write proposed candidates into. |
| `-a`, `--acq-method` | Acquisition method: `EI`, `NEI`, `KG`, or `MaxValEntropy`. |
| `-q`, `--num-candidates` | Number of candidates to propose. |
| `-c`, `--config-file` | Optional optimize-time config overrides. |
| `--propose-best` | Use the first candidate for the current surrogate posterior maximizer. |
| `--integer-fidelities` | Save the multi-fidelity fidelity column as integers. |

Legacy checkpoints without `deepopt_checkpoint` metadata still require the original training data, bounds, and model type:

```bash
deepopt optimize \
  -l legacy_learner.ckpt \
  -i sims.npz \
  -b '[[0, 1], [0, 1], [0, 1]]' \
  -m GP \
  -o suggested_inputs.npy \
  -a EI
```

## Optimize-time configuration

For self-describing checkpoints, a config file passed to `optimize` can override candidate-generation settings without restating the training inputs:

```yaml title="optimize.yaml"
optimization:
  profile: fast
  batch_limit_high: 6
  torch_num_threads: 4
```

```bash
deepopt optimize -l learner_GP.ckpt -o suggested_inputs.npy -a EI -c optimize.yaml
```

See [Configuration Settings](configuration.md#optimization-settings) for optimization profiles and thread settings.

## Multi-fidelity optimization

Multi-fidelity data use the last input column as an integer fidelity index from `0` to `num_fidelities - 1`. Multi-fidelity candidate generation supports `KG` and `MaxValEntropy`.

```bash
deepopt learn \
  -i sims_mf.npz \
  -o learner_GP_mf.ckpt \
  -b '[[0, 1], [0, 1], [0, 2]]' \
  -m GP \
  --multi-fidelity

deepopt optimize \
  -l learner_GP_mf.ckpt \
  -o suggested_inputs.npy \
  -a KG \
  --fidelity-cost '[1, 4, 16]' \
  --integer-fidelities
```

`--fidelity-cost` must contain one cost per fidelity. DeepOpt rounds the candidate fidelity column before indexing this list.

## Risk-aware optimization

Risk measures are supported with `EI`, `NEI`, and `KG`. `MaxValEntropy` does not currently support risk measures.

```bash
deepopt optimize \
  -l learner_GP.ckpt \
  -o suggested_inputs.npy \
  -a EI \
  --risk-measure CVaR \
  --risk-level 0.8 \
  --risk-n-deltas 128 \
  --X-stddev '[0.02, 0.02, 0.02]'
```

`--X-stddev` is specified in original input units. For multi-fidelity optimization, the fidelity-column perturbation is forced to zero.

## Linear constraints

Linear constraints are JSON lists of `[indices, coefficients, rhs]` entries in original input units.

```bash
deepopt optimize \
  -l learner_GP.ckpt \
  -o suggested_inputs.npy \
  -a EI \
  --inequality-constraints '[[[0, 1], [1.0, -1.0], 0.0]]' \
  --equality-constraints '[[[2], [1.0], 0.5]]'
```

Inequalities use `sum(coefficients[i] * x[indices[i]]) >= rhs`. Equalities use the same left-hand side with `== rhs`. Constraint indices must be integers. DeepOpt converts constraints from original input units into the scaled optimizer coordinates internally.

## Nonlinear constraints

Nonlinear constraints are loaded from trusted local Python files:

```python title="constraints.py"
def make_constraints():
    def inside_circle(X):
        return 0.25 - ((X[..., 0] - 0.5) ** 2 + (X[..., 1] - 0.5) ** 2)
    return [inside_circle]
```

```bash
deepopt optimize \
  -l learner_GP.ckpt \
  -o suggested_inputs.npy \
  -a EI \
  --nonlinear-inequality-constraints constraints.py:make_constraints \
  --nonlinear-mode enforce
```

The named function can be a constraint callable or a zero-argument factory returning a callable or list of callables. Constraint callables receive candidate tensors in original input units. A candidate is feasible when every constraint returns values `>= 0`.

`--nonlinear-mode enforce` passes nonlinear constraints to BoTorch and forces `batch_limit=1`. For multiple candidates, DeepOpt uses sequential `q=1` optimization where supported. `--nonlinear-mode initialization-only` uses the constraints only to find feasible initial conditions, so final candidates are not guaranteed feasible.

Current limitations:

- Nonlinear constraints are single-fidelity only.
- Nonlinear constraints are not supported for multi-fidelity mixed optimization or fixed-feature subproblems.
- Nonlinear constraints are not currently supported with `KG`.
- Equality constraints are not supported for entropy acquisition candidate sets.
