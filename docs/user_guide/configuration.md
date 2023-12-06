# Configuration Settings

!!! note

    Currently only delUQ models can be configured; GP models will be run as-is.

The DeepOpt library allows users to define custom configurations for training your model and conducting Bayesian optimization. To ensure a flexible and user-friendly experience, DeepOpt supports configuration through YAML and JSON files. This guide is designed to walk you through the available otpions and best practices for setting up your configuration files.

## The Base Configuration Options

There are several configuration options that are standard in a configuration file for delUQ models. Below is a table containing each of these options, a description of what they do, and their default values:

| Option           | Description | Default   |
| ------------     | ----------- | -------   |
| ff               | Text        | True      |
| activation       | Text        | relu      |
| n_layers         | Text        | 4         |
| hidden_dim       | Text        | 128       |
| mapping_size     | Text        | 128       |
| dropout          | Text        | True      |
| dropout_prob     | Text        | 0.2       |
| activation_first | Text        | True      |
| learning_rate    | Text        | 0.001     |
| n_epochs         | Text        | 1000      |
| batch_size       | Text        | 1000      |
| dist             | Text        | uniform   |
| opt_type         | Text        | Adam      |
| variance         | Text        | 0.0015625 |
| extrapolation_a1 | Text        | 0.05      |
| extrapolation_a2 | Text        | 0.95      |
| batchnorm        | Text        | False     |
| weight_decay     | Text        | False     |
| cutmix           | Text        | False     |
| se               | Text        | False     |
| positive_output  | Text        | False     |
