# Tutorial
In this page we'll present several tutorials on how to use different parts of the DeepOpt library.

## Tutorial: Using neural network surrogates

One of the powerful features of deepopt is the ability to use neural network surrogates in place of Gaussian process surrogates during optimization. In this tutorial, we'll repeat the [Getting Started](./index.md#getting-started-with-deepopt) example, but using neural networks in place of Gaussian process. One key difference is the need for a neural network configuration file that specifies the network architecture, activation functions, and a few other parameters. We'll describe these in detail as we move along.

### Create the initial data
Just as in [Getting Started](./index.md#getting-started-with-deepopt), we'll put some initial data in a 'sims.npz' file

```py title="generate_simulation_inputs.py" linenums="1"
import torch
import numpy as np

input_dim = 5
num_points = 10

X = torch.rand(num_points, input_dim)
y = -(X**2).sum(axis=1)

np.savez('sims.npz', X=X, y=y) # (1)
```
1. Save data to file 'sims.npz'

We can now generate the `sims.npz` file with:

```bash
python generate_simulation_inputs.py
```

### Default neural network
Neural networks in DeepOpt use the ["delta-UQ"](https://arxiv.org/abs/2110.02197) method for uncertainty quantification. The naming conventions reflect this, so to use neural networks, we set the `model_type` to "delUQ" and the model class is called `DelUQModel`.

From here we can either use the DeepOpt API or we can use the DeepOpt CLI.

If you're using the DeepOpt API, you'll first need to load the `ConfigSettings` class:

```py linenums="1" title="run_deepopt.py"
import torch
from deepopt.configuration import ConfigSettings
from deepopt.deepopt_cli import get_deepopt_model

input_dim = 5 #(1)
model_type = 'delUQ' # (2)
model_class = get_deepopt_model(model_type=model_type) # (3)
cs = ConfigSettings(model_type=model_type) #(4)
bounds = torch.FloatTensor(input_dim*[[0,1]]).T  # (5)
model = model_class(data_file='sims.npz', bounds=bounds, config_settings=cs)  # (6)
```

1. Input dimension must match data file (`sims.npz` in this case)
2. Set the model type to use throughout the script.
3. Set the model class associated with the selected model type (in this case `DelUQModel`)
4. This sets up the neural network configuration (more generally the model configuration). Since we don't pass a configuration file, the default configuration will be used.
5. Learning and optimizing will take place within these input bounds
6. Model is loaded the same way as with GP, but now we are using `DelUQModel`

Training and optimizing are done as in [Getting Started](./index.md#getting-started-with-deepopt), with the array of new points being recorded in 'suggested_inputs.npy':

=== "DeepOpt API"
    ```py title="run_deepopt.py" linenums="9"
    model.learn(outfile=f'learner_{model_type}.ckpt') # (1)
    ```

    1. Train the neural network and save its state to a checkpoint file.

=== "DeepOpt CLI"
    ```bash
    input_dim = 5
    bounds = ""
    for i in {1..input_dim-1}; do bounds+="[0,1],"; done
    bounds+="[0,1]"
    deepopt learn -i sims.npz -o learner_delUQ.ckpt -m delUQ -b $bounds
    ```

The checkpoint files saved by DeepOpt use `torch.save` under the hood. They are python dictionaries and can be viewed using `torch.load`:
```py title="view_ckpt.py" linenums="1"
import torch
model_type = 'delUQ'
ckpt = torch.load(f'learner_{model_type}.ckpt')
print(ckpt.keys())
```
The delUQ model has 4 entries in the checkpoint dictionary: 
-`epoch`: is the number of epochs the NN was trained for 
-`state_dict`: is a dictionary containing all of the values of the NN weights, biases, and other layer parameters 
-`B`: is the initial transformation to frequency space when using Fourier features 
-`opt_state_dict`: contains the optimizer parameters.

Now that we saved the trained model, we can use it to propose new candidate points:

=== "DeepOpt API"
    ```py title="run_deepopt.py" linenums="10"
    model.optimize(outfile='suggested_inputs.npy',
                   learner_file=f'learner_{model_type}.ckpt',
                   acq_method='EI') # (1)
    ```

    1. Use Expected Improvement to acquire new points based on the model saved in learner_file and save those points as a numpy array in outfile.

=== "DeepOpt CLI"
    ```bash
    deepopt optimize -i sims.npz -o suggested_inputs.npy -l learner_delUQ.ckpt \
    -m delUQ -b $bounds -a EI
    ```

The saved file `suggested_inputs.npy` is a `numpy` save file containing the array of new points with dimension Nxd (N= # of new points, d = input dimensions). We can view the file using `numpy.load`:
```bash
python -c "import numpy as np; print(np.load('suggested_inputs.npy'))"
```

### Changing the neural network configuration
Simply create a configuration yaml file with the desired entries (available settings described [here](configuration.md)). Then train and optimize the model as above, while specifying the configuration file:

=== "DeepOpt API"
    ```py title="run_deepopt.py" linenums="9"
    model.learn(outfile=f'learner_{model_type}.ckpt',
    config_file='config.yaml') # (1)

    model.optimize(outfile='suggested_inputs.npy',
                   learner_file=f'learner_{model_type}.ckpt',
                   config_file=config.yaml,
                   acq_method='EI') # (2)    
    ```

    1. Train the neural network and save its state to a checkpoint file.
    2. Use Expected Improvement to acquire new points based on the model saved in learner_file and save those points as a numpy array in outfile.

=== "DeepOpt CLI"
    ```bash
    input_dim = 5
    bounds = ""
    for i in {1..input_dim-1}; do bounds+="[0,1],"; done
    bounds+="[0,1]"
    deepopt learn -i sims.npz -o learner_delUQ.ckpt -m delUQ -b $bounds -c config.yaml # (1)

    deepopt optimize -i sims.npz -o suggested_inputs.npy -l learner_delUQ.ckpt \
    -m delUQ -b $bounds -a EI -c config.yaml # (2)
    ```

    1. Train the neural network and save its state to a checkpoint file.
    2. Use Expected Improvement to acquire new points based on the model saved in learner_delUQ.ckpt and save those points as a numpy array in suggested_inputs.npy.


## Tutorial: Iterative optimization

Typical Bayesian optimization workflows will have an iterative structure, as proposed candidates from one iteration are added to the training of the surrogate in the following iteration. We demonstrate how to use the DeepOpt API to accomplish this. A similar workflow is possible in CLI, but it is best to use a workflow manager such as [Merlin](https://merlin.readthedocs.io/en/latest/).

```py title='iterative_optimization.py'
import torch
import numpy as np
from deepopt.configuration import ConfigSettings
from deepopt.deepopt_cli import get_deepopt_model

def objective(X):
    return -(X**2).sum(axis=1)

input_dim = 5
num_initial_points = 10

X_init = torch.rand(num_initial_points, input_dim)
y_init = objective(X_init)

np.savez("points_iter0.npz",X=X_init,y=y_init) #We'll store all data in npz files with keys 'X' for inputs and 'y' for outputs

model_type = "GP"
cs = ConfigSettings(model_type=model_type)   
model_class = get_deepopt_model(model_type) 
bounds = torch.FloatTensor(input_dim*[[0,1]]).T

n_iterations = 20
for i in range(n_iterations):
    print(f"---------------\n  ITERATION {i+1}  \n---------------")
    data_prev_file = f"points_iter{i}.npz" # previous points
    candidates_file = f"suggested_inputs_iter{i+1}.npy" # save new proposed inputs to evaluate here
    ckpt_file = f"learner_{model_type}_iter{i+1}.ckpt" # save model for this iteration here

    # Train model on previous points, then propose new points using Expected Improvement
    model = model_class(data_file=data_prev_file,bounds=bounds,config_settings=cs)
    model.learn(outfile=ckpt_file)
    model.optimize(outfile=candidates_file, learner_file=ckpt_file, acq_method="EI")

    # Load previous input & output values
    data_prev = np.load(data_prev_file)
    X_prev = data_prev["X"]
    y_prev = data_prev["y"]

    X_new = np.load(candidates_file) # Load proposed inputs
    y_new = objective(X_new) # Evaluate proposed inputs

    # Concatenate new input & output values with previous and save
    X = np.concatenate([X_prev,X_new],axis=0)
    y = np.concatenate([y_prev,y_new],axis=0)
    np.savez(f"points_iter{i+1}.npz",X=X,y=y)
```

This workflow produces `n_iterations` npy files of suggested inputs and `n_iterations + 1` npz files of points. The npy files contain just the inputs to call the objective function with, so are redundant (in this example) with the npz files that contain both. We can visualize the optimization with the following script:

```py title='plot_optimization.py
import numpy as np
import matplotlib.pyplot as plt

# Make sure these match the iterative_optimization.py file
num_initial_points = 10
num_iterations = 20

pts = np.load(f'points_iter{num_iterations}.npz') # Load last collection of points
y = pts['y'] # Extract objective values
pts_per_iteration = (len(y)-num_initial_points)//num_iterations # Work out how many points were proposed per iteration
iters = np.concatenate([np.zeros(num_initial_points),np.repeat(np.arange(1,num_iterations+1),pts_per_iteration)]) # Set the iteration value of all points
plt.scatter(iters,y) # Plot all proposal results vs. iteration
plt.xlabel('Iterations')
plt.xticks(np.arange(num_iterations+1))
plt.ylabel('Objective value')

#Find running max:
max_byiter = np.zeros(num_iterations+1) # This will store max for each iteration
max_byiter[0] = np.max(y[:num_initial_points]) # Max of initial points
for i in range(1,num_iterations+1):
    max_byiter[i] = np.max(y[num_initial_points+(i-1)*pts_per_iteration:num_initial_points+i*pts_per_iteration]) # Max of each iteration (when multiple proposals)
plt.plot(np.arange(num_iterations+1),np.maximum.accumulate(max_byiter),label='running max') # Add running max to plot
plt.legend()
plt.show()
```

The plot should display a running max that converges to the objective maximum (0 in this example), while individual proposals will be a mix of high values (at or near the running max) from succesfuly exploitation/exploration and low values (far below the running max) from unsuccesful exploration.

## Tutorial: Multi-fidelity optimization
In this tutorial, we'll walk through how to accomplish multi-fidelity optimization with DeepOpt. We'll start by copying the `generate_simulation_inputs.py` and `run_deepopt.py` file from the [Getting Started page](./index.md#getting-started-with-deepopt):

<!-- Performing multi-fidelity optimization with DeepOpt requires only that the data files have the last input column as a fidelity, with integer values ranging from 0 to number of fidelities - 1, and to pass a list of fidelity costs to the "optimize" method (the length of the list must match the number of fidelities). In addition, the acquisition function must be appropriate for multi-fidelity optimization.

We show here the necessary changes to "generate_simulation_inputs.py" and the optimize call from the [Getting Started](./index.md#getting-started-with-deepopt) page: -->
```bash
cp generate_simulation_inputs.py generate_simulation_inputs_mf.py
cp run_deepopt.py run_deepopt_mf.py
```

Performing multi-fidelity optimization with DeepOpt requires that the data files have the last input column as a fidelity, with integer values ranging from 0 to number of fidelities - 1. We can generate this input file by changing two lines in the generate_simulation_inputs.py file:

```py title="generate_simulation_inputs_mf.py" linenums=0, hl_lines=8
import torch
import numpy as np

input_dim = 5 # Last dimension will be used for fidelity
num_points = 10

X = torch.rand(num_points, input_dim)
X[:,-1] = X[:,-1].round()
y = -(X**2).sum(axis=1) # (1)

np.savez('sims.npz', X=X, y=y)
```

1. This line now produces paraboloids at two fidelities, with the high fidelity paraboloid shifted one unit relative to the low fidelity paraboloid.

We run this script with 
```bash
python generate_simulation_inputs_mf.py
```

Additionally, to achieve multi-fidelity optimization with DeepOpt you must:

1. Enable multi-fidelity in your model
2. Pass a list of fidelity costs to the optimize method (the length of the list must match the number of fidelities)
3.  Select an acquisition function that's appropriate for multi-fidelity optimization (currently only Knowledge Gradient and Max Value Entropy are supported)

Below, we show how to modify the optimize call from run_deepopt.py to accommodate these changes, and also how to achieve this same functionality from the command line:

=== "DeepOpt API"

    ```py linenums=0 hl_lines=11,13-15
        import torch
        from deepopt.configuration import ConfigSettings
        from deepopt.deepopt_cli import get_deepopt_model

        input_dim = 5
        model_type = "GP"  # 
        bounds = torch.FloatTensor(input_dim*[[0,1]]).T  # 

        deepopt_model = get_deepopt_model(model_type)  #
        cs = ConfigSettings(model_type=model_type)
        model = deepopt_model(data_file="sims.npz", bounds=bounds, config_settings=cs, multi_fidelity=True)  #
        model.learn(outfile=f"learner_{model_type}.ckpt") #
        model.optimize(outfile="suggested_inputs.npy",
                    learner_file=f"learner_{model_type}.ckpt",
                    acq_method="KG",fidelity_cost=[1,6]) #

    ```

    1. We use the Knowledge Gradient (KG) multi-fidelity acquisition function with a 1:6 ratio of low:high fidelity costs.

=== "DeepOpt CLI"

    ```bash
    deepopt optimize -i sims.npz -l learner_GP.ckpt -o suggested_inputs.npy \
    -b "[[0, 1], [0, 1], [0, 1], [0, 1], [0, 1]]" \
    -a KG --multi-fidelity --fidelity-cost "[1,6]" # (1)
    ```

    1. We use the Knowledge Gradient (KG) multi-fidelity acquisition function with a 1:6 ratio of low:high fidelity costs.

The API script can be run with
```bash
python run_deepopt_mf.py
```

## Tutorial: Acquisition functions
There are currently four acquisition functions available for single-fidelity optimization: Expected Improvement (EI), Noisy Expected Improvement (NEI), Knowledge Gradient (KG), and Max Value Entropy (MaxValEntropy). The last two (KG & Max Value Entropy) are also available for multi-fidelity optimization. These acquisition functions are built around the associated (BoTorch acquisition functions)[https://botorch.org/api/acquisition.html#acquisitionfunction]: qExpectedImprovement, qNoisyExpectedImprovement, qKnowledgeGradient, and qMaxValueEntropy. We briefly describe the strengths and weaknesses of each acquisition.

**EI**: This is one of the simplest acquisition functions for Bayesian optimization. It selects points that optimize improvement over the best function value found thus far (weighted by the probability of achieving such improvement). The interpretation is straightforward, but EI tends favor exploitation over exploration and can get stuck near local optima.

**NEI**: This adapts EI to problems that are noisy (strong fluctuations in the objective function). The major change from EI is that NEI measures improvement over the best surrogate value (rather than objective function value) among points selected thus far. This allows NEI to avoid getting thrown off by noise, since the surrogate will generally be much smoother than the objective function.

**KG**: Knowledge gradient attempts to reduce EI's heavy exploitation by selecting a point such that a subsequent selection would yield the best expected improvement. Effectively it's a one-step look-ahead acquisition function. Specifically, a potential selection is evaluated by "fantasizing" at the location (drawing outputs from the probability distribution at the location and fitting a separate model to each) then, for each fantasy model, identifying how much improvement is obtained when using EI as a subsequent acquisition. Improvements are averaged over the fantasy models to assign a value to each potential selection and a final selection is made based on the best value. The need to fantasize at each potential location makes KG a fairly expensive acquisition function and it can be slow to use, but in addition to getting stuck less than EI, KG can be used for multi-fidelity optimization problems.

**MaxValEntropy**: Max Value Entropy selects points to minimize its uncertainty about the optimal value. This indirect approach allows it to heavily favor exploration before zooming in on promising spots in the input space. Its information-theoretic foundation also easily extends to the multi-fidelity setting (a low-fidelity candidate is selected if it helps minimize uncertainty about the high-fidelity optimum).

## Tutorial: Risk-averse optimization
To do risk-averse optimization, simply specify the risk_measure (CLI: --risk-measure), risk_level (CLI: --risk-level), risk_n_deltas (CLI: --risk_n_deltas), and x_stddev (CLI: --X-stddev) when calling optimize. The available risk measures are VaR (variance at risk) and CVaR (conditional variance at resk). The risk level is between 0 and 1 and sets the corresponding alpha value (see the BoTorch [example](https://botorch.org/tutorials/risk_averse_bo_with_environmental_variables) for more details). risk_n_deltas sets the number of samples to draw for input perturbations (more accuracy and longer run time for larger values). x_stddev sets the size of the input perturbations in each dimension (can provide a list to specify dimension-by-dimension or a scalar to set the same pertrubation for all inputs). Currently only EI, NEI, and KG acquisition functions support risk measures.
