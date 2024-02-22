import numpy as np
import matplotlib.pyplot as plt
import bayesflow as bf
import tensorflow_probability as tfp
from scipy import stats
from functools import partial
import seaborn as sns

"""
tfd = tfp.distributions

# Set random seed for reproducibility
np.random.seed(42)

def data():
    # Number of time steps
    num_steps = 365

    # Generate time steps
    time = np.arange(num_steps)

    # Simulate trend component
    trend = 0.1 * time

    # Simulate seasonality component
    seasonality = 10 * np.sin(2 * np.pi * time / 365)

    # Simulate noise
    noise = np.random.normal(loc=0, scale=5, size=num_steps)

    # Combine components to create time series
    time_series = trend + seasonality + noise

    return time_series

# Plot the time series
plt.figure(figsize=(10, 6))
plt.plot(time, time_series)
plt.title('Simulated Time Series Data')
plt.xlabel('Time')
plt.ylabel('Value')
plt.grid(True)
plt.show()
"""

PARAM_NAMES = [r"$\mu_d$", r"$\mu_g$", r"$\Sigma_{00}$", r"$\Sigma_{01}$", r"$\Sigma_{10}$", r"$\Sigma_{11}$"]
RNG = np.random.default_rng(2023)

def timeseries_prior_fun(rng=None):
   
    if rng is None:
        rng = np.random.default_rng()

    mu_d = rng.normal(0, 0.25)
    mu_g = rng.normal(0, 0.25)
    Q = stats.invwishart.rvs(df=10, scale=np.identity(2), random_state=rng)
    lambdas = rng.uniform(0, 3, size=2)
    sigma = np.matmul(np.matmul(np.diag(lambdas), Q), np.diag(lambdas))
    return np.concatenate([np.r_[mu_d, mu_g], sigma.flatten()])

prior = bf.simulation.Prior(prior_fun=timeseries_prior_fun, param_names=PARAM_NAMES)
print(prior(batch_size=1))


N_GROUPS = 50
N_OBS = 100

def time_series_simulator(theta, num_groups, num_obs, rng=None, *args):
    
    """Simulates time series data.

    Parameters
    ----------
    theta      : np.ndarray of shape (num_parameters, )
        Contains draws from the prior distribution for each parameter.
        For simplicity, let's assume theta contains parameters for the mean and standard deviation of the time series.
    num_groups : int
        The number of groups.
    num_obs    : int
        The number of time steps per group.

    Returns
    -------
    data     : np.ndarray of shape (num_groups, num_obs)
        The generated time series data for each group.
    """

    if rng is None:
        rng = np.random.default_rng()

    mu, sigma = theta

    # Draw parameters for each group
    params = rng.normal(mu, sigma, size=(num_groups, 2))

    # Simulate time series data for each group
    data = []
    for group_params in params:
        group_data = rng.normal(group_params[0], abs(group_params[1]), size=num_obs)
        data.append(group_data)

    return np.array(data)



prior_mu = 0
prior_sigma = 1
theta = [prior_mu, prior_sigma]

simulated_data = time_series_simulator(theta, N_GROUPS, N_OBS)
time = np.arange(N_GROUPS)

plt.figure(figsize=(10, 6))
plt.plot(time, simulated_data)
plt.title('Simulated Time Series Data')
plt.xlabel('Time')
plt.ylabel('Value')
plt.grid(True)
plt.show()
plt.close()

rng = np.random.default_rng()

model_1 = bf.simulation.GenerativeModel(
    prior=prior,
    simulator=partial(time_series_simulator, theta, N_GROUPS, N_OBS, rng),
   
    simulator_is_batched=False
)

model_output = model_1(batch_size=5)
print("Shape of data batch:", model_output["sim_data"].shape)
print("First 3 rows of first 2 participants in first data set:")
print(model_output["sim_data"][0, :2, :3])

sim_check = model_1(batch_size=20)

def get_rates(sim_data):
    """Get the hit rate and false alarm rate per participant for each data set in a batch
    of hierarchical data sets simulating binary decision (recognition) tasks.
    Assumes first half of data to cover old items and second half to cover new items."""

    num_obs_per_condition = sim_data.shape[1] // 2
    
    # Extract responses for old items (first half of the data)
    old_responses = sim_data[:, :num_obs_per_condition, 1]
    # Extract responses for new items (second half of the data)
    new_responses = sim_data[:, num_obs_per_condition:, 1]

    # Calculate hit rates (proportion of "old" responses for old items)
    hit_rates = np.mean(old_responses, axis=1)
    # Calculate false alarm rates (proportion of "old" responses for new items)
    false_alarm_rates = np.mean(new_responses, axis=1)

    return hit_rates, false_alarm_rates




rates = get_rates(sim_check["sim_data"])

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Plot hit rates histogram
axes[0].hist(rates[0], bins=20, color='blue', alpha=0.7)
axes[0].set_title('Hit Rates Histogram')
axes[0].set_xlabel('Hit Rate')
axes[0].set_ylabel('Frequency')


# Plot false alarm rates histogram
axes[1].hist(rates[1], bins=20, color='red', alpha=0.7)
axes[1].set_title('False Alarm Rates Histogram')
axes[1].set_xlabel('False Alarm Rate')
axes[1].set_ylabel('Frequency')

plt.tight_layout()
plt.show()
plt.close()

summary_net = bf.summary_networks.TimeSeriesTransformer(
   input_dim=100
)
inference_net = bf.inference_networks.InvertibleNetwork(num_params=6)
amortizer = bf.amortizers.AmortizedPosterior(inference_net, summary_net)
trainer = bf.trainers.Trainer(amortizer=amortizer, generative_model=model_1)

"""
history = trainer.train_online(epochs=5, iterations_per_epoch=100, batch_size=64, validation_sims=20)
f = bf.diagnostics.plot_losses(history["train_losses"], history["val_losses"], moving_average=True)
plt.show()
"""

losses = trainer.train_online(epochs=5, iterations_per_epoch=100, batch_size=64)
diag_plot = bf.diagnostics.plot_losses(train_losses=losses, moving_average=True, ma_window_fraction=0.05)

plt.show()
plt.close(diag_plot)

prior_mu_new = 0
prior_sigma_new = 1
theta_new = [prior_mu, prior_sigma]


prior_fixed = bf.simulation.Prior(
    prior_fun=partial(timeseries_prior_fun, rng=np.random.default_rng(2023)), param_names=PARAM_NAMES
)
fake_data_generator = bf.simulation.GenerativeModel(
    prior=prior_fixed,
    simulator=partial(
        time_series_simulator, theta_new, N_GROUPS, N_OBS, np.random.default_rng(2023)),
    skip_test=True,
    simulator_is_batched=False,
)

fake_data = fake_data_generator(batch_size=1)["sim_data"]
print(fake_data.shape)

rates = get_rates(fake_data)
f, ax = plt.subplots(1, 2, figsize=(8, 3))
sns.histplot(rates[0].flatten(), bins=20, kde=True, color="#8f2727", alpha=0.9, ax=ax[0]).set(title="Hit Rates")
sns.histplot(rates[1].flatten(), bins=20, kde=True, color="#8f2727", alpha=0.9, ax=ax[1]).set(
    title="False Alarm Rates"
)
sns.despine()
plt.show()

""" embeddings = summary_net(fake_data)
preds = inference_net.posterior_probs(embeddings)[0]
print(preds)

bayes_factor12 = preds[0] / preds[1]
print(bayes_factor12) """