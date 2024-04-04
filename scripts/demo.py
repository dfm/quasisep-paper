import sys

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import tinygp

jax.config.update("jax_enable_x64", True)


@tinygp.helpers.dataclass
class Latent(tinygp.kernels.quasisep.Wrapper):
    coeff_prim: jnp.ndarray
    coeff_deriv: jnp.ndarray

    def coord_to_sortable(self, X):
        return X[0]

    def observation_model(self, X):
        t, label = X
        design = self.kernel.design_matrix()
        obs = self.kernel.observation_model(t)
        obs_prim = jnp.asarray(self.coeff_prim)[label] * obs
        obs_deriv = jnp.asarray(self.coeff_deriv)[label] * obs @ design
        return obs_prim - obs_deriv


base_kernel = tinygp.kernels.quasisep.Matern52(
    scale=1.5
) * tinygp.kernels.quasisep.Cosine(scale=2.5)
kernel = Latent(base_kernel, [0.5, 0.02], [0.01, -0.2])

# Unlike the previous derivative observations tutorial, the datapoints here
# must be sorted in time.
random = np.random.default_rng(5678)
t = np.sort(random.uniform(0, 10, 500))
label = (random.uniform(0, 1, len(t)) < 0.5).astype(int)
X = (t, label)

gp = tinygp.GaussianProcess(kernel, X)
y = gp.sample(jax.random.PRNGKey(12345))

# Select a subset of the data as "observations"
subset = (1 + 2 * label) * random.uniform(0, 1, len(t)) < 0.3
X_obs = (X[0][subset], X[1][subset])
y_obs = y[subset] + 0.1 * random.normal(size=subset.sum())

offset = 2.5

plt.axhline(0.5 * offset, color="k", lw=1)
plt.axhline(-0.5 * offset, color="k", lw=1)
plt.plot(t[label == 0], y[label == 0] + 0.5 * offset, label="class 0")
plt.plot(t[label == 1], y[label == 1] - 0.5 * offset, label="class 1")
plt.plot(X_obs[0], y_obs + offset * (0.5 - X_obs[1]), ".k", label="measured")

plt.xlim(0, 10)
plt.ylim(-1.3 * offset, 1.3 * offset)
plt.xlabel("t")
plt.ylabel("y + offset")
plt.legend(bbox_to_anchor=(1.01, 1), loc="upper left")

plt.savefig(sys.argv[1], bbox_inches="tight")
