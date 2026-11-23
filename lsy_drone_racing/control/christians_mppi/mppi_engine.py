"""Generic Model Predictive Path Integral (MPPI) engine in JAX.

This module knows nothing about drones. It only implements the MPPI algorithm (lecture L3,
slides 29-32; Williams et al., "Information Theoretic MPC", 2017). The problem is injected as
plain functions:

    dynamics(x, u, params)         -> x_next      One discrete time step of the model.
    running_cost(x_next, u, params) -> scalar     Cost of the step that applied u and led to x_next.
    terminal_cost(x_N, params)      -> scalar     Optional cost of the final state.

``x`` may be any pytree (an array, a tuple, a dict, ...). That allows carrying extra rollout
information through the horizon, e.g. the index of the next gate, a "crashed" flag, or the
previous input for smoothness costs. ``params`` is any pytree that the functions need and that may
change between calls (model parameters, gate and obstacle positions, ...). It is a traced argument
of the jitted step, so changing its *values* never triggers a recompile. Only changing its shapes
or dtypes does.

One MPPI iteration (``MPPI.step``):

    1. Sample K noise sequences eps_k (horizon N x action dim) and perturb the nominal plan:
           U_k = clip(U + eps_k, u_min, u_max)
    2. Roll out all K sequences in parallel (vmap over samples, lax.scan over the horizon) and sum
       up their costs J_k.
    3. Weight the samples:  w_k = exp(-(J_k - min J) / lambda) / sum_j exp(-(J_j - min J) / lambda)
    4. Update the plan with the weighted average:  U <- sum_k w_k U_k
    5. Apply U[0] and shift the plan by one step as the warm start for the next call.

Usage:
    engine = MPPI(MPPIConfig(...), dynamics, running_cost, terminal_cost)
    state = engine.init(u_nominal, seed=0)              # u_nominal: (nu,) or (N, nu)
    state, u0, info = engine.step(state, x0, params)    # once per control step
"""

from __future__ import annotations

from typing import Any, Callable, NamedTuple

import jax
import jax.numpy as jnp

PyTree = Any
DynamicsFn = Callable[[PyTree, jax.Array, PyTree], PyTree]
RunningCostFn = Callable[[PyTree, jax.Array, PyTree], jax.Array]
TerminalCostFn = Callable[[PyTree, PyTree], jax.Array]


class MPPIConfig(NamedTuple):
    """Hyperparameters of the MPPI algorithm. Problem-specific settings belong into ``params``."""

    n_samples: int = 1024  # K: number of sampled input sequences per iteration
    horizon: int = 30  # N: number of steps of each rollout
    temperature: float = 1.0  # lambda: small -> greedy (best sample wins), large -> plain average
    noise_std: tuple[float, ...] = (1.0,)  # Std of the input noise, one entry per action dimension
    noise_corr: float = 0.0  # AR(1) correlation of the noise along the horizon (0 = white noise)
    u_min: tuple[float, ...] = (-1.0,)  # Lower input bounds, one entry per action dimension
    u_max: tuple[float, ...] = (1.0,)  # Upper input bounds, one entry per action dimension
    control_cost_weight: float = 0.0  # gamma: weight of the information-theoretic input cost
    keep_nominal: bool = True  # Sample 0 is the unperturbed plan (avoids losing a good plan)


class MPPIState(NamedTuple):
    """Everything MPPI carries from one control step to the next."""

    plan: jax.Array  # Nominal input sequence U, shape (N, action_dim)
    key: jax.Array  # JAX PRNG key


class MPPIInfo(NamedTuple):
    """Diagnostics of one MPPI iteration. Convert with float()/np.asarray() only when needed."""

    plan: jax.Array  # Updated input sequence before the shift, shape (N, action_dim)
    trajectory: PyTree  # States x_1..x_N of the updated plan (each leaf has a leading N axis)
    costs: jax.Array  # Total cost of every sample, shape (K,)
    weights: jax.Array  # Normalized sample weights, shape (K,)
    effective_samples: jax.Array  # 1 / sum(w^2): ~1 = one sample dominates, ~K = uniform weights


class MPPI:
    """Generic MPPI optimizer. All heavy lifting happens inside one jit-compiled function."""

    def __init__(
        self,
        cfg: MPPIConfig,
        dynamics: DynamicsFn,
        running_cost: RunningCostFn,
        terminal_cost: TerminalCostFn | None = None,
    ):
        """Store the problem definition and compile-wrap the MPPI step.

        Args:
            cfg: Algorithm hyperparameters. Fixed for the lifetime of the engine.
            dynamics: Discrete-time model ``x_next = dynamics(x, u, params)``.
            running_cost: Stage cost ``running_cost(x_next, u, params)``.
            terminal_cost: Optional final-state cost ``terminal_cost(x_N, params)``.
        """
        self.cfg = cfg
        self.dynamics = dynamics
        self.running_cost = running_cost
        self.terminal_cost = terminal_cost

        self.noise_std = jnp.asarray(cfg.noise_std, dtype=jnp.float32)
        self.u_min = jnp.asarray(cfg.u_min, dtype=jnp.float32)
        self.u_max = jnp.asarray(cfg.u_max, dtype=jnp.float32)
        self.action_dim = self.noise_std.shape[0]
        if not (self.u_min.shape == self.u_max.shape == (self.action_dim,)):
            raise ValueError("noise_std, u_min and u_max need one entry per action dimension.")

        self.step = jax.jit(self._step)
        self.rollout = jax.jit(self._rollout)

    # region Public API

    def init(self, u_nominal: jax.Array, seed: int = 0) -> MPPIState:
        """Create the initial state from a nominal input (e.g. hover) or a full input sequence."""
        u_nominal = jnp.asarray(u_nominal, dtype=jnp.float32)
        if u_nominal.ndim == 1:
            u_nominal = jnp.tile(u_nominal, (self.cfg.horizon, 1))
        if u_nominal.shape != (self.cfg.horizon, self.action_dim):
            raise ValueError(f"Expected plan shape {(self.cfg.horizon, self.action_dim)}.")
        return MPPIState(plan=u_nominal, key=jax.random.PRNGKey(seed))

    # region Algorithm

    def _step(
        self, state: MPPIState, x0: PyTree, params: PyTree
    ) -> tuple[MPPIState, jax.Array, MPPIInfo]:
        """One MPPI iteration. Returns the new state, the input to apply now, and diagnostics."""
        key, noise_key = jax.random.split(state.key)
        plan = state.plan

        # 1. Perturb the nominal plan. The noise is recomputed after clipping, so that the input
        #    cost below only sees the perturbation that was actually applied.
        noise = self._sample_noise(noise_key)
        samples = jnp.clip(plan[None] + noise, self.u_min, self.u_max)  # (K, N, action_dim)
        noise = samples - plan[None]

        # 2. Roll out all samples in parallel
        costs, _ = jax.vmap(self._rollout, in_axes=(0, None, None))(samples, x0, params)
        costs = costs + self._control_cost(plan, noise)

        # 3. Exponential weighting. Subtracting the minimum keeps exp() from underflowing to 0
        weights = jax.nn.softmax(-(costs - costs.min()) / self.cfg.temperature)

        # 4. Weighted average of the samples. Since sum(w) = 1, this equals plan + sum_k w_k eps_k
        new_plan = jnp.einsum("k,knu->nu", weights, samples)

        # 5. Apply the first input and shift the plan for the warm start (repeat the last input)
        u0 = new_plan[0]
        shifted_plan = jnp.concatenate([new_plan[1:], new_plan[-1:]], axis=0)

        _, trajectory = self._rollout(new_plan, x0, params)
        info = MPPIInfo(
            plan=new_plan,
            trajectory=trajectory,
            costs=costs,
            weights=weights,
            effective_samples=1.0 / jnp.sum(weights**2),
        )
        return MPPIState(plan=shifted_plan, key=key), u0, info

    def _rollout(self, U: jax.Array, x0: PyTree, params: PyTree) -> tuple[jax.Array, PyTree]:
        """Simulate one input sequence U (N, action_dim). Returns its total cost and the states."""

        def step(x: PyTree, u: jax.Array) -> tuple[PyTree, tuple[jax.Array, PyTree]]:
            x_next = self.dynamics(x, u, params)
            return x_next, (self.running_cost(x_next, u, params), x_next)

        x_final, (stage_costs, trajectory) = jax.lax.scan(step, x0, U)
        total = jnp.sum(stage_costs)
        if self.terminal_cost is not None:
            total = total + self.terminal_cost(x_final, params)
        return total, trajectory

    def _sample_noise(self, key: jax.Array) -> jax.Array:
        """Gaussian input noise of shape (K, N, action_dim), optionally correlated in time.

        With noise_corr = beta, each sequence follows the AR(1) process
            eps_0 = w_0,   eps_t = beta * eps_{t-1} + sqrt(1 - beta^2) * w_t,   w_t ~ N(0, sigma^2)
        which keeps the marginal std at sigma but produces smoother input sequences.
        """
        K, N, beta = self.cfg.n_samples, self.cfg.horizon, self.cfg.noise_corr
        white = jax.random.normal(key, (N, K, self.action_dim)) * self.noise_std

        def ar1(eps_prev: jax.Array, w: jax.Array) -> tuple[jax.Array, jax.Array]:
            eps = beta * eps_prev + jnp.sqrt(1.0 - beta**2) * w
            return eps, eps

        _, rest = jax.lax.scan(ar1, white[0], white[1:])
        noise = jnp.concatenate([white[:1], rest], axis=0).transpose(1, 0, 2)  # (K, N, action_dim)
        if self.cfg.keep_nominal:
            noise = noise.at[0].set(0.0)
        return noise

    def _control_cost(self, plan: jax.Array, noise: jax.Array) -> jax.Array:
        """Information-theoretic input cost gamma * sum_t u_t^T Sigma^-1 eps_t for every sample.

        It comes from the KL term in the MPPI derivation, which compares the sampling distribution
        N(U, Sigma) with the passive one N(0, Sigma). Samples whose noise points against the plan
        get a lower cost, so the term pulls the plan toward zero input (for a drone: zero thrust!).
        gamma = 0 (the default) disables it, which is common in practice.
        """
        if self.cfg.control_cost_weight == 0.0:
            return jnp.zeros(noise.shape[0])
        sampled = self.noise_std > 0.0  # Dimensions without noise carry no input cost
        inv_var = jnp.where(sampled, 1.0 / jnp.where(sampled, self.noise_std, 1.0) ** 2, 0.0)
        per_sample = jnp.einsum("nu,u,knu->k", plan, inv_var, noise)
        return self.cfg.control_cost_weight * per_sample
