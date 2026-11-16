"""Train the track-independent policy of [rl_general][lsy_drone_racing.control.rl_general] with PPO.

The PPO loop and the analytic collision check are shared with
[train_rl_racer][lsy_drone_racing.control.train_rl_racer]. What differs is the task:

* **Random tracks from a pool.** Generating a track takes much longer than a simulation step, so
  tracks come from a pre-computed pool that is replaced regularly. Part of the pool uses a wider
  distribution than level 3: other start positions, poles closer to gates, gates that have to be
  approached from the side, and poles right in front of or behind a gate. The level 2 track lies
  inside that distribution, but is never trained on, so it remains a real test of generalisation.
* **Progress along the planned line.** The progress reward and the observed points ahead come from
  the line of [plan_line][lsy_drone_racing.control.rl_general.plan_line] through the track. It
  passes around objects, so following it does not lead into them, and it handles a gate that has to
  be approached from behind. It is planned on the nominal track and repaired after each reveal, as
  in the controller. Repairs are limited per step, because they are the most expensive part of it.
* **A speed cap per episode.** Progress beyond the cap earns nothing and flying faster than the cap
  is penalised. The policy observes the cap, so one network covers the whole range of speeds.

Run a stage as:

    $ python lsy_drone_racing/control/train_rl_general.py train --total_steps 100_000_000

and measure the trade-off between speed and success on tracks the policy has not seen with:

    $ python lsy_drone_racing/control/train_rl_general.py sweep --policy <checkpoint>

Checkpoints go to ``saves/rl_general/<run_name>``. The shipped policy is the result of 688 M steps
over eight stages, each continuing from the last with ``--init_policy``. The stages also tracked the
development of this code, so new inputs were added with zero weights (`pad_inputs`). Success is on
random tracks at a speed cap of 1.5 to 3 m/s, and shows what each change contributed:

1. 98 M steps, end-to-end policy with a line that ignores objects: 30 %.
2. 63 M steps, adding the closest solid points and positions along the direction of travel: 45 %.
3. 51 M steps, adding the planned line and its points ahead: 55 %.
4. 65 M steps, adding the collision course and a stronger clearance penalty: 65 %.
5. 39 M steps, adding the contouring cost: 55 %, it follows the line even where that is wrong.
6. 52 M steps, adding line repairs after reveals: 80 %.
7. 200 M steps, adding poles right in front of or behind gates to the tracks: 85 %.
8. 120 M steps, passing poles on gate axes on the side of the next gate: 86 %, and the level 2
   track, which is never trained on, from 56 % to 85 %.

A retrain from scratch with the final code has not been done.
"""

from __future__ import annotations

import copy
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import fire
import jax
import jax.numpy as jp
import numpy as np
import optax
from crazyflow.dynamics import load_params as load_dynamics_params
from flax import struct

from lsy_drone_racing.control.rl_general import (
    LINE_DS,
    LINE_LENGTH,
    closest_solids,
    gate_frame,
    general_command,
    general_observation,
    line_ahead,
    plan_line,
    project_on_line,
    repair_line,
    to_frame,
)
from lsy_drone_racing.control.rl_racer import (
    G,
    mass_estimate_init,
    mass_estimate_update,
    mass_ratio,
    mlp_forward,
    quat_to_matrix,
)
from lsy_drone_racing.control.train_rl_racer import (
    Args,
    FastContactRaceEnv,
    critic_arrays,
    init_mlp,
    load_policy_params,
    make_env,
    make_train_iter,
    save_policy,
)
from lsy_drone_racing.envs.drone_race import VecDroneRaceEnv
from lsy_drone_racing.envs.race_core import obs as race_obs
from lsy_drone_racing.envs.randomize import build_random_track_fn
from lsy_drone_racing.utils import load_config

if TYPE_CHECKING:
    from jax import Array

    from lsy_drone_racing.envs.race_core import EnvData

ROOT = Path(__file__).parents[2]
CHECKPOINT_DIR = ROOT / "saves" / "rl_general"

CRITIC_AHEAD = (5, 10, 20, 40)  # Samples ahead on the line that the critic observes
RSI_GATE_CLEARANCE = 0.5  # [m] no reference state starts this close to a gate
RSI_POLE_CLEARANCE = 0.25  # [m] or this close to a pole
RSI_MIN_HEIGHT = 0.15  # [m] or this close to the ground
HOVER_RPM_RANGE = (0.85, 1.2)  # Rotor speeds of reference states, relative to hover
# A wider track distribution than level 3, which also contains the level 2 track
WIDE_TRACKS = {"gate_excl_r": 0.7, "obstacle_excl_r": 0.45, "yaw_range": 1.7, "drone_excl_r": 0.8}
START_SPREAD = (1.7, 0.9)  # [m] half size of the region of varied start positions
HAZARD_DISTANCE = (0.45, 0.8)  # [m] distance along the gate axis of a pole placed next to a gate
HAZARD_OFFSET = 0.12  # [m] largest lateral offset from the gate axis of such a pole
DRONE_RADIUS = 0.1  # [m] horizontal half diagonal of the collision box
POLE_RADIUS = 0.015  # [m]


@dataclass
class GeneralArgs(Args):
    """Training settings on top of the ones shared with the level 2 racer."""

    hidden: tuple[int, ...] = (192, 192, 128)
    max_episode_steps: int = 1000
    eval_steps: int = 800
    eval_envs: int = 256
    eval_every: int = 100
    rsi_start: float = 0.5
    rsi_end: float = 0.2
    rsi_speed: float = 3.0
    max_tilt: float = 1.4
    # Speed cap
    cap_low: float = 1.0  # [m/s] range of the speed cap of an episode
    cap_high: float = 4.0
    overspeed: float = 0.5  # on the squared excess of the speed over the cap
    contouring: float = 3.0  # on the squared distance to the planned line beyond the tunnel
    contouring_width: float = 0.2  # [m] half width of the tunnel, which leaves room for reveals
    clearance: float = 0.3  # on the squared relative intrusion into the margin around objects
    clearance_margin: float = 0.2  # [m] margin kept to objects, on top of the drone radius
    eval_caps: tuple[float, ...] = (1.5, 2.5, 3.5)
    # Tracks
    pool_size: int = 32768
    pool_refresh: int = 400  # iterations
    wide_tracks: float = 0.4  # share of the tracks from the wider distribution
    start_spread: float = 0.3  # share of the tracks with a varied start position
    pole_hazard: float = 0.7  # share of the wider tracks with a pole next to a gate axis
    save_every: int = 500  # iterations
    repair_slots: int = 48  # Lines repaired per step, the others wait for the next step


def line_tables(
    line: Array, gate_index: Array, gates_pos: Array, obstacles_pos: Array, sensor_range: float
) -> dict[str, Array]:
    """Which samples of a line are valid reference states, and what has been seen until there.

    Args:
        line: The samples of the planned line (l, 3).
        gate_index: The sample index of each gate pass (m,).
        gates_pos: The nominal gate positions (g, 3).
        obstacles_pos: The nominal obstacle positions (o, 3).
        sensor_range: The range at which objects are revealed.

    Returns:
        The mask ``valid`` of reference states (l,), and the first sample at which each gate and
        obstacle is in sensor range, ``gates_seen`` (g,) and ``obstacles_seen`` (o,).
    """
    gates_xy = jp.linalg.norm(line[:, None, :2] - gates_pos[None, :, :2], axis=-1)
    poles_xy = jp.linalg.norm(line[:, None, :2] - obstacles_pos[None, :, :2], axis=-1)
    gates_3d = jp.linalg.norm(line[:, None] - gates_pos[None], axis=-1)
    index = jp.arange(line.shape[0])
    from_pass = jp.abs(index[:, None] - gate_index[None]) * LINE_DS
    valid = (
        jp.all(from_pass > RSI_GATE_CLEARANCE, axis=1)
        & jp.all(gates_3d > RSI_GATE_CLEARANCE, axis=1)
        & jp.all(poles_xy > RSI_POLE_CLEARANCE, axis=1)
        & (line[:, 2] > RSI_MIN_HEIGHT)
        & (index < gate_index[-1])
    )

    def first_seen(near: Array) -> Array:
        return jp.where(near.any(axis=0), jp.argmax(near, axis=0), line.shape[0])

    return {
        "valid": valid,
        "gates_seen": first_seen(gates_xy < sensor_range),
        "obstacles_seen": first_seen(poles_xy < sensor_range),
    }


def add_pole_hazard(key: Array, gates_pos: Array, gates_quat: Array, poles: Array) -> Array:
    """Move a random pole of one track right in front of or behind a random gate.

    Args:
        key: The random key.
        gates_pos: The gate positions (g, 3).
        gates_quat: The gate orientations (g, 4).
        poles: The pole positions (o, 3).

    Returns:
        The new pole positions (o, 3).
    """
    k_gate, k_pole, k_side, k_distance, k_offset = jax.random.split(key, 5)
    gate = jax.random.randint(k_gate, (), 0, gates_pos.shape[0])
    pole = jax.random.randint(k_pole, (), 0, poles.shape[0])
    side = jax.random.choice(k_side, jp.array([-1.0, 1.0]))
    distance = jax.random.uniform(k_distance, minval=HAZARD_DISTANCE[0], maxval=HAZARD_DISTANCE[1])
    offset = jax.random.uniform(k_offset, minval=-HAZARD_OFFSET, maxval=HAZARD_OFFSET)
    rot = quat_to_matrix(jp, gates_quat[gate])
    xy = gates_pos[gate, :2] + side * distance * rot[:2, 0] + offset * rot[:2, 1]
    return poles.at[pole, :2].set(xy)


def build_pool_fn(config: Any, args: GeneralArgs, sequence: Array) -> Any:
    """Build the jitted generator of a track pool.

    Args:
        config: The race configuration.
        args: The training settings.
        sequence: The gate IDs of the gate order.

    Returns:
        A function generating the arrays of a track pool from a random key.
    """
    track = config.env.track
    gates_z = [g["pos"][2] for g in track.gates]
    obstacles_z = [o["pos"][2] for o in track.obstacles]
    low, high = track.safety_limits.pos_limit_low, track.safety_limits.pos_limit_high
    sample = jax.vmap(build_random_track_fn(gates_z, obstacles_z, low, high))
    sample_wide = jax.vmap(build_random_track_fn(gates_z, obstacles_z, low, high, **WIDE_TRACKS))
    base_start = jp.array(track.drones[0]["pos"], dtype=jp.float32)
    spread = jp.array([*START_SPREAD, 0.0], dtype=jp.float32)
    sensor_range = float(config.env.sensor_range)
    n_wide = int(args.pool_size * args.wide_tracks)

    def build_track_extras(start: Array, gates_pos: Array, gates_quat: Array, poles: Array) -> dict:
        line, gate_index = plan_line(jp, start, gates_pos, gates_quat, poles, sequence)
        tables = line_tables(line, gate_index, gates_pos, poles, sensor_range)
        return {"line": line, "gate_index": gate_index, **tables}

    @jax.jit
    def generate(key: Array) -> dict[str, Array]:
        k_spread, k_use, k_narrow, k_wide, k_hazard, k_mask = jax.random.split(key, 6)
        shape = (args.pool_size, 3)
        offset = jax.random.uniform(k_spread, shape, minval=-spread, maxval=spread)
        moved = jax.random.uniform(k_use, (args.pool_size, 1)) < args.start_spread
        start = base_start + offset * moved
        n_narrow = args.pool_size - n_wide
        narrow = sample(start[:n_narrow], jax.random.split(k_narrow, n_narrow))
        wide = sample_wide(start[n_narrow:], jax.random.split(k_wide, n_wide))
        hazards = jax.vmap(add_pole_hazard)(jax.random.split(k_hazard, n_wide), *wide)
        use_hazard = jax.random.uniform(k_mask, (n_wide, 1, 1)) < args.pole_hazard
        wide = (wide[0], wide[1], jp.where(use_hazard, hazards, wide[2]))
        gates_pos, gates_quat, poles = (jp.concatenate(x) for x in zip(narrow, wide))
        extras = jax.vmap(build_track_extras)(start, gates_pos, gates_quat, poles)
        pool = {"gates_pos": gates_pos, "gates_quat": gates_quat, "poles": poles, "start": start}
        return pool | extras

    return generate


@struct.dataclass
class GeneralState:
    """The state of the training task on top of the environment data."""

    env: EnvData
    pool: dict[str, Array]
    track: Array  # Index of the track in the pool (n,)
    line_idx: Array  # Index of the projection on the line (n,)
    line_s: Array  # Refined progress along the line (n,)
    act_prev: Array  # Last action in the world frame (n, 3)
    act_prev2: Array  # Action before the last one in the world frame (n, 3)
    thrust_cmd: Array  # Last commanded thrust (n,)
    est: dict[str, Array]  # Mass estimator
    speed_cap: Array  # Speed cap of the episode (n,)
    from_start: Array  # True for episodes that started at the regular start (n,)
    lines: Array  # The repaired line of each drone (n, LINE_LENGTH, 3)
    gate_index: Array  # The sample index of each gate pass of each drone (n, m)
    revealed: Array  # Which gates and poles were revealed at the last line repair (n, g + o)
    pending: Array  # Whether the line of a drone waits for a repair (n,)
    rng: Array


def race_state(data: EnvData) -> dict[str, Array]:
    """The race observation of the environment without the drone dimension."""
    return {k: v[:, 0] for k, v in race_obs(data).items()}


def repair_pending(
    plans: Array,
    lines: Array,
    gate_index: Array,
    pending: Array,
    data: EnvData,
    sequence: Array,
    slots: int,
) -> tuple[Array, Array]:
    """Repair the lines of up to `slots` drones that wait for it, after objects were revealed.

    Args:
        plans: The planned line of each drone (n, LINE_LENGTH, 3).
        lines: The current line of each drone (n, LINE_LENGTH, 3).
        gate_index: The sample index of each gate pass of each drone (n, m).
        pending: Whether the line of a drone waits for a repair (n,).
        data: The environment data.
        sequence: The gate IDs of the gate order (m,).
        slots: The maximum number of repairs.

    Returns:
        The updated lines and pending flags.
    """
    n = pending.shape[0]
    chosen = jp.nonzero(pending, size=slots, fill_value=n)[0]
    race = race_state(data)

    def repair_one(i: Array) -> Array:
        return repair_line(
            jp,
            plans[i],
            gate_index[i],
            sequence,
            data.nominal_gates_pos[i],
            data.nominal_gates_quat[i],
            race["gates_pos"][i],
            race["gates_quat"][i],
            race["obstacles_pos"][i],
        )

    repaired = jax.vmap(repair_one)(jp.minimum(chosen, n - 1))
    return lines.at[chosen].set(repaired, mode="drop"), pending.at[chosen].set(False, mode="drop")


def revealed_objects(data: EnvData) -> Array:
    """Which gates and poles have been in sensor range (n, g + o)."""
    return jp.concatenate([data.gates_visited[:, 0], data.obstacles_visited[:, 0]], axis=1)


def load_limits(config: Any, args: GeneralArgs) -> dict[str, float]:
    """Action limits from the drone model."""
    params = load_dynamics_params(config.sim.dynamics, config.sim.drone)
    thrust_low, thrust_high = 4 * float(params["thrust_min"]), 4 * float(params["thrust_max"])
    return {
        "thrust_mid": (thrust_low + thrust_high) / 2.0,
        "thrust_half": (thrust_high - thrust_low) / 2.0,
        "max_tilt": args.max_tilt,
        "mass_nominal": float(params["mass"]),
    }


def arena_limits(config: Any) -> tuple[Array, Array]:
    """The horizontal safety limits of the arena."""
    limits = config.env.track.safety_limits
    low = jp.array(limits.pos_limit_low[:2], dtype=jp.float32)
    high = jp.array(limits.pos_limit_high[:2], dtype=jp.float32)
    return low, high


class GeneralTask:
    """Reward, resets and observations of races on random tracks."""

    def __init__(self, env: Any, config: Any, args: GeneralArgs, limits: dict[str, float]):
        """Build the task.

        Args:
            env: The vectorized race environment, which must not randomise the track layout itself.
            config: The race configuration.
            args: The training settings.
            limits: The action limits.
        """
        self.env, self.args, self.limits = env, args, limits
        self.n = env.num_envs
        self.dt = 1.0 / config.env.freq
        self.arena = arena_limits(config)
        safety = config.env.track.safety_limits
        self.pos_low = jp.array(safety.pos_limit_low, dtype=jp.float32)
        self.pos_high = jp.array(safety.pos_limit_high, dtype=jp.float32)
        self.gate_sequence = jp.asarray(env.data.gate_sequence)
        self.n_passes = self.gate_sequence.shape[0]
        self.mass_nominal = float(limits["mass_nominal"])
        self.base_start = jp.array(config.env.track.drones[0]["pos"], dtype=jp.float32)
        self.critic_ahead = jp.array(CRITIC_AHEAD)
        self.row = jp.arange(self.n)
        params = load_dynamics_params(config.sim.dynamics, config.sim.drone)
        c = np.asarray(params["rpm2thrust"], dtype=float)
        hover = self.mass_nominal * G / 4.0
        self.rotor_hover = float(np.max(np.roots([c[2], c[1], c[0] - hover]).real))

    def init(self, pool: dict[str, Array], rng: Array) -> GeneralState:
        """Create the initial state with all episodes reset to tracks of the pool."""
        data, _ = self.env._reset(self.env.data)
        state = GeneralState(
            env=data,
            pool=pool,
            track=jp.zeros(self.n, dtype=jp.int32),
            line_idx=jp.zeros(self.n, dtype=jp.int32),
            line_s=jp.zeros(self.n, dtype=jp.float32),
            act_prev=jp.zeros((self.n, 3), dtype=jp.float32),
            act_prev2=jp.zeros((self.n, 3), dtype=jp.float32),
            thrust_cmd=jp.full((self.n,), self.mass_nominal * G, dtype=jp.float32),
            est=mass_estimate_init(jp, self.n, self.mass_nominal),
            speed_cap=jp.full((self.n,), self.args.cap_high, dtype=jp.float32),
            from_start=jp.ones(self.n, dtype=bool),
            lines=pool["line"][jp.zeros(self.n, dtype=jp.int32)],
            gate_index=pool["gate_index"][jp.zeros(self.n, dtype=jp.int32)],
            revealed=revealed_objects(data),
            pending=jp.zeros(self.n, dtype=bool),
            rng=rng,
        )
        return self.reset(state, jp.ones(self.n, dtype=bool), jp.float32(0.0))

    def reset(self, state: GeneralState, mask: Array, p_rsi: Array) -> GeneralState:
        """Reset the masked episodes to new tracks, a fraction p_rsi of them on the line."""
        args, pool = self.args, state.pool
        rng, *keys = jax.random.split(state.rng, 10)
        n_tracks = pool["start"].shape[0]
        track = jp.where(mask, jax.random.randint(keys[0], (self.n,), 0, n_tracks), state.track)

        def on_reset(reset_value: Array, value: Array) -> Array:
            return jp.where(mask.reshape(mask.shape + (1,) * (value.ndim - 1)), reset_value, value)

        # The environment reset perturbs the nominal layout, which is written here beforehand
        data = state.env.replace(
            nominal_gates_pos=on_reset(pool["gates_pos"][track], state.env.nominal_gates_pos),
            nominal_gates_quat=on_reset(pool["gates_quat"][track], state.env.nominal_gates_quat),
            nominal_obstacles_pos=on_reset(pool["poles"][track], state.env.nominal_obstacles_pos),
        )
        data, _ = self.env._reset(data, None, mask)

        # Reference states: a random valid sample of the line of the track
        rsi = mask & (jax.random.uniform(keys[1], (self.n,)) < p_rsi)
        score = jax.random.uniform(keys[2], (self.n, LINE_LENGTH)) * pool["valid"][track]
        idx = jp.argmax(score, axis=1)
        rsi = rsi & pool["valid"][track, idx]
        cap = jax.random.uniform(keys[3], (self.n,), minval=args.cap_low, maxval=args.cap_high)
        line_pos = pool["line"][track, idx]
        line_next = pool["line"][track, jp.minimum(idx + 1, LINE_LENGTH - 1)]
        tangent = line_next - line_pos
        tangent = tangent / (jp.linalg.norm(tangent, axis=-1, keepdims=True) + 1e-6)
        noise = jax.random.uniform(keys[4], (self.n, 3), minval=-0.1, maxval=0.1)
        top = jp.minimum(cap, args.rsi_speed)
        speed = 0.3 + (top - 0.3) * jax.random.uniform(keys[5], (self.n,))
        half_tilt = jax.random.uniform(keys[6], (self.n, 2), minval=-0.2, maxval=0.2)
        sr, cr = jp.sin(half_tilt[:, 0]), jp.cos(half_tilt[:, 0])
        sp, cp = jp.sin(half_tilt[:, 1]), jp.cos(half_tilt[:, 1])
        rsi_quat = jp.stack([sr * cp, cr * sp, -sr * sp, cr * cp], axis=-1)
        rotor = jax.random.uniform(
            keys[7], (self.n, 1), minval=HOVER_RPM_RANGE[0], maxval=HOVER_RPM_RANGE[1]
        )

        def select(rsi_value: Array, value: Array) -> Array:
            return jp.where(rsi.reshape(rsi.shape + (1,) * (value.ndim - 1)), rsi_value, value)

        # Regular starts move with the start position of the track
        states = data.sim_data.states
        offset = (pool["start"][track] - self.base_start) * mask[:, None]
        pos = select((line_pos + noise)[:, None], states.pos + offset[:, None])
        states = states.replace(
            pos=pos,
            vel=select((tangent * speed[:, None])[:, None], states.vel),
            quat=select(rsi_quat[:, None], states.quat),
            rotor_vel=select(self.rotor_hover * rotor[:, None], states.rotor_vel),
        )
        passed = jp.sum(idx[:, None] > pool["gate_index"][track], axis=1)
        near_gates = (
            jp.linalg.norm(pos[:, None, 0, :2] - data.gates_pos[..., :2], axis=-1)
            < data.sensor_range
        )
        near_poles = (
            jp.linalg.norm(pos[:, None, 0, :2] - data.obstacles_pos[..., :2], axis=-1)
            < data.sensor_range
        )
        gates_seen = near_gates | (rsi[:, None] & (pool["gates_seen"][track] <= idx[:, None]))
        poles_seen = near_poles | (rsi[:, None] & (pool["obstacles_seen"][track] <= idx[:, None]))
        data = data.replace(
            sim_data=data.sim_data.replace(states=states),
            n_gates_passed=select(passed[:, None], data.n_gates_passed),
            gates_visited=on_reset(gates_seen[:, None], data.gates_visited),
            obstacles_visited=on_reset(poles_seen[:, None], data.obstacles_visited),
            last_drone_pos=on_reset(pos, data.last_drone_pos),
            takeoff_pos=on_reset(pos, data.takeoff_pos),
        )
        fresh = mass_estimate_init(jp, self.n, self.mass_nominal)
        # New episodes start on the planned line. New reveals of any episode queue a line repair
        revealed = revealed_objects(data)
        pending = state.pending | mask | jp.any(revealed != state.revealed, axis=1)
        plans = pool["line"][track]
        gate_index = on_reset(pool["gate_index"][track], state.gate_index)
        lines, pending = repair_pending(
            plans,
            on_reset(plans, state.lines),
            gate_index,
            pending,
            data,
            self.gate_sequence,
            args.repair_slots,
        )
        return state.replace(
            lines=lines,
            gate_index=gate_index,
            revealed=revealed,
            pending=pending,
            env=data,
            track=track,
            line_idx=on_reset(jp.where(rsi, idx, 0), state.line_idx),
            line_s=on_reset(jp.where(rsi, idx * LINE_DS, 0.0), state.line_s),
            act_prev=on_reset(jp.zeros_like(state.act_prev), state.act_prev),
            act_prev2=on_reset(jp.zeros_like(state.act_prev2), state.act_prev2),
            thrust_cmd=on_reset(
                jp.full_like(state.thrust_cmd, self.mass_nominal * G), state.thrust_cmd
            ),
            est={k: on_reset(fresh[k], state.est[k]) for k in fresh},
            speed_cap=on_reset(cap, state.speed_cap),
            from_start=on_reset(~rsi, state.from_start),
            rng=rng,
        )

    def step(self, state: GeneralState, action: Array) -> tuple:
        """Step the race without resetting.

        Args:
            state: The current state.
            action: Policy actions in [-1, 1] (n, 3).

        Returns:
            The next state, the reward, the episode end flag and episode statistics.
        """
        args = self.args
        c, s_frame = gate_frame(jp, race_state(state.env), self.gate_sequence)
        command, world_action = general_command(jp, action, c, s_frame, self.limits)
        passed_before = state.env.n_gates_passed[:, 0]
        vel_before = state.env.sim_data.states.vel[:, 0]
        data, (_, _, disabled, truncated, _) = self.env._step(state.env, command)
        disabled, truncated = disabled[:, 0], truncated[:, 0]

        states = data.sim_data.states
        pos, vel = states.pos[:, 0], states.vel[:, 0]
        passed = data.n_gates_passed[:, 0]
        passed_gate = passed > passed_before
        finished = passed >= self.n_passes

        idx, s, off_line = project_on_line(
            jp, state.lines, state.gate_index, self.row, state.line_idx, pos, passed
        )
        outside = jp.any((pos < self.pos_low) | (pos > self.pos_high), axis=-1)
        crashed = (disabled | (off_line > args.off_line_limit) | outside) & ~finished

        progress = jp.clip(s - state.line_s, -0.25, 0.25)
        progress = jp.minimum(progress, state.speed_cap * self.dt)
        overspeed = jp.maximum(jp.linalg.norm(vel, axis=-1) - state.speed_cap, 0.0)

        # Gate-centre term, from the crossing point in the frame of the gate that was just passed
        row = jp.arange(self.n)
        gate = self.gate_sequence[jp.minimum(passed_before, self.n_passes - 1)]
        gate_rot = quat_to_matrix(jp, data.gates_quat[row, gate])
        local = jp.sum(gate_rot * (pos - data.gates_pos[row, gate])[..., None], axis=-2)
        offset = jp.maximum(jp.abs(local[:, 1]), jp.abs(local[:, 2]))
        centered = jp.clip(1.0 - (offset / 0.2) ** 2, 0.0, 1.0)

        # A soft margin around the true objects, which gives a signal before there is a crash
        solids = closest_solids(jp, pos, data.gates_pos, data.gates_quat)
        to_gate = jp.sqrt(jp.min(jp.sum(solids**2, axis=-1), axis=1) + 1e-9)
        poles = jp.linalg.norm(data.obstacles_pos[..., :2] - pos[:, None, :2], axis=-1)
        to_object = jp.minimum(to_gate, jp.min(poles, axis=1) - POLE_RADIUS) - DRONE_RADIUS
        intrusion = jp.clip(1.0 - to_object / args.clearance_margin, 0.0, 1.0)

        # The line keeps clear of objects, but progress alone would reward cutting into its corners
        contour = jp.maximum(off_line - args.contouring_width, 0.0)

        reward = args.progress * progress * ~crashed
        reward -= args.contouring * contour**2 * ~crashed
        reward -= args.overspeed * overspeed**2 * ~crashed
        reward -= args.clearance * intrusion**2 * ~crashed
        reward += passed_gate * (args.gate_bonus + args.gate_center_bonus * centered)
        reward += finished * args.finish_bonus
        reward -= crashed * args.crash
        reward -= args.action_rate * jp.sum((world_action - state.act_prev) ** 2, axis=-1)
        reward -= args.ang_vel * jp.sum(states.ang_vel[:, 0] ** 2, axis=-1)

        done = crashed | finished | truncated
        est = mass_estimate_update(
            jp,
            state.est,
            pos,
            states.quat[:, 0],
            vel,
            vel_before,
            state.thrust_cmd,
            self.limits,
            self.dt,
        )
        next_state = state.replace(
            env=data,
            line_idx=idx,
            line_s=s,
            act_prev=world_action,
            act_prev2=state.act_prev,
            thrust_cmd=command[:, 3],
            est=est,
        )
        stats = {
            "finished": finished,
            "crashed": crashed,
            "truncated": truncated & ~finished & ~crashed,
            "passed": passed,
            "time": data.steps * self.dt - self.dt,
        }
        return next_state, reward, done, stats

    def observe(self, state: GeneralState) -> tuple[Array, Array]:
        """The policy input and the critic input of the current state."""
        race = race_state(state.env)
        ratio = mass_ratio(jp, state.est, self.mass_nominal)
        policy_in = general_observation(
            jp,
            race,
            self.gate_sequence,
            state.act_prev,
            state.act_prev2,
            ratio,
            state.speed_cap,
            self.arena,
            line_ahead(jp, state.lines, self.row, state.line_idx),
        )
        # Privileged critic input: true poses, true mass and more of the line ahead
        data = state.env
        row = self.row
        pos, passed = race["pos"], race["n_gates_passed"]
        c, s = gate_frame(jp, race, self.gate_sequence)
        gate = self.gate_sequence[jp.minimum(passed, self.n_passes - 1)]
        following = self.gate_sequence[jp.minimum(passed + 1, self.n_passes - 1)]
        poles = data.obstacles_pos[..., :2] - pos[:, None, :2]
        order = jp.argsort(jp.sum(poles**2, axis=-1), axis=1)
        poles = poles[row[:, None], order]
        ahead = jp.minimum(state.line_idx[:, None] + self.critic_ahead[None], LINE_LENGTH - 1)
        ahead = state.lines[row[:, None], ahead] - pos[:, None]
        mass = data.sim_data.params.mass[:, 0, 0]
        privileged = [
            to_frame(jp, data.gates_pos[row, gate] - pos, c, s) / 2.0,
            to_frame(jp, data.gates_pos[row, following] - pos, c, s) / 2.0,
            to_frame(jp, poles, c, s).reshape(self.n, -1) / 2.0,
            to_frame(jp, ahead, c, s).reshape(self.n, -1) / 2.0,
            ((mass / self.mass_nominal - 1.0) / 0.12)[:, None],
            (state.line_s / (state.gate_index[:, -1] * LINE_DS + 1e-6))[:, None],
        ]
        return policy_in, jp.concatenate([policy_in, *privileged], axis=-1)


def pad_inputs(params: dict, old_dim: int, new_dim: int) -> dict:
    """Add inputs to a loaded policy, whose outputs stay the same because their weights are zero.

    New policy inputs follow the old ones, and the critic inputs follow the policy inputs.

    Args:
        params: The loaded parameters, with the input layers of a policy with old_dim inputs.
        old_dim: The number of old policy inputs.
        new_dim: The number of new policy inputs.

    Returns:
        The parameters for new_dim policy inputs.
    """

    def pad(layers: list, n_outputs_at: int) -> list:
        w, b = layers[0]
        zeros = jp.zeros((new_dim - old_dim, w.shape[1]), dtype=w.dtype)
        w = jp.concatenate([w[:n_outputs_at], zeros, w[n_outputs_at:]])
        return [(w, b), *layers[1:]]

    return params | {
        "actor": pad(params["actor"], old_dim),
        "critic": pad(params["critic"], old_dim),
    }


def make_real_eval(env: Any, config: Any, args: GeneralArgs, limits: dict[str, float]) -> Any:
    """Build the jitted evaluation of a policy on an unmodified environment.

    Args:
        env: The unmodified vectorized race environment.
        config: The race configuration.
        args: The settings.
        limits: The action limits.

    Returns:
        A function running one batch of episodes from a given speed cap.
    """
    n = env.num_envs
    dt = 1.0 / config.env.freq
    arena = arena_limits(config)
    sequence = jp.asarray(env.data.gate_sequence)
    n_passes = sequence.shape[0]
    mass_nominal = float(limits["mass_nominal"])
    safety = config.env.track.safety_limits
    pos_low = jp.array(safety.pos_limit_low, dtype=jp.float32)
    pos_high = jp.array(safety.pos_limit_high, dtype=jp.float32)
    plan = jax.vmap(lambda *x: plan_line(jp, *x, sequence))
    track = jp.arange(n)

    @jax.jit
    def evaluate(actor: list, data: EnvData, cap: Array) -> dict[str, Array]:
        data, _ = env._reset(data, None, jp.ones(n, dtype=bool))
        caps = jp.full((n,), cap, dtype=jp.float32)
        gates_pos, gates_quat = data.nominal_gates_pos, data.nominal_gates_quat
        plans, gate_index = plan(
            data.sim_data.states.pos[:, 0], gates_pos, gates_quat, data.nominal_obstacles_pos
        )

        def body(carry: tuple, _: None) -> tuple:
            (
                data,
                act1,
                act2,
                est,
                thrust,
                active,
                steps,
                finished,
                passed,
                path,
                left,
                idx,
                lines,
                pending,
                revealed,
            ) = carry
            now = revealed_objects(data)
            pending = pending | jp.any(now != revealed, axis=1)
            lines, pending = repair_pending(
                plans, lines, gate_index, pending, data, sequence, args.repair_slots
            )
            race = race_state(data)
            idx, _, _ = project_on_line(
                jp, lines, gate_index, track, idx, race["pos"], race["n_gates_passed"]
            )
            policy_in = general_observation(
                jp,
                race,
                sequence,
                act1,
                act2,
                mass_ratio(jp, est, mass_nominal),
                caps,
                arena,
                line_ahead(jp, lines, track, idx),
            )
            action = mlp_forward(jp, actor, policy_in)
            c, s = gate_frame(jp, race, sequence)
            command, world_action = general_command(jp, action, c, s, limits)
            vel_before, pos_before = race["vel"], race["pos"]
            data, (_, _, disabled, truncated, _) = env._step(data, command)
            pos = data.sim_data.states.pos[:, 0]
            vel = data.sim_data.states.vel[:, 0]
            quat = data.sim_data.states.quat[:, 0]
            est = mass_estimate_update(jp, est, pos, quat, vel, vel_before, thrust, limits, dt)
            n_passed = data.n_gates_passed[:, 0]
            done_now = n_passed >= n_passes
            alive = active & ~disabled[:, 0]
            path = path + jp.linalg.norm(pos - pos_before, axis=-1) * alive
            left = left | (alive & jp.any((pos < pos_low) | (pos > pos_high), axis=-1))
            steps = steps + active
            finished = finished | (active & done_now)
            passed = jp.where(active, n_passed, passed)
            active = active & ~(disabled[:, 0] | truncated[:, 0] | done_now)
            carry = (
                data,
                world_action,
                act1,
                est,
                command[:, 3],
                active,
                steps,
                finished,
                passed,
                path,
                left,
                idx,
                lines,
                pending,
                now,
            )
            return carry, None

        zeros_i = jp.zeros(n, dtype=jp.int32)
        carry = (
            data,
            jp.zeros((n, 3), dtype=jp.float32),
            jp.zeros((n, 3), dtype=jp.float32),
            mass_estimate_init(jp, n, mass_nominal),
            jp.full((n,), mass_nominal * G, dtype=jp.float32),
            jp.ones(n, dtype=bool),
            zeros_i,
            jp.zeros(n, dtype=bool),
            zeros_i,
            jp.zeros(n, dtype=jp.float32),
            jp.zeros(n, dtype=bool),
            zeros_i,
            plans,
            jp.ones(n, dtype=bool),
            revealed_objects(data),
        )
        carry, _ = jax.lax.scan(body, carry, None, length=args.eval_steps)
        data, *_, active, steps, finished, passed, path, left, _, _, _, _ = carry
        return {
            "data": data,
            "steps": steps,
            "finished": finished,
            "passed": passed,
            "path": path,
            "left": left,
            "active": active,
            "gates_pos": gates_pos,
            "gates_quat": gates_quat,
        }

    return evaluate


def summarize(result: dict[str, Array], dt: float) -> dict[str, float]:
    """Success rate, lap time and speed of a batch of evaluation episodes."""
    finished = np.asarray(result["finished"])
    laps = (np.asarray(result["steps"])[finished] - 1) * dt
    path = np.asarray(result["path"])[finished]
    return {
        "success": float(finished.mean()),
        "lap": float(laps.mean()) if len(laps) else np.nan,
        "lap_std": float(laps.std()) if len(laps) else 0.0,
        "speed": float((path / np.maximum(laps, dt)).mean()) if len(laps) else np.nan,
        "gates": float(np.asarray(result["passed"]).mean()),
        "left": float(np.asarray(result["left"]).mean()),
        "strict": float((finished & ~np.asarray(result["left"])).mean()),
    }


def make_race_config(config: str) -> Any:
    """Load a race configuration for headless batched simulation with the attitude interface."""
    race_config = load_config(ROOT / "config" / config)
    race_config.sim.render = False
    race_config.env.control_mode = "attitude"
    return race_config


def train(config: str = "level3.toml", **kwargs: Any):
    """Train the policy.

    Args:
        config: The race configuration in ``config/``.
        **kwargs: Overrides of the settings in `GeneralArgs`.
    """
    args = GeneralArgs(**kwargs)
    print(asdict(args), flush=True)
    race_config = make_race_config(config)
    limits = load_limits(race_config, args)

    # The training environment gets its layouts from the pool, the evaluation is unmodified
    train_config = copy.deepcopy(race_config)
    train_config.env.track.randomize = False
    FastContactRaceEnv.drone_inflate = args.drone_inflate
    env = make_env(FastContactRaceEnv, train_config, args, args.num_envs, args.seed)
    task = GeneralTask(env, train_config, args, limits)
    eval_env = make_env(VecDroneRaceEnv, race_config, args, args.eval_envs, args.seed + 1000)
    evaluate = make_real_eval(eval_env, race_config, args, limits)
    eval_data = eval_env.data

    rng = jax.random.PRNGKey(args.seed)
    rng, k_pool, k_state, k_actor, k_critic = jax.random.split(rng, 5)
    generate = build_pool_fn(train_config, args, task.gate_sequence)
    start_pool = time.perf_counter()
    pool = generate(k_pool)
    jax.block_until_ready(pool)
    print(f"Track pool generated in {time.perf_counter() - start_pool:.1f} s", flush=True)
    state = task.init(pool, k_state)

    obs_pi, obs_vf = task.observe(state)
    print(f"Observation sizes: actor {obs_pi.shape[-1]}, critic {obs_vf.shape[-1]}", flush=True)
    sizes = list(args.hidden)
    params = {
        "actor": init_mlp(k_actor, [obs_pi.shape[-1], *sizes, 3], 0.01),
        "critic": init_mlp(k_critic, [obs_vf.shape[-1], *sizes, 1], 1.0),
        "log_std": jp.full(3, args.log_std_init, dtype=jp.float32),
    }
    if args.init_policy:
        new_dim = params["actor"][0][0].shape[0]
        params = load_policy_params(args.init_policy, params, args.max_tilt)
        old_dim = params["actor"][0][0].shape[0]
        if old_dim != new_dim:
            params = pad_inputs(params, old_dim, new_dim)
        print(f"Continuing from {args.init_policy} ({old_dim} -> {new_dim} inputs)", flush=True)

    batch = args.num_envs * args.num_steps
    iterations = args.total_steps // batch
    n_updates = iterations * args.epochs * args.minibatches
    schedule = optax.linear_schedule(args.lr, args.lr_final, n_updates)
    optimizer = optax.adam(schedule, eps=1e-5)
    opt_state = optimizer.init(params)
    train_iter = make_train_iter(task, args, optimizer)
    reset_all = jax.jit(lambda s: task.reset(s, jp.ones(task.n, dtype=bool), jp.float32(0.0)))

    run_dir = CHECKPOINT_DIR / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    arena = {k: np.asarray(v) for k, v in zip(("arena_low", "arena_high"), task.arena)}
    best = -np.inf
    start = time.perf_counter()
    window: dict[str, float] = {}
    for iteration in range(1, iterations + 1):
        if iteration % args.pool_refresh == 0:
            rng, key = jax.random.split(rng)
            state = reset_all(state.replace(pool=generate(key)))
        frac = (iteration - 1) / max(iterations - 1, 1)
        p_rsi = jp.float32(args.rsi_start + frac * (args.rsi_end - args.rsi_start))
        params, opt_state, state, metrics = train_iter(params, opt_state, state, p_rsi)
        metrics = {k: float(v) for k, v in metrics.items()}
        for k in ("ended", "finished", "crashed", "lap_sum", "passed_sum"):
            window[k] = window.get(k, 0.0) + metrics[k]

        if iteration % 10 == 0 or iteration == 1:
            elapsed = time.perf_counter() - start
            ended = max(window["ended"], 1.0)
            lap = window["lap_sum"] / max(window["finished"], 1.0)
            print(
                f"{iteration:5d}/{iterations} {iteration * batch / 1e6:6.1f}M "
                f"{iteration * batch / elapsed:7.0f} sps | rew {metrics['reward']:6.3f} "
                f"val {metrics['value']:6.2f} vloss {metrics['value_loss']:.2f} "
                f"kl {metrics['kl']:.4f} "
                f"std {np.exp(np.asarray(params['log_std'])).round(3)} | "
                f"start episodes {window['ended']:5.0f} finish {window['finished'] / ended:5.1%} "
                f"crash {window['crashed'] / ended:5.1%} gates {window['passed_sum'] / ended:4.2f} "
                f"lap {lap:5.2f}",
                flush=True,
            )
            window = {}

        if iteration % args.eval_every == 0 or iteration == iterations:
            lines, success = [], []
            for cap in args.eval_caps:
                result = evaluate(params["actor"], eval_data, jp.float32(cap))
                eval_data = result["data"]
                summary = summarize(result, task.dt)
                success.append(summary["success"])
                lines.append(
                    f"cap {cap:.1f}: {summary['success']:5.1%} lap {summary['lap']:5.2f} "
                    f"speed {summary['speed']:4.2f} gates {summary['gates']:4.2f}"
                )
            print(f"  EVAL {iteration * batch / 1e6:6.1f}M | " + " | ".join(lines), flush=True)
            extra = critic_arrays(params) | arena
            save_policy(run_dir / "latest.npz", params, limits, extra)
            if iteration % args.save_every == 0:
                name = f"step_{iteration * batch // 1000000}M.npz"
                save_policy(run_dir / name, params, limits, extra)
            value = float(np.mean(success))
            if value > best:
                best = value
                save_policy(run_dir / "best.npz", params, limits, extra)
                print(f"  new best {best:.3f}", flush=True)


def sweep(
    policy: str,
    config: str = "level3.toml",
    caps: tuple[float, ...] = (1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0),
    episodes: int = 2048,
    steps: int = 1500,
    seed: int = 12345,
):
    """Measure speed against success on an unmodified environment.

    Args:
        policy: The stored policy.
        config: The race configuration in ``config/``.
        caps: The speed caps to evaluate.
        episodes: The number of episodes per cap.
        steps: The maximum number of steps per episode, 1500 as in ``scripts/sim.py``.
        seed: The random seed of the environment.
    """
    args = GeneralArgs(max_episode_steps=steps, eval_steps=steps, eval_envs=episodes)
    race_config = make_race_config(config)
    ckpt = np.load(policy)
    limits = {k: float(ckpt[k]) for k in ("thrust_mid", "thrust_half", "max_tilt", "mass_nominal")}
    n_layers = int(ckpt["n_layers"])
    actor = [(jp.asarray(ckpt[f"w{i}"]), jp.asarray(ckpt[f"b{i}"])) for i in range(n_layers)]
    env = make_env(VecDroneRaceEnv, race_config, args, episodes, seed)
    evaluate = make_real_eval(env, race_config, args, limits)
    data = env.data
    print(f"{config}, {episodes} episodes per cap")
    for cap in caps:
        result = evaluate(actor, data, jp.float32(cap))
        data = result["data"]
        s = summarize(result, 1.0 / race_config.env.freq)
        print(
            f"cap {cap:3.1f} m/s: success {s['success']:6.1%}  lap {s['lap']:5.2f} s "
            f"(std {s['lap_std']:4.2f})  mean speed {s['speed']:4.2f} m/s  "
            f"gates {s['gates']:4.2f}  left safety box {s['left']:5.1%}  "
            f"finished inside it {s['strict']:6.1%}",
            flush=True,
        )


if __name__ == "__main__":
    fire.Fire({"train": train, "sweep": sweep})
