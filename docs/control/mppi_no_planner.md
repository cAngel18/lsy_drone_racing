# control.mppi_no_planner

The same MPPI controller as [control.mppi_controller](mppi_controller.md), but **without a path
planner**: no grid search, no spline, no speed profile. MPPI only knows the gates (position,
orientation, order and passing direction) and the obstacles.

```bash
python scripts/sim.py --config level3.toml --controller mppi_no_planner.py --render True
```

## What is different

| | `mppi_controller.py` | `mppi_no_planner.py` |
| --- | --- | --- |
| Model, sampling, weighting | identical | identical |
| Collision, bounds and smoothness costs | identical (`safety_cost`) | identical (`safety_cost`) |
| Guidance | Progress along a planned path, inside a corridor | *Distance to go*, computed from the gate geometry inside every rollout |
| Speed limit | Planner's curvature-based speed profile | One soft speed limit (slower close to unobserved gates) |
| Re-planning | When the observed track changes | Not needed, the cost uses the current observation directly |

## Distance to go

Every rollout is scored by how far the drone still has to fly at the end of the horizon to pass all
remaining gates (`w_progress` per meter). For the target gate, the estimate depends on where the
drone is:

* **In front of the gate:** distance to the gate center, where lateral and vertical offsets count
  more (`lateral_weight`). This forms a funnel that leads into the opening along the passing
  direction.
* **Behind the gate:** the drone has to fly around the frame. The estimate is the distance to a
  point next to the frame (`corner_margin`), from there to an entry point in front of the gate
  (`entry_dist`), and from there to the center.

The rollouts track the gate passes themselves: a rollout that crosses the target gate inside the
opening switches to the next gate. Passing a gate therefore immediately pays off, and MPPI already
plans the exit of a gate towards the next one. The distance between the following gates is added
as an offset, so the cost stays continuous when a gate is passed.

## When it struggles

Obstacles only appear through the collision costs. MPPI looks 0.8 s ahead, so a single pole in the
way is no problem: some rollouts curve around it. If several obstacles form a wall between the
drone and the gate, though, no short rollout finds the way around, and the drone would stop in
front of the wall. The planner of `mppi_controller.py` exists for these cases. On the random
level-3 tracks and the level-2 track we tested, the planner-free version never got stuck, but it
crashed more often (see below).

## Results

Headless simulation on an Apple Silicon laptop (CPU only), same tracks for both controllers:

| Config | Runs | With planner | Without planner |
| --- | --- | --- | --- |
| Level 3 (random tracks) | 51 | 50 finished (98 %), mean 8.2 s | 47 finished (92 %), mean 7.7 s (6.2 - 10.3 s) |
| Level 2 (competition track) | 20 | 18 finished (90 %), mean 6.9 s | 18 finished (90 %), mean 6.5 s (6.2 - 7.2 s) |

All failures of the planner-free version are crashes (three into obstacles, three into gate frames
while flying around a gate), none are stalls. It needs ~11.9 ms per control step. A wider way
around the gate frames (`corner_margin = 0.4`) and an earlier obstacle cost (`r_obs_safe = 0.28`)
gave 48/51 and 18/20, but 0.4 s slower on level 2, so the defaults were kept.

::: lsy_drone_racing.control.mppi_no_planner
    options:
      inherited_members: true
