# PD-RSAC: Primal-Dual Robust Soft Actor-Critic

## PD-RSAC

**PD-RSAC** stands for **Primal-Dual Robust Soft Actor-Critic**. It is our fleet-level EV fleet control model that combines **Semi-MDP duration-aware backups**, **Wasserstein-1 DRO** for robustness to demand shift, and **MILP feasible-action projection** so final actions satisfy SoC, charger, and feeder constraints. In this repository, PD-RSAC is the fleet SAC path with **`--wdro`**, **`--semi-mdp`**, and **`--milp`** enabled.

---

## Repository layout

```text
.
├── README.md                               - overview of PD-RSAC and how to run it
├── requirements.txt                        - Python dependencies for training, evaluation, H3, and MILP
├── preprocess_full_data.py                 - preprocesses raw taxi parquet data into H3-based trip parquet data
└── gpu_core/
    ├── __init__.py                         - package marker
    ├── assignment/
    │   ├── __init__.py                     - package marker
    │   └── hex_vehicle_assigner.py         - maps fleet-level hex decisions to individual idle vehicles
    ├── config/
    │   ├── __init__.py                     - config exports
    │   ├── base.py                         - dataclass definitions for environment, training, replay, logging, and model config
    │   └── loader.py                       - YAML/CLI config loading and override logic
    ├── data/
    │   ├── __init__.py                     - package marker
    │   └── real_trip_loader.py             - loads, filters, and serves real parquet trip data
    ├── features/
    │   ├── __init__.py                     - package marker
    │   ├── builder.py                      - builds vehicle, hex, and global context features for the agent
    │   ├── replay_buffer.py                - replay buffer for SAC / fleet SAC training
    │   ├── ppo_buffer.py                   - rollout buffer for MAPPO training
    │   └── maddpg_buffer.py                - replay buffer for MADDPG training
    ├── networks/
    │   ├── __init__.py                     - network exports
    │   ├── critic.py                       - legacy flat critic variants and value-network utilities
    │   ├── gcn.py                          - shared graph convolution encoder over the hex graph
    │   ├── gcn_actor.py                    - legacy GCN actor and active FleetGCNActor
    │   ├── gcn_critic.py                   - legacy GCN critic and active fleet GCN twin critic
    │   ├── maddpg_actor.py                 - MADDPG actor network
    │   ├── maddpg_agent.py                 - MADDPG agent wrapper with target networks
    │   ├── maddpg_critic.py                - MADDPG centralized critic
    │   ├── mappo_actor.py                  - MAPPO actor network
    │   ├── mappo_critic.py                 - MAPPO centralized critic
    │   ├── ppo_agent.py                    - PPO / MAPPO agent wrapper
    │   └── sac.py                          - legacy SACAgent and active FleetSACAgent
    ├── scripts/
    │   ├── __init__.py                     - package marker
    │   ├── train.py                        - main training entrypoint for fleet SAC, PD-RSAC, MAPPO, and MADDPG
    │   ├── evaluate.py                     - evaluates trained checkpoints on real trip data
    │   ├── evaluate_maddpg.py              - evaluates MADDPG checkpoints
    │   ├── evaluate_mappo.py               - evaluates MAPPO checkpoints
    │   ├── heuristic_matching.py           - heuristic fleet-control / matching baseline script
    │   ├── powergrid_evaluate.py           - evaluates charging-power and feeder-related behavior
    │   ├── powergrid_evaluate_mappo.py     - power-grid evaluation for MAPPO checkpoints
    │   ├── config_wdro_mod.yaml            - recommended main PD-RSAC config
    │   ├── config_wdro.yaml                - WDRO fleet config with paper-oriented defaults and notes
    │   ├── config_sac.yaml                 - fleet SAC baseline config
    │   ├── config_mappo.yaml               - MAPPO baseline config
    │   └── config_maddpg.yaml              - MADDPG baseline config
    ├── simulator/
    │   ├── __init__.py                     - simulator exports
    │   ├── action_processor.py             - executes serve, charge, and reposition actions on vehicles
    │   ├── baseline_reward_attribution.py  - per-vehicle reward attribution for baseline algorithms
    │   ├── dynamics.py                     - energy, travel-time, and SoC transition dynamics
    │   ├── environment.py                  - main GPU EV fleet environment (GPUEnvironmentV2)
    │   ├── reward.py                       - computes fleet reward components
    │   └── trip_manager.py                 - manages trip spawning, assignment, completion, and dropping
    ├── spatial/
    │   ├── __init__.py                     - spatial exports
    │   ├── assignment.py                   - trip/station assignment utilities
    │   ├── distance.py                     - distance-matrix construction and access
    │   ├── grid.py                         - hex-grid structure, geometry, and adjacency
    │   └── neighbors.py                    - k-hop neighbor computation and padded neighbor indices
    ├── state/
    │   ├── __init__.py                     - state exports
    │   ├── fleet.py                        - tensor fleet-state definitions and vehicle status enums
    │   ├── stations.py                     - tensor charging-station state definitions
    │   └── trips.py                        - tensor trip-state definitions
    ├── training/
    │   ├── __init__.py                     - training exports and aliases
    │   ├── assignment_loss.py              - auxiliary reposition / assignment-related supervision losses
    │   ├── distributed.py                  - distributed and mixed-precision training helpers
    │   ├── enhanced_collector.py           - collector for fleet SAC / Semi-MDP / WDRO training
    │   ├── enhanced_trainer.py             - main trainer for Semi-MDP + WDRO fleet SAC
    │   ├── episode_collector.py            - standard SAC / fleet episode collection logic
    │   ├── maddpg_collector.py             - rollout collection for MADDPG
    │   ├── maddpg_trainer.py               - MADDPG training loop
    │   ├── milp_assignment.py              - rolling MILP feasible-action projection
    │   ├── ppo_collector.py                - rollout collection for MAPPO
    │   ├── ppo_trainer_enhanced.py         - MAPPO training loop
    │   ├── semi_mdp.py                     - duration-aware discounting and action-duration logic
    │   ├── trainer.py                      - standard SAC and older fleet trainer implementations
    │   └── wdro.py                         - Wasserstein adversary, graph metric, and primal-dual robustness updates
    └── utils/
        ├── __init__.py                     - package marker
        ├── profiler.py                     - profiling helpers
        └── visualizer.py                   - visualization helpers
```

---

## Run PD-RSAC training

```bash
python gpu_core/scripts/train.py --config gpu_core/scripts/config_wdro_mod.yaml --real-data <PROCESSED_TRIPS_PARQUET> --trip-sample 0.2 --episodes 500 --start-date <TRAIN_START_DATE> --end-date <TRAIN_END_DATE> --eval-start-date <EVAL_START_DATE> --eval-end-date <EVAL_END_DATE> --target-h3-resolution 8 --milp --wdro --semi-mdp --checkpoint-dir <CHECKPOINT_DIR> --tensorboard
```

---

## Resume PD-RSAC training

```bash
python gpu_core/scripts/train.py --config gpu_core/scripts/config_wdro_mod.yaml --real-data <PROCESSED_TRIPS_PARQUET> --trip-sample 0.2 --start-date <TRAIN_START_DATE> --end-date <TRAIN_END_DATE> --eval-start-date <EVAL_START_DATE> --eval-end-date <EVAL_END_DATE> --target-h3-resolution 8 --milp --wdro --semi-mdp --resume <CHECKPOINT_PATH>
```

---

## Evaluate a PD-RSAC checkpoint

```bash
python gpu_core/scripts/evaluate.py --checkpoint <CHECKPOINT_PATH> --config gpu_core/scripts/config_wdro_mod.yaml --real-data <PROCESSED_TRIPS_PARQUET> --start-date <YYYY-MM-DD> --end-date <YYYY-MM-DD> --milp --deterministic
```

---

## Run power-grid evaluation

```bash
python gpu_core/scripts/powergrid_evaluate.py --checkpoint <CHECKPOINT_PATH> --config gpu_core/scripts/config_wdro_mod.yaml --real-data <PROCESSED_TRIPS_PARQUET> --day <YYYY-MM-DD> --milp --out-csv <OUTPUT_CSV> --out-json <OUTPUT_JSON>
```

---

## Preprocess taxi data

```bash
python preprocess_full_data.py --input <RAW_TAXI_PARQUET> --output-dir <PROCESSED_DATA_DIR> --sample 1.0 --resolution 9
```

---

## Requirements / conditions

- Install Python dependencies from `requirements.txt`
- GPU training is the intended setup - 17GB VRAM needed to run all settings
- `--milp` requires **Gurobi** and a valid license
- training and evaluation commands expect a processed parquet file such as `<PROCESSED_TRIPS_PARQUET>`
