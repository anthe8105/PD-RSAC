"""
Training entry point of every learned method: PD-RSAC and its ablations, SAC, MAPPO and MADDPG.
All of them use the vehicle-level stack (gpu_core/learning): the same encoder, actor network,
candidates, executor / MILP projection, replay of per-vehicle transitions and evaluation; the
learner is chosen by ``training.algo`` of the family YAML (sac, mappo, maddpg).

Usage:
    python train.py --preset full                    # PD-RSAC (see gpu_core/config/presets.py)
    python train.py --preset mappo_milp --episodes 500
    python train.py --config gpu_core/scripts/config_wdro.yaml --milp --wdro --semi-mdp
    python train.py --config ... --synthetic         # tiny smoke run without data
"""

import argparse
import os
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from gpu_core.config import Config, ConfigLoader  # noqa: E402
from gpu_core.data.dataset import resolve_path  # noqa: E402
from gpu_core.data.factory import SIM_DEVICE, build_env  # noqa: E402
from gpu_core.utils.run_record import build_run_record, write_run_record  # noqa: E402

VARIANT_FLAGS = ('milp', 'wdro', 'semi_mdp', 'wdro_metric', 'critic_population')


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--config', type=str, default=None, help='family YAML (config_{sac,wdro,mappo,maddpg}.yaml)')
    parser.add_argument('--preset', type=str, default=None,
                        help='experiment preset (full, no_milp, no_graph, diag_only, iac, sac_milp, sac, mappo, '
                             'mappo_milp, maddpg, maddpg_milp); sets the family YAML and every variant switch (F4). '
                             'Not combinable with --config or with the variant flags')
    parser.add_argument('--num-vehicles', type=int, default=None)
    parser.add_argument('--num-hexes', type=int, default=None, help='synthetic runs only (real runs use the manifest)')
    parser.add_argument('--episodes', type=int, default=None, help='total training episodes')
    parser.add_argument('--episode-duration-hours', type=float, default=None)
    parser.add_argument('--batch-size', type=int, default=None)
    parser.add_argument('--warmup-steps', type=int, default=None, help='decision epochs of uniform intentions')
    parser.add_argument('--gpus', type=str, default=None, help='GPU id(s); the first one is used')
    parser.add_argument('--distributed', action='store_true', help='refused (F55)')
    parser.add_argument('--mixed-precision', action='store_true')
    parser.add_argument('--checkpoint-dir', type=str, default=None, help='default: checkpoint.checkpoint_dir of the YAML')
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--save-interval', type=int, default=None, help='default: checkpoint.save_interval of the YAML')
    parser.add_argument('--log-interval', type=int, default=None, help='default: logging.log_interval of the YAML')
    parser.add_argument('--eval-interval', type=int, default=50, help='validation episode every N episodes')
    parser.add_argument('--eval-split', type=str, default='val', help='split used to select the best checkpoint')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--debug', action='store_true', help='autograd anomaly detection')
    parser.add_argument('--tensorboard', action='store_true')
    parser.add_argument('--tensorboard-dir', type=str, default='runs')
    parser.add_argument('--synthetic', action='store_true', help='synthetic trips on a small H3 disk (smoke tests only)')
    parser.add_argument('--manifest', type=str, default=None, help='override data.manifest_path')
    parser.add_argument('--processed-root', type=str, default=None, help='override data.processed_root')
    parser.add_argument('--sample-ratio', type=float, default=None,
                        help='override data.sample_ratio (trip sample of every split; stored in the checkpoint)')
    # variant switches (refused together with --preset)
    parser.add_argument('--milp', dest='milp', action='store_true', default=None,
                        help='MILP feasible-action projection (Eq. 12-16, requires Gurobi); sets milp.enabled')
    parser.add_argument('--no-milp', dest='milp', action='store_false', help='raw executor (milp.enabled = false)')
    parser.add_argument('--semi-mdp', dest='semi_mdp', action='store_true', default=None, help='gamma^Delta bootstrap')
    parser.add_argument('--no-semi-mdp', dest='semi_mdp', action='store_false', help='fixed-gamma bootstrap')
    parser.add_argument('--wdro', dest='wdro', action='store_true', default=None, help='Wasserstein-robust target')
    parser.add_argument('--no-wdro', dest='wdro', action='store_false', help='non-robust target')
    parser.add_argument('--wdro-rho', type=float, default=None, help='override wdro.rho')
    parser.add_argument('--wdro-metric', type=str, default=None, choices=['graph', 'diag', 'identity'],
                        help='override wdro.metric (graph = PD-RSAC, identity = w/o Graph, diag = N11 ablation)')
    parser.add_argument('--critic-population', dest='critic_population', action='store_true', default=None,
                        help='local population critic Q(o, a, m, g)')
    parser.add_argument('--no-critic-population', dest='critic_population', action='store_false',
                        help='independent critic without m_i (IAC ablation)')
    return parser.parse_args(argv)


def get_device(args) -> torch.device:
    if args.gpus:
        return torch.device(f'cuda:{int(args.gpus.split(",")[0])}')
    return torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def create_config(args) -> Config:
    preset = getattr(args, 'preset', None)
    if preset:
        from gpu_core.config.presets import get_preset, resolve_preset
        p = get_preset(preset)
        if args.config:
            raise SystemExit(f"--preset {preset} already selects config_{p.family}.yaml; drop --config")
        given = [f for f in VARIANT_FLAGS if getattr(args, f, None) is not None]
        if given:
            raise SystemExit(f"--preset {preset} fixes the variant switches; drop --{', --'.join(g.replace('_', '-') for g in given)}")
        if not p.trainable:
            raise SystemExit(f"preset '{preset}' is not trained ({p.role}); run it with scripts/run_protocol.py --preset {preset}")
        if not p.available:
            raise SystemExit(f"preset '{preset}' is not available")
        config = resolve_preset(preset)
    elif args.config:
        config = ConfigLoader.from_yaml(args.config)
    else:
        config = Config()
    # CLI overrides the YAML only when given explicitly
    for flag, key in (('num_vehicles', 'environment.num_vehicles'), ('num_hexes', 'environment.num_hexes'),
                      ('episodes', 'training.total_episodes'), ('episode_duration_hours', 'episode.duration_hours'),
                      ('batch_size', 'training.batch_size'), ('warmup_steps', 'training.warmup_steps'),
                      ('checkpoint_dir', 'checkpoint.checkpoint_dir'), ('save_interval', 'checkpoint.save_interval'),
                      ('log_interval', 'logging.log_interval'), ('semi_mdp', 'agent.semi_mdp'), ('wdro', 'wdro.enabled'),
                      ('wdro_metric', 'wdro.metric'),
                      ('critic_population', 'agent.critic_population'), ('manifest', 'data.manifest_path'),
                      ('processed_root', 'data.processed_root'), ('sample_ratio', 'data.sample_ratio')):
        value = getattr(args, flag, None)
        if value is not None:
            section, name = key.split('.')
            setattr(getattr(config, section), name, value)
    if getattr(args, 'wdro_rho', None) is not None:
        # one budget for the target and the dual (round 7), metric steps of rho / 2 (round 8)
        config.wdro.rho = config.wdro.rho_target = float(args.wdro_rho)
        config.wdro.step_size_d = float(args.wdro_rho) / 2.0
    if getattr(args, 'milp', None) is not None:
        config.milp.enabled = bool(args.milp)
    if args.mixed_precision:
        config.training.mixed_precision = True
    return config


def train_single_gpu(args):
    torch.manual_seed(args.seed)
    device = get_device(args)
    config = create_config(args)
    # Data: spatial manifest + training split, or explicit synthetic trips (no silent fallback)
    env, train_data, manifest = build_env(config, SIM_DEVICE, split='train', synthetic=args.synthetic, verbose=False)
    # Provenance: resolved config + manifest, written once and embedded in every checkpoint.
    log_dir = os.path.join(config.logging.log_dir, f"run_{int(time.time())}")
    run_record = build_run_record(
        config, args,
        data_path=None if args.synthetic else str(resolve_path(config.data.manifest_path)),
        hex_ids=getattr(env.hex_grid, '_hex_ids', None),
        preset=getattr(args, 'preset', None),
    )
    write_run_record(log_dir, run_record)
    from gpu_core.learning.run import train_vehicle_sac
    return train_vehicle_sac(args, config, device, env, train_data, manifest, run_record, log_dir)


def main(argv=None):
    args = parse_args(argv)
    if args.debug:
        torch.autograd.set_detect_anomaly(True)
    if args.distributed:
        # The previous DDP path never wrapped the trainers, so each rank trained an independent
        # model (audit F55). Refuse until real DDP exists.
        raise SystemExit("--distributed is not supported: gradients were never synchronised across ranks (F55). "
                         "Use single-GPU training.")
    return train_single_gpu(args)


if __name__ == '__main__':
    main()
