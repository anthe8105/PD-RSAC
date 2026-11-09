"""Loading vehicle-level checkpoints for evaluation (shared by evaluate.py and powergrid_evaluate.py)."""

from typing import Dict

import torch

from .factory import make_agent


def is_vehicle_checkpoint(checkpoint: Dict) -> bool:
    return checkpoint.get("algo") == "vehicle_sac"


def load_vehicle_agent(checkpoint: Dict, config, env, device, projected: bool):
    """Agent with the checkpoint's weights; refuses a checkpoint trained on another manifest (E13)."""
    if not is_vehicle_checkpoint(checkpoint):
        raise ValueError("not a vehicle-level SAC / PD-RSAC checkpoint")
    trained_on = checkpoint.get("manifest_infra_hash")
    if trained_on and env.manifest is not None and trained_on != env.manifest.infra_hash:
        raise ValueError("checkpoint was trained on a different spatial manifest (cells, stations or ports)")
    agent = make_agent(env, config, device, projected=projected)
    if checkpoint.get("agent_algo", "sac") != agent.algo:
        raise ValueError(f"checkpoint holds a {checkpoint.get('agent_algo', 'sac')} agent but the config says {agent.algo}")
    agent.load_state_dict(checkpoint["agent"], with_optim=False)
    for m in agent.eval_modules():
        if m is not None:
            m.eval()
    return agent


def result_row(env, rep) -> Dict:
    m = env.get_metrics()
    loaded = max(1, m["trips_loaded"])
    return {
        **m, "service_rate": m["trips_served"] / loaded, "steps": rep.steps,
        "intention_share": rep.intention_share, "executed_share": rep.executed_share,
        "serve_when_available": rep.serve_when_available, "low_energy_stay_share": rep.low_energy_idle_share,
        "mean_energy_end": rep.mean_energy_end, "stuck_vehicles_end": rep.stuck_vehicles_end,
        "pi_hat_match": rep.pi_hat_match, "projection": rep.projection, "seconds": rep.seconds,
    }
