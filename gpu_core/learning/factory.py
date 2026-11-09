"""Vehicle-level learner selected by ``training.algo`` (one stack for every learned method, G)."""

from .agent import VehicleSAC
from .maddpg import VehicleMADDPG
from .mappo import VehicleMAPPO

ALGOS = {"sac": VehicleSAC, "maddpg": VehicleMADDPG, "mappo": VehicleMAPPO, "ppo": VehicleMAPPO}
NAMES = {"sac": "SAC / PD-RSAC", "maddpg": "MADDPG", "mappo": "MAPPO", "ppo": "MAPPO"}


def make_agent(env, config, device, projected: bool):
    algo = str(config.training.algo).lower()
    if algo not in ALGOS:
        raise ValueError(f"unknown training.algo '{algo}'; known: {', '.join(ALGOS)}")
    return ALGOS[algo](env, config, device, projected=projected)
