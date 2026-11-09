"""Experiment presets (workstream F4): one name per row of the preset table in PD-RSAC_FIX_PLAN.md.

A preset fixes the algorithm family (the YAML) and sets every variant switch explicitly, so the
resolved configs of two presets differ only in ``VARIANT_KEYS`` (plus, across families, the
algorithm family itself) and in where the run writes (``LOCATION_KEYS``). Tests compare resolved
configs to enforce this. The preset name is stored in the run record.

    cfg = resolve_preset("full")           # PD-RSAC
    cfg = resolve_preset("no_graph")       # ablation: identity metric
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

from .base import Config
from .loader import ConfigError, ConfigLoader

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"

VARIANT_KEYS = ("milp.enabled", "wdro.enabled", "wdro.metric", "agent.semi_mdp", "agent.critic_population")
LOCATION_KEYS = ("checkpoint.checkpoint_dir", "logging.log_dir")


@dataclass(frozen=True)
class Preset:
    name: str
    family: Optional[str]                 # YAML family (config_<family>.yaml); None = base only
    variant: Dict[str, Any] = field(default_factory=dict)
    role: str = ""
    trainable: bool = True                # False for the heuristic
    available: bool = True                # False until the baseline is rebuilt (workstream G)


def _pd_rsac(**changes) -> Dict[str, Any]:
    v = {"milp.enabled": True, "wdro.enabled": True, "wdro.metric": "graph",
         "agent.semi_mdp": True, "agent.critic_population": True}
    v.update({k.replace("__", "."): x for k, x in changes.items()})
    return v


def _fixed_gamma(milp: bool) -> Dict[str, Any]:
    """Baseline learners: no WDRO, fixed gamma per decision (Q12)."""
    return {"milp.enabled": milp, "wdro.enabled": False, "wdro.metric": "graph",
            "agent.semi_mdp": False, "agent.critic_population": True}


def _semi_mdp_baseline(milp: bool) -> Dict[str, Any]:
    """MAPPO / MADDPG: no WDRO, gamma^Delta per transition like PD-RSAC."""
    return {**_fixed_gamma(milp), "agent.semi_mdp": True}


PRESETS: Dict[str, Preset] = {p.name: p for p in (
    Preset("full", "wdro", _pd_rsac(), "PD-RSAC"),
    Preset("no_milp", "wdro", _pd_rsac(milp__enabled=False), "ablation: raw executor instead of the MILP"),
    Preset("no_graph", "wdro", _pd_rsac(wdro__metric="identity"), "ablation: identity scenario metric"),
    Preset("diag_only", "wdro", _pd_rsac(wdro__metric="diag"), "ablation: diag(w) metric without graph (N11)"),
    Preset("iac", "wdro", _pd_rsac(agent__critic_population=False), "ablation: critic without m_i"),
    Preset("sac_milp", "sac", _pd_rsac(wdro__enabled=False), "ablation 'w/o WDRO': PD-RSAC without the robust target (semi-MDP kept, Q12b round 7)"),
    Preset("sac", "sac", _fixed_gamma(False), "baseline"),
    Preset("mappo", "mappo", _semi_mdp_baseline(False), "baseline (semi-MDP)"),
    Preset("mappo_milp", "mappo", _semi_mdp_baseline(True), "baseline (semi-MDP)"),
    Preset("maddpg", "maddpg", _semi_mdp_baseline(False), "baseline (semi-MDP)"),
    Preset("maddpg_milp", "maddpg", _semi_mdp_baseline(True), "baseline (semi-MDP)"),
    Preset("greedy", None, _fixed_gamma(False), "heuristic (Tian et al. 2025)", trainable=False),
)}


def get_preset(name: str) -> Preset:
    if name not in PRESETS:
        raise ConfigError(f"unknown preset '{name}'; known presets: {', '.join(PRESETS)}")
    return PRESETS[name]


def get_key(config: Config, dotted: str) -> Any:
    obj = config
    for part in dotted.split("."):
        obj = getattr(obj, part)
    return obj


def set_key(config: Config, dotted: str, value: Any) -> None:
    *head, last = dotted.split(".")
    obj = config
    for part in head:
        obj = getattr(obj, part)
    if not hasattr(obj, last):
        raise ConfigError(f"unknown config key '{dotted}'")
    setattr(obj, last, value)


def resolve_preset(name: str, scripts_dir: Optional[Path] = None) -> Config:
    """Family YAML (or base.yaml for the heuristic) + the preset's variant switches + per-preset
    output directories, so variants of one family never overwrite each other's checkpoints."""
    p = get_preset(name)
    root = Path(scripts_dir) if scripts_dir is not None else SCRIPTS
    if p.family is None:
        config = Config()
        ConfigLoader._update_dataclass(config, ConfigLoader.load_yaml(str(root / "base.yaml")))
    else:
        config = ConfigLoader.from_yaml(str(root / f"config_{p.family}.yaml"))
    for key, value in p.variant.items():
        set_key(config, key, value)
    config.checkpoint.checkpoint_dir = f"checkpoint/{name}"
    config.logging.log_dir = f"logs/{name}"
    return config


def flat_config(config: Config) -> Dict[str, Any]:
    """Resolved config as {dotted key: value} (for comparisons and run tables)."""
    out: Dict[str, Any] = {}

    def walk(d, prefix):
        for k, v in d.items():
            if isinstance(v, dict):
                walk(v, f"{prefix}{k}.")
            else:
                out[prefix + k] = v
    walk(ConfigLoader.to_dict(config), "")
    return out


def config_diff(a: Config, b: Config) -> Dict[str, Any]:
    fa, fb = flat_config(a), flat_config(b)
    return {k: (fa.get(k), fb.get(k)) for k in sorted(set(fa) | set(fb)) if fa.get(k) != fb.get(k)}
