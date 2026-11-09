# PD-RSAC

Primal-Dual Robust Soft Actor-Critic for EV fleet control: semi-MDP backups, Wasserstein DRO and a MILP feasible-action projection.

## Setup

```bash
pip install -r requirements.txt
export GRB_LICENSE_FILE=<GUROBI_LICENSE>
```

## Data

```bash
python -m gpu_core.data.preprocess --input <RAW_TRIPS> --output-root <PROCESSED_DIR>
python -m gpu_core.spatial.manifest --processed-root <PROCESSED_DIR> --datasets <DATASET> \
    --train-start <START_DATE> --train-end <END_DATE> --out <MANIFEST>
```

## Train

```bash
python gpu_core/scripts/train.py --preset <PRESET>
```

## Evaluate

```bash
python gpu_core/scripts/run_protocol.py --checkpoint <CHECKPOINT> --label <LABEL> --out-dir <OUT_DIR>
python gpu_core/scripts/powergrid_evaluate.py --checkpoint <CHECKPOINT> --start-day <START_DAY> --days <N_DAYS> --out-csv <OUT_CSV>
```
