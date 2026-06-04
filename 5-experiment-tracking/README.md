# 5. Experiment Tracking

Track metrics across runs so you can compare experiments and catch regressions. All three tools below follow the same pattern: **only rank 0 logs**, everything else trains normally.

## TensorBoard

Built into PyTorch, no account needed. Logs are written locally and visualised via the LUMI web interface.

```python
from torch.utils.tensorboard import SummaryWriter

if rank == 0:
    writer = SummaryWriter("runs")

# in training loop (rank 0 only):
writer.add_scalar("training loss", loss, epoch)
writer.add_scalar("validation accuracy", accuracy, epoch)
```

```bash
sbatch run_tensorboard.sh
```

Logs are written to `runs/`. To visualise, open the [LUMI web interface](https://www.lumi.csc.fi), go to **Apps → TensorBoard**, and point it at your `runs/` directory.

## MLflow

Open source, no account needed. Runs are stored locally (directory or SQLite) and visualised via the LUMI web interface.

```python
import mlflow

if rank == 0:
    mlflow.set_tracking_uri("sqlite:///" + os.environ["PWD"] + "/mlruns.db")
    mlflow.start_run(run_name=os.getenv("SLURM_JOB_ID"))

# in training loop (rank 0 only):
mlflow.log_metric("loss", loss, step=epoch)
mlflow.log_metric("accuracy", accuracy, step=epoch)
```

```bash
sbatch run_mlflow.sh
```

To visualise, go to **Apps → MLflow** in the LUMI web interface and point it at your tracking URI.

## Weights & Biases

Cloud-hosted dashboard with automatic system metrics (GPU utilisation, power, memory). Requires a free account and API key.

```bash
export WANDB_API_KEY=<your_api_key>
```

```python
import wandb

if rank == 0:
    wandb.init(project="my-project", config={"lr": 0.001, "epochs": 10})

# in training loop (rank 0 only):
wandb.log({"loss": loss, "acc": accuracy})
```

```bash
sbatch run_wandb.sh
```

Results appear at [wandb.ai](https://wandb.ai). W&B also automatically captures `rocm-smi` system metrics during the run.

## Choosing a tool

| | TensorBoard | MLflow | W&B |
|---|---|---|---|
| Account required | No | No | Yes (free) |
| Stored locally | Yes | Yes | No (cloud) |
| System metrics | No | No | Yes (automatic) |
| LUMI web UI | Yes | Yes | No |

## Next

You have completed the core guide. For advanced topics — fine-tuning, inference, RAG, scaling patterns — see the [extension track](../extension-track/README.md).
