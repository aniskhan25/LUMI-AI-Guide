# 7. TensorBoard visualization

## Goal

Track and visualize training metrics and sample data from distributed PyTorch runs with TensorBoard.

## Assumptions

- You can run the DDP example from [5. Multi-GPU and Multi-Node Training](../5-multi-gpu-and-node/README.md).
- TensorBoard logging dependencies are available in your runtime environment.
- You can access the LUMI web interface Apps menu.

## What changes from baseline

- Baseline you already have: distributed training runs that print metrics to stdout.
- This lesson adds: rank-aware TensorBoard logging and visualization through LUMI web apps.
- Expected output/artifact: TensorBoard event logs in `runs/` and interactive metric dashboards.

[TensorBoard](https://www.tensorflow.org/tensorboard) is a tool for providing the measurements and visualizations needed during the machine learning workflow. It enables tracking experiment metrics like loss and accuracy, visualizing the model graph, projecting embeddings to a lower dimensional space, and much more.

## Collecting logs

TensorBoard can be used to [visualize models, data, and training with PyTorch](https://pytorch.org/tutorials/intermediate/tensorboard_tutorial.html). We can also configure TensorBoard to collect metrics from distributed execution. We will use the Vision Transformer classification example, which utilizes Distributed Data Parallel for distributed execution, and adapt it to collect some metrics to TensorBoard.

Since during distributed runs we use multiple processes, we set one of the processes to be responsible for collecting the logs. This can be done by using the `rank` environment variable assigned to every process created by Slurm. We can use this variable to assign the task of logging to the first process (rank 0). With just a few additions we can display some of the training images and loss, but additional metrics can be added using similar methods. 

We can start the logger, called SummaryWriter, on the first process to generate logs into the directory `runs` as follows:
```python
from torch.utils.tensorboard import SummaryWriter
    
if rank == 0:
    writer = SummaryWriter('runs')
```

After having configured the logger, we can visualize some of the images we use as a grid with Matplotlib like so:

```python
import matplotlib.pyplot as plt
import numpy as np

def matplotlib_imshow(img, one_channel=False):
    if one_channel:
        img = img.mean(dim=0)
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    if one_channel:
        plt.imshow(npimg, cmap="Greys")
    else:
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
...
if rank == 0:
    dataiter = iter(train_loader)
    images, labels = next(dataiter)
    # create grid of images
    img_grid = torchvision.utils.make_grid(images)
    # show images
    matplotlib_imshow(img_grid, one_channel=True)
    # write to tensorboard
    writer.add_image('images', img_grid)
```

The images will then be visualized in TensorBoard similar to the following:

![Image title](../assets/images/view_images.png)

Graphs of the training loss and validation accuracy can also be gathered with the addition of 2 lines of code:
```python
if rank == 0:
    print(f'Epoch {epoch+1}, Loss: {running_loss/len(train_loader)}')
    writer.add_scalar('training loss', running_loss / len(train_loader), epoch)
...
if rank == 0:
    print(f'Accuracy: {100 * correct / total}%')
    writer.add_scalar('validation accuracy', 100*correct/total, epoch)
```
In TensorBoard, the collected data will be visualized similar to the following:

![Image title](../assets/images/loss.png)

For a full example that integrates TensorBoard to the DDP script, have a look at [visiontransformer_ddp_tensorboard.py](visiontransformer_ddp_tensorboard.py). For the batch script there are no changes required except for replacing [visiontransformer_ddp.py](../5-multi-gpu-and-node/visiontransformer_ddp.py) with [visiontransformer_ddp_tensorboard.py](visiontransformer_ddp_tensorboard.py).

## Visualizing the logs

TensorBoard can be used on LUMI via the [web interface](https://docs.lumi-supercomputer.eu/runjobs/webui/) by selecting "TensorBoard" from the "Apps" menu. Once you have the logs generated during execution, you can launch the TensorBoard server on a compute node, display the GUI and analyze the run.

![Image title](../assets/images/web_interface_tensorboard.png)

To launch it, select the log directory where you have data to visualize, which in this case would be the path to the `runs` directory, and the resources for the Slurm job.

Note that TensorBoard is very memory intensive but has low CPU usage. Thus, in the case of performance problems, adding more memory during allocation can help.

## Verify

Confirm all of the following:

- Rank-0 process creates TensorBoard logs in the configured `runs/` directory.
- Training loss and validation accuracy are visible in TensorBoard.
- You can launch TensorBoard from the LUMI web interface and load the run data.

## Troubleshooting

- Empty dashboard: verify the selected log directory and confirm rank-0 is writing events.
- Missing logs in distributed mode: ensure only one process initializes `SummaryWriter` for shared outputs.
- Slow or unstable UI: allocate more memory for the TensorBoard job.

## Navigation

- Previous: [6. Monitoring and Profiling jobs](../6-monitoring-and-profiling/README.md)
- Next: [8. MLflow visualization](../8-mlflow-visualization/README.md)
