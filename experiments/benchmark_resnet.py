"""Experiment 2: ResNet-18 (CIFAR variant) training on CIFAR-10.

Identical across backends: model + initial weights, data subset, per-epoch shuffling and
augmentation (done on the CPU with a seeded generator before the timer starts), optimizer,
learning rate, batch size, epochs, seed.

Timing per training step = sync -> [host->device copy, forward, loss, backward, optimizer step,
loss.item()] -> sync. Epoch time is the sum of its step times; augmentation and validation are
timed separately and stored in `extra` (aug_s, val_s). A few warmup steps run first on a copy of
the weights that is restored afterwards, so warmup does not change the training trajectory.
"""
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from src.config import resolve_path  # noqa: E402
from src.devices import assert_on_device  # noqa: E402
from src.metrics import MemoryTracker, summarize  # noqa: E402
from src.models import count_params, resnet18_cifar  # noqa: E402
from src.runner import RunContext, check_finite, experiment_main  # noqa: E402
from src.timing import DeviceTimer  # noqa: E402
from src.workloads import augment_epoch, load_cifar10  # noqa: E402

EXPERIMENT = "cnn"


def make_optimizer(model, cfg):
    return torch.optim.SGD(model.parameters(), lr=cfg["lr"], momentum=cfg["momentum"], weight_decay=cfg["weight_decay"])


def train_step(model, opt, xb, yb, device):
    xb, yb = xb.to(device), yb.to(device)
    opt.zero_grad(set_to_none=True)
    loss = F.cross_entropy(model(xb), yb)
    loss.backward()
    opt.step()
    return float(loss.item())


@torch.no_grad()
def evaluate(model, vx, vy, device, batch_size):
    model.eval()
    correct = 0
    for i in range(0, vx.shape[0], batch_size):
        out = model(vx[i:i + batch_size].to(device))
        correct += (out.argmax(1).cpu() == vy[i:i + batch_size]).sum().item()
    model.train()
    return correct / vx.shape[0]


def run_batch_size(ctx: RunContext, data, bs: int) -> None:
    cfg, dev = ctx.exp_cfg, ctx.device
    tx, ty, vx, vy = data
    aug = cfg.get("augmentation", {})
    torch.manual_seed(ctx.seed)  # identical initial weights for every backend / batch size
    model = resnet18_cifar().to(dev)
    n_params = count_params(model)
    assert_on_device(dev, model)
    model.train()
    common = dict(model="resnet18_cifar", dataset="cifar10", batch_size=bs, precision="fp32", n_params=n_params,
                  train_samples=int(tx.shape[0]), lr=cfg["lr"], optimizer=cfg["optimizer"])

    # -- warmup on a throw-away copy of the initial state
    init_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    opt = make_optimizer(model, cfg)
    for i in range(cfg.get("warmup_steps", 5)):
        with DeviceTimer(ctx.backend) as t:
            train_step(model, opt, tx[i * bs:(i + 1) * bs], ty[i * bs:(i + 1) * bs], dev)
        ctx.record(**common, phase="warmup", iteration=i, latency_ms=t.elapsed_ms, throughput=bs / (t.elapsed_ms / 1000),
                   throughput_unit="images/s", granularity="step")
    model.load_state_dict(init_state)
    opt = make_optimizer(model, cfg)
    del init_state
    ctx.cleanup()

    steps_per_epoch = tx.shape[0] // bs  # drop_last: identical step count on every backend
    with MemoryTracker(ctx.backend) as mem:
        for epoch in range(cfg["epochs"]):
            t_aug = time.perf_counter()
            ex, ey = augment_epoch(tx, ty, ctx.seed * 1000 + epoch, aug.get("random_crop_padding", 0), aug.get("horizontal_flip", False))
            aug_s = time.perf_counter() - t_aug
            step_ms, losses = [], []
            for s in range(steps_per_epoch):
                with DeviceTimer(ctx.backend) as t:
                    loss = train_step(model, opt, ex[s * bs:(s + 1) * bs], ey[s * bs:(s + 1) * bs], dev)
                check_finite(loss, f"(epoch {epoch}, step {s})")
                step_ms.append(t.elapsed_ms)
                losses.append(loss)
            epoch_s = sum(step_ms) / 1000.0
            images = steps_per_epoch * bs
            t_val = time.perf_counter()
            acc = evaluate(model, vx, vy, dev, cfg["val_batch_size"])
            val_s = time.perf_counter() - t_val
            st = summarize(step_ms)
            ctx.record(**common, phase="measure", iteration=epoch, latency_ms=epoch_s * 1000.0, throughput=images / epoch_s,
                       throughput_unit="images/s", loss=statistics.fmean(losses), accuracy=acc,
                       memory_mb=None, granularity="epoch", epoch_time_s=epoch_s, images=images, aug_s=aug_s, val_s=val_s,
                       step_ms_median=st["median"], step_ms_mean=st["mean"], step_ms_std=st["std"], step_ms_min=st["min"],
                       step_ms_max=st["max"], steps=steps_per_epoch)
            print(f"    bs={bs:4d} epoch {epoch}: {epoch_s:6.1f}s  {images / epoch_s:8.1f} img/s  loss {statistics.fmean(losses):.3f}  val_acc {acc:.3f}", flush=True)
    assert_on_device(dev, model)
    # memory is one number for the whole configuration; attach it to a dedicated config row
    ctx.record(**common, phase="config", iteration=None, memory_mb=mem.result["memory_mb"], memory_kind=mem.result["memory_kind"],
               granularity="memory_summary", **{k: v for k, v in mem.result.items() if k not in ("memory_mb", "memory_kind")})


def run(ctx: RunContext) -> None:
    cfg = ctx.exp_cfg
    data = load_cifar10(resolve_path(cfg["data_dir"]), cfg.get("train_samples"), cfg.get("val_samples"), ctx.seed)
    print(f"    data: train {tuple(data[0].shape)} val {tuple(data[2].shape)}", flush=True)
    for bs in cfg["batch_sizes"]:
        try:
            run_batch_size(ctx, data, bs)
        except Exception as exc:
            ctx.record_failure(exc, model="resnet18_cifar", dataset="cifar10", batch_size=bs, precision="fp32")
        ctx.cleanup()


def main(argv=None) -> int:
    return experiment_main(EXPERIMENT, run, description="ResNet-18 / CIFAR-10 training benchmark", argv=argv)


if __name__ == "__main__":
    sys.exit(main())
