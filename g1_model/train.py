"""The ONE training loop, the ONE loss, and the ONE evaluation path.

WHY THIS IS SHARED, AND WHY IT IS BUILT BEFORE ANY MODEL
---------------------------------------------------------
RQ2 and RQ3 are differences between BC, ACT and ACT-LSTM. A difference is only
attributable to the architecture if everything else is held fixed, and "the
training loop" is one of the things that has to be fixed. If BC gets its own
loop and ACT gets another, then the measured gap contains the loop difference -
a different shuffle, a different reduction, a different early stop - and no
amount of care afterwards separates the two. Chapter 4 would be reporting the
sum of an architecture effect and an implementation accident, with no way to say
which is which.

So the loop is written first, with no model in it, and the model is a PARAMETER.
`train()` knows about `nn.Module`, batches, an optimizer and a loss. It does not
know what BC is. Adding chunked BC, ACT or ACT-LSTM later must not require
touching anything here except passing a different module.

THE LOSS IS L1 FOR A SPECIFIC REASON
-------------------------------------
ACT's objective is `L1 + beta * KL`. If BC trained under MSE and ACT under L1,
the reconstruction terms would be on different scales and in different units,
and "ACT beat BC" would partly mean "L1 and MSE are different numbers". The
reconstruction term is therefore identical across every stage, and ACT's KL is
ADDED to it rather than replacing it.

TWO MASKS, AND THEY COMPOSE
----------------------------
  `spec.ACTION_MASK`   the 16 trainable dims of 22. The other 6 are constant by
                       measurement (2026-09-10 audit): wrist yaws and the waist
                       are hard zeros, and `a_gR` is bit-identical to `a_gL`.
                       Training on them teaches a model to reproduce a constant,
                       which is free, and inflates any average that includes it.
  the padding mask     real timesteps only. Chunks near the end of an episode
                       are padded to K (loader B3); a padded step is not data.

The reduction is over the INTERSECTION, divided by the number of contributing
elements. Not by `B*K*22`, which would make the loss depend on how much padding
a batch happened to contain, so the same model would score differently at K=1
and K=100 and the scaling comparison would be measuring the padding.
"""
from __future__ import annotations

import datetime as _dt
import json
import os
import platform
import random
import time
from dataclasses import asdict, dataclass, field
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from g1_data import spec
from g1_data.paths import ROOT, repo_relpath
from g1_data.recorder import _git_commit
from g1_model.loader import ChunkDataset, collate_chunks

RUNS = os.path.join(ROOT, "runs")


class TrainError(AssertionError):
    """A training setup that would produce a number nobody should cite."""


# ─── B2: the device ───────────────────────────────────────────────────────────
def select_device(prefer_cuda: bool = True) -> torch.device:
    if prefer_cuda and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def device_report(device: torch.device) -> dict:
    """What was selected, and what it can hold. Printed at every startup.

    The VRAM figure is here because it is the batch-size ceiling and because a
    silent fallback to CPU is otherwise invisible until a run that should take
    two minutes takes forty.
    """
    r = dict(device=str(device), torch=torch.__version__,
             cuda_available=bool(torch.cuda.is_available()),
             cuda_version=torch.version.cuda, python=platform.python_version())
    if device.type == "cuda":
        p = torch.cuda.get_device_properties(device)
        r.update(name=p.name, total_vram_gb=round(p.total_memory / 1024 ** 3, 2),
                 capability="%d.%d" % (p.major, p.minor),
                 multiprocessors=p.multi_processor_count)
    else:
        r.update(name=platform.processor() or "cpu", total_vram_gb=None)
    return r


# ─── B3: determinism ──────────────────────────────────────────────────────────
def set_determinism(seed: int, strict: bool = True) -> dict:
    """Seed everything, and report HONESTLY what could not be made deterministic.

    `torch.use_deterministic_algorithms(True)` makes any op without a
    deterministic implementation RAISE rather than quietly vary. That is the
    behaviour worth having: a nondeterministic op that nobody knew about is
    exactly how "the same run gave a different number" happens, and a run that
    refuses to start is cheaper than a result that cannot be reproduced.

    cuBLAS needs `CUBLAS_WORKSPACE_CONFIG` set BEFORE its handle is created. It
    is set here, before any CUDA work, and the report says whether that
    succeeded - if the handle already exists, the setting is silently ignored by
    the library, so it is verified rather than assumed.

    The return value is written into the run metadata. Where determinism was not
    achievable, that fact travels with the run instead of being forgotten.
    """
    notes: List[str] = []
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False       # benchmark picks by timing: varies
    deterministic_algorithms = False
    if strict:
        try:
            torch.use_deterministic_algorithms(True)
            deterministic_algorithms = True
        except Exception as e:                              # noqa: BLE001
            notes.append("use_deterministic_algorithms refused: %s" % e)
    else:
        notes.append("strict determinism NOT requested by this run")
    # TF32 changes the arithmetic, not just the order, so a run that enables it
    # is not comparable with one that does not. Off, and recorded.
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    return dict(seed=int(seed),
                deterministic_algorithms=deterministic_algorithms,
                cudnn_deterministic=True, cudnn_benchmark=False, tf32=False,
                cublas_workspace=os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
                notes=notes)


def seeded_build(train_cfg: "TrainConfig", factory: Callable[..., nn.Module],
                 /, **kw) -> nn.Module:
    """Seed, THEN construct the model. Use this instead of building it yourself.

    MEASURED, 2026-09-21: two "identical" 3-epoch runs differed by 6.0e-3 in
    final loss. The cause was not a nondeterministic kernel - it was that
    `train()` seeds at ENTRY, by which time the caller has already constructed
    the model, so WEIGHT INITIALIZATION consumed whatever RNG state the process
    happened to be in. The second run started from the state the first run's
    training had left behind, so the two models began from different weights and
    every later number differed.

    That is the whole class of determinism bug worth worrying about here: not an
    op that varies, but a seed that lands after the thing it was meant to
    control. `train()` still seeds - the shuffle order needs it - and seeding
    twice with the same value is harmless.
    """
    set_determinism(train_cfg.seed, strict=train_cfg.strict_determinism)
    return factory(**kw)


def seed_worker(worker_id: int) -> None:
    """Per-worker seeding, so `num_workers > 0` does not reintroduce variance."""
    s = torch.initial_seed() % 2 ** 32
    np.random.seed(s)
    random.seed(s)


# ─── B4: the loss ─────────────────────────────────────────────────────────────
def masked_l1(pred: torch.Tensor, target: torch.Tensor,
              pad_mask: torch.Tensor,
              dim_mask: Optional[torch.Tensor] = None,
              reduce: bool = True):
    """Mean |pred - target| over (real timesteps) x (trainable dims).

    pred, target : (B, K, A)
    pad_mask     : (B, K)   True on real timesteps
    dim_mask     : (A,)     True on trainable dims; `spec.ACTION_MASK` by default

    Returns the scalar loss and the number of contributing elements. The count
    is returned rather than hidden because an epoch mean has to be weighted by
    it: averaging per-batch means would give a short final batch the same weight
    as a full one, which is a real error of a few percent and a completely
    invisible one.

    A fully padded sample contributes nothing to either the numerator or the
    denominator, so it is exactly zero rather than a small number - and a batch
    that is entirely padding yields 0.0 instead of a NaN from 0/0.
    """
    if dim_mask is None:
        dim_mask = torch.from_numpy(
            np.asarray(spec.ACTION_MASK, dtype=bool).copy()).to(pred.device)
    if pred.shape != target.shape:
        raise TrainError("pred %s and target %s must have the same shape"
                         % (tuple(pred.shape), tuple(target.shape)))
    if pred.shape[-1] != spec.ACTION_DIM:
        raise TrainError("actions must have last dimension %d, got %s"
                         % (spec.ACTION_DIM, tuple(pred.shape)))
    m = pad_mask.bool().unsqueeze(-1) & dim_mask.bool().view(1, 1, -1)
    err = (pred - target).abs() * m
    n = m.sum()
    if not reduce:
        return err, m
    total = err.sum()
    return total / n.clamp(min=1), n


# ─── B6: the baselines ────────────────────────────────────────────────────────
def baselines(ds: ChunkDataset, device: Optional[torch.device] = None,
              batch_size: int = 512) -> dict:
    """What "near zero" is measured against, in the loss's own units.

    ZERO      predict all zeros. In normalized space zero IS the training mean,
              so this is the loss of the constant predictor and the number any
              model must beat to have learned that actions vary at all.
    COPY      predict the previous tick's action, held across the whole chunk.
              The persistence baseline. At 25 Hz consecutive actions are very
              similar, so this is a STRONG baseline and the meaningful one: a
              model that beats ZERO but not COPY has learned the average pose,
              not the task. At tick 0 there is no previous action, so zeros are
              predicted there - the honest choice, since using the target itself
              would let the baseline peek at the answer.

    Computed through the SAME loss as training, over the same samples, so the
    numbers are directly comparable rather than approximately comparable.
    """
    device = device or torch.device("cpu")
    dim_mask = torch.from_numpy(
        np.asarray(spec.ACTION_MASK, dtype=bool).copy()).to(device)
    sums = dict(zero=0.0, copy=0.0)
    n_total = 0
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                        collate_fn=collate_chunks, num_workers=0)
    offset = 0
    for batch in loader:
        target = batch["action"].to(device)
        pad = batch["action_mask"].to(device)
        B, K, A = target.shape

        # the previous action for each sample, read from the episode arrays
        prev = np.zeros((B, A), dtype=np.float32)
        for b in range(B):
            ei, t = (int(v) for v in ds.index[offset + b])
            if t > 0:
                prev[b] = ds.actions[ei][t - 1]
        offset += B
        copy_pred = torch.from_numpy(prev).to(device).unsqueeze(1).expand(B, K, A)

        for name, pred in (("zero", torch.zeros_like(target)),
                           ("copy", copy_pred)):
            err, m = masked_l1(pred, target, pad, dim_mask, reduce=False)
            sums[name] += float(err.sum())
        n_total += int((pad.bool().unsqueeze(-1)
                        & dim_mask.view(1, 1, -1)).sum())
    return dict(zero=sums["zero"] / max(n_total, 1),
                copy=sums["copy"] / max(n_total, 1),
                elements=n_total, samples=len(ds))


# ─── B1/B5: the loop ──────────────────────────────────────────────────────────
@dataclass
class TrainConfig:
    """Everything a run needs that is not the model or the data.

    Every field is written verbatim into the run directory, so a run can be
    reproduced from its own metadata rather than from someone's memory of which
    flags they passed.
    """

    epochs: int
    batch_size: int
    lr: float
    weight_decay: float = 0.0
    optimizer: str = "adamw"
    grad_clip: Optional[float] = 1.0
    seed: int = 0
    num_workers: int = 0
    prefer_cuda: bool = True
    strict_determinism: bool = True
    log_every: int = 1
    checkpoint_every: int = 0        # 0 = only the last and the best
    run_name: str = "run"
    #: Free-form. The gate runner puts its data-provenance statement here, and
    #: it ends up in metadata.json (C5).
    notes: Dict[str, object] = field(default_factory=dict)


def make_optimizer(model: nn.Module, cfg: TrainConfig) -> torch.optim.Optimizer:
    if cfg.optimizer.lower() == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=cfg.lr,
                                 weight_decay=cfg.weight_decay)
    if cfg.optimizer.lower() == "adam":
        return torch.optim.Adam(model.parameters(), lr=cfg.lr,
                                weight_decay=cfg.weight_decay)
    if cfg.optimizer.lower() == "sgd":
        return torch.optim.SGD(model.parameters(), lr=cfg.lr, momentum=0.9,
                               weight_decay=cfg.weight_decay)
    raise TrainError("unknown optimizer %r" % cfg.optimizer)


def make_run_dir(cfg: TrainConfig, root: str = RUNS) -> str:
    stamp = _dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    path = os.path.join(root, "%s_%s" % (stamp, cfg.run_name))
    os.makedirs(path, exist_ok=True)
    return path


def dataset_provenance(ds: ChunkDataset) -> dict:
    """WHICH DATA this ran on (C5). Written into every run's metadata.

    Every gate in Phase 4 runs on SCRIPTED episodes, which are smoother and more
    consistent than piloted data will be, and which cannot exercise `tracking_ok`
    at all because there was no camera. A gate result from here is provisional on
    the data, not only on the hyperparameters. That statement lives in the run
    directory rather than in a report, because reports get lost and run
    directories are what someone re-reads when they want to cite a number.
    """
    sources = sorted({str(s) for s in ds.sources}) or ["unknown"]
    real = bool(ds.real_demonstration) and all(ds.real_demonstration)
    return dict(
        episode_sources=sources,
        episodes=len(ds.lengths), seeds=list(ds.seeds), ticks=int(sum(ds.lengths)),
        samples=len(ds),
        tracking_measured_episodes=int(sum(ds.tracking_is_measured)),
        real_demonstrations=bool(real),
        caveat=(
            "SCRIPTED demonstrator data. It is smoother and more consistent "
            "than piloted teleoperation will be, and tracking_ok cannot be "
            "exercised by episodes recorded without a camera. NO GATE RESULT "
            "FROM THIS RUN MAY BE CITED without a re-run on piloted data."
            if not real else
            "Real teleoperated demonstrations."))


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device,
             dim_mask: torch.Tensor) -> float:
    """The ONE evaluation path. Weighted by contributing elements, not batches."""
    model.eval()
    total, n = 0.0, 0
    for batch in loader:
        obs = batch["obs"].to(device, non_blocking=True)
        target = batch["action"].to(device, non_blocking=True)
        pad = batch["action_mask"].to(device, non_blocking=True)
        pred = model(obs)
        err, m = masked_l1(pred, target, pad, dim_mask, reduce=False)
        total += float(err.sum())
        n += int(m.sum())
    return total / max(n, 1)


def train(model: nn.Module, train_ds: ChunkDataset, cfg: TrainConfig,
          val_ds: Optional[ChunkDataset] = None,
          run_dir: Optional[str] = None,
          on_epoch: Optional[Callable[[dict], None]] = None) -> dict:
    """Train any model on any ChunkDataset. Knows nothing about BC or ACT.

    The model's contract is the whole interface: it takes `obs` of shape
    (B, W_o, 47) and returns actions of shape (B, K, 22). BC is K=1 and W_o=1;
    ACT is K=100. Nothing here changes between them.

    BUILD THE MODEL WITH `seeded_build`, not by calling its constructor. Seeding
    happens here, at entry, which is too late to control the weight
    initialization the caller already did - see `seeded_build` for the 6.0e-3
    divergence that produced.
    """
    det = set_determinism(cfg.seed, strict=cfg.strict_determinism)
    device = select_device(cfg.prefer_cuda)
    dev = device_report(device)
    run_dir = run_dir or make_run_dir(cfg)
    os.makedirs(run_dir, exist_ok=True)

    model = model.to(device)
    opt = make_optimizer(model, cfg)
    dim_mask = torch.from_numpy(
        np.asarray(spec.ACTION_MASK, dtype=bool).copy()).to(device)

    g = torch.Generator()
    g.manual_seed(cfg.seed)
    tl = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,
                    collate_fn=collate_chunks, num_workers=cfg.num_workers,
                    generator=g, worker_init_fn=seed_worker, drop_last=False)
    vl = (DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False,
                     collate_fn=collate_chunks, num_workers=cfg.num_workers)
          if val_ds is not None else None)

    base = baselines(train_ds, device)
    prov = dataset_provenance(train_ds)

    meta = dict(
        config=asdict(cfg), device=dev, determinism=det,
        git_commit=_git_commit(), spec_version=spec.SPEC_VERSION,
        started_utc=_dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        model=dict(cls=type(model).__name__,
                   parameters=int(sum(p.numel() for p in model.parameters())),
                   trainable=int(sum(p.numel() for p in model.parameters()
                                     if p.requires_grad)),
                   repr=str(model)),
        loader=dict(chunk_size=train_ds.cfg.chunk_size,
                    obs_window=train_ds.cfg.obs_window,
                    tracking=asdict(train_ds.cfg.tracking)),
        data=prov, baselines=base,
        val=dict(episodes=len(val_ds.lengths), seeds=list(val_ds.seeds),
                 samples=len(val_ds)) if val_ds is not None else None,
    )
    with open(os.path.join(run_dir, "metadata.json"), "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=1, default=str)

    print("run           %s" % repo_relpath(run_dir))
    print("device        %s%s" % (dev["device"],
                                  ("  %s, %.2f GB VRAM" % (dev["name"], dev["total_vram_gb"]))
                                  if dev["total_vram_gb"] else "  (%s)" % dev["name"]))
    print("determinism   seed %d, deterministic_algorithms=%s%s"
          % (det["seed"], det["deterministic_algorithms"],
             ("  NOTES: " + "; ".join(det["notes"])) if det["notes"] else ""))
    print("model         %s, %d parameters"
          % (meta["model"]["cls"], meta["model"]["parameters"]))
    print("data          %d episodes, %d samples, K=%d W_o=%d"
          % (prov["episodes"], prov["samples"], train_ds.cfg.chunk_size,
             train_ds.cfg.obs_window))
    print("              SOURCE: %s" % ", ".join(prov["episode_sources"]))
    print("              %s" % prov["caveat"])
    print("baselines     zero %.6f   copy %.6f   (normalized L1, trainable dims)"
          % (base["zero"], base["copy"]))

    metrics_path = os.path.join(run_dir, "metrics.jsonl")
    history: List[dict] = []
    best = float("inf")
    t0 = time.perf_counter()

    for epoch in range(1, int(cfg.epochs) + 1):
        model.train()
        e0 = time.perf_counter()
        total, n = 0.0, 0
        for batch in tl:
            obs = batch["obs"].to(device, non_blocking=True)
            target = batch["action"].to(device, non_blocking=True)
            pad = batch["action_mask"].to(device, non_blocking=True)
            pred = model(obs)
            loss, count = masked_l1(pred, target, pad, dim_mask)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            if cfg.grad_clip:
                nn.utils.clip_grad_norm_(model.parameters(), float(cfg.grad_clip))
            opt.step()
            total += float(loss.detach()) * int(count)
            n += int(count)
        train_loss = total / max(n, 1)
        val_loss = evaluate(model, vl, device, dim_mask) if vl is not None else None

        row = dict(epoch=epoch, train_loss=train_loss, val_loss=val_loss,
                   lr=float(opt.param_groups[0]["lr"]),
                   epoch_seconds=round(time.perf_counter() - e0, 3),
                   elapsed_seconds=round(time.perf_counter() - t0, 3),
                   frac_of_zero_baseline=train_loss / base["zero"] if base["zero"] else None,
                   frac_of_copy_baseline=train_loss / base["copy"] if base["copy"] else None,
                   git_commit=meta["git_commit"])
        history.append(row)
        with open(metrics_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(row) + "\n")
        watch = val_loss if val_loss is not None else train_loss
        if watch < best:
            best = watch
            save_checkpoint(os.path.join(run_dir, "best.pt"), model, cfg, meta,
                            epoch, train_loss, val_loss)
        if cfg.log_every and (epoch % cfg.log_every == 0 or epoch == cfg.epochs):
            print("  epoch %4d  train %.6f%s  (%.3fx zero, %.3fx copy)  %.1fs"
                  % (epoch, train_loss,
                     "  val %.6f" % val_loss if val_loss is not None else "",
                     row["frac_of_zero_baseline"], row["frac_of_copy_baseline"],
                     row["epoch_seconds"]))
        if on_epoch:
            on_epoch(row)

    save_checkpoint(os.path.join(run_dir, "last.pt"), model, cfg, meta,
                    len(history), history[-1]["train_loss"],
                    history[-1]["val_loss"])
    result = dict(run_dir=run_dir, history=history, baselines=base,
                  metadata=meta, best=best,
                  final_train_loss=history[-1]["train_loss"],
                  final_val_loss=history[-1]["val_loss"],
                  wall_seconds=round(time.perf_counter() - t0, 3))
    with open(os.path.join(run_dir, "result.json"), "w", encoding="utf-8") as fh:
        json.dump({k: v for k, v in result.items() if k != "metadata"}, fh,
                  indent=1, default=str)
    return result


# ─── checkpoints ──────────────────────────────────────────────────────────────
def save_checkpoint(path: str, model: nn.Module, cfg: TrainConfig, meta: dict,
                    epoch: int, train_loss: float,
                    val_loss: Optional[float]) -> str:
    torch.save(dict(state_dict=model.state_dict(), config=asdict(cfg),
                    model_cls=type(model).__name__,
                    model_kwargs=getattr(model, "hparams", {}),
                    spec_version=spec.SPEC_VERSION,
                    git_commit=meta.get("git_commit"),
                    data=meta.get("data"), baselines=meta.get("baselines"),
                    epoch=int(epoch), train_loss=float(train_loss),
                    val_loss=None if val_loss is None else float(val_loss)),
               path)
    return path


def load_checkpoint(path: str, model_factory: Callable[..., nn.Module]) -> tuple:
    """(model, checkpoint). Refuses a checkpoint from another spec version -
    the dimension layout may differ, so the weights would be read into a
    different meaning of the same 47 and 22 numbers."""
    ck = torch.load(path, map_location="cpu", weights_only=False)
    spec.assert_spec_version(ck["spec_version"], where=os.path.basename(path))
    model = model_factory(**(ck.get("model_kwargs") or {}))
    model.load_state_dict(ck["state_dict"])
    model.eval()
    return model, ck
