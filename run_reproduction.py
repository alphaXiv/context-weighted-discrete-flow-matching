"""Compact QM9 reproduction for Context-weighted Discrete Flow Matching.

Every formal result is printed as a JSON record prefixed by ``ORX_RESULT`` so
that it survives in the OpenResearch terminal log.
"""

from __future__ import annotations

import json
import math
import os
import random
import re
import tarfile
import time
import urllib.request
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")


DATA_URL = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/gdb9.tar.gz"
TOKEN_RE = re.compile(
    r"(\[[^\]]+\]|Br|Cl|Si|Se|Na|Li|Mg|Al|Ca|[BCNOFPSI]|[bcnops]|"
    r"\%\d{2}|\d|=|#|-|\+|\\\\|/|\(|\)|\.|:|@)"
)
SPECIAL = ["<pad>", "<bos>", "<eos>", "<mask>"]


def emit(kind: str, **values) -> None:
    print("ORX_RESULT " + json.dumps({"kind": kind, **values}, sort_keys=True), flush=True)


def tokenize(smiles: str) -> list[str] | None:
    tokens = TOKEN_RE.findall(smiles)
    return tokens if "".join(tokens) == smiles else None


def load_qm9(cache: Path, sequence_length: int) -> list[str]:
    cache.mkdir(parents=True, exist_ok=True)
    smiles_file = cache / "canonical_qm9.smi"
    if not smiles_file.exists():
        archive = cache / "gdb9.tar.gz"
        if not archive.exists():
            print(f"Downloading public QM9 from {DATA_URL}", flush=True)
            urllib.request.urlretrieve(DATA_URL, archive)
        sdf_path = cache / "gdb9.sdf"
        if not sdf_path.exists():
            with tarfile.open(archive, "r:gz") as tf:
                member = next(m for m in tf.getmembers() if m.name.endswith("gdb9.sdf"))
                source = tf.extractfile(member)
                assert source is not None
                with sdf_path.open("wb") as destination:
                    while chunk := source.read(1024 * 1024):
                        destination.write(chunk)
        supplier = Chem.ForwardSDMolSupplier(str(sdf_path), removeHs=True)
        canonical: set[str] = set()
        for mol in supplier:
            if mol is None:
                continue
            smi = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
            toks = tokenize(smi)
            if toks is not None and len(toks) + 1 <= sequence_length - 1:
                canonical.add(smi)
        smiles_file.write_text("\n".join(sorted(canonical)) + "\n")
    return [x for x in smiles_file.read_text().splitlines() if x]


class SmilesData:
    def __init__(self, smiles: list[str], sequence_length: int, seed: int):
        counts = Counter(tok for s in smiles for tok in (tokenize(s) or []))
        self.itos = SPECIAL + sorted(counts)
        self.stoi = {tok: i for i, tok in enumerate(self.itos)}
        self.pad = self.stoi["<pad>"]
        self.bos = self.stoi["<bos>"]
        self.eos = self.stoi["<eos>"]
        self.mask = self.stoi["<mask>"]
        self.sequence_length = sequence_length
        encoded = [self.encode(s) for s in smiles]
        order = np.random.default_rng(1729).permutation(len(encoded))
        self.encoded = torch.tensor(np.asarray(encoded, dtype=np.int64)[order])
        self.smiles = [smiles[i] for i in order]
        self.generator = torch.Generator().manual_seed(seed)

    def encode(self, smiles: str) -> list[int]:
        toks = tokenize(smiles)
        assert toks is not None
        ids = [self.bos] + [self.stoi[t] for t in toks] + [self.eos]
        return ids + [self.pad] * (self.sequence_length - len(ids))

    def decode(self, ids: list[int]) -> str:
        out: list[str] = []
        for idx in ids[1:]:
            tok = self.itos[idx]
            if tok == "<eos>":
                break
            if tok in SPECIAL:
                continue
            out.append(tok)
        return "".join(out)

    def batch(self, size: int, train_size: int, device: torch.device) -> torch.Tensor:
        idx = torch.randint(train_size, (size,), generator=self.generator)
        return self.encoded[idx].to(device, non_blocking=True)


class DFMTransformer(nn.Module):
    def __init__(self, vocab: int, length: int, cfg: dict):
        super().__init__()
        d = cfg["d_model"]
        self.token = nn.Embedding(vocab, d)
        self.position = nn.Parameter(torch.randn(1, length, d) * 0.02)
        self.time = nn.Sequential(nn.Linear(1, d), nn.SiLU(), nn.Linear(d, d))
        layer = nn.TransformerEncoderLayer(
            d_model=d,
            nhead=cfg["n_heads"],
            dim_feedforward=4 * d,
            dropout=cfg["dropout"],
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.blocks = nn.TransformerEncoder(layer, cfg["n_layers"], enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d)
        self.head = nn.Linear(d, vocab, bias=False)
        self.head.weight = self.token.weight

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        h = self.token(x) + self.position + self.time(t[:, None])[:, None, :]
        return self.head(self.norm(self.blocks(h)))


def local_counts(visible: torch.Tensor, radius: int) -> torch.Tensor:
    kernel = torch.ones(1, 1, 2 * radius + 1, device=visible.device)
    return F.conv1d(visible.float()[:, None], kernel, padding=radius)[:, 0] - visible.float()


def corrupt(clean: torch.Tensor, t: torch.Tensor, mask_id: int) -> tuple[torch.Tensor, torch.Tensor]:
    reveal = torch.rand_like(clean.float()) < t[:, None].square()
    reveal[:, 0] = True
    noisy = torch.where(reveal, clean, mask_id)
    return noisy, ~reveal


def loss_value(
    logits: torch.Tensor,
    clean: torch.Tensor,
    masked: torch.Tensor,
    noisy: torch.Tensor,
    cfg: dict,
    mask_id: int,
) -> torch.Tensor:
    token_loss = F.cross_entropy(logits.transpose(1, 2), clean, reduction="none")
    active = masked.float()
    if cfg["loss"] == "sce":
        visible = noisy.ne(mask_id)
        score = local_counts(visible, cfg["sce_radius"])
        score = score.masked_fill(~masked, -1e9)
        weights = torch.softmax(cfg["sce_scale"] * score, dim=-1)
        weights = weights * masked.sum(dim=-1, keepdim=True).clamp_min(1)
        active = torch.where(masked, weights, torch.zeros_like(weights))
    return (token_loss * active).sum() / active.sum().clamp_min(1)


@torch.inference_mode()
def sample(
    model: nn.Module,
    data: SmilesData,
    n: int,
    nfe: int,
    method: str,
    cfg: dict,
    device: torch.device,
) -> list[str]:
    output: list[str] = []
    batch_size = cfg["eval_batch_size"]
    forbidden = torch.tensor([data.bos, data.mask], device=device)
    for start in range(0, n, batch_size):
        b = min(batch_size, n - start)
        x = torch.full((b, data.sequence_length), data.mask, dtype=torch.long, device=device)
        x[:, 0] = data.bos
        for step in range(nfe):
            t0 = step / nfe
            t1 = (step + 1) / nfe
            k0, k1 = t0 * t0, t1 * t1
            p = min(1.0, (k1 - k0) / max(1e-8, 1.0 - k0))
            t = torch.full((b,), t0, device=device)
            logits = model(x, t) / cfg["temperature"]
            logits[..., forbidden] = -torch.inf
            probs = torch.softmax(logits, dim=-1)
            masked = x.eq(data.mask)
            masked[:, 0] = False
            if method == "euler":
                choose = (torch.rand_like(x.float()) < p) & masked
            else:
                counts = torch.distributions.Binomial(masked.sum(-1).float(), p).sample().long()
                if method == "neighbor":
                    scores = cfg["neighbor_scale"] * local_counts(~masked, cfg["neighbor_radius"])
                elif method == "entropy":
                    entropy = -(probs * probs.clamp_min(1e-12).log()).sum(-1)
                    scores = -cfg["entropy_scale"] * entropy
                else:
                    raise ValueError(method)
                scores = scores.masked_fill(~masked, -torch.inf)
                choose = torch.zeros_like(masked)
                for row, count in enumerate(counts.tolist()):
                    if count:
                        picked = torch.multinomial(torch.softmax(scores[row], -1), count, replacement=False)
                        choose[row, picked] = True
            proposed = torch.multinomial(probs.reshape(-1, probs.shape[-1]), 1).reshape(b, -1)
            x = torch.where(choose, proposed, x)
        output.extend(data.decode(row) for row in x.cpu().tolist())
    return output


def molecule_metrics(samples: list[str], train_set: set[str]) -> dict:
    valid = 0
    novel = 0
    canonical_samples: list[str] = []
    for smi in samples:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        canonical = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
        valid += 1
        canonical_samples.append(canonical)
        novel += canonical not in train_set
    return {
        "valid": valid,
        "novel": novel,
        "unique_valid": len(set(canonical_samples)),
        "unique_novel": len(set(canonical_samples) - train_set),
        "total": len(samples),
    }


@torch.inference_mode()
def context_diagnostic(
    model: nn.Module, data: SmilesData, clean: torch.Tensor, device: torch.device
) -> list[dict]:
    rows = []
    for t_value in (0.25, 0.5, 0.75):
        t = torch.full((clean.shape[0],), t_value, device=device)
        noisy, masked = corrupt(clean, t, data.mask)
        logits = model(noisy, t)
        probs = torch.softmax(logits, -1)
        entropy = -(probs * probs.clamp_min(1e-12).log()).sum(-1)
        nll = F.cross_entropy(logits.transpose(1, 2), clean, reduction="none")
        neighbors = local_counts(noisy.ne(data.mask), 2).long()
        for count in range(5):
            pick = masked & neighbors.eq(count)
            if pick.any():
                rows.append(
                    {
                        "t": t_value,
                        "neighbors": count,
                        "count": int(pick.sum()),
                        "entropy": float(entropy[pick].mean()),
                        "nll": float(nll[pick].mean()),
                    }
                )
    return rows


def main() -> None:
    started = time.time()
    cfg = json.loads(Path("config.json").read_text())
    seed = cfg["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda")
    assert torch.cuda.is_available(), "Formal reproduction requires the configured Kubernetes GPU"
    emit(
        "environment",
        config=cfg,
        device=torch.cuda.get_device_name(0),
        torch=torch.__version__,
        cuda=torch.version.cuda,
        hostname=os.uname().nodename,
        started_unix=started,
    )

    smiles = load_qm9(Path("/tmp/qm9"), cfg["sequence_length"])
    data = SmilesData(smiles, cfg["sequence_length"], seed)
    required = cfg["train_size"] + cfg["validation_size"]
    if len(data.encoded) < required:
        raise RuntimeError(f"Only {len(data.encoded)} canonical length-compatible QM9 molecules; need {required}")
    train_set = set(data.smiles[: cfg["train_size"]])
    emit(
        "dataset",
        source=DATA_URL,
        raw_canonical_count=len(smiles),
        train_size=cfg["train_size"],
        validation_size=cfg["validation_size"],
        vocab=data.itos,
        canonicalization="RDKit canonical isomeric SMILES; molecules over 30 tokens excluded",
    )

    model = DFMTransformer(len(data.itos), data.sequence_length, cfg).to(device)
    params = sum(p.numel() for p in model.parameters())
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["learning_rate"],
        betas=(0.9, 0.999),
        weight_decay=cfg["weight_decay"],
    )
    scaler = torch.amp.GradScaler("cuda")
    emit("model", parameters=params, architecture="bidirectional masked Transformer")
    model.train()
    for step in range(1, cfg["steps"] + 1):
        clean = data.batch(cfg["batch_size"], cfg["train_size"], device)
        t = torch.rand(cfg["batch_size"], device=device).clamp_(1e-4, 1 - 1e-4)
        noisy, masked = corrupt(clean, t, data.mask)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(noisy, t)
            loss = loss_value(logits, clean, masked, noisy, cfg, data.mask)
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        if step <= cfg["warmup_steps"]:
            lr_scale = step / cfg["warmup_steps"]
        else:
            progress = (step - cfg["warmup_steps"]) / (cfg["steps"] - cfg["warmup_steps"])
            floor = cfg["min_learning_rate"] / cfg["learning_rate"]
            lr_scale = floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * progress))
        for group in optimizer.param_groups:
            group["lr"] = cfg["learning_rate"] * lr_scale
        if step == 1 or step % 500 == 0:
            emit(
                "training",
                step=step,
                loss=float(loss),
                lr=optimizer.param_groups[0]["lr"],
                elapsed_seconds=time.time() - started,
            )

    model.eval()
    val_clean = data.encoded[cfg["train_size"] : cfg["train_size"] + 1024].to(device)
    emit("context_diagnostic", rows=context_diagnostic(model, data, val_clean, device))
    for nfe in cfg["nfe_values"]:
        for method in ("euler", "neighbor", "entropy"):
            eval_seed = 100000 + seed * 1000 + nfe * 10 + ("euler", "neighbor", "entropy").index(method)
            torch.manual_seed(eval_seed)
            torch.cuda.manual_seed_all(eval_seed)
            samples = sample(model, data, cfg["eval_samples"], nfe, method, cfg, device)
            metrics = molecule_metrics(samples, train_set)
            emit(
                "generation",
                seed=seed,
                loss=cfg["loss"],
                sce_radius=cfg["sce_radius"] if cfg["loss"] == "sce" else None,
                method=method,
                nfe=nfe,
                **metrics,
            )
    emit(
        "summary",
        status="success",
        elapsed_seconds=time.time() - started,
        parameters=params,
        peak_gpu_memory_bytes=torch.cuda.max_memory_allocated(),
        gpu=torch.cuda.get_device_name(0),
    )


if __name__ == "__main__":
    main()
