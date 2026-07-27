# Pretrained property encoders (`observation=gnn`)

Frozen property predictors used as the DQN's *teacher*: the encoder embeds a molecule
and its scalar prediction `r_hat` is appended as the last observation feature, on top of
which a zero-initialised residual head is trained (`src/models/gnn_teacher.py`).
The encoders are **never** updated by the Bellman loss.

| file | model | target | obs dim | notes |
|---|---|---|---|---|
| `gnn_qed.pt` | GINE, 4 layers, hidden 256 | QED | 256 + 1 | trained on `Data/zinc_10000.txt` labels |
| `linear_qed.pt` | Morgan(r=3, 2048) → MLP | QED | 2048 + 1 | fingerprint counterpart, same labels |

Each file is a plain `torch.save` dict: `state_dict`, `hidden`, `layers`, `bounded`.
No molecule data is stored in the checkpoints.

## Use

```bash
python main.py env.observation=gnn env.gnn_ckpt=gnn_models/gnn_qed.pt \
               train.aux_distill=1.0 reward=qed
```

`train.aux_distill` is the candidate-set distillation weight; it is what makes the
frozen encoder's ranking survive Bellman training (see `rep_gnn/RESULTS.md`).

## Training

`train_supervised.py` (QED / logP) and `train_combined.py` (multi-target), both in
`rep_gnn/` on the **`rep`** branch — research scaffolding that no production code
imports, so it is not merged here. Its write-up is `docs/gnn_qnetwork_study.md`
(kept local, out of history, like the rest of `docs/`).

Research-time checkpoints live in `rep_gnn/ckpt/`; the two files here are the ones the
production configs point at. BDE/IP encoders are not published because their labels
come from proprietary data.
