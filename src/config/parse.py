"""Build the typed :class:`Config` from a Hydra ``DictConfig``.

``parse_config(cfg)`` converts enum strings to enums, parses ``maintain_OH``
(yaml ``null``/``'none'``/``'exist'``/``'same'``/int -> :class:`MaintainOH`), and
turns the named reward-weight blocks into :class:`BdeIpWeights` / :class:`QedWeights`.
All validation is done by the dataclasses' ``__post_init__``.
"""
from __future__ import annotations

from typing import Any

from src.config.enums import (
    Backend,
    ObservationType,
    Optimizer,
    OHMode,
    RewardType,
    Starter,
)
from src.config.schema import (
    BdeIpWeights,
    CacheCfg,
    CkptCfg,
    Config,
    DistCfg,
    EnvCfg,
    EtkdgCfg,
    ExperimentCfg,
    MaintainOH,
    MolsCfg,
    OptimCfg,
    QedWeights,
    RecordCfg,
    RewardCfg,
    RunCfg,
    TrainCfg,
)


def _get(node: Any, key: str, default: Any = None) -> Any:
    """Read ``key`` from a DictConfig/dict node, tolerating a missing key."""
    if node is None:
        return default
    if hasattr(node, "get"):
        try:
            return node.get(key, default)
        except Exception:
            pass
    try:
        return node[key]
    except Exception:
        return getattr(node, key, default)


def parse_maintain_oh(value: Any) -> MaintainOH:
    """yaml none/'none'/'exist'/'same'/int -> MaintainOH."""
    if value is None:
        return MaintainOH(OHMode.NO_LIMIT)
    if isinstance(value, bool):
        raise ValueError(f"maintain_OH cannot be a bool: {value!r}")
    if isinstance(value, int):
        return MaintainOH(OHMode.EXACT, count=int(value))
    if isinstance(value, str):
        s = value.strip().lower()
        if s in ("none", "null", ""):
            return MaintainOH(OHMode.NO_LIMIT)
        if s == "exist":
            return MaintainOH(OHMode.AT_LEAST_ONE)
        if s == "same":
            return MaintainOH(OHMode.SAME_AS_INIT)
        # numeric string -> EXACT
        try:
            return MaintainOH(OHMode.EXACT, count=int(s))
        except ValueError:
            raise ValueError(
                f"maintain_OH must be null/'none'/'exist'/'same'/int, got {value!r}")
    raise ValueError(f"maintain_OH has unsupported type: {value!r}")


def _enum_from(enum_cls, value, *, allow_none=False):
    if value is None:
        if allow_none:
            return None
        raise ValueError(f"{enum_cls.__name__} value cannot be None")
    if isinstance(value, enum_cls):
        return value
    return enum_cls(str(value))


def _bde_ip_weights(node) -> BdeIpWeights:
    if node is None:
        return BdeIpWeights()
    return BdeIpWeights(
        bde=float(_get(node, "bde", BdeIpWeights.bde)),
        ip=float(_get(node, "ip", BdeIpWeights.ip)),
        rrab=float(_get(node, "rrab", BdeIpWeights.rrab)),
    )


def _qed_weights(node) -> QedWeights:
    if node is None:
        return QedWeights()
    return QedWeights(
        qed=float(_get(node, "qed", QedWeights.qed)),
        sa=float(_get(node, "sa", QedWeights.sa)),
    )


def parse_config(cfg) -> Config:
    """Build a typed :class:`Config` from the Hydra structured ``DictConfig``."""
    exp = _get(cfg, "experiment")
    train = _get(cfg, "train")
    optim = _get(cfg, "optim")
    reward = _get(cfg, "reward")
    env = _get(cfg, "env")
    mols = _get(cfg, "mols")
    dist = _get(cfg, "dist")
    ckpt = _get(cfg, "ckpt")
    record = _get(cfg, "record")
    run = _get(cfg, "run")

    etkdg_node = _get(env, "etkdg")
    cache_node = _get(env, "cache")

    config = Config(
        experiment=ExperimentCfg(
            experiment=str(_get(exp, "experiment", ExperimentCfg.experiment)),
            note=_get(exp, "note", None),
            trial=int(_get(exp, "trial", ExperimentCfg.trial)),
        ),
        train=TrainCfg(
            iteration=int(_get(train, "iteration", TrainCfg.iteration)),
            max_steps_per_episode=int(
                _get(train, "max_steps_per_episode", TrainCfg.max_steps_per_episode)),
            update_episodes=int(_get(train, "update_episodes", TrainCfg.update_episodes)),
            eps_threshold=float(_get(train, "eps_threshold", TrainCfg.eps_threshold)),
            eps_decay=float(_get(train, "eps_decay", TrainCfg.eps_decay)),
            gamma=float(_get(train, "gamma", TrainCfg.gamma)),
            polyak=float(_get(train, "polyak", TrainCfg.polyak)),
        ),
        optim=OptimCfg(
            optimizer=_enum_from(Optimizer, _get(optim, "optimizer", Optimizer.ADAM)),
            lr=float(_get(optim, "lr", OptimCfg.lr)),
            min_batch_size=int(_get(optim, "min_batch_size", OptimCfg.min_batch_size)),
            max_batch_size=int(_get(optim, "max_batch_size", OptimCfg.max_batch_size)),
            replay_buffer_size=int(
                _get(optim, "replay_buffer_size", OptimCfg.replay_buffer_size)),
        ),
        reward=RewardCfg(
            type=_enum_from(RewardType, _get(reward, "type", RewardType.QED)),
            bde_ip=_bde_ip_weights(_get(reward, "bde_ip")),
            qed=_qed_weights(_get(reward, "qed")),
            penalty_weight=float(_get(reward, "penalty_weight", RewardCfg.penalty_weight)),
            discount_factor=float(
                _get(reward, "discount_factor", RewardCfg.discount_factor)),
            bde_factor=float(_get(reward, "bde_factor", RewardCfg.bde_factor)),
            ip_factor=float(_get(reward, "ip_factor", RewardCfg.ip_factor)),
            reward_of_invalid_mol=float(
                _get(reward, "reward_of_invalid_mol", RewardCfg.reward_of_invalid_mol)),
        ),
        env=EnvCfg(
            atom_types=[str(a) for a in _get(env, "atom_types", EnvCfg().atom_types)],
            allowed_ring_sizes=[
                int(r) for r in _get(env, "allowed_ring_sizes", EnvCfg().allowed_ring_sizes)],
            fingerprint_radius=int(
                _get(env, "fingerprint_radius", EnvCfg.fingerprint_radius)),
            fingerprint_length=int(
                _get(env, "fingerprint_length", EnvCfg.fingerprint_length)),
            allow_removal=bool(_get(env, "allow_removal", EnvCfg.allow_removal)),
            allow_no_modification=bool(
                _get(env, "allow_no_modification", EnvCfg.allow_no_modification)),
            allow_bonds_between_rings=bool(
                _get(env, "allow_bonds_between_rings", EnvCfg.allow_bonds_between_rings)),
            observation=_enum_from(
                ObservationType, _get(env, "observation", ObservationType.LIST)),
            maintain_oh=parse_maintain_oh(_get(env, "maintain_OH", None)),
            cache=CacheCfg(
                bde=bool(_get(cache_node, "bde", CacheCfg.bde)),
                ip=bool(_get(cache_node, "ip", CacheCfg.ip)),
            ),
            etkdg=EtkdgCfg(
                max_attempts_cache=int(
                    _get(etkdg_node, "max_attempts_cache", EtkdgCfg.max_attempts_cache)),
                max_attempts_uncache=int(
                    _get(etkdg_node, "max_attempts_uncache", EtkdgCfg.max_attempts_uncache)),
                threads=int(_get(etkdg_node, "threads", EtkdgCfg.threads)),
            ),
            multi_threading=bool(_get(env, "multi_threading", EnvCfg.multi_threading)),
            lru_cache_capacity=int(
                _get(env, "lru_cache_capacity", EnvCfg.lru_cache_capacity)),
        ),
        mols=MolsCfg(
            init_mol=_get(mols, "init_mol", None),
            init_mol_path=_get(mols, "init_mol_path", None),
            num_init_mol=(
                int(_get(mols, "num_init_mol")) if _get(mols, "num_init_mol") is not None
                else None),
            init_mol_start=int(_get(mols, "init_mol_start", MolsCfg.init_mol_start)),
            gpu_list=[int(g) for g in _get(mols, "gpu_list", MolsCfg().gpu_list)],
        ),
        dist=DistCfg(
            starter=_enum_from(Starter, _get(dist, "starter", None), allow_none=True),
            backend=_enum_from(Backend, _get(dist, "backend", Backend.GLOO)),
            mp_world_size=int(_get(dist, "mp_world_size", DistCfg.mp_world_size)),
            mp_master_port=int(_get(dist, "mp_master_port", DistCfg.mp_master_port)),
            init_method=_get(dist, "init_method", None),
            torch_num_threads=int(
                _get(dist, "torch_num_threads", DistCfg.torch_num_threads)),
            seed_offset=int(_get(dist, "seed_offset", DistCfg.seed_offset)),
        ),
        ckpt=CkptCfg(
            checkpoint=_get(ckpt, "checkpoint", None),
            use_checkpoint_eps=bool(
                _get(ckpt, "use_checkpoint_eps", CkptCfg.use_checkpoint_eps)),
        ),
        record=RecordCfg(
            record_top_path=int(_get(record, "record_top_path", RecordCfg.record_top_path)),
            record_last_path=int(
                _get(record, "record_last_path", RecordCfg.record_last_path)),
            record_all_path=bool(
                _get(record, "record_all_path", RecordCfg.record_all_path)),
            save_reward_freq=int(_get(record, "save_reward_freq", RecordCfg.save_reward_freq)),
            save_model_freq=int(_get(record, "save_model_freq", RecordCfg.save_model_freq)),
            save_path_freq=int(_get(record, "save_path_freq", RecordCfg.save_path_freq)),
        ),
        run=RunCfg(
            test=bool(_get(run, "test", RunCfg.test)),
            debug=bool(_get(run, "debug", RunCfg.debug)),
        ),
    )
    return config
