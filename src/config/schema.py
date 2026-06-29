"""Typed, validated single-source config (dataclasses + enums).

This replaces the two parallel config sources (``configs/config.yaml`` -> ``cfg.X``
and ``src/config_defaults.py`` -> ``hyp.X``). Field defaults are the resolved
values: on any yaml/hyp conflict the hyp value wins, EXCEPT ``discount_factor``
which stays 1.0 (the live reward value; the hyp 0.9 was read only by the dead
``MolDQN_2``).

Import-time constants: ``environment.py`` and ``features/fingerprints.py`` build
module-level globals (the cenv environment, the Morgan generator) BEFORE any
runtime config exists. Those constants (``atom_types``, ``allowed_ring_sizes``,
``fingerprint_radius``, ``fingerprint_length``) are NOT user-overridable; they are
sourced from ``EnvCfg`` field defaults via the module-level ``ENV_DEFAULTS``
constant below, so they remain available at import with identical values.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from src.config.enums import (
    Backend,
    ObservationType,
    Optimizer,
    OHMode,
    RewardType,
    Starter,
)


# --------------------------------------------------------------------------- #
# maintain_OH policy
# --------------------------------------------------------------------------- #
@dataclass
class MaintainOH:
    """O-H bond maintenance policy. Preserves the old -2/-1/k semantics verbatim.

    The env derives the per-mol cenv flag from this + the init molecules:
      NO_LIMIT      -> -2  (every mol)
      AT_LEAST_ONE  -> -1  (every mol)
      SAME_AS_INIT  -> count_OH(init_mol_i)  (per-mol)
      EXACT         -> count  (every mol)
    """

    mode: OHMode = OHMode.NO_LIMIT
    count: int | None = None  # required iff mode == EXACT

    def __post_init__(self):
        if not isinstance(self.mode, OHMode):
            raise ValueError(f"MaintainOH.mode must be OHMode, got {self.mode!r}")
        if self.mode is OHMode.EXACT:
            if self.count is None or int(self.count) < 0:
                raise ValueError(
                    f"MaintainOH(mode=EXACT) requires count >= 0, got {self.count!r}")
            self.count = int(self.count)

    def per_mol_flag(self, init_mol, count_oh) -> int:
        """The old cenv int flag for one init molecule.

        ``count_oh`` is the ``count_OH`` function (kept in the env to avoid a
        circular import on rdkit here).
        """
        if self.mode is OHMode.NO_LIMIT:
            return -2
        if self.mode is OHMode.AT_LEAST_ONE:
            return -1
        if self.mode is OHMode.SAME_AS_INIT:
            return count_oh(init_mol)
        # EXACT
        return int(self.count)


# --------------------------------------------------------------------------- #
# reward weights (named fields, no list-length magic)
# --------------------------------------------------------------------------- #
@dataclass
class BdeIpWeights:
    bde: float = 0.8
    ip: float = 0.2
    rrab: float = 0.5


@dataclass
class QedWeights:
    qed: float = 0.8
    sa: float = 0.2


# --------------------------------------------------------------------------- #
# grouped config
# --------------------------------------------------------------------------- #
@dataclass
class ExperimentCfg:
    experiment: str = "da"
    note: str | None = None
    trial: int = 1


@dataclass
class TrainCfg:
    iteration: int = 200000
    max_steps_per_episode: int = 40
    update_episodes: int = 1
    eps_threshold: float = 1.0
    eps_decay: float = 0.999
    gamma: float = 0.95    # RL/Bellman discount (hyp-only); distinct from reward.discount_factor
    polyak: float = 0.995  # target-net Polyak averaging (hyp-only)

    def __post_init__(self):
        if self.iteration < 0:
            raise ValueError(f"iteration must be >= 0, got {self.iteration}")
        if self.max_steps_per_episode <= 0:
            raise ValueError(
                f"max_steps_per_episode must be > 0, got {self.max_steps_per_episode}")
        if self.update_episodes <= 0:
            raise ValueError(f"update_episodes must be > 0, got {self.update_episodes}")
        if not 0.0 <= self.polyak <= 1.0:
            raise ValueError(f"polyak must be in [0,1], got {self.polyak}")


@dataclass
class OptimCfg:
    optimizer: Optimizer = Optimizer.ADAM
    lr: float = 1e-4
    min_batch_size: int = 128
    max_batch_size: int = 128
    replay_buffer_size: int = 4000  # hyp-only

    def __post_init__(self):
        if not isinstance(self.optimizer, Optimizer):
            raise ValueError(f"optimizer must be Optimizer, got {self.optimizer!r}")
        if self.lr <= 0:
            raise ValueError(f"lr must be > 0, got {self.lr}")
        if self.min_batch_size <= 0 or self.max_batch_size <= 0:
            raise ValueError("batch sizes must be > 0")
        if self.replay_buffer_size <= 0:
            raise ValueError(f"replay_buffer_size must be > 0, got {self.replay_buffer_size}")


@dataclass
class RewardCfg:
    type: RewardType = RewardType.QED
    bde_ip: BdeIpWeights = field(default_factory=BdeIpWeights)
    qed: QedWeights = field(default_factory=QedWeights)
    penalty_weight: float = 0.0
    discount_factor: float = 1.0  # SINGLE live use (reward shaping); stays 1.0 per spec
    bde_factor: float = 0.9       # hyp-only
    ip_factor: float = 0.8        # hyp-only
    reward_of_invalid_mol: float = -1000.0  # hyp-only
    ip_ensemble: bool = True      # True: avg all 5 aimnet-nse cv models; False: cv4 only

    def __post_init__(self):
        if not isinstance(self.type, RewardType):
            raise ValueError(f"reward.type must be RewardType, got {self.type!r}")
        if not isinstance(self.ip_ensemble, bool):
            raise ValueError(f"reward.ip_ensemble must be a bool, got {self.ip_ensemble!r}")


@dataclass
class CacheCfg:
    bde: bool = True
    ip: bool = False  # IP is never cached (random ETKDG conformer)


@dataclass
class EtkdgCfg:
    max_attempts_cache: int = 7
    max_attempts_uncache: int = 7
    threads: int = 1  # intra-rank ETKDG thread pool (>1 parallelizes embedding)
    # Seed ETKDG by molecule identity (crc32 of canonical SMILES) instead of
    # batch position -> IP becomes reproducible per molecule and cache.ip exact.
    deterministic_seed: bool = False

    def __post_init__(self):
        if self.threads < 1:
            raise ValueError(f"etkdg.threads must be >= 1, got {self.threads}")
        if not isinstance(self.deterministic_seed, bool):
            raise ValueError(f"etkdg.deterministic_seed must be a bool, got {self.deterministic_seed!r}")


@dataclass
class EnvCfg:
    # ---- import-time constants (NOT user-overridable; sourced as dataclass defaults)
    atom_types: list[str] = field(default_factory=lambda: ["C", "O", "N"])
    allowed_ring_sizes: list[int] = field(default_factory=lambda: [3, 5, 6])
    fingerprint_radius: int = 3
    fingerprint_length: int = 2048
    # ---- runtime env knobs
    allow_removal: bool = True
    allow_no_modification: bool = True
    allow_bonds_between_rings: bool = False
    observation: ObservationType = ObservationType.LIST
    maintain_oh: MaintainOH = field(default_factory=MaintainOH)
    cache: CacheCfg = field(default_factory=CacheCfg)
    etkdg: EtkdgCfg = field(default_factory=EtkdgCfg)
    multi_threading: bool = True
    lru_cache_capacity: int = 128  # hyp-only

    def __post_init__(self):
        if not isinstance(self.observation, ObservationType):
            raise ValueError(f"observation must be ObservationType, got {self.observation!r}")
        if not isinstance(self.maintain_oh, MaintainOH):
            raise ValueError(f"maintain_oh must be MaintainOH, got {self.maintain_oh!r}")
        if self.fingerprint_length <= 0 or self.fingerprint_radius < 0:
            raise ValueError("fingerprint length/radius out of range")


@dataclass
class MolsCfg:
    init_mol: str | None = None
    init_mol_path: str | None = None
    num_init_mol: int | None = None
    init_mol_start: int = 0
    gpu_list: list[int] = field(default_factory=lambda: [0])


@dataclass
class DistCfg:
    starter: Starter | None = None
    backend: Backend = Backend.GLOO
    mp_world_size: int = 1
    mp_master_port: int = 6500
    init_method: str | None = None
    torch_num_threads: int = 1
    seed_offset: int = -1

    def __post_init__(self):
        if self.starter is not None and not isinstance(self.starter, Starter):
            raise ValueError(f"starter must be Starter or None, got {self.starter!r}")
        if not isinstance(self.backend, Backend):
            raise ValueError(f"backend must be Backend, got {self.backend!r}")
        if self.mp_world_size <= 0:
            raise ValueError(f"mp_world_size must be > 0, got {self.mp_world_size}")


@dataclass
class CkptCfg:
    checkpoint: str | None = None
    use_checkpoint_eps: bool = False


@dataclass
class RecordCfg:
    record_top_path: int = 10
    record_last_path: int = 5
    record_all_path: bool = False
    save_reward_freq: int = 100
    save_model_freq: int = 1000
    save_path_freq: int = 1000


@dataclass
class RunCfg:
    test: bool = False
    debug: bool = False


@dataclass
class Config:
    experiment: ExperimentCfg = field(default_factory=ExperimentCfg)
    train: TrainCfg = field(default_factory=TrainCfg)
    optim: OptimCfg = field(default_factory=OptimCfg)
    reward: RewardCfg = field(default_factory=RewardCfg)
    env: EnvCfg = field(default_factory=EnvCfg)
    mols: MolsCfg = field(default_factory=MolsCfg)
    dist: DistCfg = field(default_factory=DistCfg)
    ckpt: CkptCfg = field(default_factory=CkptCfg)
    record: RecordCfg = field(default_factory=RecordCfg)
    run: RunCfg = field(default_factory=RunCfg)


# Import-time constants for the module-level globals (cenv environment, Morgan gen).
# These read the EnvCfg field DEFAULTS — they are not affected by the runtime config.
ENV_DEFAULTS = EnvCfg()
