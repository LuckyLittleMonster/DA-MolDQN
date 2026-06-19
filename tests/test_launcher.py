from src.launch import get_launcher


def test_torchrun(monkeypatch):
    monkeypatch.setenv("WORLD_SIZE", "4")
    monkeypatch.setenv("LOCAL_RANK", "2")
    assert get_launcher("torchrun").resolve() == (2, 4)


def test_slurm(monkeypatch):
    monkeypatch.setenv("SLURM_NPROCS", "8")
    monkeypatch.setenv("SLURM_PROCID", "3")
    assert get_launcher("slurm").resolve() == (3, 8)


def test_single():
    assert get_launcher("single").resolve() == (0, 1)
    assert get_launcher(None).resolve() == (0, 1)


def test_fork_resolve():
    assert get_launcher("fork", world_size=4).resolve() == (None, 4)
