import gzip
import pickle

from src.persistence.recorder import Recorder


def test_merge_to_single_gz(tmp_path):
    base = str(tmp_path / "Experiments")
    for r in (0, 1):
        rec = Recorder(base, exp="e", trial="t", rank=r, world_size=2)
        rec.record_metrics({"rewards": [r], "batch_losses": [0.1 * r]})
        rec.record_paths(top=[("CCO", 1.0)], last=[], all_smiles=[f"S{r}"])
        rec.flush()

    out = Recorder.merge(base, exp="e", trial="t", world_size=2)

    assert out.endswith("e_t/e_t.pickle.gz")
    with gzip.open(out, "rb") as f:
        data = pickle.load(f)

    assert set(data["metrics"].keys()) == {0, 1}
    assert data["metrics"][1]["rewards"] == [1]
    assert set(data["paths"].keys()) == {0, 1}
    assert data["paths"][0]["all"] == ["S0"]

    # shards cleaned up
    import glob
    import os
    assert not glob.glob(os.path.join(base, "e_t", "_rank*_*.pickle"))


def test_config_and_run_dir(tmp_path):
    base = str(tmp_path / "Experiments")
    rec = Recorder(base, exp="qed", trial=7, rank=0, world_size=1)
    rec.save_config("a: 1\n")
    import os
    assert os.path.isfile(os.path.join(base, "qed_7", "config.yaml"))
