import json
import math
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from src import devices  # noqa: E402
from src.config import apply_overrides, load_config  # noqa: E402
from src.metrics import speedup, summarize, throughput_degradation  # noqa: E402
from src.reporting import COLUMNS, ResultWriter, latest_runs, load_results, validate  # noqa: E402
from src.runner import NonFiniteLoss, check_finite, classify_exception  # noqa: E402
from src.timing import DeviceTimer, time_fn  # noqa: E402


def test_summarize_statistics():
    s = summarize([1.0, 2.0, 3.0, 4.0, 100.0])
    assert s["median"] == 3.0 and s["min"] == 1.0 and s["max"] == 100.0 and s["n"] == 5
    assert s["mean"] == pytest.approx(22.0)
    assert s["std"] == pytest.approx(43.5, rel=0.01)
    assert s["cv"] == pytest.approx(s["std"] / s["mean"])
    assert summarize([5.0])["std"] is None
    assert summarize([])["median"] is None
    assert summarize([1.0, float("nan"), 3.0])["n"] == 2  # non-finite values never enter statistics


def test_speedup_is_baseline_over_backend():
    assert speedup(10.0, 2.0) == 5.0
    assert speedup(None, 2.0) is None and speedup(10.0, 0) is None


def test_throughput_degradation_sign():
    falling = throughput_degradation(list(range(100, 0, -1)))
    assert falling["degradation_pct"] > 0  # positive = slower at the end
    flat = throughput_degradation([50.0] * 20)
    assert flat["degradation_pct"] == 0.0
    assert throughput_degradation([1.0])["degradation_pct"] is None


def test_time_fn_counts_and_streams_every_sample():
    seen = []
    res = time_fn(lambda: sum(range(1000)), "cpu", warmup=2, iterations=5, on_sample=lambda p, i, ms: seen.append((p, i)))
    assert len(res.warmup_ms) == 2 and len(res.samples_ms) == 5
    assert seen == [("warmup", 0), ("warmup", 1)] + [("measure", i) for i in range(5)]
    assert all(x > 0 for x in res.samples_ms)


def test_device_timer_measures_elapsed():
    import time

    with DeviceTimer("cpu") as t:
        time.sleep(0.02)
    assert 15 < t.elapsed_ms < 200


@pytest.mark.parametrize("exc,status", [
    (RuntimeError("MPS backend out of memory (MPS allocated: 1 GB)"), "oom"),
    (RuntimeError("Invalid buffer size: 20.00 GiB"), "oom"),
    (NotImplementedError("The operator 'aten::foo' is not currently implemented for the MPS device"), "unsupported"),
    (TypeError("Trying to convert Float64 to the MPS backend but it does not have support for that dtype"), "unsupported"),
    (RuntimeError("something else entirely"), "error"),
])
def test_classify_exception(exc, status):
    assert classify_exception(exc)[0] == status


def test_check_finite():
    check_finite(1.0)
    with pytest.raises(NonFiniteLoss):
        check_finite(float("nan"))


def test_resolve_backends_reports_unavailable_instead_of_hiding():
    auto = devices.resolve_backends("auto")
    assert [b.backend for b in auto] == ["cpu", "mps", "cuda"]
    cpu = devices.resolve_backends("cpu")[0]
    assert cpu.available and cpu.reason is None
    for b in auto:
        if not b.available:
            assert b.reason  # always explains why
            with pytest.raises(RuntimeError):
                devices.get_device(b)  # never silently substitutes another device
    with pytest.raises(ValueError):
        devices.resolve_backends("tpu")


def test_assert_on_device_detects_wrong_placement():
    t = torch.zeros(2)
    devices.assert_on_device(torch.device("cpu"), t, torch.nn.Linear(2, 2))
    with pytest.raises(RuntimeError):
        devices.assert_on_device(torch.device("meta"), t)


def test_mps_fallback_env_is_removed_on_import():
    import os

    assert "PYTORCH_ENABLE_MPS_FALLBACK" not in os.environ


def test_config_overrides():
    cfg = load_config(None, ["matmul.iterations=7", "matmul.sizes=[64,128]", "sustained.duration_minutes=0.5"])
    assert cfg["matmul"]["iterations"] == 7 and cfg["matmul"]["sizes"] == [64, 128]
    assert cfg["sustained"]["duration_minutes"] == 0.5 and cfg["seed"] == 1234
    with pytest.raises(ValueError):
        apply_overrides({}, ["novalue"])


def _status():
    return devices.resolve_backends("cpu")[0]


def test_writer_roundtrip_and_extra_fields(tmp_path):
    meta = {"run_id": "r1", "experiment": "demo", "backend": "cpu", "torch": "x", "device_name": "d", "seed": 1}
    with ResultWriter(tmp_path, "demo", _status(), meta) as w:
        w.write(phase="measure", iteration=0, latency_ms=1.5, throughput=10.0, custom_field=42)
        w.write(status="oom", phase="config", error_type="OutOfMemoryError", error="boom", batch_size=64)
        w.write(phase="measure", iteration=1, latency_ms=float("nan"), throughput=None)  # NaN never reaches the file
    df, metas = load_results(tmp_path)
    assert list(df.columns[:4]) == COLUMNS[:4] and len(df) == 3
    assert df.loc[0, "x_custom_field"] == 42
    assert df.loc[1, "status"] == "oom" and df.loc[1, "error"] == "boom"
    assert pd.isna(df.loc[2, "latency_ms"])
    assert "demo|cpu|r1" in metas and json.loads(df.loc[0, "metadata"])["seed"] == 1


def test_validate_flags_problems(tmp_path):
    meta = {"run_id": "r1", "experiment": "demo", "backend": "cpu"}
    with ResultWriter(tmp_path, "demo", _status(), meta) as w:
        w.write(phase="measure", iteration=0, latency_ms=-1.0, throughput=1.0)
        w.write(status="error", phase="config")  # failure without an error message
        w.write(status="bogus", phase="measure", iteration=1, latency_ms=1.0)
    issues = validate(load_results(tmp_path)[0])
    text = " ".join(issues)
    assert "negative" in text and "no error message" in text and "unknown status" in text


def test_latest_runs_keeps_crash_marker_with_its_run():
    df = pd.DataFrame({"experiment": ["m"] * 4, "backend": ["cpu"] * 4, "run_id": ["a", "b", "b", "b_crash"], "v": [1, 2, 3, 4]})
    assert sorted(latest_runs(df)["v"]) == [2, 3, 4]


def test_matmul_end_to_end_on_cpu(tmp_path):
    from experiments.benchmark_matmul import main

    rc = main(["--backend", "cpu,cuda" if not torch.cuda.is_available() else "cpu", "--results-dir", str(tmp_path), "--run-id", "t",
               "--set", "matmul.sizes=[32]", "--set", "matmul.iterations=3", "--set", "matmul.warmup=1", "--set", "matmul.overrides_by_size={}"])
    assert rc == 0
    df, _ = load_results(tmp_path / "raw")
    cpu = df[(df["backend"] == "cpu") & (df["status"] == "ok")]
    assert len(cpu) == 4 and set(cpu["phase"]) == {"warmup", "measure"}  # every measurement, warmup included
    assert (cpu["throughput_unit"] == "GFLOPS").all()
    assert validate(df) == []
    if not torch.cuda.is_available():
        un = df[df["backend"] == "cuda"]
        assert list(un["status"]) == ["unavailable"] and un["error"].notna().all()


def test_time_budget_reduces_slow_configurations_and_flags_it():
    import time

    seen = []
    slow = time_fn(lambda: time.sleep(0.05), "cpu", warmup=5, iterations=50, budget_s=0.5, on_sample=lambda p, i, ms: seen.append(p))
    assert slow.reduced_for_budget and len(slow.warmup_ms) == 1 and 3 <= len(slow.samples_ms) < 50
    fast = time_fn(lambda: None, "cpu", warmup=5, iterations=50, budget_s=5.0)
    assert not fast.reduced_for_budget and len(fast.warmup_ms) == 5 and len(fast.samples_ms) == 50
    unbudgeted = time_fn(lambda: time.sleep(0.01), "cpu", warmup=2, iterations=4)
    assert not unbudgeted.reduced_for_budget and len(unbudgeted.samples_ms) == 4
