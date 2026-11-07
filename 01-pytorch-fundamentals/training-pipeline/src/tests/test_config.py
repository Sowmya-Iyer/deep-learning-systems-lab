import pytest

from dlsl.config import (
    config_hash,
    get,
    load_config,
    save_config,
)


def write_yaml(tmp_path, text):
    path = tmp_path / "config.yaml"
    path.write_text(text, encoding="utf-8")
    return path


def test_overrides_change_values_and_types(tmp_path):
    path = write_yaml(
        tmp_path,
        """
train:
  batch_size: 2
  amp: false
optim:
  lr: 0.01
""",
    )

    cfg = load_config(
        path,
        overrides=[
            "train.batch_size=4",
            "train.amp=true",
            "optim.lr=1e-4",
        ],
    )

    assert cfg["train"]["batch_size"] == 4
    assert type(cfg["train"]["batch_size"]) is int

    assert cfg["train"]["amp"] is True

    assert cfg["optim"]["lr"] == pytest.approx(0.0001)
    assert type(cfg["optim"]["lr"]) is float

    # Overrides must not modify the source file.
    original = load_config(path)
    assert original["train"]["batch_size"] == 2


def test_quoted_numeric_override_stays_string(tmp_path):
    path = write_yaml(tmp_path, "experiment: baseline\n")

    cfg = load_config(
        path,
        overrides=['experiment="1e-4"'],
    )

    assert cfg["experiment"] == "1e-4"
    assert isinstance(cfg["experiment"], str)


@pytest.mark.parametrize(
    "override",
    [
        "train.batch_sze=4",
        "unknown.batch_size=4",
        "train.batch_size.extra=4",
    ],
)
def test_unknown_key_raises(tmp_path, override):
    path = write_yaml(
        tmp_path,
        "train:\n  batch_size: 2\n",
    )

    with pytest.raises(KeyError, match="Unknown config key"):
        load_config(path, overrides=[override])


def test_override_requires_equals_sign(tmp_path):
    path = write_yaml(tmp_path, "seed: 0\n")

    with pytest.raises(ValueError, match="Expected key=value"):
        load_config(path, overrides=["seed"])


@pytest.mark.parametrize("value", [None, ""])
def test_env_default_when_unset_or_empty(tmp_path, monkeypatch, value):
    if value is None:
        monkeypatch.delenv("RUNS_ROOT", raising=False)
    else:
        monkeypatch.setenv("RUNS_ROOT", value)

    path = write_yaml(
        tmp_path,
        "output:\n  runs_root: ${RUNS_ROOT:-runs}\n",
    )

    cfg = load_config(path)

    assert cfg["output"]["runs_root"] == "runs"


def test_env_value_takes_precedence_over_default(tmp_path, monkeypatch):
    monkeypatch.setenv("RUNS_ROOT", "C:/experiments")

    path = write_yaml(
        tmp_path,
        "output:\n  runs_root: ${RUNS_ROOT:-runs}\n",
    )

    cfg = load_config(path)

    assert cfg["output"]["runs_root"] == "C:/experiments"


def test_required_env_variable(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_ROOT", "C:/datasets")

    path = write_yaml(
        tmp_path,
        "data:\n  images: ${DATA_ROOT}/loco\n",
    )

    cfg = load_config(path)

    assert cfg["data"]["images"] == "C:/datasets/loco"


def test_missing_env_names_config_key(tmp_path, monkeypatch):
    monkeypatch.delenv("DATA_ROOT", raising=False)

    path = write_yaml(
        tmp_path,
        "data:\n  images: ${DATA_ROOT}/loco\n",
    )

    with pytest.raises(
        ValueError,
        match=r"data\.images.*DATA_ROOT",
    ):
        load_config(path)


def test_override_happens_before_env_resolution(tmp_path, monkeypatch):
    monkeypatch.delenv("DATA_ROOT", raising=False)

    path = write_yaml(
        tmp_path,
        "data:\n  images: ${DATA_ROOT}/loco\n",
    )

    cfg = load_config(
        path,
        overrides=["data.images=C:/other/loco"],
    )

    assert cfg["data"]["images"] == "C:/other/loco"


def test_env_resolution_inside_lists(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_ROOT", "C:/datasets")

    path = write_yaml(
        tmp_path,
        """
paths:
  - ${DATA_ROOT}/train
  - ${DATA_ROOT}/val
""",
    )

    cfg = load_config(path)

    assert cfg["paths"] == [
        "C:/datasets/train",
        "C:/datasets/val",
    ]


def test_repo_root_is_discovered(tmp_path, monkeypatch):
    (tmp_path / ".git").mkdir()

    config_dir = tmp_path / "configs"
    config_dir.mkdir()

    monkeypatch.setenv("REPO_ROOT", "incorrect")

    path = config_dir / "config.yaml"
    path.write_text(
        "root: ${REPO_ROOT}\n",
        encoding="utf-8",
    )

    cfg = load_config(path)

    assert cfg["root"] == tmp_path.as_posix()


def test_save_then_load_preserves_resolved_config(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_ROOT", "C:/datasets")

    path = write_yaml(
        tmp_path,
        """
seed: 42
data:
  images: ${DATA_ROOT}/loco
  subset_size: null
train:
  batch_size: 2
  amp: true
optim:
  lr: 0.01
""",
    )

    cfg = load_config(path, overrides=["optim.lr=1e-4"])

    run_dir = tmp_path / "runs" / "example"
    save_config(cfg, run_dir)

    # Saved paths must no longer depend on the original environment.
    monkeypatch.delenv("DATA_ROOT", raising=False)

    restored = load_config(run_dir / "config.yaml")

    assert restored == cfg
    assert config_hash(restored) == config_hash(cfg)


def test_dotted_get():
    cfg = {
        "train": {
            "batch_size": 4,
            "optional": None,
        }
    }

    assert get(cfg, "train.batch_size") == 4
    assert get(cfg, "train.missing", default=8) == 8
    assert get(cfg, "missing.value") is None

    # An existing None value should not be replaced by the default.
    assert get(cfg, "train.optional", default=8) is None

    assert get(cfg, "train.batch_size.extra", default=8) == 8


def test_hash_ignores_dictionary_key_order():
    first = {
        "seed": 42,
        "train": {"batch_size": 4, "amp": True},
    }
    second = {
        "train": {"amp": True, "batch_size": 4},
        "seed": 42,
    }

    assert config_hash(first) == config_hash(second)
    assert len(config_hash(first)) == 12


def test_hash_changes_when_config_changes():
    first = {"train": {"batch_size": 4}}
    second = {"train": {"batch_size": 8}}

    assert config_hash(first) != config_hash(second)


@pytest.mark.parametrize("text", ["", "- first\n- second\n", "42\n"])
def test_config_must_be_a_mapping(tmp_path, text):
    path = write_yaml(tmp_path, text)

    with pytest.raises(ValueError, match="YAML mapping"):
        load_config(path)