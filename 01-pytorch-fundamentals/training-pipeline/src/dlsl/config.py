"""Load, override, resolve, save, and identify experiment configurations."""

import hashlib
import os
import re
from pathlib import Path

import yaml


_ENV_PATTERN = re.compile(
    r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}"
)

_SCIENTIFIC = re.compile(
    r"^[+-]?\d+(?:\.\d*)?[eE][+-]?\d+$"
)


def _parse_value(text):
    """Parse an override value while preserving explicitly quoted strings."""
    value = yaml.safe_load(text)

    # PyYAML may treat unquoted 1e-4 as a string.
    if isinstance(value, str) and _SCIENTIFIC.fullmatch(text.strip()):
        return float(text)

    return value


def _apply_override(cfg, override):
    """Apply one key=value override. The full key must already exist."""
    key, separator, raw_value = override.partition("=")

    if not separator:
        raise ValueError(
            f"Expected key=value, got: {override!r}"
        )

    key = key.strip()
    parts = key.split(".")

    if not key or any(not part for part in parts):
        raise ValueError(f"Invalid override key: {key!r}")

    current = cfg

    for part in parts[:-1]:
        if not isinstance(current, dict) or part not in current:
            raise KeyError(f"Unknown config key: {key}")

        current = current[part]

    final_key = parts[-1]

    if not isinstance(current, dict) or final_key not in current:
        raise KeyError(f"Unknown config key: {key}")

    current[final_key] = _parse_value(raw_value)


def _find_repo_root(config_path):
    """Walk upward from the config file to a directory containing .git."""
    start = Path(config_path).resolve().parent

    for directory in (start, *start.parents):
        # .git can be a directory or a file in a Git worktree.
        if (directory / ".git").exists():
            return directory

    raise FileNotFoundError(
        f"Could not find a repository root above {start}"
    )


def _contains_repo_root(value):
    """Check whether any config value references REPO_ROOT."""
    if isinstance(value, dict):
        return any(
            _contains_repo_root(child)
            for child in value.values()
        )

    if isinstance(value, list):
        return any(_contains_repo_root(child) for child in value)

    if isinstance(value, str):
        return any(
            match.group(1) == "REPO_ROOT"
            for match in _ENV_PATTERN.finditer(value)
        )

    return False


def _resolve_env(value, env, key=""):
    """Resolve placeholders in nested values, preserving other types."""
    if isinstance(value, dict):
        return {
            name: _resolve_env(
                child,
                env,
                f"{key}.{name}" if key else str(name),
            )
            for name, child in value.items()
        }

    if isinstance(value, list):
        return [
            _resolve_env(child, env, f"{key}[{index}]")
            for index, child in enumerate(value)
        ]

    if not isinstance(value, str):
        return value

    def replace(match):
        variable = match.group(1)
        default = match.group(2)
        resolved = env.get(variable)

        if resolved is None or resolved == "":
            if default is not None:
                return default

            raise ValueError(
                f"Config key '{key}' requires environment "
                f"variable '{variable}'"
            )

        return resolved

    # Substituted values stay strings and are not reparsed as YAML.
    return _ENV_PATTERN.sub(replace, value)


def load_config(path, overrides=()):
    """Read YAML, apply existing-key overrides, and resolve environment values."""
    path = Path(path)

    with path.open("r", encoding="utf-8") as file:
        cfg = yaml.safe_load(file)

    if not isinstance(cfg, dict):
        raise ValueError(
            f"Config must contain a YAML mapping: {path}"
        )

    for override in overrides:
        _apply_override(cfg, override)

    env = dict(os.environ)

    # Temporary standalone configs need no Git repository unless they
    # actually reference REPO_ROOT.
    if _contains_repo_root(cfg):
        env["REPO_ROOT"] = _find_repo_root(path).as_posix()

    return _resolve_env(cfg, env)


def save_config(cfg, run_dir):
    """Save the resolved dictionary returned by load_config."""
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    with (run_dir / "config.yaml").open("w", encoding="utf-8") as file:
        yaml.safe_dump(
            cfg,
            file,
            sort_keys=True,
            allow_unicode=True,
        )


def get(cfg, key, default=None):
    """Look up a dotted dictionary key, returning default if absent."""
    current = cfg

    for part in key.split("."):
        if not isinstance(current, dict) or part not in current:
            return default

        current = current[part]

    return current


def config_hash(cfg):
    """Return a 12-character SHA-256 identifier for the resolved config."""
    # Sorted keys make the hash independent of dictionary insertion order.
    serialized = yaml.safe_dump(
        cfg,
        sort_keys=True,
        allow_unicode=True,
    )

    return hashlib.sha256(
        serialized.encode("utf-8")
    ).hexdigest()[:12]