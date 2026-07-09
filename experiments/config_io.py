"""YAML experiment configuration: generic dataclass <-> dict conversion.

The YAML schema mirrors the config dataclasses one-to-one: a top-level key
per ``ExperimentConfig`` field, nested keys per dataclass field. The
``system`` section is special-cased only in that ``system.type`` selects the
system definition whose ``default_params()`` / ``default_noise()`` provide
the defaults for ``system.params`` / ``system.noise``.

Unknown keys raise ``ValueError`` (typos must fail loudly); absent keys keep
their dataclass defaults. Round-trip invariant:
``load_experiment_config(write_config_snapshot(c)) == c``.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import yaml


def dataclass_to_dict(instance):
    result = {}
    for field in dataclasses.fields(instance):
        result[field.name] = _plain_value(getattr(instance, field.name))
    return result


def _plain_value(value):
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return dataclass_to_dict(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return [_plain_value(item) for item in value]
    return value


def dataclass_from_dict(defaults, data, context="config"):
    if data is None:
        data = {}
    if not isinstance(data, dict):
        raise ValueError(f"{context} must be a mapping, got {type(data).__name__}.")
    field_map = {field.name: field for field in dataclasses.fields(defaults)}
    unknown = sorted(set(data) - set(field_map))
    if unknown:
        raise ValueError(
            f"unknown key(s) in {context}: {', '.join(unknown)}; "
            f"valid keys: {', '.join(sorted(field_map))}."
        )
    updates = {}
    for name, value in data.items():
        updates[name] = _coerce_value(
            getattr(defaults, name),
            field_map[name],
            value,
            f"{context}.{name}",
        )
    return dataclasses.replace(defaults, **updates)


def _coerce_value(current, field, value, context):
    if value is None:
        return None
    if dataclasses.is_dataclass(current) and not isinstance(current, type):
        return dataclass_from_dict(current, value, context)
    if isinstance(current, Path) or (current is None and "Path" in str(field.type)):
        return Path(value)
    if isinstance(current, bool):
        if not isinstance(value, bool):
            raise ValueError(f"{context} must be a boolean, got {value!r}.")
        return value
    if isinstance(current, tuple):
        if not isinstance(value, (list, tuple)):
            raise ValueError(f"{context} must be a list, got {value!r}.")
        if len(current) and all(isinstance(item, float) for item in current):
            return tuple(float(item) for item in value)
        return tuple(value)
    if isinstance(current, float):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{context} must be a number, got {value!r}.")
        return float(value)
    if isinstance(current, int):
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"{context} must be an integer, got {value!r}.")
        return value
    if isinstance(current, str):
        if not isinstance(value, str):
            raise ValueError(f"{context} must be a string, got {value!r}.")
        return value
    return value


def load_experiment_config(path, default_config, get_system):
    """Read a YAML config file onto ``default_config``.

    ``get_system`` resolves ``system.type`` to a system definition; when the
    type differs from the default, that definition's ``default_params()`` /
    ``default_noise()`` become the merge base for ``system.params`` /
    ``system.noise``.
    """
    path = Path(path)
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise ValueError(f"config file {path} must contain a mapping at top level.")
    data = dict(raw)

    system_section = data.pop("system", None) or {}
    if not isinstance(system_section, dict):
        raise ValueError("system section must be a mapping.")
    system_section = dict(system_section)
    system_type = system_section.pop("type", default_config.system.type)
    definition = get_system(system_type)
    if system_type == default_config.system.type:
        base_params = default_config.system.params
        base_noise = default_config.system.noise
    else:
        base_params = definition.default_params()
        base_noise = definition.default_noise()
    params = dataclass_from_dict(
        base_params, system_section.pop("params", None), "system.params"
    )
    noise = dataclass_from_dict(
        base_noise, system_section.pop("noise", None), "system.noise"
    )
    if system_section:
        raise ValueError(
            f"unknown key(s) in system: {', '.join(sorted(system_section))}; "
            "valid keys: noise, params, type."
        )
    config = dataclasses.replace(
        default_config,
        system=dataclasses.replace(
            default_config.system, type=system_type, params=params, noise=noise
        ),
    )

    section_names = [
        field.name for field in dataclasses.fields(config) if field.name != "system"
    ]
    unknown = sorted(set(data) - set(section_names))
    if unknown:
        raise ValueError(
            f"unknown config section(s): {', '.join(unknown)}; "
            f"valid sections: system, {', '.join(sorted(section_names))}."
        )
    for name in section_names:
        if name in data:
            config = dataclasses.replace(
                config,
                **{name: dataclass_from_dict(getattr(config, name), data[name], name)},
            )
    return config


def config_to_yaml_str(config):
    return yaml.safe_dump(
        dataclass_to_dict(config),
        sort_keys=False,
        default_flow_style=False,
    )


def write_config_snapshot(config, path):
    path = Path(path)
    path.write_text(config_to_yaml_str(config), encoding="utf-8")
    return path
