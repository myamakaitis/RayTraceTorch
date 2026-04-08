import json
import os


def save_project(path: str, element_configs: list, bundle_configs: list, settings: dict) -> None:
    """
    Serialises the full scene state to a .rtt JSON file.

    element_configs : list of {'config': {'name', 'class', 'params'}} dicts
    bundle_configs  : list of {'N_rays': int, 'config': {...}} dicts
    settings        : {'device': str, 'Nbounces': int}
    """
    data = {
        "version": "1.0",
        "settings": settings,
        "elements": element_configs,
        "bundles": bundle_configs,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def load_project(path: str) -> dict:
    """
    Deserialises a .rtt JSON file.
    Returns {'version', 'settings', 'elements', 'bundles'}.
    Raises ValueError if the file is missing the version key.
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if "version" not in data:
        raise ValueError(f"Not a valid .rtt project file: {os.path.basename(path)}")
    return migrate_project(data)


def migrate_project(data: dict) -> dict:
    """
    Upgrades older project schemas to the current format.
    Add elif branches here when the schema version changes.
    """
    version = data.get("version", "0.0")

    if version == "1.0":
        return data

    raise ValueError(f"Unsupported project version: {version}")
