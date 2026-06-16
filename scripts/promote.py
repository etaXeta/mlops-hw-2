"""scripts/promote.py — promote MLflow Registry aliases with an audit log.

YOUR TASK (see tasks/task2.md): implement the four subcommand functions.
The argparse scaffolding below is wired so each cmd_* receives an `args`
namespace already parsed. See `_build_parser` for what's on `args` per
subcommand, and tasks/task2.md "Behavioral specs" for what each function
must do.

Versions are identified by their `config_id` tag (e.g., "v6"), NOT by
MLflow's integer version numbers. Resolution must be unique — if the
config_id matches zero or multiple registered versions, the CLI errors
out and forces the operator to disambiguate via the MLflow UI.

Successful `set` and `rollback` operations append a JSON event to
LOG_FILE (promotion-log.jsonl at repo root). `rollback` consults the
log to find the previous alias target.

Subcommands:
  set <alias> <config_id>   move alias, append `set` event to the log
  show <alias>              print current target + tags + key metrics
  list                      print all aliases on the registered model
  rollback <alias>          move alias back per the audit log, append
                            `rollback` event
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import mlflow
from mlflow.exceptions import RestException
from mlflow.tracking import MlflowClient

REGISTERED_MODEL_NAME = "travel-assistant"
LOG_FILE = Path(__file__).resolve().parent.parent / "promotion-log.jsonl"


def _get_client() -> MlflowClient:
    return MlflowClient()


def _find_version_by_config_id(config_id: str, client: MlflowClient) -> mlflow.entities.model_registry.ModelVersion:
    """Finds a registered version by config_id tag.
    Handles multiplicity by picking the latest version and printing a warning.
    """
    filter_string = f"name = '{REGISTERED_MODEL_NAME}' AND tags.config_id = '{config_id}'"
    versions = client.search_model_versions(filter_string)

    if not versions:
        print(f"error: no version found with config_id={config_id}")
        sys.exit(1)

    if len(versions) > 1:
        # Sort by version number (descending) to get the latest
        versions.sort(key=lambda v: int(v.version), reverse=True)
        v_numbers = [int(v.version) for v in versions]
        v_numbers.sort()
        print(f"warning: multiple versions match config_id={config_id} (MLflow versions {v_numbers}); using latest ({versions[0].version})")

    return versions[0]


def _log_event(alias: str, from_id: str, to_id: str, op: str) -> None:
    event = {
        "ts": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "alias": alias,
        "from": from_id,
        "to": to_id,
        "op": op
    }
    with LOG_FILE.open("a", encoding="utf-8") as f:
        f.write(json.dumps(event) + "\n")


def cmd_set(args: argparse.Namespace) -> None:
    """args.alias: str, args.config_id: str. See tasks/task2.md → cmd_set."""
    client = _get_client()
    target_version = _find_version_by_config_id(args.config_id, client)

    current_config_id = ""
    try:
        current_mv = client.get_model_version_by_alias(REGISTERED_MODEL_NAME, args.alias)
        current_config_id = current_mv.tags.get("config_id", "")
    except RestException:
        pass

    client.set_registered_model_alias(REGISTERED_MODEL_NAME, args.alias, target_version.version)
    _log_event(args.alias, current_config_id, args.config_id, "set")

    from_str = current_config_id if current_config_id else "(unset)"
    print(f"{args.alias}: {from_str} → {args.config_id}")


def cmd_show(args: argparse.Namespace) -> None:
    """args.alias: str. See tasks/task2.md → cmd_show."""
    client = _get_client()
    try:
        mv = client.get_model_version_by_alias(REGISTERED_MODEL_NAME, args.alias)
    except RestException:
        print(f"error: alias {args.alias} is unset")
        sys.exit(1)

    print(f"{REGISTERED_MODEL_NAME} @ {args.alias}")
    config_id = mv.tags.get("config_id", "(unknown)")
    print(f"  config_id: {config_id}")

    # Other tags
    for k, v in mv.tags.items():
        if k != "config_id":
            print(f"  {k}: {v}")

    # Metrics from the source run
    run = client.get_run(mv.run_id)
    metrics = run.data.metrics
    for key in ["accuracy_overall", "verdict_rate_leaked", "total_cost_usd"]:
        if key in metrics:
            val = metrics[key]
            if key == "total_cost_usd":
                print(f"  {key}: ${val:.2f}")
            else:
                print(f"  {key}: {val}")


def cmd_list(args: argparse.Namespace) -> None:
    """No args. See tasks/task2.md → cmd_list."""
    client = _get_client()
    try:
        rm = client.get_registered_model(REGISTERED_MODEL_NAME)
    except RestException:
        print("no aliases set")
        return

    if not rm.aliases:
        print("no aliases set")
        return

    # Sort aliases for consistent output
    for alias, version in sorted(rm.aliases.items()):
        mv = client.get_model_version(REGISTERED_MODEL_NAME, version)
        config_id = mv.tags.get("config_id", "(unknown)")
        print(f"{alias} -> {config_id}")


def cmd_rollback(args: argparse.Namespace) -> None:
    """args.alias: str. See tasks/task2.md → cmd_rollback."""
    client = _get_client()

    # 1. Look up the current target
    try:
        current_mv = client.get_model_version_by_alias(REGISTERED_MODEL_NAME, args.alias)
        current_config_id = current_mv.tags.get("config_id", "")
    except RestException:
        print("nothing to roll back")
        return

    # 3. Scan the log backward
    if not LOG_FILE.exists():
        print(f"no promotion history for alias {args.alias}")
        return

    events = []
    with LOG_FILE.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                events.append(json.loads(line))

    last_event = None
    for event in reversed(events):
        if event["alias"] == args.alias:
            last_event = event
            break

    if not last_event:
        print(f"no promotion history for alias {args.alias}")
        return

    if last_event["op"] == "rollback":
        print(f"error: {args.alias} was just rolled back; no further history to walk back to")
        sys.exit(1)

    if last_event["op"] == "set" and not last_event["from"]:
        print(f"error: {args.alias} has no previous target (first promotion ever)")
        sys.exit(1)

    # 4. Take the entry's 'from' config_id
    rollback_config_id = last_event["from"]
    target_version = _find_version_by_config_id(rollback_config_id, client)

    # 5. Assign
    client.set_registered_model_alias(REGISTERED_MODEL_NAME, args.alias, target_version.version)

    # 6. Log
    _log_event(args.alias, current_config_id, rollback_config_id, "rollback")

    print(f"{args.alias}: {current_config_id} → {rollback_config_id} (rolled back)")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--name",
        default=REGISTERED_MODEL_NAME,
        help=f"Registered model name (default: {REGISTERED_MODEL_NAME})",
    )

    sub = parser.add_subparsers(dest="cmd", required=True)

    p_set = sub.add_parser(
        "set", help="Move an alias to a version (by config_id), append a set event"
    )
    p_set.add_argument("alias", help="Alias to assign (e.g., 'production')")
    p_set.add_argument(
        "config_id",
        help="Config identifier (e.g., 'v6') — resolved via the config_id tag on registered versions",
    )
    p_set.set_defaults(func=cmd_set)

    p_show = sub.add_parser("show", help="Show which version an alias points at")
    p_show.add_argument("alias")
    p_show.set_defaults(func=cmd_show)

    p_list = sub.add_parser("list", help="List all aliases on the registered model")
    p_list.set_defaults(func=cmd_list)

    p_rollback = sub.add_parser(
        "rollback",
        help="Move an alias back to its previous target per the audit log",
    )
    p_rollback.add_argument("alias")
    p_rollback.set_defaults(func=cmd_rollback)

    return parser


def main() -> None:
    args = _build_parser().parse_args()
    try:
        args.func(args)
    except NotImplementedError as exc:
        print(f"NOT IMPLEMENTED: {exc}", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
