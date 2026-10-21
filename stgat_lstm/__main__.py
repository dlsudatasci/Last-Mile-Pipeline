"""Single command entry point for the rider-preference pipeline."""

from __future__ import annotations

import importlib
import sys


COMMANDS = {
    "audit-data": ("preprocessing.audit_rider_data", "inspect exported rider data"),
    "audit-network": ("preprocessing.audit_road_network", "check the OSM graph and feature join"),
    "audit-matching": ("preprocessing.audit_map_matching", "diagnose GPS map matching"),
    "build-deviations": ("preprocessing.build_deviation_candidates", "build strict deviation candidates"),
    "build-decisions": ("preprocessing.build_decision_candidates", "build follow/deviate candidates"),
    "review-map": ("preprocessing.create_review_map", "create the local review map"),
    "review": ("preprocessing.review_decisions", "initialize or finalize review decisions"),
    "traffic": ("preprocessing.mapbox_traffic", "collect one Mapbox traffic observation"),
    "gps-traffic": ("preprocessing.gps_traffic", "derive historical road-speed context from rider GPS"),
    "synthetic-hotspot": ("generation.synthetic_hotspot", "build a controlled Taft learning experiment"),
    "train": ("modeling.train_model", "train the outgoing-road choice model"),
    "evaluate": ("modeling.evaluate_model", "run rider-disjoint evaluation"),
    "predict": ("routing.predict_route", "predict preferred roads and generate a route"),
    "dashboard": ("routing.dashboard", "serve the local routing dashboard"),
}


def _usage() -> str:
    width = max(len(command) for command in COMMANDS)
    rows = ["Usage: python -m stgat_lstm <command> [arguments]", "", "Commands:"]
    rows.extend(
        f"  {command:<{width}}  {description}"
        for command, (_, description) in COMMANDS.items()
    )
    rows.extend(["", "Use: python -m stgat_lstm <command> --help"])
    return "\n".join(rows)


def main() -> None:
    if len(sys.argv) < 2 or sys.argv[1] in {"-h", "--help"}:
        print(_usage())
        return
    command = sys.argv[1]
    if command not in COMMANDS:
        print(f"Unknown command: {command}\n\n{_usage()}", file=sys.stderr)
        raise SystemExit(2)
    module_name = COMMANDS[command][0]
    module = importlib.import_module(f"{__package__}.{module_name}")
    sys.argv = [f"python -m stgat_lstm {command}", *sys.argv[2:]]
    module.main()


if __name__ == "__main__":
    main()
