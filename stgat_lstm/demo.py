"""Run the isolated synthetic training and learned-cost routing demonstration."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

from .features import build_model_input
from .fixture import prepare_choices, read_fixture
from .routing import shortest_preference_path
from .training import train_preference_model


def run_demo(fixture_path: Path, epochs: int, output_dir: Path | None) -> dict:
    torch.set_num_threads(1)
    graph, scenarios = read_fixture(fixture_path)
    choices = prepare_choices(graph, scenarios)
    fixture_hash = hashlib.sha256(fixture_path.read_bytes()).hexdigest()
    report = {
        "provenance": "SYNTHETIC ONLY - in-sample software demonstration, no real rider validation",
        "fixture_sha256": fixture_hash,
        "usable_preference_pairs": sum(choice.preferred_edge_ids is not None for choice in choices),
        "excluded_or_unlabeled_cases": [choice.case_id for choice in choices if choice.preferred_edge_ids is None],
        "models": {},
    }
    for architecture in ("gcn", "gat", "stgat_lstm"):
        model, summary = train_preference_model(graph, choices, architecture=architecture, epochs=epochs)
        case_results = {}
        model.eval()
        with torch.no_grad():
            for case_id in ("syn_case_01", "syn_case_02"):
                choice = next(item for item in choices if item.case_id == case_id)
                inputs = build_model_input(graph, choice)
                scores = model(inputs)
                costs = {edge_id: float(scores[index]) for index, edge_id in enumerate(inputs.edge_ids)}
                recommended = shortest_preference_path(
                    graph, costs, choice.origin_node_id, choice.destination_node_id
                )
                case_results[case_id] = {
                    "recommended_edge_ids": list(recommended.edge_ids),
                    "learned_preference_cost": recommended.total_preference_cost,
                    "observed_edge_ids": list(choice.observed_edge_ids),
                }
        report["models"][architecture] = {
            "training_examples": summary.examples,
            "epochs": summary.epochs,
            "initial_training_loss": summary.initial_loss,
            "final_training_loss": summary.final_loss,
            "in_sample_ranking_correct": summary.in_sample_correct,
            "routes": case_results,
        }
        if output_dir is not None:
            output_dir.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "architecture": architecture,
                    "hidden_channels": model.hidden_channels,
                    "model_config": model.configuration(),
                    "fixture_sha256": fixture_hash,
                    "provenance": "synthetic-only",
                },
                output_dir / f"{architecture}_synthetic.pt",
            )
    if output_dir is not None:
        (output_dir / "synthetic_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fixture",
        type=Path,
        default=Path("data/synthetic/rider_preference_fixture.json"),
    )
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/synthetic_demo"))
    args = parser.parse_args()
    report = run_demo(args.fixture, args.epochs, args.output_dir)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
