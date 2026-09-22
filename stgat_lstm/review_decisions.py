"""Record auditable human decisions for extracted real preference candidates."""

from __future__ import annotations

import argparse
import hashlib
import json
from copy import deepcopy
from pathlib import Path


VALID_DECISIONS = {"approve", "reject"}


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _review_collection(document: dict) -> tuple[str, str, list[dict]]:
    if "candidates" in document:
        return "candidates", "event_key", document["candidates"]
    if "examples" in document:
        return "examples", "example_key", document["examples"]
    raise ValueError("Artifact contains neither candidates nor decision examples")


def create_decision_template(candidate_bytes: bytes) -> dict:
    document = json.loads(candidate_bytes)
    _, key_field, records = _review_collection(document)
    review_rows = []
    for index, record in enumerate(records, start=1):
        if "target" in record:
            context = {
                "candidate_number": index,
                "expected_label": record["target"].get("label"),
                "label_basis": record["target"].get("basis"),
                "suggested_roads": record.get("suggested_path", {}).get("road_names", []),
                "suggested_length_m": record.get("suggested_path", {}).get("length_m"),
                "observed_roads": record.get("observed_path_label_evidence", {}).get("road_names", []),
                "observed_length_m": record.get("observed_path_label_evidence", {}).get("length_m"),
                "quality": record.get("quality", {}),
            }
        else:
            context = {
                "candidate_number": index,
                "expected_label": "deviation preference pair",
                "label_basis": record.get("survey", {}).get("primary_reason"),
                "suggested_roads": record.get("rejected_prior_suggestion", {}).get("road_names", []),
                "suggested_length_m": record.get("rejected_prior_suggestion", {}).get("length_m"),
                "observed_roads": record.get("preferred_observed_path", {}).get("road_names", []),
                "observed_length_m": record.get("preferred_observed_path", {}).get("length_m"),
                "quality": record.get("quality", {}),
            }
        review_rows.append(
            {
                key_field: record[key_field],
                "review_context": context,
                "decision": "pending",
                "review_note": "",
            }
        )
    return {
        "schema_version": 1,
        "candidate_artifact_sha256": _sha256_bytes(candidate_bytes),
        "instructions": "Set every decision to approve or reject and add a short review note.",
        "decisions": review_rows,
    }


def apply_decisions(candidate_bytes: bytes, decisions: dict) -> dict:
    if decisions.get("candidate_artifact_sha256") != _sha256_bytes(candidate_bytes):
        raise ValueError("Review decisions belong to a different candidate artifact")
    document = json.loads(candidate_bytes)
    collection_field, key_field, records = _review_collection(document)
    candidates = {candidate[key_field]: candidate for candidate in records}
    provided = decisions.get("decisions", [])
    if len(provided) != len(candidates) or {row.get(key_field) for row in provided} != set(candidates):
        raise ValueError("Review decisions must cover every candidate exactly once")
    if any(row.get("decision") not in VALID_DECISIONS for row in provided):
        raise ValueError("Every candidate decision must be approve or reject")

    approved_document = deepcopy(document)
    decisions_by_event = {row[key_field]: row for row in provided}
    approved_count = 0
    for candidate in approved_document[collection_field]:
        review = decisions_by_event[candidate[key_field]]
        candidate["status"] = "approved" if review["decision"] == "approve" else "rejected"
        candidate["review_note"] = str(review.get("review_note", "")).strip()
        approved_count += int(candidate["status"] == "approved")
    if approved_count == 0:
        raise ValueError("At least one candidate must be approved to create a training artifact")
    approved_document["status"] = "approved_for_training"
    approved_document["review"] = {
        "candidate_artifact_sha256": decisions["candidate_artifact_sha256"],
        "approved": approved_count,
        "rejected": len(candidates) - approved_count,
        "method": "manual path and event-context review",
    }
    return approved_document


def _atomic_json_write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    initialize = subparsers.add_parser("init", help="Create a pending decision template")
    initialize.add_argument("candidates", type=Path)
    initialize.add_argument("--output", type=Path, required=True)
    finalize = subparsers.add_parser("finalize", help="Apply completed decisions")
    finalize.add_argument("candidates", type=Path)
    finalize.add_argument("decisions", type=Path)
    finalize.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    candidate_bytes = args.candidates.read_bytes()
    if args.command == "init":
        result = create_decision_template(candidate_bytes)
    else:
        decision_document = json.loads(args.decisions.read_text(encoding="utf-8"))
        result = apply_decisions(candidate_bytes, decision_document)
    _atomic_json_write(args.output, result)
    print(json.dumps({"output": str(args.output), "status": result.get("status", "pending_review")}, indent=2))


if __name__ == "__main__":
    main()
