"""Validate extracted real preference candidates and record an audit trail."""

from __future__ import annotations

import argparse
import hashlib
import json
from copy import deepcopy
from pathlib import Path


VALID_DECISIONS = {"approve", "reject"}
ACCEPTED_INTENTIONAL_REASONS = {
    "Avoid Intersection",
    "Road Blockage/Hazard (Flood, Accident, Poor road condition)",
    "Shortcut/Faster Route/Personal Preference/Familiar Road",
    "Traffic Congestion",
}


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _candidate_sha256(record: dict) -> str:
    """Hash the candidate itself, excluding fields added during review."""
    candidate = {key: value for key, value in record.items() if key not in {"status", "review_note"}}
    canonical = json.dumps(candidate, sort_keys=True, separators=(",", ":")).encode()
    return _sha256_bytes(canonical)


def _review_collection(document: dict) -> tuple[str, str, list[dict]]:
    if "candidates" in document:
        return "candidates", "event_key", document["candidates"]
    if "examples" in document:
        return "examples", "example_key", document["examples"]
    raise ValueError("Artifact contains neither candidates nor decision examples")


def create_decision_template(candidate_bytes: bytes, previous_approved_bytes: bytes | None = None) -> dict:
    document = json.loads(candidate_bytes)
    _, key_field, records = _review_collection(document)
    previous = {}
    if previous_approved_bytes is not None:
        approved_document = json.loads(previous_approved_bytes)
        if approved_document.get("status") != "approved_for_training":
            raise ValueError("Carry-forward input is not an approved training artifact")
        _, previous_key, previous_records = _review_collection(approved_document)
        if previous_key != key_field:
            raise ValueError("Carry-forward artifact uses a different candidate type")
        previous = {record[key_field]: record for record in previous_records}

    review_rows = []
    carried_forward = 0
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
        decision = "pending"
        note = ""
        prior = previous.get(record[key_field])
        if (prior is not None and _candidate_sha256(prior) == _candidate_sha256(record)
                and prior.get("status") in {"approved", "rejected"}):
            decision = "approve" if prior["status"] == "approved" else "reject"
            note = str(prior.get("review_note", "")).strip()
            carried_forward += 1
        review_rows.append({
            key_field: record[key_field],
            "candidate_sha256": _candidate_sha256(record),
            "review_context": context,
            "decision": decision,
            "review_note": note,
        })
    return {
        "schema_version": 2,
        "candidate_artifact_sha256": _sha256_bytes(candidate_bytes),
        "carried_forward_from_sha256": (_sha256_bytes(previous_approved_bytes)
                                          if previous_approved_bytes is not None else None),
        "instructions": "Review every pending row. Set it to approve or reject and add a short note.",
        "summary": {"candidates": len(records), "carried_forward": carried_forward,
                    "pending": len(records) - carried_forward},
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
        "carried_forward": int(decisions.get("summary", {}).get("carried_forward", 0)),
        "carried_forward_from_sha256": decisions.get("carried_forward_from_sha256"),
        "method": "manual path and event-context review",
    }
    return approved_document


def _path_is_connected(path: dict) -> bool:
    edges = path.get("edges", [])
    return bool(edges) and all(
        str(left.get("v")) == str(right.get("u"))
        for left, right in zip(edges, edges[1:])
    )


def _automatic_candidate_decision(record: dict) -> tuple[str, str]:
    """Apply conservative survey and geometry rules without subjective scoring."""
    target = record.get("target", {})
    label = target.get("label")
    suggested = record.get("suggested_path", {})
    observed = record.get("observed_path_label_evidence", {})
    if label not in {"followed", "deviated"}:
        return "reject", "automatic: missing or unsupported route-choice label"
    if not _path_is_connected(suggested) or not _path_is_connected(observed):
        return "reject", "automatic: suggested or observed edge sequence is empty or disconnected"

    common_od = record.get("common_od", {})
    for path_name, path in (("suggested", suggested), ("observed", observed)):
        edges = path["edges"]
        if (str(edges[0].get("u")) != str(common_od.get("origin_node_id"))
                or str(edges[-1].get("v")) != str(common_od.get("destination_node_id"))):
            return "reject", f"automatic: {path_name} path does not match the declared common endpoints"

    survey = record.get("survey_evidence", {})
    quality = record.get("quality", {})
    if label == "followed":
        if any(int(survey.get(field, 0) or 0) != 0 for field in (
                "reported_deviation_count", "intentional_deviation_count", "missing_response_count")):
            return "reject", "automatic: followed example has conflicting deviation evidence"
        if float(quality.get("observed_edge_agreement", 0.0) or 0.0) < 0.8:
            return "reject", "automatic: GPS agreement is below the followed-example threshold"
        if float(quality.get("gps_mean_match_distance_m", float("inf"))) > 20.0:
            return "reject", "automatic: mean GPS map-match distance exceeds 20 m"
        return "approve", "automatic: no deviation report and GPS follows the suggested branch"

    reason = str(survey.get("primary_reason", "")).strip()
    if reason not in ACCEPTED_INTENTIONAL_REASONS:
        return "reject", f"automatic: excluded or unsupported deviation reason: {reason or 'missing'}"
    if reason == "Traffic Congestion":
        severity = str(survey.get("traffic_severity", "")).strip()
        if not severity or severity[0] not in "12345":
            return "reject", "automatic: traffic deviation has no valid severity response"
    if float(quality.get("gps_edge_overlap_with_regenerated", 0.0) or 0.0) <= float(
            quality.get("gps_edge_overlap_with_prior", 0.0) or 0.0):
        return "reject", "automatic: GPS does not support the observed alternative over the suggestion"
    if float(quality.get("mean_gps_to_matched_edge_m", float("inf"))) > 20.0:
        return "reject", "automatic: mean GPS map-match distance exceeds 20 m"
    return "approve", f"automatic: intentional {reason.lower()} with GPS-supported divergence and rejoin"


def apply_automatic_rules(candidate_bytes: bytes) -> dict:
    """Create a training artifact using reproducible survey and geometry rules."""
    document = json.loads(candidate_bytes)
    collection_field, _, records = _review_collection(document)
    approved_document = deepcopy(document)
    approved_count = 0
    reasons: dict[str, int] = {}
    for candidate in approved_document[collection_field]:
        decision, note = _automatic_candidate_decision(candidate)
        candidate["status"] = "approved" if decision == "approve" else "rejected"
        candidate["review_note"] = note
        approved_count += int(decision == "approve")
        reasons[note] = reasons.get(note, 0) + 1
    if approved_count == 0:
        raise ValueError("Automatic validation rejected every candidate")
    approved_document["status"] = "approved_for_training"
    approved_document["review"] = {
        "candidate_artifact_sha256": _sha256_bytes(candidate_bytes),
        "approved": approved_count,
        "rejected": len(records) - approved_count,
        "method": "automatic survey, GPS, directed-path and common-endpoint validation",
        "rules_version": 1,
        "decision_counts": reasons,
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
    initialize.add_argument("--carry-forward", type=Path,
                            help="Reuse decisions for unchanged candidates from an earlier approved artifact")
    finalize = subparsers.add_parser("finalize", help="Apply completed decisions")
    finalize.add_argument("candidates", type=Path)
    finalize.add_argument("decisions", type=Path)
    finalize.add_argument("--output", type=Path, required=True)
    automatic = subparsers.add_parser("auto", help="Approve or reject with reproducible data rules")
    automatic.add_argument("candidates", type=Path)
    automatic.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    candidate_bytes = args.candidates.read_bytes()
    if args.command == "init":
        previous = args.carry_forward.read_bytes() if args.carry_forward else None
        result = create_decision_template(candidate_bytes, previous)
    elif args.command == "auto":
        result = apply_automatic_rules(candidate_bytes)
    else:
        decision_document = json.loads(args.decisions.read_text(encoding="utf-8"))
        result = apply_decisions(candidate_bytes, decision_document)
    _atomic_json_write(args.output, result)
    print(f"Review file saved to {args.output}")
    if args.command == "init":
        summary = result["summary"]
        print(f"Candidates: {summary['candidates']} | carried forward: {summary['carried_forward']} | "
              f"pending review: {summary['pending']}")
    else:
        review = result["review"]
        print(f"Status: {result['status']} | approved: {review['approved']} | rejected: {review['rejected']}")
        print("Only approved rows will be loaded for training.")


if __name__ == "__main__":
    main()
