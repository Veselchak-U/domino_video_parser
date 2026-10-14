"""Resolve observed placements against a unique legal journal before decoding more frames."""

from copy import deepcopy
from itertools import permutations, product
from math import factorial

from .stone_recovery import StoneRecovery

DECK = frozenset(f"{a}-{b}" for a in range(7) for b in range(a, 7))
NAMES = [f"Seat {i}" for i in range(1, 5)]
SEARCH_LIMIT = 128


def _terminal_slot(raw: dict, observations: list, missing: str) -> dict | None:
    """Require independent before/after evidence, not just a last turn indicator."""
    if raw.get("unresolved") or raw.get("issues") or not raw.get("events"):
        return None
    end = raw.get("end")
    if end is None or not all(raw.get("remaining_confirmed", [])):
        return None
    empty = [seat for seat, hand in enumerate(raw["remaining"]) if not hand]
    if len(empty) != 1:
        return None
    seat = empty[0]
    previous = max(event["time"] for event in raw["events"])
    before = [
        obs
        for obs in observations
        if previous <= obs.time < end
        and not obs.reveal
        and obs.supported
        and obs.hands[seat] == [missing]
        and (obs.hand_regions_valid is None or obs.hand_regions_valid[seat])
    ]
    after = [
        obs
        for obs in observations
        if end <= obs.time
        and obs.reveal
        and obs.supported
        and not obs.hands[seat]
        and obs.reveal_valid is not None
        and obs.reveal_valid[seat]
    ]
    # Do not carry reveal data into another round or accept one animation frame.
    boundary = next(
        (obs.time for obs in observations if obs.time > end and not obs.reveal and obs.board),
        float("inf"),
    )
    after = [obs for obs in after if obs.time < boundary]
    if len({obs.time for obs in before}) < 2 or len({obs.time for obs in after}) < 2:
        return None
    interval = [before[-1].time, end]
    return dict(
        time=interval[0],
        time_is_anchor=True,
        interval=interval,
        seat=seat,
        seats=[seat],
        action=None,
        stone=None,
        candidates=[missing],
        logical_terminal_evidence=dict(
            before=[missing],
            after=[],
            before_times=[o.time for o in before],
            after_times=[o.time for o in after],
        ),
    )


def _variants(raw: dict, observations: list) -> list[dict]:
    if not raw.get("complete") or raw.get("remaining") is None:
        return []
    candidate = deepcopy(raw)
    used = [e["stone"] for e in candidate["events"] if e.get("stone") is not None]
    used += [stone for hand in candidate["remaining"] for stone in hand]
    if len(used) != len(set(used)):
        return []
    missing = sorted(DECK - set(used))
    if not missing:
        return [candidate]
    if not all(candidate.get("remaining_confirmed", [False] * 4)):
        return []
    unknown = [e for e in candidate["events"] if e.get("stone") is None]
    if len(missing) == 1:
        # The legacy visual slot remains usable. A new terminal slot needs an
        # independently visible one-tile hand and a fully visible empty reveal.
        if not unknown and not candidate.get("unresolved"):
            recovery = StoneRecovery()
            proofs = [
                proof
                for proof in candidate.get("remaining_evidence", [])
                if proof.get("method") == "zero_reveal_points"
            ]
            final_empty = recovery._final_empty_evidence(
                candidate, candidate["events"], missing, proofs
            )
            if final_empty is not None:
                if not recovery.exclusion(candidate, candidate["events"], missing):
                    return []
                candidate["events"][-1]["time_is_anchor"] = True
                return [candidate]
            slot = _terminal_slot(candidate, observations, missing[0])
            if slot is None:
                return []
            candidate["events"].append(slot)
        if StoneRecovery().exclusion(candidate, candidate["events"], missing):
            return [candidate]
        return []
    if len(unknown) != len(missing) or factorial(len(missing)) > SEARCH_LIMIT:
        return []
    result = []
    for values in permutations(missing):
        if any(
            e.get("candidates") and value not in e["candidates"]
            for e, value in zip(unknown, values)
        ):
            continue
        variant = deepcopy(candidate)
        for event, value in zip((e for e in variant["events"] if e["stone"] is None), values):
            event["stone"] = value
        result.append(variant)
    return result


def resolve_logically(
    reconstructor, rounds: list[dict], observations: list, variant: str, limit: int
) -> bool:
    """Commit only unique legal completed prefixes; failed attempts never mutate evidence."""
    from .reconstruct import ReconstructionError

    choices = [_variants(raw, observations) for raw in rounds]
    resolved = False
    # A later unfinished/unreadable round must not force recovery for a uniquely
    # validated earlier prefix. Cross-round starter/scoring checks still apply.
    for count in range(len(rounds), 0, -1):
        selected = choices[:count]
        if not all(selected):
            continue
        combinations = 1
        for options in selected:
            combinations *= len(options)
        if combinations > SEARCH_LIMIT:
            continue
        valid = []
        for combination in product(*selected):
            candidate = deepcopy(list(combination))
            try:
                game = reconstructor.build(candidate, NAMES, variant, limit, require_finished=False)
            except (ReconstructionError, ValueError, KeyError, TypeError):
                continue
            valid.append((candidate, game))
        if len(valid) != 1:
            continue
        candidates, game = valid[0]
        for raw, candidate, calculated in zip(rounds, candidates, game["rounds"]):
            moves = [move for move in calculated["moves"] if move["action"] != "pass"]
            if len(moves) != len(candidate["events"]):
                return False
            for event, move in zip(candidate["events"], moves):
                event.update(
                    seat=NAMES.index(move["player"]),
                    seats=[NAMES.index(move["player"])],
                    action=move["action"],
                    orientation=move["stone"],
                )
            known = {event["stone"] for event in candidate["events"]}
            candidate["unresolved"] = [
                e for e in candidate.get("unresolved", []) if e.get("stone") not in known
            ]
            candidate["logically_resolved"] = True
            candidate.setdefault("logical_resolution", []).append(
                dict(
                    stage="logical",
                    status="rules_validated",
                    interval=[raw["start"], raw["end"]],
                    evidence=dict(played=sorted(known), remaining=deepcopy(raw["remaining"])),
                    fallback_reason=None,
                )
            )
            raw.clear()
            raw.update(candidate)
        resolved = count == len(rounds)
        break
    return resolved
