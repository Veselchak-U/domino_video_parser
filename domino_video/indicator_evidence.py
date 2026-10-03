"""Keep a weak visual cue separate from proof of the player who moved."""


def mark_weak_indicator(rnd: dict, observations) -> None:
    if any(
        o.supported
        and not o.reveal
        and o.active_method in {"timer", "timer_absent", "ambiguous_timer"}
        for o in observations
    ):
        rnd["digital_indicator_observed"] = True
    conflicts = {
        o.time
        for o in observations
        if o.supported and not o.reveal and o.active_method == "ambiguous_timer"
    }
    if conflicts:
        rnd["timer_conflicts"] = sorted(set(rnd.get("timer_conflicts", [])) | conflicts)
    readings = {
        (o.time, o.active, o.timer_value)
        for o in observations
        if o.supported
        and not o.reveal
        and o.active_method == "timer"
        and o.active is not None
        and getattr(o, "timer_value", None) is not None
    }
    if readings:
        readings.update(tuple(row) for row in rnd.get("timer_evidence", []))
        rnd["timer_evidence"] = [list(row) for row in sorted(readings)]
    times = {
        o.time
        for o in observations
        if o.supported
        and not o.reveal
        and o.active is not None
        and o.active_method == "avatar_color"
    }
    if not times:
        return
    rnd["indicator_unreliable"] = True
    evidence = rnd.setdefault("indicator_evidence", {"method": "avatar_color", "times": []})
    evidence["times"] = sorted(set(evidence["times"]) | times)


def preserve_player_candidates(rnd: dict) -> None:
    independent = {
        proof["stone"]: proof["seat"] - 1
        for proof in rnd.get("stone_recovery", [])
        if proof.get("method") == "hand_difference" and proof.get("seat") is not None
    }
    for event in rnd.get("events", []) + rnd.get("unresolved", []):
        proof = event.get("recovery", {})
        if proof.get("method") == "hand_difference":
            continue
        event.pop("timer_confirmed", None)
        when = event.get("time")
        readings = [
            r
            for r in rnd.get("timer_evidence", [])
            if when is not None and when - 1.5 <= r[0] <= when - 0.3
        ]
        reliable = (
            len(readings) >= 2
            # A late tile reading can occur during the next player's countdown.
            # Keep counter checks while the placement interval is unresolved.
            and not event.get("early_reading_interval")
            and proof.get("evidence", {}).get("motion") == "incoming"
            and len({r[1] for r in readings}) == 1
            and readings[-1][0] >= when - 0.9
            and not any(readings[0][0] <= t <= when for t in rnd.get("timer_conflicts", []))
            and not any(
                when - 0.3 < r[0] <= when and r[1] != readings[-1][1]
                for r in rnd.get("timer_evidence", [])
            )
            and all(0 <= a[2] - b[2] <= 2 for a, b in zip(readings, readings[1:]))
        )
        if event.get("stone") in independent:
            seat = independent[event["stone"]]
            event["seat"], event["seats"] = seat, [seat]
        elif reliable:
            seat = readings[-1][1]
            event["seat"], event["seats"] = seat, [seat]
            event["timer_confirmed"] = True
        elif rnd.get("indicator_unreliable") or rnd.get("digital_indicator_observed"):
            event["seat"], event["seats"] = None, [0, 1, 2, 3]
        if proof:
            proof["seat"] = event["seat"] + 1 if event["seat"] is not None else None
