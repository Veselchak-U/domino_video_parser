"""Independent evidence of a fully dealt round before its first placement."""

from collections import defaultdict


def start_windows(rounds):
    windows = []
    previous_end = 0.0
    for rnd in rounds:
        first = min((e["time"] for e in rnd["events"]), default=rnd["start"])
        windows.append((max(previous_end, first - 2.0), first))
        previous_end = rnd.get("end") or first
    return windows


def _candidates(observations, window):
    start, stop = window
    return sorted(
        (
            o
            for o in observations
            if start < o.time < stop
            and o.supported
            and not o.reveal
            and not o.result_table
            and not o.board
            and not o.uncertain_board
            and (o.hand_regions_valid is None or o.hand_regions_valid[0])
            and len(set(o.hands[0])) == 7
        ),
        key=lambda o: o.time,
    )


def start_requests(rounds, observations):
    # The last empty-looking frames can already contain an incoming first tile.
    # Prefer the earliest complete own hand after the deal animation.
    return {
        o.time for window in start_windows(rounds) for o in _candidates(observations, window)[:3]
    }


def confirm_round_starts(rounds, observations):
    for rnd, window in zip(rounds, start_windows(rounds)):
        hands = defaultdict(list)
        for obs in _candidates(observations, window):
            if obs.counts is not None and tuple(obs.counts) == (7, 7, 7, 7):
                hands[tuple(sorted(obs.hands[0]))].append(obs.time)
        confirmed = [(hand, times) for hand, times in hands.items() if len(set(times)) >= 2]
        if len(confirmed) == 1:
            hand, times = confirmed[0]
            rnd["start_observed"] = True
            rnd["initial_hand"] = list(hand)
            rnd["start_evidence"] = dict(
                method="full_deal", times=sorted(set(times)), counts=[7] * 4
            )
