"""Чистые операции начисления по kozel-game-v1."""


class ScoringRules:
    def hand_value(self, hand: list[str]) -> int:
        if hand == ["0-0"]:
            return 10
        return sum(int(value) for stone in hand for value in stone.split("-"))

    def game_result(self, scores: dict[str, int], limit: int) -> dict | None:
        if max(scores.values()) < limit:
            return None
        a, b = scores
        if scores[a] == scores[b]:
            return {"draw": True}
        return {"winner": min(scores, key=scores.get)}

    def settle(self, variant, reason, winner, remaining, scores, fences, eggs):
        scores, fences = dict(scores), dict(fences)
        penalty = dict.fromkeys(scores, 0)
        if reason == "fish" and variant == "withEggs":
            if len(set(remaining.values())) == 1:
                reason, winner = "eggs", None
                eggs += sum(remaining.values())
            else:
                winner = min(remaining, key=remaining.get)
        if reason != "eggs":
            for team in scores:
                if team == winner:
                    fences[team] = 0
                    continue
                own = remaining[team]
                penalty[team] = own + eggs
                if scores[team] > 0:
                    scores[team] += penalty[team]
                elif own >= 13:
                    scores[team] = fences[team] + penalty[team]
                    fences[team] = 0
                else:
                    fences[team] += penalty[team]
            eggs = 0
        result = dict(reason=reason, winner_team=winner, round_score=penalty, total_score=scores)
        if any(fences.values()):
            result["fence"] = {team: value for team, value in fences.items() if value}
        if eggs:
            result["eggs"] = eggs
        return result
