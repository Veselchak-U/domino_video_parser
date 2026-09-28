import pytest


def test_replaces_event_and_rejects_wrong_hash():
    from domino_video.corrections import Corrections

    raw = [
        {
            "events": [{"seat": 0, "action": "start", "stone": "1-1", "time": 1}],
            "remaining": [[], [], [], []],
        }
    ]
    document = {
        "sources": [
            {
                "sha256": "abc",
                "games": [
                    {
                        "number": 1,
                        "rounds": [
                            {
                                "number": 1,
                                "events": [
                                    {
                                        "op": "replace",
                                        "index": 1,
                                        "value": {
                                            "seat": 1,
                                            "action": "start",
                                            "stone": "1-1",
                                            "time": 1,
                                        },
                                    }
                                ],
                            }
                        ],
                    }
                ],
            }
        ]
    }
    assert Corrections(document).apply("abc", 1, raw)[0][0]["events"][0]["seat"] == 1
    assert raw[0]["events"][0]["seat"] == 0
    with pytest.raises(ValueError):
        Corrections(document).check_sources({"other"})


def test_rejects_conflicting_operations():
    from domino_video.corrections import Corrections

    operation = {"op": "delete", "index": 1}
    document = {
        "sources": [
            {
                "sha256": "abc",
                "games": [
                    {"number": 1, "rounds": [{"number": 1, "events": [operation, operation]}]}
                ],
            }
        ]
    }
    with pytest.raises(ValueError):
        Corrections(document).apply("abc", 1, [{"events": [{"stone": "1-1"}]}])


@pytest.mark.parametrize("index", [0, 2, 100])
def test_rejects_unknown_event_reference(index):
    from domino_video.corrections import Corrections

    document = {
        "sources": [
            {
                "sha256": "abc",
                "games": [
                    {
                        "number": 1,
                        "rounds": [{"number": 1, "events": [{"op": "delete", "index": index}]}],
                    }
                ],
            }
        ]
    }
    with pytest.raises(ValueError):
        Corrections(document).apply("abc", 1, [{"events": [{"stone": "1-1"}]}])
