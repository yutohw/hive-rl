"""Curriculum track selection: track 0 first, widening to all five tracks."""

import random


def curriculum_track_pool(episode: int, num_episodes: int) -> list[int]:
    progress = episode / num_episodes
    if progress < 0.08:
        return [0]
    elif progress < 0.16:
        return [0, 1]
    elif progress < 0.24:
        return [0, 1, 2]
    elif progress < 0.32:
        return [0, 1, 2, 3]
    else:
        return [0, 1, 2, 3, 4]


# Shuffled deck of tracks, refilled whenever it runs out or the pool changes.
_track_deck: list[int] = []
_track_deck_pool: list[int] | None = None


def next_track(episode: int, num_episodes: int) -> int:
    global _track_deck, _track_deck_pool

    pool = curriculum_track_pool(episode, num_episodes)

    if pool != _track_deck_pool or not _track_deck:
        _track_deck_pool = pool
        _track_deck = pool.copy()
        random.shuffle(_track_deck)

    return _track_deck.pop()


def next_track_batch(episode: int, num_episodes: int, n: int) -> list[int]:
    return [next_track(episode, num_episodes) for _ in range(n)]
