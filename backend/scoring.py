"""Rolling-window smoothing, signal bands and stability per listening session."""

import statistics
import threading
import time
from collections import OrderedDict, deque
from dataclasses import dataclass, field
from typing import Literal

import config

Band = Literal["idle", "green", "amber", "red"]

MAX_SESSIONS = 500


def band_for(score: float) -> Band:
    if score > config.BAND_RED_MIN:
        return "red"
    if score >= config.BAND_GREEN_MAX:
        return "amber"
    return "green"


def stability_of(history: list[float]) -> float:
    """1.0 = consistent readings, 0.0 = wildly varying. Needs 2+ readings to mean anything."""
    if len(history) < 2:
        return 0.0
    # Scores live in [0, 1], so population std-dev is at most 0.5.
    return round(max(0.0, 1.0 - statistics.pstdev(history) / 0.5), 2)


@dataclass
class SessionState:
    history: deque = field(default_factory=lambda: deque(maxlen=config.HISTORY_WINDOWS))
    audio: deque = field(default_factory=lambda: deque(maxlen=config.LIVE_CONTEXT_WINDOWS))  # recent windows
    band: Band = "idle"
    samples_seen: int = 0
    last_seen: float = field(default_factory=time.time)

    @property
    def smoothed(self) -> float | None:
        return sum(self.history) / len(self.history) if self.history else None

    @property
    def stability(self) -> float:
        return stability_of(list(self.history))


class SessionStore:
    def __init__(self) -> None:
        self._sessions: OrderedDict[str, SessionState] = OrderedDict()
        self._lock = threading.Lock()

    def get(self, session_id: str) -> SessionState:
        with self._lock:
            state = self._sessions.get(session_id)
            if state is None:
                state = self._sessions[session_id] = SessionState()
                while len(self._sessions) > MAX_SESSIONS:
                    self._sessions.popitem(last=False)
            self._sessions.move_to_end(session_id)
            state.last_seen = time.time()
            return state

    def push(self, session_id: str, score: float) -> SessionState:
        state = self.get(session_id)
        with self._lock:
            state.history.append(score)
            state.samples_seen += 1
            band = band_for(state.smoothed)  # type: ignore[arg-type]
            if band == "red" and len(state.history) < config.MIN_SAMPLES_FOR_RED:
                band = "amber"
            state.band = band
        return state
