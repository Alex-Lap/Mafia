from __future__ import annotations
from typing import Annotated, Literal
from dataclasses import dataclass, field
import operator


# ---------------------------------------------------------------------------
# Role & Phase enums
# ---------------------------------------------------------------------------

Role = Literal["villager", "mafia", "detective", "doctor"]
Phase = Literal["setup", "day", "vote", "night", "game_over"]


# ---------------------------------------------------------------------------
# Player
# ---------------------------------------------------------------------------

@dataclass
class Player:
    id: str                          
    name: str                        
    role: Role = "villager"
    is_alive: bool = True
    model: str = "claude-haiku-4-5-20251001"       

    def __repr__(self) -> str:
        status = "alive" if self.is_alive else "dead"
        return f"Player({self.name}, {self.role}, {status})"


# ---------------------------------------------------------------------------
# Chat message
# ---------------------------------------------------------------------------

@dataclass
class Message:
    sender_id: str                   # player id or "narrator"
    content: str
    phase: Phase                     # which phase this was said in
    round: int                       # game round number
    is_public: bool = True           # False = only mafia / night actions can see


# ---------------------------------------------------------------------------
# Game state  (this is what flows through the LangGraph nodes)
# ---------------------------------------------------------------------------

@dataclass
class GameState:
    # --- Players ---
    players: list[Player] = field(default_factory=list)

    # --- Game progress ---
    phase: Phase = "setup"
    round: int = 0
    winner: Literal["mafia", "village", None] = None

    # --- Votes (player_id -> player_id they voted for) ---
    votes: Annotated[dict[str, str], operator.or_] = field(default_factory=dict)

    # --- Night actions ---
    mafia_target: str | None = None       # player_id mafia wants to kill
    doctor_save: str | None = None        # player_id doctor protects
    detective_result: str | None = None  # "mafia" or "innocent"

    # --- Chat log ---
    messages: Annotated[list[Message], operator.add] = field(default_factory=list)

    # --- Reasoning traces (for debugging / replay) ---
    reasoning: Annotated[dict[str, str], operator.or_] = field(default_factory=dict)

    # ---------------------------------------------------------------------------
    # Convenience helpers
    # ---------------------------------------------------------------------------

    def alive_players(self) -> list[Player]:
        return [p for p in self.players if p.is_alive]

    def get_player(self, player_id: str) -> Player | None:
        return next((p for p in self.players if p.id == player_id), None)

    def mafia_players(self) -> list[Player]:
        return [p for p in self.players if p.role == "mafia" and p.is_alive]

    def village_players(self) -> list[Player]:
        return [p for p in self.players if p.role != "mafia" and p.is_alive]

    def check_win_condition(self) -> Literal["mafia", "village", None]:
        """
        Mafia wins  → mafia count >= village count
        Village wins → all mafia are dead
        """
        alive_mafia = len(self.mafia_players())
        alive_village = len(self.village_players())

        if alive_mafia == 0:
            return "village"
        if alive_mafia >= alive_village:
            return "mafia"
        return None

    def public_history(self, max_messages: int = 20) -> str:
        """Return recent public messages as a readable string for LLM context."""
        public = [m for m in self.messages if m.is_public][-max_messages:]
        lines = []
        for m in public:
            lines.append(f"[Round {m.round} | {m.phase}] {m.sender_id}: {m.content}")
        return "\n".join(lines) if lines else "(no messages yet)"
    
    def validate(self) -> None:
        ids = [p.id for p in self.players]
        names = [p.name for p in self.players]
        if len(ids) != len(set(ids)):
            raise ValueError(f"Duplicate player IDs: {ids}")

    def __repr__(self) -> str:
        return (
            f"GameState(phase={self.phase}, round={self.round}, "
            f"alive={len(self.alive_players())}/{len(self.players)}, "
            f"winner={self.winner})"
        )