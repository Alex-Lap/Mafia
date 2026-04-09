from __future__ import annotations
import random
from langgraph.graph import StateGraph, END
from game.state import GameState, Player, Message, Role


# ---------------------------------------------------------------------------
# Helper: narrator message
# ---------------------------------------------------------------------------

def narrate(state: GameState, text: str) -> Message:
    return Message(
        sender_id="narrator",
        content=text,
        phase=state.phase,
        round=state.round,
        is_public=True,
    )


# ---------------------------------------------------------------------------
# Node: setup
# ---------------------------------------------------------------------------

ROLE_DISTRIBUTION: dict[int, list[Role]] = {
    4: ["mafia", "detective", "villager", "villager"],
    5: ["mafia", "detective", "doctor", "villager", "villager"],
    6: ["mafia", "mafia", "detective", "doctor", "villager", "villager"],
    7: ["mafia", "mafia", "detective", "doctor", "villager", "villager", "villager"],
    8: ["mafia", "mafia", "detective", "doctor", "villager", "villager", "villager", "villager"],
}


def setup_node(state: GameState) -> dict:
    n = len(state.players)
    if n not in ROLE_DISTRIBUTION:
        raise ValueError(f"Unsupported player count: {n}. Use 4–8 players.")

    roles = ROLE_DISTRIBUTION[n].copy()
    random.shuffle(roles)

    updated_players = []
    for player, role in zip(state.players, roles):
        updated_players.append(Player(
            id=player.id,
            name=player.name,
            role=role,
            is_alive=True,
            model=player.model,
        ))

    msg = narrate(state, f"A new game of Mafia begins with {n} players. Roles have been assigned.")

    return {
        "players": updated_players,
        "phase": "day",
        "round": 1,
        "messages": [msg],
        "votes": {},
        "mafia_target": None,
        "doctor_save": None,
        "detective_result": None,
        "winner": None,
    }


# ---------------------------------------------------------------------------
# Node: day — LLMs discuss publicly
# ---------------------------------------------------------------------------

def day_node(state: GameState) -> dict:
    from game.agents import day_discussion

    alive = state.alive_players()
    messages = [narrate(state, (
        f"Round {state.round} — Day phase begins. "
        f"{len(alive)} players remain: {', '.join(p.name for p in alive)}."
    ))]

    for player in alive:
        msg = day_discussion(player, state)
        messages.append(msg)
        state.messages.append(msg)

    return {
        "phase": "vote",
        "messages": messages,
        "votes": {},
    }


# ---------------------------------------------------------------------------
# Node: vote — LLMs reason and vote
# ---------------------------------------------------------------------------

def vote_node(state: GameState) -> dict:
    from game.agents import cast_vote

    alive = state.alive_players()
    votes: dict[str, str] = {}
    reasoning: dict[str, str] = {}
    messages = []

    for voter in alive:
        candidates = [p for p in alive if p.id != voter.id]
        if not candidates:
            continue
        target_id, reason = cast_vote(voter, state)
        votes[voter.id] = target_id
        reasoning[voter.id] = reason

        target = state.get_player(target_id)
        messages.append(Message(
            sender_id=voter.id,
            content=f"{voter.name} votes to eliminate {target.name if target else target_id}.",
            phase="vote",
            round=state.round,
            is_public=True,
        ))

    # Tally votes
    tally: dict[str, int] = {}
    for target_id in votes.values():
        tally[target_id] = tally.get(target_id, 0) + 1

    eliminated_id = max(tally, key=lambda k: tally[k]) if tally else None
    eliminated = state.get_player(eliminated_id) if eliminated_id else None

    updated_players = [
        Player(id=p.id, name=p.name, role=p.role,
               is_alive=False if p.id == eliminated_id else p.is_alive,
               model=p.model)
        for p in state.players
    ]

    if eliminated:
        messages.append(narrate(state, (
            f"The village voted. {eliminated.name} ({eliminated.role}) "
            f"has been eliminated with {tally[eliminated_id]} vote(s)."
        )))
    else:
        messages.append(narrate(state, "No one was eliminated this round."))

    return {
        "phase": "night",
        "votes": votes,
        "reasoning": reasoning,
        "players": updated_players,
        "messages": messages,
    }


# ---------------------------------------------------------------------------
# Node: night — mafia kills, doctor saves, detective investigates
# ---------------------------------------------------------------------------

def night_node(state: GameState) -> dict:
    from game.agents import mafia_night_discussion, doctor_pick_save, detective_investigate

    messages = []
    alive = state.alive_players()

    # --- Mafia picks a target ---
    mafia = state.mafia_players()
    village = state.village_players()
    mafia_target_id: str | None = None

    if mafia and village:
        mafia_target_id, mafia_messages = mafia_night_discussion(mafia, state)
        messages.extend(mafia_messages)

    # --- Doctor saves ---
    doctor = next((p for p in alive if p.role == "doctor"), None)
    doctor_save_id: str | None = None
    if doctor:
        doctor_save_id = doctor_pick_save(doctor, state)

    # --- Detective investigates ---
    detective = next((p for p in alive if p.role == "detective"), None)
    detective_result: str | None = None
    if detective:
        target_id, result = detective_investigate(detective, state)
        detective_result = result
        investigated = state.get_player(target_id)
        messages.append(Message(
            sender_id="narrator",
            content=f"Detective investigated {investigated.name if investigated else target_id}: they are {result}.",
            phase="night",
            round=state.round,
            is_public=False,
        ))

    # --- Resolve kill vs save ---
    killed_id: str | None = None
    if mafia_target_id and mafia_target_id != doctor_save_id:
        killed_id = mafia_target_id
        killed = state.get_player(killed_id)
        messages.append(narrate(state, (
            f"Night falls. {killed.name if killed else killed_id} was found dead in the morning."
        )))
    elif mafia_target_id and mafia_target_id == doctor_save_id:
        saved = state.get_player(doctor_save_id)
        messages.append(narrate(state, (
            f"Night falls. {saved.name if saved else doctor_save_id} was targeted but miraculously survived."
        )))
    else:
        messages.append(narrate(state, "Night passes quietly. No one was killed."))

    updated_players = [
        Player(id=p.id, name=p.name, role=p.role,
               is_alive=False if p.id == killed_id else p.is_alive,
               model=p.model)
        for p in state.players
    ]

    return {
        "phase": "day",
        "round": state.round + 1,
        "players": updated_players,
        "mafia_target": mafia_target_id,
        "doctor_save": doctor_save_id,
        "detective_result": detective_result,
        "messages": messages,
    }


# ---------------------------------------------------------------------------
# Win condition
# ---------------------------------------------------------------------------

def should_continue(state: GameState) -> str:
    winner = state.check_win_condition()
    return "game_over" if winner else "day"


def game_over_node(state: GameState) -> dict:
    winner = state.check_win_condition()
    msg = narrate(state, (
        f"Game over! The {winner} wins! "
        f"Surviving players: {', '.join(p.name + ' (' + p.role + ')' for p in state.alive_players())}."
    ))
    return {
        "phase": "game_over",
        "winner": winner,
        "messages": [msg],
    }


# ---------------------------------------------------------------------------
# Build the graph
# ---------------------------------------------------------------------------

def build_graph() -> StateGraph:
    graph = StateGraph(GameState)

    graph.add_node("setup", setup_node)
    graph.add_node("day", day_node)
    graph.add_node("vote", vote_node)
    graph.add_node("night", night_node)
    graph.add_node("game_over", game_over_node)

    graph.set_entry_point("setup")

    graph.add_edge("setup", "day")
    graph.add_edge("day", "vote")
    graph.add_edge("vote", "night")

    graph.add_conditional_edges(
        "night",
        should_continue,
        {
            "day": "day",
            "game_over": "game_over",
        }
    )

    graph.add_edge("game_over", END)

    return graph.compile()