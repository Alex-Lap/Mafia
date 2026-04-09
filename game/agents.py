from __future__ import annotations
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import SystemMessage, HumanMessage
from game.state import GameState, Player, Message


# ---------------------------------------------------------------------------
# Build a model instance for a player
# ---------------------------------------------------------------------------

def get_model(model: str) -> ChatAnthropic:
    return ChatAnthropic(model=model, max_tokens=512, temperature=0.9)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _strip_spoken_section(text: str) -> str:
    """Remove any spoken/dialogue section that leaks into private reasoning."""
    dividers = ["---", "**Spoken", "**SPOKEN", "Spoken dialogue", "Spoken:", "**Speaking"]
    for divider in dividers:
        if divider in text:
            text = text[:text.index(divider)].strip()
    return text


def _strip_command_line(text: str, command: str) -> str:
    """Remove the SAVE:/INVESTIGATE: line from reasoning content."""
    return "\n".join(
        line for line in text.splitlines()
        if not line.strip().upper().startswith(command.upper())
    ).strip()


def _build_history(state: GameState, round_history: list[Message]) -> str:
    """Combine past public history with current round messages."""
    history = state.public_history()
    current = "\n".join(
        f"{state.get_player(m.sender_id).name if state.get_player(m.sender_id) else m.sender_id}: {m.content}"
        for m in round_history
        if m.is_public
    )
    if current:
        return (history + "\n\n" + current).strip()
    return history


# ---------------------------------------------------------------------------
# System prompts
# ---------------------------------------------------------------------------

def build_system_prompt(player: Player, state: GameState) -> str:
    """Build a role-aware system prompt for this player."""

    role_instructions = {
        "villager": (
            "You are a villager. You do NOT know who the mafia are. "
            "Your goal is to identify and vote out the mafia through discussion and reasoning. "
            "Pay close attention to what others say — inconsistencies are suspicious."
        ),
        "mafia": (
            "You are a mafia member. You must HIDE your identity from the village. "
            "During the day, act like an innocent villager. Deflect suspicion onto others. "
            "At night you will secretly choose a villager to eliminate. "
            f"Your mafia allies are: {', '.join(p.name for p in state.mafia_players() if p.id != player.id) or 'none (you are alone)'}."
        ),
        "detective": (
            "You are the detective. Each night you secretly investigate one player "
            "and learn whether they are mafia or innocent. "
            "Use this information carefully — revealing yourself makes you a mafia target. "
            "Guide the village subtly without exposing your role too early. "
            + (
                "Your investigation findings so far:\n" +
                "\n".join(f"  - {f}" for f in state.detective_findings)
                if state.detective_findings
                else "You have no investigation results yet."
            )
        ),
        "doctor": (
            "You are the doctor. Each night you choose one player to protect. "
            "If the mafia targets that player, they survive. "
            "You can protect yourself. Keep your role secret as long as possible."
        ),
    }

    alive_players = [p for p in state.alive_players() if p.id != player.id]

    return f"""You are playing a game of Mafia. You are {player.name}.

{role_instructions[player.role]}

Currently alive players (excluding you): {', '.join(p.name for p in alive_players)}.
It is Round {state.round}.

Stay in character at all times. Be concise — 2-3 sentences max per message.
Do not reveal your role unless it strategically benefits you (detectives may choose to reveal late game).
Respond only with your spoken dialogue — no stage directions or meta-commentary."""


# ---------------------------------------------------------------------------
# Day discussion: private reasoning then public statement
# ---------------------------------------------------------------------------

def day_discussion(player: Player, state: GameState, round_history: list[Message]) -> tuple[Message, Message]:
    """Returns (private_reasoning_message, public_statement_message)."""
    model = get_model(player.model)
    system = build_system_prompt(player, state)
    history = _build_history(state, round_history)

    # --- Step 1: private reasoning (bullet points only, no spoken dialogue) ---
    reasoning_prompt = (
        f"It is the day discussion phase. Here is what has been said so far:\n\n"
        f"{history}\n\n"
        f"Reason privately about your situation in 2-3 sentences:\n"
        f"- What do you know or suspect about other players?\n"
        f"- Who seems suspicious and why?\n"
        f"- What is your strategy this round?\n"
        f"Keep it concise. Do NOT write any spoken dialogue here."
    )

    reasoning_response = model.invoke([SystemMessage(content=system), HumanMessage(content=reasoning_prompt)])
    private_reasoning = _strip_spoken_section(reasoning_response.content.strip())

    private_msg = Message(
        sender_id=player.id,
        content=private_reasoning,
        phase="day",
        round=state.round,
        is_public=False,
    )

    # --- Step 2: public statement informed by private reasoning ---
    public_prompt = (
        f"You just reasoned privately:\n\n{private_reasoning}\n\n"
        f"Now speak to the village. Your statement should reflect your strategy "
        f"but stay in character — do not reveal hidden information unless you decided to. "
        f"Be concise, 2-3 sentences max. Respond with dialogue only."
    )

    public_response = model.invoke([SystemMessage(content=system), HumanMessage(content=public_prompt)])

    public_msg = Message(
        sender_id=player.id,
        content=public_response.content.strip(),
        phase="day",
        round=state.round,
        is_public=True,
    )

    return private_msg, public_msg


# ---------------------------------------------------------------------------
# Voting: each player reasons privately then picks someone to eliminate
# ---------------------------------------------------------------------------

def cast_vote(player: Player, state: GameState) -> tuple[str, str]:
    """Returns (target_player_id, reasoning)."""
    model = get_model(player.model)

    system = build_system_prompt(player, state)
    history = state.public_history()

    alive_others = [p for p in state.alive_players() if p.id != player.id]
    candidates = "\n".join(f"- {p.name} (id: {p.id})" for p in alive_others)

    prompt = (
        f"It is time to vote. Here is the discussion so far:\n\n"
        f"{history}\n\n"
        f"You must vote to eliminate one player. Your candidates:\n{candidates}\n\n"
        f"Reason briefly about who is most suspicious (1-2 sentences). "
        f"Use player names only in your reasoning. "
        f"Then on the final line write exactly: VOTE: <player_id>"
    )

    response = model.invoke([SystemMessage(content=system), HumanMessage(content=prompt)])
    text = response.content.strip()

    target_id = _parse_vote(text, alive_others, player)
    return target_id, text


def _parse_vote(text: str, candidates: list[Player], voter: Player) -> str:
    """Extract player_id from the LLM's VOTE: line, fall back to first candidate."""
    for line in reversed(text.splitlines()):
        if line.strip().upper().startswith("VOTE:"):
            candidate_id = line.split(":", 1)[1].strip()
            if any(p.id == candidate_id for p in candidates):
                return candidate_id
    print(f"[warn] {voter.name} gave unparseable vote, falling back to first candidate.")
    return candidates[0].id if candidates else voter.id


# ---------------------------------------------------------------------------
# Night actions
# ---------------------------------------------------------------------------

def mafia_night_discussion(mafia: list[Player], state: GameState) -> tuple[str, list[Message]]:
    """
    Mafia members privately discuss who to eliminate, then each votes.
    Returns (agreed_target_id, list of private Messages).
    """
    village = state.village_players()
    if not village:
        return mafia[0].id, []

    candidates_text = "\n".join(f"- {p.name} (id: {p.id})" for p in village)
    messages: list[Message] = []
    discussion_log: list[str] = []

    for member in mafia:
        model = get_model(member.model)
        system = build_system_prompt(member, state)
        ally_chat = "\n".join(discussion_log) if discussion_log else "(you speak first)"

        prompt = (
            f"It is night. You are meeting secretly with your mafia allies.\n\n"
            f"Villagers you can target:\n{candidates_text}\n\n"
            f"What your allies said so far:\n{ally_chat}\n\n"
            f"Argue for who you think is the biggest threat to eliminate tonight. "
            f"Use player names only. Be strategic — 2-3 sentences."
        )

        response = model.invoke([SystemMessage(content=system), HumanMessage(content=prompt)])
        speech = response.content.strip()
        discussion_log.append(f"{member.name}: {speech}")
        messages.append(Message(
            sender_id=member.id,
            content=speech,
            phase="night",
            round=state.round,
            is_public=False,
        ))

    votes: dict[str, int] = {}
    for member in mafia:
        model = get_model(member.model)
        system = build_system_prompt(member, state)
        full_discussion = "\n".join(discussion_log)

        prompt = (
            f"Your mafia team discussed:\n{full_discussion}\n\n"
            f"Villagers to target:\n{candidates_text}\n\n"
            f"Cast your final vote. Write exactly: TARGET: <player_id>"
        )

        response = model.invoke([SystemMessage(content=system), HumanMessage(content=prompt)])
        text = response.content.strip()

        for line in reversed(text.splitlines()):
            if line.strip().upper().startswith("TARGET:"):
                target_id = line.split(":", 1)[1].strip()
                if any(p.id == target_id for p in village):
                    votes[target_id] = votes.get(target_id, 0) + 1
                    break

    final_target = max(votes, key=lambda k: votes[k]) if votes else village[0].id
    target = state.get_player(final_target)
    messages.append(Message(
        sender_id="narrator",
        content=f"The mafia agreed to target {target.name if target else final_target}.",
        phase="night",
        round=state.round,
        is_public=False,
    ))

    return final_target, messages


def doctor_pick_save(player: Player, state: GameState) -> tuple[str, Message]:
    """Doctor reasons about who to protect. Returns (player_id, reasoning_message)."""
    model = get_model(player.model)
    system = build_system_prompt(player, state)

    alive = state.alive_players()
    candidates = "\n".join(f"- {p.name}" for p in alive)
    example_name = alive[0].name

    prompt = (
        f"It is night. As the doctor, choose one player to protect from the mafia.\n"
        f"You can protect yourself. Candidates:\n{candidates}\n\n"
        f"Reason in 2-3 bullet points about who the mafia is most likely to target. "
        f"Use player names only in your reasoning. "
        f"Do NOT write any spoken dialogue. "
        f"You MUST end your response with a SAVE line, for example:\n"
        f"SAVE: {example_name}"
    )

    response = model.invoke([SystemMessage(content=system), HumanMessage(content=prompt)])
    text = response.content.strip()

    reasoning_content = _strip_command_line(text, "SAVE:")
    reasoning_msg = Message(
        sender_id=player.id,
        content=reasoning_content,
        phase="night",
        round=state.round,
        is_public=False,
    )

    for line in reversed(text.splitlines()):
        if line.strip().upper().startswith("SAVE:"):
            save_name = line.split(":", 1)[1].strip()
            match = next((p for p in alive if p.name.lower() == save_name.lower()), None)
            if match:
                return match.id, reasoning_msg

    return player.id, reasoning_msg  # default: protect self


def detective_investigate(player: Player, state: GameState) -> tuple[str, str, Message]:
    """Detective reasons about who to investigate. Returns (target_id, finding, reasoning_message)."""
    model = get_model(player.model)
    system = build_system_prompt(player, state)

    suspects = [p for p in state.alive_players() if p.id != player.id]
    candidates = "\n".join(f"- {p.name}" for p in suspects)
    example_name = suspects[0].name if suspects else ""

    prompt = (
        f"It is night. As the detective, choose one player to investigate.\n"
        f"Candidates:\n{candidates}\n\n"
        f"Reason in 2-3 bullet points about who seems most suspicious. "
        f"Use player names only in your reasoning. "
        f"Do NOT write any spoken dialogue. "
        f"You MUST end your response with an INVESTIGATE line, for example:\n"
        f"INVESTIGATE: {example_name}"
    )

    response = model.invoke([SystemMessage(content=system), HumanMessage(content=prompt)])
    text = response.content.strip()

    reasoning_content = _strip_command_line(text, "INVESTIGATE:")
    reasoning_msg = Message(
        sender_id=player.id,
        content=reasoning_content,
        phase="night",
        round=state.round,
        is_public=False,
    )

    target_id = suspects[0].id if suspects else player.id
    for line in reversed(text.splitlines()):
        if line.strip().upper().startswith("INVESTIGATE:"):
            investigate_name = line.split(":", 1)[1].strip()
            match = next((p for p in suspects if p.name.lower() == investigate_name.lower()), None)
            if match:
                target_id = match.id
                break

    target = state.get_player(target_id)
    is_mafia = target.role == "mafia" if target else False
    finding = f"Round {state.round}: {target.name if target else target_id} is {'MAFIA' if is_mafia else 'innocent'}"

    return target_id, finding, reasoning_msg