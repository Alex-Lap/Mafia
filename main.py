from dotenv import load_dotenv
load_dotenv()

from colorama import init, Fore, Back, Style
init(autoreset=True)

from game.state import GameState, Player, Message
from game.graph import build_graph

import textwrap


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------

PHASE_COLORS = {
    "setup":     Fore.WHITE,
    "day":       Fore.YELLOW,
    "vote":      Fore.MAGENTA,
    "night":     Fore.BLUE,
    "game_over": Fore.RED,
}

ROLE_COLORS = {
    "mafia":     Fore.RED,
    "villager":  Fore.GREEN,
    "detective": Fore.CYAN,
    "doctor":    Fore.BLUE,
}

WIDTH = 60

def wrap(text: str, width: int = WIDTH) -> str:
    return "\n".join(textwrap.fill(line, width) for line in text.splitlines())

def phase_banner(phase: str, round: int = 0) -> None:
    color = PHASE_COLORS.get(phase, Fore.WHITE)
    label = phase.upper().replace("_", " ")
    if round and phase not in ("setup", "game_over"):
        label = f"ROUND {round} — {label}"
    border = "═" * WIDTH
    print(f"\n{color}{border}")
    print(f"{color}  {label}")
    print(f"{color}{border}\n")


def print_message(msg: Message, players: list[Player]) -> None:
    # Find sender name
    if msg.sender_id == "narrator":
        name = "Narrator"
        name_color = Fore.WHITE + Style.BRIGHT
    else:
        player = next((p for p in players if p.id == msg.sender_id), None)
        name = player.name if player else msg.sender_id
        name_color = ROLE_COLORS.get(player.role if player else "", Fore.WHITE)

    # Private messages are dimmed
    if not msg.is_public:
        prefix = f"{Fore.BLACK + Style.BRIGHT}[private] "
        content_color = Fore.BLACK + Style.BRIGHT
    else:
        prefix = ""
        content_color = Fore.WHITE

    print(f"{prefix}{name_color}{name}:\n{content_color}{wrap(msg.content)}\n")


def print_result(winner: str, players: list[Player]) -> None:
    color = Fore.RED if winner == "mafia" else Fore.GREEN
    border = "═" * WIDTH
    print(f"\n{color}{border}")
    print(f"{color}  GAME OVER — {winner.upper()} WINS!")
    print(f"{color}{border}\n")

    print(f"{Style.BRIGHT}Final player roles:\n")
    for p in players:
        status = f"{Fore.GREEN}alive" if p.is_alive else f"{Fore.RED}eliminated"
        role_color = ROLE_COLORS.get(p.role, Fore.WHITE)
        print(f"  {Fore.WHITE}{p.name:<12} {role_color}{p.role:<12} {status}")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    players = [
        Player(id="p1", name="Alice",  model="claude-haiku-4-5-20251001"),
        Player(id="p2", name="Bob",    model="claude-haiku-4-5-20251001"),
        Player(id="p3", name="Carol",  model="claude-haiku-4-5-20251001"),
        Player(id="p4", name="Dave",   model="claude-haiku-4-5-20251001"),
        Player(id="p5", name="Eve",    model="claude-haiku-4-5-20251001"),
        Player(id="p6", name="Frank",  model="claude-haiku-4-5-20251001"),
        Player(id="p7", name="Greg",  model="claude-haiku-4-5-20251001"),
        Player(id="p8", name="Hannah",  model="claude-haiku-4-5-20251001"),
    ]

    initial_state = GameState(players=players)
    initial_state.validate()
    graph = build_graph()

    border = "═" * WIDTH
    print(f"\n{Fore.CYAN}{border}")
    print(f"{Fore.CYAN}  MAFIA — LangGraph Edition")
    print(f"{Fore.CYAN}{border}\n")

    final_state = graph.invoke(initial_state)
    all_players = final_state["players"]

    # Print transcript grouped by phase transitions
    current_phase = None
    current_round = None

    for msg in final_state["messages"]:
        # Print a banner whenever phase or round changes
        if msg.phase != current_phase or msg.round != current_round:
            phase_banner(msg.phase, msg.round)
            current_phase = msg.phase
            current_round = msg.round

        print_message(msg, all_players)

    print_result(final_state["winner"], all_players)


if __name__ == "__main__":
    main()