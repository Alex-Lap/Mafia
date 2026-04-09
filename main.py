from dotenv import load_dotenv 
load_dotenv()
 
from game.state import GameState, Player
from game.graph import build_graph
 
 
def main():
    # Create 6 players
    players = [
        Player(id="p1", name="Alice",   model="claude-haiku-4-5-20251001"),
        Player(id="p2", name="Bob",     model="claude-haiku-4-5-20251001"),
        Player(id="p3", name="Carol",   model="claude-haiku-4-5-20251001"),
        Player(id="p4", name="Dave",    model="claude-haiku-4-5-20251001"),
        Player(id="p5", name="Eve",     model="claude-haiku-4-5-20251001"),
        Player(id="p6", name="Frank",   model="claude-haiku-4-5-20251001"),
        Player(id="p7", name="Greg",   model="claude-haiku-4-5-20251001"),
        Player(id="p8", name="Harold",   model="claude-haiku-4-5-20251001"),
    ]
 
    initial_state = GameState(players=players)
    initial_state.validate()
    graph = build_graph()
 
    print("=" * 50)
    print("  MAFIA — LangGraph Edition")
    print("=" * 50)
 
    final_state = graph.invoke(initial_state)
 
    print("\n--- GAME TRANSCRIPT ---\n")
    for msg in final_state["messages"]:
        visibility = "public" if msg.is_public else "private"
        print(f"[{visibility}] {msg.sender_id}: {msg.content}\n")
 
    print(f"\nResult: {final_state['winner']} wins!")
 
 
if __name__ == "__main__":
    main()