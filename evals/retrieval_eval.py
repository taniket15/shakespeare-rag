"""Retrieval eval: does the right work (and, for quotes, the right scene) show up in the top k?

Run from the repo root:  python -m evals.retrieval_eval
Add --expand to also test query expansion (uses the OpenAI API to write alternative phrasings).
"""
import argparse

from src.retriever import HybridRetriever
from src.vectorstore import FaissVectorStore

TOP_K = 5  # kept at 5 so results stay comparable as the eval grows; the app answers from 8

# (question, expected title, expected section or None for "any section of the work")
CASES = [
    # Plot and character questions
    ("How does Lady Macbeth persuade Macbeth to kill Duncan?", "Macbeth", None),
    ("What bond does Shylock demand from Antonio?", "The Merchant of Venice", None),
    ("How does Puck cause trouble in the forest?", "A Midsummer Night's Dream", None),
    ("Who is Caliban?", "The Tempest", None),
    ("How does Iago make Othello jealous?", "Othello", None),
    ("Why does Lear divide his kingdom?", "King Lear", None),
    ("What happens when Viola disguises herself as Cesario?", "Twelfth Night", None),
    ("Who killed Polonius?", "Hamlet", None),
    ("How does Juliet fake her death?", "Romeo and Juliet", None),
    ("Why does Brutus join the conspiracy against Caesar?", "Julius Caesar", None),
    ("How does Petruchio tame Katherine?", "The Taming of the Shrew", None),
    # Plot overviews: the play's synopsis should be retrieved
    ("What happens in As You Like It?", "As You Like It", "Synopsis"),
    ("Summarize the plot of Othello", "Othello", "Synopsis"),
    ("What is The Winter's Tale about?", "The Winter's Tale", "Synopsis"),
    # Sonnets by number
    ("What is Sonnet 18 about?", "Shakespeare's Sonnets", "Sonnet 18"),
    ("What does Sonnet 130 say about the speaker's mistress?", "Shakespeare's Sonnets", "Sonnet 130"),
    # Shakespeare's life (Wikipedia biography)
    ("When and where was Shakespeare born?", "William Shakespeare (biography)", None),
    ("Who was Shakespeare's wife?", "William Shakespeare (biography)", None),  # Anne Hathaway appears in several sections
    ("When did Shakespeare die?", "William Shakespeare (biography)", None),
    ("What is the First Folio?", "William Shakespeare (biography)", None),
    # Known miss: casual wording doesn't match the section's vocabulary ("authorship", "doubts")
    ("Did someone else write Shakespeare's plays?", "William Shakespeare (biography)", "Speculation about Shakespeare › Authorship"),
    # Famous quotes: the exact scene must be retrieved
    ("Who says 'The quality of mercy is not strained' and why?", "The Merchant of Venice", "Act 4, Scene 1"),
    ("Who says 'To be or not to be'?", "Hamlet", "Act 3, Scene 1"),
    ("Who says 'Now is the winter of our discontent'?", "Richard III", "Act 1, Scene 1"),
    ("Who says 'Friends, Romans, countrymen, lend me your ears'?", "Julius Caesar", "Act 3, Scene 2"),
    ("Who says 'Out, damned spot'?", "Macbeth", "Act 5, Scene 1"),
    ("Where does Juliet say 'What's in a name'?", "Romeo and Juliet", "Act 2, Scene 2"),
    ("Who says 'Once more unto the breach, dear friends'?", "Henry V", "Act 3, Scene 1"),
    ("Who says 'We few, we happy few, we band of brothers'?", "Henry V", "Act 4, Scene 3"),
    ("When is 'Et tu, Brutè' said?", "Julius Caesar", "Act 3, Scene 1"),
    ("Who chants 'Double, double toil and trouble'?", "Macbeth", "Act 4, Scene 1"),
    ("Who says 'All the world's a stage'?", "As You Like It", "Act 2, Scene 7"),
    ("Who says 'If music be the food of love, play on'?", "Twelfth Night", "Act 1, Scene 1"),
]


def hit(results, title, section) -> bool:
    return any(
        r["metadata"].get("title") == title and (section is None or r["metadata"].get("section") == section)
        for r in results
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--expand", action="store_true", help="also evaluate hybrid search with query expansion")
    args = parser.parse_args()

    store = FaissVectorStore()
    store.load()
    hybrid = HybridRetriever(store)
    systems = {
        "vector": lambda q: store.query(q, top_k=TOP_K),
        "hybrid": lambda q: hybrid.search(q, top_k=TOP_K),
    }
    if args.expand:
        from src.search import RAGSearch
        rag = RAGSearch()
        rag.vectorstore, rag.retriever = store, hybrid  # reuse the loaded index
        systems["expanded"] = lambda q: hybrid.search(q, top_k=TOP_K, alternatives=rag.plan(q, []).alternatives)
    totals = {name: 0 for name in systems}
    for question, title, section in CASES:
        row = []
        for name, search in systems.items():
            ok = hit(search(question), title, section)
            totals[name] += ok
            row.append(f"{name}:{'OK ' if ok else 'MISS'}")
        target = f"{title}, {section}" if section else title
        print(f"{'  '.join(row)}  {question[:55]:55} -> {target}")
    print()
    for name, total in totals.items():
        print(f"{name:7} top-{TOP_K} accuracy: {total}/{len(CASES)} ({total / len(CASES):.0%})")


if __name__ == "__main__":
    main()
