"""Answer eval: are the app's answers complete, faithful to the retrieved passages, and cited?

An LLM judge compares each answer with key facts a good answer should cover and with the
passages it was given. Run from the repo root:  python -m evals.answer_eval
"""
import argparse
import json
import time
from pathlib import Path

from pydantic import BaseModel, Field

from src.search import RAGSearch

# Questions written for this eval and not used while tuning retrieval or prompts.
# key_facts=None means the texts can't answer it, so the app should say so instead of answering.
CASES = [
    ("Why does Othello kill Desdemona?",
     ["Othello believes Desdemona has been unfaithful (with Cassio)", "Iago deceives / manipulates Othello",
      "the handkerchief is used as false evidence"]),
    ("How does Hamlet test whether Claudius killed his father?",
     ["Hamlet stages a play (The Mousetrap / The Murder of Gonzago) re-enacting the murder",
      "Claudius reacts guiltily and stops the play"]),
    ("What happens to Ophelia?",
     ["Ophelia goes mad", "Ophelia drowns"]),
    ("Why are Prospero and Miranda on the island?",
     ["Prospero's brother Antonio usurped his dukedom of Milan", "Prospero and Miranda were set adrift at sea"]),
    ("How is Malvolio tricked in Twelfth Night?",
     ["Maria forges a letter in Olivia's handwriting", "Malvolio believes Olivia loves him",
      "he wears yellow stockings / cross-garters or is locked up as mad"]),
    ("What is the casket test in The Merchant of Venice?",
     ["suitors choose between gold, silver and lead caskets", "the right casket (lead) holds Portia's portrait",
      "Bassanio chooses the lead casket"]),
    ("How does Romeo and Juliet end?",
     ["Romeo believes Juliet is dead and poisons himself", "Juliet wakes and kills herself with Romeo's dagger",
      "the Montagues and Capulets are reconciled"]),
    ("Who are Goneril and Regan?",
     ["they are King Lear's daughters", "they flatter Lear to get his kingdom", "they turn against / mistreat Lear"]),
    ("What happens to Bottom in A Midsummer Night's Dream?",
     ["Puck gives Bottom an ass's (donkey's) head", "Titania falls in love with Bottom"]),
    ("What happens to Hermione in The Winter's Tale?",
     ["Leontes falsely accuses Hermione of adultery", "Hermione is reported dead",
      "Hermione is revealed alive as a 'statue' at the end"]),
    ("Who is Falstaff in Henry IV, Part 1?",
     ["Falstaff is a fat, witty knight and companion of Prince Hal", "the Gad's Hill robbery / tavern life"]),
    ("What does Sonnet 116 say about love?",
     ["true love does not change or alter", "love is not Time's fool / lasts until the edge of doom"]),
    ("How does Antony die in Antony and Cleopatra?",
     ["Antony hears a false report that Cleopatra is dead", "Antony falls on / stabs himself with his sword",
      "Antony dies with Cleopatra (in her monument / arms)"]),
    ("What does Beatrice ask Benedick to do after Hero is shamed?",
     ["Beatrice asks Benedick to kill Claudio"]),
    ("Why is Coriolanus banished from Rome?",
     ["Coriolanus shows contempt for the common people / plebeians", "the tribunes turn the people against him"]),
    ("In what year was Hamlet first performed?", None),
    ("What is the capital of France?", None),
]


class Judgment(BaseModel):
    facts_in_passages: list[bool] = Field(description="For each key fact, in order: do the passages state it?")
    facts_covered: list[bool] = Field(description="For each key fact, in order: does the answer state it?")
    faithful: bool = Field(description="True if every claim in the answer is supported by the passages")
    unsupported_claims: list[str] = Field(description="Claims in the answer that the passages don't support")
    cites_sources: bool = Field(description="True if the answer cites passages with [n] markers")
    declined: bool = Field(description="True if the answer says the passages don't contain the answer")


JUDGE_PROMPT = """You grade answers from a question-answering app about Shakespeare.
You get the question, the passages the app retrieved, the app's answer, and key facts a good answer covers.
Judge strictly from the passages for faithfulness: a claim the passages don't support is unsupported,
even if it's true. A key fact counts as covered if the answer states it, in any wording."""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--top-k", type=int, default=8, help="passages retrieved per question")
    parser.add_argument("--name", default="answer_eval", help="results file name")
    args = parser.parse_args()

    rag = RAGSearch()
    judge = rag.llm.with_structured_output(Judgment)
    rows = []
    for question, key_facts in CASES:
        start = time.time()
        answer, results, _ = rag.answer(question, top_k=args.top_k)
        seconds = time.time() - start
        passages = "\n\n".join(f"[{i}] {r['metadata']['text']}" for i, r in enumerate(results, 1))
        facts = "\n".join(f"- {f}" for f in key_facts) if key_facts else "(none: the passages can't answer this)"
        try:
            j = judge.invoke([
                ("system", JUDGE_PROMPT),
                ("human", f"Question: {question}\n\nPassages:\n{passages}\n\nAnswer:\n{answer}\n\nKey facts:\n{facts}"),
            ])
        except Exception as e:
            print(f"  judge failed, skipping: {question[:55]} ({type(e).__name__})")
            continue
        if key_facts:
            covered = sum(j.facts_covered[:len(key_facts)]) / len(key_facts)
            recall = sum(j.facts_in_passages[:len(key_facts)]) / len(key_facts)
        else:
            covered = 1.0 if j.declined else 0.0
            recall = None
        rows.append({
            "question": question, "answerable": key_facts is not None, "answer": answer,
            "completeness": covered, "context_recall": recall, "faithful": j.faithful, "cites_sources": j.cites_sources,
            "unsupported_claims": j.unsupported_claims, "words": len(answer.split()), "seconds": round(seconds, 1),
        })
        flag = "" if j.faithful else "  UNFAITHFUL"
        in_passages = f"{recall:4.0%} in passages" if recall is not None else "  unanswerable  "
        print(f"{in_passages}  {covered:4.0%} complete  {len(answer.split()):3} words  {question[:55]}{flag}")

    answerable = [r for r in rows if r["answerable"]]
    print("\nSummary")
    print(f"  context recall (answerable): {sum(r['context_recall'] for r in answerable) / len(answerable):.0%}"
          "  <- key facts present in the retrieved passages")
    print(f"  completeness (answerable):  {sum(r['completeness'] for r in answerable) / len(answerable):.0%}")
    print(f"  faithful to passages:       {sum(r['faithful'] for r in rows)}/{len(rows)}")
    print(f"  cites sources (answerable): {sum(r['cites_sources'] for r in answerable)}/{len(answerable)}")
    print(f"  declines unanswerable:      {sum(r['completeness'] == 1 for r in rows if not r['answerable'])}"
          f"/{sum(not r['answerable'] for r in rows)}")
    print(f"  average answer length:      {sum(r['words'] for r in answerable) / len(answerable):.0f} words")

    out = Path(__file__).parent / "results" / f"{args.name}.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(rows, indent=2, ensure_ascii=False))
    print(f"\nFull answers and judgments saved to {out.relative_to(Path.cwd())}")


if __name__ == "__main__":
    main()
