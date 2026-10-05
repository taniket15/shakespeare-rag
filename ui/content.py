"""Static content for the chat UI: the works catalog, example questions and HTML snippets."""
from pathlib import Path

WORKS = {
    "Comedies": [
        "All's Well That Ends Well", "As You Like It", "The Comedy of Errors", "Love's Labor's Lost",
        "Measure for Measure", "The Merchant of Venice", "The Merry Wives of Windsor",
        "A Midsummer Night's Dream", "Much Ado About Nothing", "The Taming of the Shrew",
        "Twelfth Night", "The Two Gentlemen of Verona",
    ],
    "Tragedies": [
        "Antony and Cleopatra", "Coriolanus", "Hamlet", "Julius Caesar", "King Lear", "Macbeth",
        "Othello", "Romeo and Juliet", "Timon of Athens", "Titus Andronicus", "Troilus and Cressida",
    ],
    "Histories": [
        "Henry IV, Part 1", "Henry IV, Part 2", "Henry V", "Henry VI, Part 1", "Henry VI, Part 2",
        "Henry VI, Part 3", "Henry VIII", "King John", "Richard II", "Richard III",
    ],
    "Romances": ["Cymbeline", "Pericles", "The Tempest", "The Winter's Tale", "The Two Noble Kinsmen"],
    "Poems": ["Shakespeare's Sonnets", "Venus and Adonis", "Lucrece", "The Phoenix and Turtle"],
}

EXAMPLES = [
    "What happens in As You Like It?",
    "Why does Macbeth murder King Duncan?",
    "What bond does Shylock demand from Antonio?",
    "How does Puck cause trouble in the forest?",
    "Who is Caliban in The Tempest?",
    "How does Romeo and Juliet end?",
    "Who was Shakespeare's wife?",
]

HEADER = """
<div class="bard-header">
  <span class="corner tl">✥</span><span class="corner tr">✥</span>
  <span class="corner bl">✥</span><span class="corner br">✥</span>
  <div class="bard-title">Shakespeare RAG</div>
  <div class="bard-subtitle">Ask anything about Shakespeare's plays and poems</div>
  <div class="bard-rule">❦</div>
</div>
"""

EPIGRAPH = """
<div class="bard-epigraph">“All the world’s a stage, and all the men and women merely players.”</div>
<div class="bard-epigraph-source">As You Like It, Act 2, Scene 7</div>
<div class="bard-hint">Ask a question below, or pick an example from the sidebar.</div>
"""


def styles() -> str:
    """The page CSS (ui/styles.css), wrapped for st.markdown."""
    return f"<style>{(Path(__file__).parent / 'styles.css').read_text()}</style>"
