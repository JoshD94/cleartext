"""
POS tagger for the sentence simplification project.

Setup (run these once in your terminal):
    pip install spacy
    python -m spacy download en_core_web_sm

Usage from your teammate's code:
    from pos_tag import pos_tag
    tokens = pos_tag("The committee utilized innovative methodologies.")
"""

import spacy

# Load the model once when this file is imported (loading is the slow part).
nlp = spacy.load("en_core_web_sm")


def pos_tag(sentence):
    """
    Tag every word in a sentence with its part of speech.

    Returns a list of dictionaries, one per word, for example:
        {"index": 2, "text": "utilized", "lemma": "utilize", "pos": "VERB",
         "tag": "VBD", "dep": "ROOT", "head": 2, "start": 14, "whitespace": " "}

    Fields:
        index      - position of the word in the sentence (0, 1, 2, ...)
        text       - the word exactly as it appears in the sentence
        lemma      - the base form of the word ("utilized" -> "utilize")
        pos        - coarse part of speech (NOUN, VERB, ADJ, ADV, PROPN, ...)
        tag        - detailed tag (VBD = past-tense verb, NNS = plural noun, ...)
        dep        - grammatical role (nsubj = subject, dobj = object, ROOT = main verb, ...)
        head       - index of the word this one attaches to in the sentence structure
        start      - character position where the word begins in the sentence
        whitespace - the space after the word ("" if none), for rebuilding text
    """
    doc = nlp(sentence)
    return [_token_info(token) for token in doc]


def _token_info(token):
    return {
        "index": token.i,
        "text": token.text,
        "lemma": token.lemma_,
        "pos": token.pos_,
        "tag": token.tag_,
        "dep": token.dep_,
        "head": token.head.i,
        "start": token.idx,
        "whitespace": token.whitespace_,
    }


def print_tags(sentence):
    """Print the tags as a readable table (handy for testing)."""
    print(f"\nSentence: {sentence}")
    print(f"{'#':<4}{'WORD':<15}{'LEMMA':<15}{'POS':<8}{'TAG':<6}{'DEP':<10}HEAD")
    print("-" * 70)
    tokens = pos_tag(sentence)
    for t in tokens:
        head_word = tokens[t["head"]]["text"]
        print(f"{t['index']:<4}{t['text']:<15}{t['lemma']:<15}{t['pos']:<8}"
              f"{t['tag']:<6}{t['dep']:<10}{head_word}")


if __name__ == "__main__":
    print_tags("The committee utilized innovative methodologies.")
    print_tags("I will book a flight to read a book.")

    while True:
        sentence = input("\nEnter a sentence (or press Enter to quit): ").strip()
        if not sentence:
            break
        print_tags(sentence)