"""
POS tagger using spacy library in Python
"""

import spacy

# Load the model once when this file is imported (loading is the slow part).
nlp = spacy.load("en_core_web_sm")


def pos_tag(sentence):
    """
    Tag every word in a sentence with its part of speech.

    Returns a list of dictionaries, one per word, for example:
        {"text": "utilized", "lemma": "utilize", "pos": "VERB", "tag": "VBD",
         "start": 14, "whitespace": " "}

    Fields:
        text       - the word exactly as it appears in the sentence
        lemma      - the base form of the word ("utilized" -> "utilize")
        pos        - coarse part of speech (NOUN, VERB, ADJ, ADV, PROPN, ...)
        tag        - detailed tag (VBD = past-tense verb, NNS = plural noun, ...)
        start      - character position where the word begins in the sentence
        whitespace - the space after the word ("" if none), for rebuilding text
    """
    doc = nlp(sentence)
    results = []
    for token in doc:
        results.append({
            "text": token.text,
            "lemma": token.lemma_,
            "pos": token.pos_,
            "tag": token.tag_,
            "start": token.idx,
            "whitespace": token.whitespace_,
        })
    return results


def print_tags(sentence):
    """Print the tags as a readable table (handy for testing)."""
    print(f"\nSentence: {sentence}")
    print(f"{'WORD':<15}{'LEMMA':<15}{'POS':<8}{'TAG':<6}MEANING")
    print("-" * 70)
    for t in pos_tag(sentence):
        meaning = spacy.explain(t["tag"]) or ""
        print(f"{t['text']:<15}{t['lemma']:<15}{t['pos']:<8}{t['tag']:<6}{meaning}")


if __name__ == "__main__":
    # Try a few example sentences.
    print_tags("The committee utilized innovative methodologies.")
    print_tags("I will book a flight to read a book.")

    # Or type your own sentences. Press Enter on an empty line to quit.
    while True:
        sentence = input("\nEnter a sentence (or press Enter to quit): ").strip()
        if not sentence:
            break
        print_tags(sentence)