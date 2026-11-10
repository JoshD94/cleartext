"""
Shows which complex phrases are detected in each sentence and what they change to.

Usage:
    python test_phrase_finder.py

Edit SENTENCES below to try your own text.
"""

from phrase_finder import find_complex_phrases

SENTENCES = [
    "To what extent is climate change caused by human activity?",
    "We left early in order to avoid traffic.",
    "Prior to the meeting, we ate lunch.",
    "They were able to finish on time.",
    "The committee made a decision in order to reduce costs prior to the merger.",
    "Due to the fact that it was raining, the event was postponed.",
    "The school is located in the vicinity of a large park.",
    "A large number of participants were unable to attend the conference.",
    "The researchers carried out an investigation with regard to the incident.",
    "In the event that the system fails, a backup will be utilized.",
    "The government has the ability to implement new regulations at this point in time.",
    "Students are required to submit their assignments in a timely manner.",
    "It is of the utmost importance that we take into account all of the evidence.",
]


def apply_replacements(sentence, phrases):
    """Swap each detected phrase for its suggestion (no tense/grammar fixing)."""
    result = sentence
    for p in reversed(phrases):          # work backwards so positions stay valid
        new = p["suggestion"]
        if p["sentence_start"]:
            new = new[0].upper() + new[1:]
        result = result[:p["start_char"]] + new + result[p["end_char"]:]
    return result


for i, sentence in enumerate(SENTENCES, 1):
    phrases = find_complex_phrases(sentence)
    print(f"\n{i}. Before: {sentence}")
    if not phrases:
        print("   (no complex phrases detected)")
        continue
    for p in phrases:
        print(f'   - "{p["text"]}" -> "{p["suggestion"]}"   [{p["source"]}]')
    print(f"   After:  {apply_replacements(sentence, phrases)}")