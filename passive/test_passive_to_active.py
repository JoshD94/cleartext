"""Run with:  python test_passive_to_active.py   (or pytest)"""

from passive_to_active import to_active

CASES = [
    # --- tenses
    ("The cake was eaten by John.", "John ate the cake."),
    ("The cakes were eaten by the children.", "The children ate the cakes."),
    ("The letter is written by my sister.", "My sister writes the letter."),
    ("The letters are written by the students.", "The students write the letters."),
    ("The house is being painted by the workers.", "The workers are painting the house."),
    ("The road was being repaired by the city.", "The city was repairing the road."),
    ("The house has been sold by the owners.", "The owners have sold the house."),
    ("The report had been finished by Maria.", "Maria had finished the report."),
    ("The game will be won by our team.", "Our team will win the game."),
    ("The form must be signed by a parent.", "A parent must sign the form."),
    ("The work will have been completed by the team.", "The team will have completed the work."),
    # --- pronouns
    ("I was helped by them.", "They helped me."),
    ("She was praised by me.", "I praised her."),
    ("We were invited by him.", "He invited us."),
    # --- negation and adverbs
    ("The email was not read by the manager.", "The manager did not read the email."),
    ("The door isn't locked by the guard.", "The guard doesn't lock the door."),
    ("The cake was quickly eaten by John.", "John quickly ate the cake."),
    ("The film has never been seen by my parents.", "My parents have never seen the film."),
    # --- structure around the clause
    ("Yesterday, the window was broken by a ball.", "Yesterday, a ball broke the window."),
    ("The book that was written by Tolkien is long.", "The book that Tolkien wrote is long."),
    ("The trash was picked up by the neighbors.", "The neighbors picked up the trash."),
    ("The song was sung by John and Mary.", "John and Mary sang the song."),
    ("The bridge, which was designed by a famous architect, was damaged by the storm.",
     "The storm damaged the bridge, which a famous architect designed."),
    ("It was decided by the committee that the event would be canceled.",
     "The committee decided that the event would be canceled."),
    ("The results were analyzed by researchers, and the paper was published by Nature.",
     "Researchers analyzed the results, and Nature published the paper."),
    # --- left alone
    ("Mistakes were made.", "Mistakes were made."),
    ("Was the cake eaten by John?", "Was the cake eaten by John?"),
    ("John ate the cake.", "John ate the cake."),
]


def test_cases():
    failures = []
    for src, expected in CASES:
        got = to_active(src).output
        if got != expected:
            failures.append((src, expected, got))
    for src, exp, got in failures:
        print(f"FAIL  {src}\n      expected: {exp}\n      got:      {got}")
    print(f"{len(CASES) - len(failures)}/{len(CASES)} passed")
    assert not failures


def test_placeholder():
    assert to_active("Mistakes were made.", placeholder="someone").output == "Someone made mistakes."


if __name__ == "__main__":
    try:
        test_cases()
    except AssertionError:
        pass
    test_placeholder()
