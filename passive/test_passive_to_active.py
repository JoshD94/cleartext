"""Run with:  python test_passive_to_active.py   (or pytest)"""

from passive_to_active import PassiveToActive, to_active

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
    # --- questions
    ("Was the cake eaten by John?", "Did John eat the cake?"),
    ("Were the cookies baked by your mom?", "Did your mom bake the cookies?"),
    ("Is the letter written by your sister?", "Does your sister write the letter?"),
    ("Has the house been sold by the owners?", "Have the owners sold the house?"),
    ("Is the house being painted by the workers?", "Are the workers painting the house?"),
    ("Will the game be won by our team?", "Will our team win the game?"),
    ("Wasn't the email read by the manager?", "Didn't the manager read the email?"),
    ("Was I helped by them?", "Did they help me?"),
    ("Why was the cake eaten by John?", "Why did John eat the cake?"),
    ("When will the report be finished by Maria?", "When will Maria finish the report?"),
    ("Who was the cake eaten by?", "Who ate the cake?"),
    ("By whom was the letter written?", "Who wrote the letter?"),
    ("What was eaten by John?", "What did John eat?"),
    ("Which song was sung by them?", "Which song did they sing?"),
    ("Who was helped by them?", "Who did they help?"),
    ("How was the bridge built by the engineers?", "How did the engineers build the bridge?"),
    ("Can the form be signed by a parent?", "Can a parent sign the form?"),
    # --- left alone
    ("Mistakes were made.", "Mistakes were made."),
    ("Was the cake eaten?", "Was the cake eaten?"),
    ("Is John happy?", "Is John happy?"),
    ("John ate the cake.", "John ate the cake."),
]


def test_cases():
    # placeholder=None: test the core rules; agentless passives stay unchanged.
    failures = []
    for src, expected in CASES:
        got = to_active(src, placeholder=None).output
        if got != expected:
            failures.append((src, expected, got))
    for src, exp, got in failures:
        print(f"FAIL  {src}\n      expected: {exp}\n      got:      {got}")
    print(f"{len(CASES) - len(failures)}/{len(CASES)} passed")
    assert not failures


AUTO_CASES = [
    # --- converted: "people" for general present-tense facts
    ("English is spoken in Canada.", "People speak English in Canada."),
    ("Rice is eaten in many countries.", "People eat rice in many countries."),
    # --- converted: "someone" for everything else
    ("The thief was caught.", "Someone caught the thief."),
    ("Mistakes were made.", "Someone made mistakes."),
    ("The house was built in 1900.", "Someone built the house in 1900."),
    ("The car has been stolen.", "Someone has stolen the car."),
    ("The form must be signed.", "Someone must sign the form."),
    ("The window wasn't closed.", "Someone didn't close the window."),
    ("Was the thief caught?", "Did someone catch the thief?"),
    # --- left alone: states, not actions
    ("She was born in 1990.", "She was born in 1990."),
    ("They are married.", "They are married."),
    ("The store is located downtown.", "The store is located downtown."),
    ("The table is made of wood.", "The table is made of wood."),
    ("He is known for his music.", "He is known for his music."),
    ("I am interested in art.", "I am interested in art."),
    # --- left alone: followed by another verb
    ("He is said to be rich.", "He is said to be rich."),
    ("You are not allowed to smoke here.", "You are not allowed to smoke here."),
    # --- a "by" phrase still wins
    ("The table was made by my father.", "My father made the table."),
]


def test_placeholder():
    assert to_active("Mistakes were made.", placeholder="they").output == "They made mistakes."
    assert to_active("Mistakes were made.", placeholder=None).output == "Mistakes were made."
    assert to_active("Mistakes were made.").output == "Someone made mistakes."  # default: auto


def test_class():
    p2a = PassiveToActive()
    assert p2a.transform("The cake was eaten by John.") == "John ate the cake."
    assert p2a.transform("John ate the cake.") == "John ate the cake."
    assert p2a.transform("The thief was caught.") == "Someone caught the thief."
    assert p2a.transform("") == ""
    assert p2a.transform_text(
        "The thief was caught. He was sent to jail by the judge. It rained."
    ) == "Someone caught the thief. The judge sent him to jail. It rained."
    assert p2a.is_passive("The cake was eaten by John.")
    assert p2a.is_passive("Was the cake eaten by John?")
    assert p2a.is_passive("She was born in 1990.")
    assert not p2a.is_passive("John ate the cake.")
    assert PassiveToActive(placeholder=None).transform("The thief was caught.") == "The thief was caught."
    print("class tests passed")


def test_auto_placeholder():
    failures = []
    for src, expected in AUTO_CASES:
        got = to_active(src, placeholder="auto").output
        if got != expected:
            failures.append((src, expected, got))
    for src, exp, got in failures:
        print(f"FAIL  {src}\n      expected: {exp}\n      got:      {got}")
    print(f"{len(AUTO_CASES) - len(failures)}/{len(AUTO_CASES)} auto-placeholder cases passed")
    assert not failures


if __name__ == "__main__":
    for test in (test_cases, test_placeholder, test_auto_placeholder, test_class):
        try:
            test()
        except AssertionError:
            print(f"{test.__name__} FAILED")