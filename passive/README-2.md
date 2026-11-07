# Passive → Active Voice Converter

A rule-based tool that rewrites passive sentences in active voice to make them easier to read. It is one part of our sentence simplification project.

```
The cake was eaten by John.                 →  John ate the cake.
The house has been sold by them.            →  They have sold the house.
The email was not read by the manager.      →  The manager did not read the email.
```

## Setup

Requires Python 3.9+.

```bash
pip install spacy lemminflect
python -m spacy download en_core_web_sm
```

## Usage

### Plugging it into the simplification tool

Use the `PassiveToActive` class. It's the only part of the file the rest of the tool needs to know about.

```python
from passive_to_active import PassiveToActive

p2a = PassiveToActive()

p2a.transform("The cake was eaten by John.")
# "John ate the cake."

p2a.transform("John ate the cake.")
# "John ate the cake."   (not passive: returned unchanged)

p2a.transform_text("The thief was caught. He was sent to jail by the judge.")
# "Someone caught the thief. The judge sent him to jail."
```

| Method | What it does |
|--------|--------------|
| `transform(sentence)` | One sentence in, one sentence out. |
| `transform_text(text)` | Splits a paragraph into sentences and transforms each. |
| `is_passive(sentence)` | `True` if the sentence has a passive clause, even one it won't convert. |
| `explain(sentence)` | A `Result` with `.output`, `.changed`, `.changes` (before/after clauses) and `.skipped` (clause, reason). Useful for debugging. |

Sentences that aren't passive, or can't be converted safely, come back exactly as they went in. So the bigger tool can run every sentence through `transform` without checking first.

### Passives with no "by" phrase

Passives like "The thief was caught." have no one to make the subject. The `placeholder` setting decides what happens:

```python
PassiveToActive()                    # default, same as placeholder="auto"
PassiveToActive(placeholder=None)    # leave these sentences unchanged
PassiveToActive(placeholder="they")  # always use "they"
```

| `placeholder` | Behavior |
|---------------|----------|
| `"auto"` (default) | **"people"** for general facts in the simple present, **"someone"** for everything else. |
| `None` | Leave them unchanged. |
| any word | Always use that word as the subject. |

```
English is spoken in Canada.   →  People speak English in Canada.
The thief was caught.          →  Someone caught the thief.
She was born in 1990.          →  She was born in 1990.   (unchanged: describes a state)
```

Whatever the setting, these are always left alone:

- **Passives that describe a state, not an action**: "She was born in 1990", "They are married", "The store is located downtown", "The table is made of wood", "He is known for his music". The lists are `STATIVE_LEMMAS` and `STATIVE_WITH_PREP` at the top of `passive_to_active.py`; add to them as you find more.
- **Passives followed by another verb**: "He is said to be rich", "You are not allowed to smoke".

A "by" phrase always wins: "The table was made by my father" → "My father made the table."

Note that a passive is often used because the actor is unknown or unimportant, and "Someone caught the thief" isn't always easier to read. If that turns out to be true for our texts, switch to `placeholder=None`.

### From the command line

```bash
python passive_to_active.py "The cake was eaten by John." "The thief was caught."
python passive_to_active.py --placeholder=none "The thief was caught."
```

```
The cake was eaten by John.
  -> John ate the cake.
The thief was caught.
  -> Someone caught the thief.
```

### Running the tests

```bash
python test_passive_to_active.py   # or: pytest
```

## How it works

1. **Find passive clauses.** spaCy's dependency parser labels the passive subject (`nsubjpass`), the passive "be" (`auxpass`), and the "by" phrase (`agent`). A verb with both a passive subject and a passive "be" is a passive clause. Main clauses are converted before inner ones, because moving an inner clause as one block keeps the next parse cleaner.

2. **Swap subject and object.** The noun in the "by" phrase becomes the subject, and the old subject becomes the object. Pronouns change form: "by him" → "he", "I" → "me".

3. **Rebuild the verb.** The form of the passive "be" tells us what form the active verb needs:

   | Passive            | Active          | Rule                                   |
   |--------------------|-----------------|----------------------------------------|
   | was eaten          | ate             | past "be" → past verb                  |
   | is eaten           | eats            | present "be" → present verb, re-agreed |
   | is being eaten     | is eating       | "being" → -ing form                    |
   | has been eaten     | has eaten       | "been" → past participle               |
   | will be eaten      | will eat        | bare "be" → base form                  |
   | was not eaten      | did not eat     | negation adds "do"                     |

   Helper verbs are re-agreed with the new subject ("The house has been sold by them" → "They **have** sold"). Verb forms come from `lemminflect`.

4. **Keep everything else in place.** Words outside the clause ("Yesterday,", "after the meeting") stay where they were. Adverbs, negation and particles ("picked **up**") are placed back around the new verb.

### Special cases

- **Relative clauses:** "the book that was written by Tolkien" → "the book that Tolkien wrote" (the relative pronoun stays in front).
- **"It … that" sentences:** "It was decided by the committee that…" → "The committee decided that…" (the placeholder "it" is dropped).
- **Multiple passives** in one sentence are converted one at a time, re-parsing after each.

### Questions

| Passive question                       | Active question                     |
|----------------------------------------|-------------------------------------|
| Was the cake eaten by John?            | Did John eat the cake?              |
| Has the house been sold by the owners? | Have the owners sold the house?     |
| Wasn't the email read by the manager?  | Didn't the manager read the email?  |
| Why was the cake eaten by John?        | Why did John eat the cake?          |
| Who was the cake eaten by?             | Who ate the cake?                   |
| By whom was the letter written?        | Who wrote the letter?               |
| What was eaten by John?                | What did John eat?                  |

The small spaCy model often misparses inverted questions, so for questions like "Was the cake eaten by John?" the tool:

1. Undoes the inversion to get a statement ("the cake was eaten by John."), which parses reliably.
2. Converts the statement to active voice.
3. Puts it back in question order, adding "do" when there is no other helper verb ("John ate" → "Did John eat").

When the question word asks about the agent ("Who … by?", "By whom …?"), a placeholder agent is used in step 1 and then replaced by "Who". Questions where the question word is the passive subject ("What was eaten by John?") parse correctly already and are converted directly.

## What it leaves unchanged

| Case                              | Example                        | Why                                   |
|-----------------------------------|--------------------------------|---------------------------------------|
| No "by" phrase, with `placeholder=None` | Mistakes were made.      | No one to use as the subject          |
| State passives                    | She was born in 1990.          | No natural active form                |
| "get" passives                    | He got fired by his boss.      | Not handled yet                       |
| Infinitive passives               | The form needs to be signed.   | Not handled yet                       |

Each skipped clause is listed in `result.skipped` with a reason.

## Known limitations

- **Parser errors.** The tool is only as good as the parse. The small spaCy model sometimes misreads long or unusual sentences. `en_core_web_trf` is more accurate but slower; change the model name at the top of `passive_to_active.py` to try it.
- **"by" + place.** "The house was built by the river" can be misread as the river building the house if spaCy labels the phrase as an agent.
- **Question subjects.** In inverted questions the subject is found with a simple rule (it ends at the next helper verb, participle or adverb). Long subjects with their own verbs, like "Was the man who lives next door helped by them?", can be split in the wrong place.
- **Capitalization.** When the old subject started the sentence, it's lowercased unless it's a name. Names that are also ordinary words ("Bill", "Mark") are lowercased by mistake when they start a sentence on their own.
- **"people" vs "someone".** The choice is based only on tense, so a one-time event in the present ("The door is locked every night") still gets "people".
- **Agreement.** Subject–verb agreement is basic. Collective nouns ("the team", "the police") may get the wrong verb form.
- **Meaning and emphasis.** Passive voice is sometimes the better choice (when the actor is unknown or unimportant). The tool does not judge whether a conversion actually improves readability.

## Files

| File                          | Contents                                  |
|-------------------------------|-------------------------------------------|
| `passive_to_active.py`        | The converter (`PassiveToActive` is the public API) |
| `test_passive_to_active.py`   | 46 test sentences, plus 18 for passives without "by" |

## Next steps

- Run the converter on a few hundred sentences from our target texts and add each failure to the tests.
- Handle "get" passives and infinitive passives.
- Merge into the main tool through `PassiveToActive`, and combine with the other simplification rules (sentence splitting, relative clauses) and the lexical simplification step.