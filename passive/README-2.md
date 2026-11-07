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

### In Python

```python
from passive_to_active import to_active

result = to_active("The window was broken by a ball.")
print(result.output)    # A ball broke the window.
print(result.changed)   # True
print(result.changes)   # [(before_clause, after_clause), ...]
print(result.skipped)   # [(clause, reason), ...] for passives left unchanged
```

To convert passives that have no "by" phrase, pass a placeholder subject:

```python
to_active("Mistakes were made.", placeholder="someone").output
# Someone made mistakes.
```

### From the command line

```bash
python passive_to_active.py "The cake was eaten by John." "The thief was caught."
```

```
The cake was eaten by John.
  -> John ate the cake.
The thief was caught.
  -> The thief was caught.
     (skipped: no agent (no 'by ...' phrase))
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

## What it leaves unchanged

| Case                              | Example                        | Why                                   |
|-----------------------------------|--------------------------------|---------------------------------------|
| No "by" phrase                    | Mistakes were made.            | No one to use as the subject          |
| Questions / inverted order        | Was the cake eaten by John?    | Word order rules differ               |
| "get" passives                    | He got fired by his boss.      | Not handled yet                       |
| Infinitive passives               | The form needs to be signed.   | Not handled yet                       |

Each skipped clause is listed in `result.skipped` with a reason.

## Known limitations

- **Parser errors.** The tool is only as good as the parse. The small spaCy model sometimes misreads long or unusual sentences. `en_core_web_trf` is more accurate but slower; change the model name at the top of `passive_to_active.py` to try it.
- **"by" + place.** "The house was built by the river" can be misread as the river building the house if spaCy labels the phrase as an agent.
- **Agreement.** Subject–verb agreement is basic. Collective nouns ("the team", "the police") may get the wrong verb form.
- **Meaning and emphasis.** Passive voice is sometimes the better choice (when the actor is unknown or unimportant). The tool does not judge whether a conversion actually improves readability.

## Files

| File                          | Contents                                  |
|-------------------------------|-------------------------------------------|
| `passive_to_active.py`        | The converter (`to_active` is the entry point) |
| `test_passive_to_active.py`   | 28 test sentences plus a placeholder test |

## Next steps

- Run the converter on a few hundred sentences from our target texts and add each failure to the tests.
- Handle "get" passives and infinitive passives.
- Combine with the other simplification rules (sentence splitting, relative clauses) and the lexical simplification step.
