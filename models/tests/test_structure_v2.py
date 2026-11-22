"""Structure v2 operations: realization and skip rules on fixed sentences.

These need only the pipeline's spaCy model (en_core_web_sm) and lemminflect.
"""

import pytest

from cleartext.structure_v2 import StructureV2
from cleartext.structure_v2.common import parse
from cleartext.structure_v2.passive import ActiveVoice


def run(op, sentence, context=()):
    return StructureV2(ops=(op,)).rewrite_sentence(sentence, [parse(c) for c in context])[0]


@pytest.mark.parametrize("before, after", [
    ("The cake was eaten by John.", "John ate the cake."),
    ("The house has been sold by them.", "They have sold the house."),
    ("The email was not read by the manager.", "The manager did not read the email."),
    ("The bill wasn't signed by the governor.", "The governor didn't sign the bill."),
    ("The project is being reviewed by the committee.", "The committee is reviewing the project."),
    ("He was given a book by his mother.", "His mother gave him a book."),
    ("It was decided by the board that the plant would close.", "The board decided that the plant would close."),
    ("An investigation into the cause of the incident has been launched by the UN.",
     "The UN has launched an investigation into the cause of the incident."),
    ("Every year, there are up to 2,000 quakes that can be felt by people.",
     "Every year, there are up to 2,000 quakes that people can feel."),
    # Plural agreement for a coordinated agent; hyphenated words kept intact.
    ("The pair are subsequently joined by family, friends and colleagues of Rebecca.",
     "Family, friends and colleagues of Rebecca subsequently join the pair."),
    ("Spain's tanks were replaced by Italian anti-tank vehicles.",
     "Italian anti-tank vehicles replaced Spain's tanks."),
    # A manner adverb after the participle moves before the verb.
    ("The natural environment of the area has been shaped largely by its geology.",
     "Its geology has largely shaped the natural environment of the area."),
])
def test_agentive_realization(before, after):
    assert run("passive", before) == after


@pytest.mark.parametrize("sentence, reason", [
    ("Qyra was found by a passerby at noon, and was pronounced dead by authorities at the scene.",
     "coordinated predicate shares the subject"),
    ("On May 31, Boyle was admitted by ambulance to the clinic in London.", "bare noun agent (a means, not a doer)"),
    ("After suffering from health problems, he was replaced by the former manager Alex McLeish.",
     "fronted modifier would attach to the new subject"),
    ("The titles were acquired by Roberts & Vinter, who hired Kyril Bonfiglioli to edit the magazine.",
     "bracket, comma or list inside the agent"),
    ("The final year of the magazine, as it was titled by that time, included good material.",
     "time phrase, not an agent"),
])
def test_agentive_skips(sentence, reason):
    sent = list(parse(sentence).sents)[0]
    reasons = [r["reason"] for r in ActiveVoice().candidates(sent)]
    assert reason in reasons
    assert run("passive", sentence) == sentence


def test_recovered_we_needs_context_and_author_verb():
    s = "The samples were analyzed using mass spectrometry."
    assert run("passive_recovered", s) == s  # no "we" in the context
    ctx = ["We collected blood from 40 volunteers in 2020."]
    assert run("passive_recovered", s, ctx) == "We analyzed the samples using mass spectrometry."
    # Not an author action: the doer is someone else.
    other = "Most of the comments were directed at two issues."
    assert run("passive_recovered", other, ctx) == other
    # Agentive passives are left to the agentive arm.
    assert run("passive_recovered", "The cake was eaten by John.", ctx) == "The cake was eaten by John."


@pytest.mark.parametrize("op, before, after", [
    ("appositive", "Joel Anderson, a member of the State Assembly, won the primary on Tuesday.",
     "Joel Anderson won the primary on Tuesday. Joel Anderson was a member of the State Assembly."),
    ("appositive", "The plan, which was approved by the council in May, cuts costs for most residents.",
     "The plan cuts costs for most residents. The plan was approved by the council in May."),
    ("appositive", "Most of the legal team is based in Amman, the capital of Jordan.",
     "Most of the legal team is based in Amman. Amman is the capital of Jordan."),
    ("fronted_split", "Although the company reported strong earnings in the first quarter, its shares fell sharply.",
     "The company reported strong earnings in the first quarter. But its shares fell sharply."),
    ("fronted_split", "Because the uptake rate did not develop as we expected, our business was harmed badly.",
     "The uptake rate did not develop as we expected. So our business was harmed badly."),
    ("fronted_move", "If the flags sub-TLV is not included in these cases, receivers will be unable to find the source.",
     "Receivers will be unable to find the source if the flags sub-TLV is not included in these cases."),
    ("frames", "There are many people who live in the old town near the river.",
     "Many people live in the old town near the river."),
    ("frames", "There were five people killed in the attack on the market.",
     "Five people were killed in the attack on the market."),
    ("frames", "It was the board that approved the plan in 2019.", "The board approved the plan in 2019."),
    ("nominal", "The agency will conduct an evaluation of the program next year.",
     "The agency will evaluate the program next year."),
    ("nominal", "The committee made a decision on the matter.", "The committee decided on the matter."),
    ("nominal", "We provide assistance to applicants who apply online.", "We assist applicants who apply online."),
])
def test_other_operations(op, before, after):
    assert run(op, before) == after


@pytest.mark.parametrize("op, sentence", [
    ("appositive", "All prisoners were released after talks between Forster and Arnold, who brought a large force."),
    ("appositive", "Grimaldi starred in a pantomime version of Robinson Crusoe, which confirmed his credentials."),
    ("appositive", 'One witness, who would not release his name, said, "It was a direct hit.'),
    ("fronted_split", "Because of the benzyl alcohol in the diluent, the powder should be mixed with sterile water."),
    ("fronted_move", "When the input is longer than the limit and exceeds the resources, then the mechanism MUST fail."),
    ("frames", "There is no evidence that the drug causes harm in adults."),
    ("frames", "It is important that you read the label before use."),
    ("nominal", "The agency made a careful assessment of the risks."),
    ("nominal", "Students take a test at the end of the year."),
])
def test_other_operations_abstain(op, sentence):
    assert run(op, sentence) == sentence


def test_hard_checks_drop_an_edit():
    # Repeating an anchor that contains a number changes the number count.
    s = "The trial has been adjourned until 16 October, which is when the court is expected to rule."
    out, accepted, rejected = StructureV2(ops=("appositive",)).rewrite_sentence(s)
    assert out == s and not accepted


def test_stage_passage_shape():
    text = "We measured the samples twice. The results were checked by an independent lab. Nothing else changed."
    out, edits = StructureV2(ops=("passive",)).rewrite(text)
    assert out == "We measured the samples twice. An independent lab checked the results. Nothing else changed."
    assert [e["op"] for e in edits] == ["passive"]
