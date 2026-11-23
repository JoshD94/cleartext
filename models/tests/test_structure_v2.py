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
    # Plural agreement for a coordinated agent; hyphenated words kept intact.
    ("The pair are subsequently joined by family, friends and colleagues of Rebecca.",
     "Family, friends and colleagues of Rebecca subsequently join the pair."),
    ("Spain's tanks were replaced by Italian anti-tank vehicles.",
     "Italian anti-tank vehicles replaced Spain's tanks."),
    # A manner adverb after the participle moves before the verb.
    ("The natural environment of the area has been shaped largely by the regional geology.",
     "The regional geology has largely shaped the natural environment of the area."),
    # A sentence-final abbreviation keeps the sentence's period.
    ("It was released as the lead single on November 6, 1990, by Sire and Warner Bros.",
     "Sire and Warner Bros. released it as the lead single on November 6, 1990."),
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
    # Round 2 (failure types from the round-1 ratings).
    ("Every year, there are up to 2,000 quakes that can be felt by people.",
     "relative pronoun subject (would become an object relative)"),
    ("Gandhi was 14 when his grandmother, Indira Gandhi, was assassinated by her security guards.",
     "pronoun in the agent would precede its antecedent"),
    ("These are the costs of discharges that are not covered by any financial protection on hand.",
     "relative pronoun subject (would become an object relative)"),
    ("The salts are readily absorbed in the gut by direct absorption of the iron by the bloodstream.",
     "a second by-phrase inside the agent (means, not a doer)"),
    ("Celebrity shows were criticised by public and politicians alike last month.",
     "word right after the agent suggests a misparse"),
    ("Elimination Isoproterenol is metabolized primarily in the liver by COMT.",
     "run-in heading at the sentence start"),
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
    ("appositive_v2", "Results appear in the journal Cancer, a journal published by the American Cancer Society.",
     "Results appear in the journal Cancer. Cancer is a journal published by the American Cancer Society."),
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
    # Round 2 (failure types from the round-1 ratings).
    ("frames_v2", "It is a state trunkline highway that links the two cities."),
    ("frames_v2", "There were also comments asking the agency to expand the data."),
    ("frames_v2", "There were 3 deaths attributed to the disease."),
    ("appositive_v2", "The commenters included a legal advocacy organization, a labor rights organization, and an individual."),
    ("appositive_v2", "He is a member of the Nehru-Gandhi family, the most prominent political family in India."),
    ("appositive_v2", "CTW allowed songwriters to retain the rights to the songs, which allowed them to earn profits."),
    ("appositive_v2", "The campaign was designed to appeal to a Hispanic audience, who were considered at risk."),
    ("nominal", "We have made changes in other areas to reduce the burden on institutions."),
    ("nominal", "This will help them make decisions about how to balance work and study."),
    ("fronted_split", "Although it may not have been identified as such, Congress intended for knowledge to count."),
    ("fronted_split", "Although some comments disagreed, we have kept the sentence, but we have changed its tense."),
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


@pytest.mark.parametrize("op, sentence", [
    # Round 3 (failure types from the round-2 ratings).
    ("passive_v3", "Lidocaine HCl is metabolized rapidly by the liver."),
    ("passive_v3", "Elimination half-life was prolonged by about 60% in cirrhotic subjects."),
    ("passive_v3", "It is flanked by a statue of William Prescott to the south and an entrance building to the north."),
    ("passive_v3", "Bioavailability is also slightly enhanced by the presence of food but unaffected by antacids."),
    ("passive_v3", "Care should be taken since this may be followed by a small slough."),
    ("passive_v3", "If these values are weighted by mass percentage, then the blend has a GWP of about 287."),
    ("passive_v3", "Metabolism — Tadalafil is predominantly metabolized by CYP3A4 to a catechol metabolite."),
    ("appositive_v3", "Ryan supported the American Health Care Act, the House Republican plan to repeal the ACA."),
])
def test_round3_abstains(op, sentence):
    assert run(op, sentence) == sentence


@pytest.mark.parametrize("op, before, after", [
    ("passive_v3", "The execution method has been criticised by some.", "Some have criticised the execution method."),
    ("passive_v3", "The cake was eaten by John.", "John ate the cake."),
    ("appositive_v3", "Marcellus Leonard, a merchant from nearby Saranac Lake, proposed the highway in the 1890s.",
     "Marcellus Leonard proposed the highway in the 1890s. Marcellus Leonard was a merchant from nearby Saranac Lake."),
    ("appositive_v3", "Dementia, a progressive neurodegenerative disorder, affects memory and reasoning.",
     "Dementia affects memory and reasoning. Dementia is a progressive neurodegenerative disorder."),
])
def test_round3_operations(op, before, after):
    assert run(op, before) == after
