import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from cleartext.rewrites import structural,phrase_rewrite
def test_passive_roles_and_tense():
    assert structural('The report was published by the committee.')[0]=='The committee published the report.'
    assert structural('The meals are prepared by the chefs.')[0]=='The chefs prepare the meals.'
def test_split_requires_independent_subject():
    assert structural('The engineer tested the software, and the manager reviewed the report.')[0]=='The engineer tested the software. The manager reviewed the report.'
    text='The engineer tested the software and reviewed the report.'
    assert structural(text)[0]==text
def test_does_not_drop_scope_or_modality():
    for text in ['If the server fails, the backup may start.','The report was not published by the committee.',
                 'The report may be published by the committee.','Was the report published by the committee?',
                 '"The report was published by the committee."','The report was published by the committee yesterday.']:
        assert structural(text)[0]==text
def test_phrase_boundaries_and_no_forced_rewrite():
    scorer=lambda s:.5 if len(s.split())>1 else .1
    assert phrase_rewrite('The office is closed at the present time.',scorer)[0]=='The office is closed now.'
    assert phrase_rewrite('The office is closed at the present times.',scorer)[0]=='The office is closed at the present times.'
    assert phrase_rewrite('She studies in order to pass.',lambda s:.2)[0]=='She studies in order to pass.'
def test_keep_preserves_whitespace():
    text='  Simple text.\n'
    assert structural(text)[0]==text
    assert phrase_rewrite(text,lambda s:.1)[0]==text
