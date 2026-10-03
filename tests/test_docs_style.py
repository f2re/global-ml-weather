"""Formal documentation checks do not certify scientific or linguistic quality."""
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('check_docs', ROOT/'scripts/check_docs.py')
docs = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(docs)


def test_wrapped_long_sentence_is_detected():
    text = ' '.join(['слово']*20)+'\n'+' '.join(['слово']*11)+'.'
    assert docs.sentence_issues(text)[0]['words'] == 31


def test_multiple_short_sentences_remain_separate():
    text = (' '.join(['слово']*20)+'. ')*3
    assert not docs.sentence_issues(text)


def test_procedure_has_own_limit():
    text = '1. '+' '.join(['слово']*26)+'.'
    assert docs.sentence_issues(text)[0]['limit'] == 25


def test_fenced_code_frontmatter_and_inline_ids_do_not_create_prose_failures():
    text = '---\nname: '+('x '*80)+'\n---\n```python\n'+('x '*80)+'\n```\nПрочитайте `path with many words`.'
    assert not docs.sentence_issues(text)


def test_table_links_are_checked(tmp_path):
    text = '| [Документ](missing.md) | значение |\n'
    issues, count = docs.link_issues(text, tmp_path/'README.md', tmp_path)
    assert count == 1 and issues[0]['rule'] == 'local_link'


def test_examples_do_not_create_broken_link_failures(tmp_path):
    text = '```text\n[example](missing.md)\n```\n'
    assert docs.link_issues(text, tmp_path/'README.md', tmp_path) == ([], 0)


def test_parent_link_inside_repository_is_valid(tmp_path):
    (tmp_path/'README.md').write_text('root')
    (tmp_path/'docs').mkdir()
    issues, count = docs.link_issues('[Главная](../README.md)', tmp_path/'docs/a.md', tmp_path)
    assert count == 1 and issues == []


def test_outside_root_is_not_a_valid_link(tmp_path):
    issues, _ = docs.link_issues('[Файл](../../outside)', tmp_path/'README.md', tmp_path)
    assert issues


def test_empty_scope_does_not_pass(tmp_path):
    config = tmp_path/'style.json'
    config.write_text(json.dumps({'profile':'RU-TECH-1', 'include':['absent.md'],
                                 'limits':{'procedure_words':25, 'description_words':30}}))
    assert docs.check_repository(tmp_path, config)['status'] == 'failed'


def test_unclosed_code_is_not_silently_ignored():
    try:
        docs.prose_blocks('```python\nunclosed')
    except ValueError:
        pass
    else:
        raise AssertionError('An unclosed example must not hide remaining prose.')


def test_repository_docs_pass_declared_profile():
    report = docs.check_repository(ROOT)
    assert report['files_checked'] >= 35
    assert report['status'] == 'passed', json.dumps(report['issues'], ensure_ascii=False, indent=2)
    assert report['certified_standard_compliance'] is False
    assert report['manual_review_required'] is True
