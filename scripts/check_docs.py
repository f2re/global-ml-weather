"""Check selected RU-TECH-1 rules, not official ASD-STE100 compliance.

No network or third-party dependencies. Tables, headings, fenced examples and
front matter are excluded from sentence length; their meaning needs review.
Local link paths are checked, but fragment identifiers and external URLs are not.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import re
from urllib.parse import unquote, urlsplit

LINK = re.compile(r'(?<!!)\[([^\]\n]+)\]\(([^)\n]+)\)')
INLINE_CODE = re.compile(r'`+[^`\n]*`+')
WORDS = re.compile(r"[\w]+(?:[-’'][\w]+)*", re.UNICODE)
ITEM = re.compile(r'^\s*(?:(\d+)[.)]|[-*+])\s+')


def prose_blocks(text):
    """Return line-numbered prose blocks; preserve wrapped paragraphs."""
    blocks = []
    parts, start, procedural = [], 0, False
    fence = None
    front_matter = False
    def flush():
        nonlocal parts
        if parts:
            blocks.append((start, procedural, ' '.join(parts)))
            parts = []
    for number, line in enumerate(text.splitlines(), 1):
        stripped = line.strip()
        if number == 1 and stripped == '---':
            front_matter = True
            continue
        if front_matter:
            if stripped == '---': front_matter = False
            continue
        if fence:
            if re.fullmatch(re.escape(fence[0])+r'{'+str(fence[1])+r',}\s*', stripped):
                fence = None
            continue
        marker = re.match(r'^(`{3,}|~{3,})', stripped)
        if marker:
            flush(); fence = (marker[1][0], len(marker[1])); continue
        if not stripped or stripped.startswith(('#', '|', '<!--')) or re.fullmatch(r'[-*_]{3,}', stripped):
            flush(); continue
        match = ITEM.match(line)
        if match:
            flush(); start = number; procedural = match[1] is not None
            parts = [line[match.end():]]
        else:
            if not parts: start = number; procedural = False
            parts.append(stripped)
    flush()
    if fence or front_matter:
        raise ValueError('Unclosed fenced block or front matter.')
    return blocks


def sentence_issues(text, *, procedure_words=25, description_words=30):
    issues = []
    for line, procedural, block in prose_blocks(text):
        block = INLINE_CODE.sub('CODE', block)
        block = LINK.sub(lambda m: m[1], block)
        block = re.sub(r'https?://\S+', 'URL', block)
        limit = procedure_words if procedural else description_words
        for sentence in re.split(r'(?<=[.!?])\s+', block):
            count = len(WORDS.findall(sentence))
            if count > limit:
                issues.append(dict(line=line, rule='sentence_length', words=count, limit=limit,
                                   text=sentence[:180]))
    return issues


def visible_markdown(text):
    """Keep link-bearing tables/headings, but ignore fenced code and inline code."""
    out, fence, front = [], None, False
    for number, line in enumerate(text.splitlines(), 1):
        stripped = line.strip()
        if number == 1 and stripped == '---': front = True; continue
        if front:
            if stripped == '---': front = False
            continue
        if fence:
            if re.fullmatch(re.escape(fence[0])+r'{'+str(fence[1])+r',}\s*', stripped): fence = None
            continue
        marker = re.match(r'^(`{3,}|~{3,})', stripped)
        if marker: fence = (marker[1][0], len(marker[1])); continue
        out.append((number, INLINE_CODE.sub('CODE', line)))
    return out


def link_issues(text, path, root):
    issues, checked = [], 0
    root = Path(root).resolve()
    for number, line in visible_markdown(text):
        for match in LINK.finditer(line):
            value = match[2].strip().split(' ', 1)[0].strip('<>')
            parsed = urlsplit(value)
            if parsed.scheme or parsed.netloc or not parsed.path: continue
            checked += 1
            target = (Path(path).parent/unquote(parsed.path)).resolve()
            if not target.is_relative_to(root) or not target.exists():
                issues.append(dict(line=number, rule='local_link', target=value))
    return issues, checked


def check_repository(root, config_path=None):
    root = Path(root).resolve()
    config_path = Path(config_path) if config_path else root/'configs/docs_style.json'
    if not config_path.is_file(): raise ValueError('Missing documentation configuration.')
    config = json.loads(config_path.read_text(encoding='utf-8'))
    patterns = config.get('include')
    if config.get('profile') != 'RU-TECH-1' or not isinstance(patterns, list) or not patterns:
        raise ValueError('Expected RU-TECH-1 configuration with explicit nonempty scope.')
    limits = config.get('limits', {})
    for key in ('procedure_words', 'description_words'):
        if type(limits.get(key)) is not int or not 1 <= limits[key] <= 60:
            raise ValueError('Invalid configured sentence limit.')
    paths, issues = set(), []
    for pattern in patterns:
        if not isinstance(pattern,str) or Path(pattern).is_absolute() or '..' in Path(pattern).parts:
            raise ValueError('Scope must remain inside the repository.')
        matches = [p for p in root.glob(pattern) if p.is_file() and p.suffix == '.md']
        if not matches: issues.append(dict(file=pattern, rule='empty_scope_pattern'))
        paths.update(matches)
    checked_links = 0
    hashes = {}
    for path in sorted(paths):
        relative = path.relative_to(root).as_posix()
        if path.is_symlink() or not path.resolve().is_relative_to(root):
            issues.append(dict(file=relative, rule='unsafe_path')); continue
        raw = path.read_bytes(); hashes[relative] = hashlib.sha256(raw).hexdigest()
        text = raw.decode('utf-8')
        try:
            found = sentence_issues(text, **limits)
            links, count = link_issues(text, path, root)
            checked_links += count
            issues.extend(dict(file=relative, **item) for item in found+links)
        except ValueError as exc:
            issues.append(dict(file=relative, rule='markdown_structure', error=str(exc)))
    if not paths: issues.append(dict(file='*', rule='no_documents_checked'))
    return dict(profile='RU-TECH-1', status='failed' if issues else 'passed', files_checked=len(paths),
                local_links_checked=checked_links, issues=issues, source_sha256=hashes,
                certified_standard_compliance=False, manual_review_required=True,
                excluded_checks=['Russian grammar', 'one action per instruction', 'scientific truth',
                                 'official STE dictionary', 'external URLs', 'link fragments', 'table sentence lengths'])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--config', type=Path)
    parser.add_argument('--strict', action='store_true')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args(argv)
    try:
        report = check_repository(args.root, args.config)
    except (ValueError, OSError) as exc:
        report = dict(profile='RU-TECH-1', status='failed', error=str(exc),
                      certified_standard_compliance=False, manual_review_required=True)
    rendered = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)+'\n'
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding='utf-8')
    print(rendered, end='')
    return 1 if report['status'] == 'failed' and args.strict else 0


if __name__ == '__main__': raise SystemExit(main())
