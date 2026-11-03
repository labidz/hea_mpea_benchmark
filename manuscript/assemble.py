"""Assemble manuscript spec (JSON) from text modules + results, number citations, then build .docx with node."""
import json
import re
import subprocess
import sys
from references import REFS

MINUS = re.compile(r'(?<![A-Za-z0-9\-/._])-(?=\d)')
TOK = re.compile(r'(\*\*.+?\*\*|\*.+?\*|\^\{.+?\}|_\{.+?\})')


def parse_runs(text):
    """Markup -> runs. *italic*, **bold**, ^{sup}, _{sub}. Nested: sup/sub content may contain *italic*."""
    runs = []
    for part in TOK.split(text):
        if not part:
            continue
        if part.startswith('**') and part.endswith('**'):
            runs += [dict(r, b=True) for r in parse_runs(part[2:-2])]
        elif part.startswith('*') and part.endswith('*') and len(part) > 2:
            runs += [dict(r, i=True) for r in parse_runs(part[1:-1])]
        elif part.startswith('^{'):
            runs += [dict(r, sup=True) for r in parse_runs(part[2:-1])]
        elif part.startswith('_{'):
            runs += [dict(r, sub=True) for r in parse_runs(part[2:-1])]
        else:
            runs.append({'t': part})
    return runs


class Citer:
    def __init__(self):
        self.order = []

    def num(self, key):
        if key not in REFS:
            raise KeyError(f"unknown reference key {key}")
        if key not in self.order:
            self.order.append(key)
        return self.order.index(key) + 1

    def sub(self, text):
        return MINUS.sub('−', self._sub(text))

    def _sub(self, text):
        def rep(m):
            keys = [k.strip() for k in m.group(1).split(',')]
            nums = sorted(self.num(k) for k in keys)
            # compress ranges
            out, i = [], 0
            while i < len(nums):
                j = i
                while j + 1 < len(nums) and nums[j + 1] == nums[j] + 1:
                    j += 1
                out.append(f"{nums[i]}–{nums[j]}" if j - i >= 2 else ",".join(map(str, nums[i:j + 1])))
                i = j + 1
            return "[" + ",".join(out) + "]"
        return re.sub(r'\[@([^\]]+)\]', rep, text)


def build(blocks, meta, out_json, out_docx):
    citer = Citer()
    spec_blocks = []
    eqn = 0
    for b in blocks:
        kind = b[0]
        if kind in ('h1', 'h2', 'h3', 'p', 'title', 'bullet'):
            spec_blocks.append({'type': kind, 'runs': parse_runs(citer.sub(b[1]))})
        elif kind == 'plain':
            opts = b[2] if len(b) > 2 else {}
            spec_blocks.append(dict({'type': 'plain', 'runs': parse_runs(citer.sub(b[1]))}, **opts))
        elif kind == 'eq':
            eqn += 1
            spec_blocks.append({'type': 'eq', 'name': b[1], 'num': eqn})
        elif kind == 'figure':
            f = b[1]
            spec_blocks.append({'type': 'figure', 'path': f['path'], 'width_mm': f['width_mm'], 'aspect': f['aspect'],
                                'alt': f['alt'], 'caption': parse_runs(citer.sub(f['caption']))})
        elif kind == 'table':
            t = b[1]
            spec_blocks.append({'type': 'table', 'caption': parse_runs(citer.sub(t['caption'])),
                                'header': [parse_runs(citer.sub(h)) for h in t['header']],
                                'rows': [[parse_runs(citer.sub(str(c))) for c in r] for r in t['rows']],
                                'widths': t['widths'],
                                'notes': parse_runs(citer.sub(t['notes'])) if t.get('notes') else None})
        elif kind == 'pagebreak':
            spec_blocks.append({'type': 'pagebreak'})
        elif kind == 'references':
            spec_blocks.append({'type': 'h1', 'runs': parse_runs('References')})
            for i, k in enumerate(citer.order):
                spec_blocks.append({'type': 'ref', 'runs': parse_runs(f"[{i + 1}] {REFS[k]}")})
        else:
            raise ValueError(kind)
    json.dump({'meta': meta, 'blocks': spec_blocks}, open(out_json, 'w'), ensure_ascii=False, indent=0)
    r = subprocess.run(['node', '/home/claude/hea/manuscript/build_docx.js', out_json, out_docx], capture_output=True, text=True)
    print(r.stdout, r.stderr[-2000:])
    return citer.order
