"""Highlights file (separate upload required by Elsevier; each bullet <= 85 characters)."""
from assemble import build
from collect import data_stats
from res import Res
from text_front import HIGHLIGHTS


def main(out="/home/claude/hea/manuscript/out/highlights.docx"):
    H = HIGHLIGHTS(data_stats(), Res())
    assert all(len(h) <= 85 for h in H), [len(h) for h in H]
    B = [('h1', 'Highlights')] + [('bullet', h) for h in H]
    build(B, {'creator': 'Rohul Amin', 'title': 'Highlights', 'description': 'Highlights'}, out.replace('.docx', '.json'), out)


if __name__ == '__main__':
    main()
