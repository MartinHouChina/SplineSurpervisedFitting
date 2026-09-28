"""Render the historical 30-case measurements without modifying source values."""
from pathlib import Path
import re
import textwrap
from reportlab.pdfgen import canvas
import fitz

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'outputs/figures/postprune_paper_usgs_alternative_20260923/six_methods_case_values.md'
OUT = ROOT / 'output/pdf'
OUT.mkdir(parents=True, exist_ok=True)
groups = []
for section in SOURCE.read_text(encoding='utf-8').split('## ')[1:]:
    name = section.splitlines()[0].strip()
    rows = []
    for line in section.splitlines():
        if re.match(r'\| (Ours|Park|Liang|Dung|Kang|Luo) \|', line):
            cells = [x.strip() for x in line.strip('|').split('|')]
            rows.append((cells[0], int(cells[1]), float(cells[2]), float(cells[3]), float(cells[4]), cells[5]))
    assert len(rows) == 6
    groups.append((name, rows))
assert len(groups) == 5
names = {'Synthetic': 'Synthetic', 'UJI': 'UJI handwriting', 'NaturalEarth': 'Natural Earth', 'USGS': 'USGS contours', 'IndustrialOffset': 'Industrial offsets'}
notes = [
    'Selected illustrative cases (one per source), not dataset averages. K denotes internal knots; capacity is 32.',
    'MSE and MaxSE are mean and maximum pointwise squared Euclidean errors in normalized coordinates (no square root).',
    'Pass requires MSE <= 5e-5 and MaxSE <= 5e-4. Total time includes measured numerical postprocessing.',
    'Historical protocol: all six methods use insertion repair; only Ours additionally uses post-pruning. Baselines are repository adaptations.',
    'Bold marks the minimum in each numerical column within a case (ties included); lower error alone does not imply the best simplification.'
]
pdf = OUT / 'six_methods_selected_cases_table.pdf'
c = canvas.Canvas(str(pdf), pagesize=(650, 940))
c.setTitle('Six-method comparison on five selected curve cases')
c.setFont('Times-Bold', 14)
c.drawString(32, 908, 'Table 1. Six-method comparison on five selected curve cases')
c.setFont('Times-Roman', 10)
c.drawString(32, 890, 'Historical 32-internal-knot experiment | Errors without square roots')
xs = [32, 173, 287, 371, 466, 549, 604]
c.setLineWidth(1.2); c.line(32, 876, 618, 876)
c.setFont('Times-Bold', 10)
for x, h in zip(xs, ['Curve source', 'Method', 'K', 'MSE', 'MaxSE', 'Time', 'Pass']):
    (c.drawString if x < 200 else c.drawRightString)(x, 860, h)
c.setFont('Times-Roman', 9)
for x, h in [(371, '(x 10^-5)'), (466, '(x 10^-4)'), (549, '(ms)')]:
    c.drawRightString(x, 846, h)
c.setLineWidth(.6); c.line(32, 838, 618, 838)
y = 821
tex = [r'\documentclass{article}', r'\usepackage[margin=1.5cm]{geometry}', r'\usepackage{booktabs}', r'\begin{document}', r'\begin{table}[htbp]\centering\small', r'\caption{Six-method comparison on five selected curve cases (historical 32-knot experiment).}', r'\label{tab:selected-cases}', r'\begin{tabular}{llrrrrl}', r'\toprule', r'Source & Method & $K$ & MSE ($10^{-5}$) & MaxSE ($10^{-4}$) & Time (ms) & Pass \\', r'\midrule']
for gi, (name, rows) in enumerate(groups):
    mins = [min(r[j] for r in rows) for j in range(1, 5)]
    for ri, row in enumerate(rows):
        method, k, mse, maxse, ms, passed = row
        vals = [str(k), f'{mse/1e-5:.3f}', f'{maxse/1e-4:.3f}', f'{ms:.2f}']
        c.setFont('Times-Roman', 10)
        if ri == 0: c.drawString(xs[0], y, names[name])
        c.drawString(xs[1], y, method)
        tvals = []
        for j, val in enumerate(vals):
            best = row[j+1] == mins[j]
            c.setFont('Times-Bold' if best else 'Times-Roman', 10)
            c.drawRightString(xs[j+2], y, val)
            tvals.append(r'\textbf{' + val + '}' if best else val)
        c.setFont('Times-Roman', 10); c.drawRightString(xs[6], y, passed.capitalize())
        tex.append(' & '.join([names[name] if ri == 0 else '', method, *tvals, passed.capitalize()]) + r' \\')
        y -= 18
    if gi < 4:
        c.setLineWidth(.35); c.line(32, y+7, 618, y+7)
        tex.append(r'\midrule')
        y -= 7
c.setLineWidth(1.2); c.line(32, y+6, 618, y+6)
y -= 12
c.setFont('Times-Roman', 9)
for note in notes:
    for line in textwrap.wrap(note, width=115):
        c.drawString(32, y, line); y -= 11
    y -= 3
assert y > 25, y
c.save()
tex.extend([r'\bottomrule', r'\end{tabular}', r'\par\smallskip\begin{minipage}{\textwidth}\footnotesize', *[n.replace('<=', r'$\leq$') + r'\par' for n in notes], r'\end{minipage}', r'\end{table}', r'\end{document}'])
(OUT / 'six_methods_selected_cases_table.tex').write_text('\n'.join(tex), encoding='utf-8')
doc = fitz.open(pdf)
assert len(doc) == 1
assert 'Industrial offsets' in doc[0].get_text()
doc[0].get_pixmap(matrix=fitz.Matrix(3, 3)).save(str(OUT / 'six_methods_selected_cases_table.png'))
print(pdf)
print('Verified: 30 rows, five sources, one PDF page; PNG and LaTeX exported.')
