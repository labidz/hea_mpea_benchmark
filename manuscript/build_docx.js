// Build the Elsevier-style manuscript .docx from manuscript/spec.json
const fs = require('fs');
const path = require('path');
const {
  Document, Packer, Paragraph, TextRun, HeadingLevel, AlignmentType, ImageRun, Table, TableRow, TableCell,
  WidthType, BorderStyle, ShadingType, Footer, PageNumber, LineNumberRestartFormat, TabStopType, VerticalAlign,
  Math: OMath, MathRun, MathSubScript, MathSuperScript, MathSubSuperScript, MathFraction, MathSum, MathRoundBrackets,
  MathSquareBrackets, MathCurlyBrackets, PageBreak, Header,
} = require('docx');

const spec = JSON.parse(fs.readFileSync(process.argv[2], 'utf8'));
const OUT = process.argv[3];
const FONT = 'Times New Roman';
const SZ = 24;              // 12 pt body
const TEXT_W = 9026;        // A4 (11906) minus 2 x 1440 margins
const LINE = 360;           // 1.5 line spacing

function runs(rs, base = {}) {
  return rs.map(r => new TextRun({
    text: r.t, italics: !!r.i || !!base.italics, bold: !!r.b || !!base.bold,
    superScript: !!r.sup, subScript: !!r.sub, font: r.mono ? 'Courier New' : FONT,
    size: base.size || SZ, color: base.color,
  }));
}

function para(rs, opts = {}) {
  return new Paragraph({
    children: runs(rs, opts.base || {}),
    alignment: opts.align || AlignmentType.JUSTIFIED,
    spacing: { line: opts.line || LINE, after: opts.after === undefined ? 120 : opts.after, before: opts.before || 0 },
    indent: opts.indent,
    keepNext: opts.keepNext, keepLines: opts.keepLines,
    heading: opts.heading,
  });
}

// ---------------- equations (OMML via docx Math components)
const M = t => new MathRun(t);
const sub = (b, s) => new MathSubScript({ children: [M(b)], subScript: [M(s)] });
const sup = (b, s) => new MathSuperScript({ children: [M(b)], superScript: [M(s)] });
const subsup = (b, s, p) => new MathSubSuperScript({ children: [M(b)], subScript: [M(s)], superScript: [M(p)] });
const EQ = {
  cvplus: () => [
    sub('Ĉ', 'α'), M('(𝐱) = '),
    new MathSquareBrackets({ children: [
      subsup('q̂', 'α', '−'), new MathCurlyBrackets({ children: [sub('μ̂', '−k(i)'), M('(𝐱) − '), sub('R', 'i')] }), M(' , '),
      subsup('q̂', 'α', '+'), new MathCurlyBrackets({ children: [sub('μ̂', '−k(i)'), M('(𝐱) + '), sub('R', 'i')] }),
    ] }),
  ],
  ncvp: () => [
    subsup('Ĉ', 'α', 'norm'), M('(𝐱) = '),
    new MathSquareBrackets({ children: [
      subsup('q̂', 'α', '−'), new MathCurlyBrackets({ children: [sub('μ̂', '−k(i)'), M('(𝐱) − '),
        new MathFraction({ numerator: [sub('R', 'i')], denominator: [sub('s', 'i')] }), sub('s', '−k(i)'), M('(𝐱)')] }), M(' , '),
      subsup('q̂', 'α', '+'), new MathCurlyBrackets({ children: [sub('μ̂', '−k(i)'), M('(𝐱) + '),
        new MathFraction({ numerator: [sub('R', 'i')], denominator: [sub('s', 'i')] }), sub('s', '−k(i)'), M('(𝐱)')] }),
    ] }),
  ],
  ad: () => [
    sub('d', '5'), M('(𝐱) = '), new MathFraction({ numerator: [M('1')], denominator: [M('5')] }),
    new MathSum({ children: [M('‖𝐳(𝐱) − 𝐳('), sub('𝐱', 'j'), M(')‖')], subScript: [M('j ∈ '), sub('𝒩', '5'), M('(𝐱)')] }),
  ],
};

function equation(name, num) {
  return new Paragraph({
    children: [new TextRun({ text: '\t', font: FONT, size: SZ }), new OMath({ children: EQ[name]() }),
      new TextRun({ text: `\t(${num})`, font: FONT, size: SZ })],
    tabStops: [{ type: TabStopType.CENTER, position: Math.round(TEXT_W / 2) }, { type: TabStopType.RIGHT, position: TEXT_W }],
    spacing: { before: 120, after: 120, line: LINE },
  });
}

// ---------------- tables
const thin = { style: BorderStyle.SINGLE, size: 4, color: '000000' };
const none = { style: BorderStyle.NONE, size: 0, color: 'FFFFFF' };
function table(t) {
  const W = t.widths.reduce((a, b) => a + b, 0);
  const scale = TEXT_W / W;
  const widths = t.widths.map(w => Math.floor(w * scale));
  widths[widths.length - 1] += TEXT_W - widths.reduce((a, b) => a + b, 0);
  const cell = (rs, w, isHead, isLastRow, isFirstBody, align) => new TableCell({
    width: { size: w, type: WidthType.DXA },
    margins: { top: 30, bottom: 30, left: 70, right: 70 },
    verticalAlign: VerticalAlign.CENTER,
    borders: { left: none, right: none, top: isHead ? thin : none, bottom: (isHead || isLastRow) ? thin : none },
    children: [new Paragraph({ children: runs(rs, { size: 18, bold: isHead }), alignment: align,
      keepNext: !isLastRow || !!t.notes, spacing: { line: 240, after: 0 } })],
  });
  const rows = [new TableRow({ tableHeader: true, children: t.header.map((h, j) => cell(h, widths[j], true, false, false,
    j === 0 ? AlignmentType.LEFT : AlignmentType.CENTER)) })];
  const isNum = rs => /^[\s\d.,±–\-−+%()\[\]†a-z*^{}]*$/i.test(rs.map(x => x.t).join('')) && /\d|–/.test(rs.map(x => x.t).join('')) && rs.map(x => x.t).join('').length < 28;
  t.rows.forEach((r, i) => rows.push(new TableRow({ cantSplit: true, children: r.map((c, j) => cell(c, widths[j], false,
    i === t.rows.length - 1, i === 0, (j === 0 || !isNum(c)) ? AlignmentType.LEFT : AlignmentType.CENTER)) })));
  return new Table({ width: { size: TEXT_W, type: WidthType.DXA }, columnWidths: widths, rows,
    borders: { top: none, bottom: none, left: none, right: none, insideHorizontal: none, insideVertical: none } });
}

// ---------------- assemble
const children = [];
const H = (rs, level) => para(rs, { heading: level, align: AlignmentType.LEFT, base: { bold: true, size: level === HeadingLevel.HEADING_1 ? 26 : 24, italics: level === HeadingLevel.HEADING_3 }, before: 240, after: 120, keepNext: true });
for (const b of spec.blocks) {
  if (b.type === 'title') children.push(para(b.runs, { align: AlignmentType.LEFT, base: { bold: true, size: 32 }, line: 300, after: 240 }));
  else if (b.type === 'plain') children.push(para(b.runs, { align: b.align === 'left' ? AlignmentType.LEFT : AlignmentType.JUSTIFIED, line: b.line || LINE, after: b.after === undefined ? 120 : b.after, base: { size: b.size || SZ, italics: b.italic, bold: b.bold } }));
  else if (b.type === 'h1') children.push(H(b.runs, HeadingLevel.HEADING_1));
  else if (b.type === 'h2') children.push(H(b.runs, HeadingLevel.HEADING_2));
  else if (b.type === 'h3') children.push(H(b.runs, HeadingLevel.HEADING_3));
  else if (b.type === 'p') children.push(para(b.runs));
  else if (b.type === 'bullet') children.push(para([{ t: '•  ' }].concat(b.runs), { indent: { left: 360, hanging: 360 }, after: 60, align: AlignmentType.LEFT }));
  else if (b.type === 'ref') children.push(para(b.runs, { indent: { left: 540, hanging: 540 }, after: 60, line: 276, align: AlignmentType.LEFT, base: { size: 22 } }));
  else if (b.type === 'eq') children.push(equation(b.name, b.num));
  else if (b.type === 'pagebreak') children.push(new Paragraph({ children: [new PageBreak()] }));
  else if (b.type === 'figure') {
    const img = fs.readFileSync(b.path);
    const wpx = Math.round(b.width_mm / 25.4 * 96);
    const hpx = Math.round(wpx * b.aspect);
    children.push(new Paragraph({ alignment: AlignmentType.CENTER, keepNext: true, spacing: { before: 200, after: 80 },
      children: [new ImageRun({ type: 'png', data: img, transformation: { width: wpx, height: hpx },
        altText: { title: b.alt, description: b.alt, name: b.alt } })] }));
    children.push(para(b.caption, { align: AlignmentType.JUSTIFIED, line: 276, after: 240, keepLines: true, base: { size: 20 } }));
  } else if (b.type === 'table') {
    children.push(para(b.caption, { align: AlignmentType.LEFT, line: 276, after: 80, keepNext: true, base: { size: 20 } }));
    children.push(table(b));
    if (b.notes) children.push(para(b.notes, { align: AlignmentType.LEFT, line: 240, after: 240, base: { size: 18 } }));
    else children.push(new Paragraph({ children: [], spacing: { after: 200 } }));
  }
}

const doc = new Document({
  creator: spec.meta.creator, title: spec.meta.title, description: spec.meta.description,
  styles: { default: { document: { run: { font: FONT, size: SZ } } },
    paragraphStyles: [
      { id: 'Heading1', name: 'Heading 1', basedOn: 'Normal', next: 'Normal', quickFormat: true, run: { font: FONT, size: 26, bold: true }, paragraph: { spacing: { before: 240, after: 120 }, outlineLevel: 0 } },
      { id: 'Heading2', name: 'Heading 2', basedOn: 'Normal', next: 'Normal', quickFormat: true, run: { font: FONT, size: 24, bold: true }, paragraph: { spacing: { before: 200, after: 100 }, outlineLevel: 1 } },
      { id: 'Heading3', name: 'Heading 3', basedOn: 'Normal', next: 'Normal', quickFormat: true, run: { font: FONT, size: 24, italics: true }, paragraph: { spacing: { before: 160, after: 80 }, outlineLevel: 2 } },
    ] },
  sections: [{
    properties: {
      page: { size: { width: 11906, height: 16838 }, margin: { top: 1440, bottom: 1440, left: 1440, right: 1440 } },
      lineNumbers: { countBy: 1, restart: LineNumberRestartFormat.CONTINUOUS, distance: 300 },
    },
    footers: { default: new Footer({ children: [new Paragraph({ alignment: AlignmentType.CENTER,
      children: [new TextRun({ children: [PageNumber.CURRENT], font: FONT, size: 20 })] })] }) },
    children,
  }],
});
Packer.toBuffer(doc).then(buf => { fs.writeFileSync(OUT, buf); console.log('wrote', OUT, buf.length); });
