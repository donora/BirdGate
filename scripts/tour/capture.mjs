// Capture TOUR.md's screenshots from the running app, with numbered markers.
//   node scripts/tour/capture.mjs <port> <outdir>      (make_tour.sh runs this)
//
// Each shot is a tab, a crop from one heading to the next, and markers placed on
// elements found by their text, so they follow the layout as it changes. A marker
// whose element is missing is reported and left out.
import { chromium } from 'playwright-core';

const [port, out] = process.argv.slice(2);
const SHOTS = [
  { name: 'splits', tab: 'Splits', from: 'Recording-level split', to: 'Traceability',
    marks: [[1, 'label', 'test'], [2, 'testid', 'stDataFrame']] },
  { name: 'occupancy', tab: 'Occupancy', from: 'How much of this data', to: 'The silent-gap distribution',
    marks: [[1, 'label', 'Bird-seconds (union)'], [2, 'text', 'Occupancy varies enormously']] },
  { name: 'gaps', tab: 'Occupancy', from: 'The silent-gap distribution', to: null, marks: [] },
  { name: 'oracle', tab: 'Oracle', from: 'A gate driven by ground truth', to: 'Per-species retention',
    marks: [[1, 'label', 'Hangover (s)'], [2, 'label', 'Min recording (s)'], [3, 'label', 'Saving'],
            [4, 'heading', 'Saving vs (pre-roll']] },
  { name: 'oracle-hour', tab: 'Oracle', from: 'What one hour looks like', to: null, marks: [] },
  { name: 'cochlea', tab: 'Cochlea', from: 'The CAR-IHC front-end', to: 'A cochleagram of real birdsong',
    marks: [[1, 'label', 'Front-end'], [2, 'label', 'Compute'], [3, 'heading', 'Measured channel responses']] },
  { name: 'cochleagram', tab: 'Cochlea', from: 'A cochleagram of real birdsong', to: 'The template vector',
    marks: [[1, 'label', 'Species'], [2, 'testid', 'stAudio'], [3, 'testid', 'stImage', 1]] },
  { name: 'compare', tab: 'Compare', from: 'What each arm would actually have recorded', to: 'How much pure silence',
    marks: [[1, 'label', 'Recording'], [2, 'label', 'Threshold'], [3, 'label', 'Target retention'],
            [4, 'label', 'Background anchor (quantile)'], [5, 'text', 'Teal = birdsong kept']] },
  { name: 'compare-zoom', tab: 'Compare', from: 'Zoom in and listen', to: null, marks: [] },
  { name: 'silicon', tab: 'Silicon', from: 'Where the part actually goes', to: 'Flash —',
    marks: [[1, 'label', 'Arm'], [2, 'label', 'Internal SRAM'], [3, 'heading', 'Internal SRAM —']] },
  { name: 'silicon-core', tab: 'Silicon', from: 'Core time —', to: 'Every stage, with the operation count',
    marks: [] },
];

const b = await chromium.launch({ channel: 'chrome', headless: true });
const p = await b.newPage({ viewport: { width: 1400, height: 5600 }, deviceScaleFactor: 1.5 });
await p.goto(`http://localhost:${port}/`);
await p.getByRole('tab').first().waitFor({ timeout: 300000 });
const idle = async () => {
  await p.waitForTimeout(1500);
  for (let k = 0; k < 600; k++) {
    if (!(await p.locator('[data-testid="stStatusWidget"]').isVisible().catch(() => false))) break;
    await p.waitForTimeout(500);
  }
  await p.waitForTimeout(2500);
};
await idle();

const panel = () => p.locator('[role="tabpanel"]:visible').first();
const heading = (t) => panel().locator('h1,h2,h3,h4,h5,h6').filter({ hasText: t }).first();
function find(kind, t, nth = 0) {
  if (kind === 'heading') return heading(t);
  if (kind === 'testid') return panel().locator(`[data-testid="${t}"]`).nth(nth);
  if (kind === 'label') return panel().locator('[data-testid="stWidgetLabel"], [data-testid="stMetricLabel"]')
    .filter({ hasText: new RegExp('^\\s*' + t.replace(/[()]/g, '\\$&') + '\\s*$') }).first();
  return panel().getByText(t).first();
}

for (const s of SHOTS) {
  await p.getByRole('tab').filter({ hasText: s.tab }).first().click();
  await idle();
  const pb = await panel().boundingBox();
  const top = (await heading(s.from).boundingBox()).y - 14;
  const bottom = s.to ? (await heading(s.to).boundingBox()).y - 18 : pb.y + pb.height + 8;
  const ids = [];
  for (const [n, kind, t, nth] of s.marks) {
    const bb = await find(kind, t, nth).boundingBox({ timeout: 3000 }).catch(() => null);
    if (!bb) { console.log(`  ${s.name}: marker ${n} (${t}) not found`); continue; }
    ids.push(await p.evaluate(([n, x, y]) => {
      const d = document.createElement('div');
      d.className = 'tour-mark';
      d.textContent = n;
      Object.assign(d.style, { position: 'fixed', left: `${x}px`, top: `${y}px`, width: '26px',
        height: '26px', borderRadius: '13px', background: '#e34948', color: '#fff',
        font: '700 15px/26px -apple-system, Helvetica, sans-serif', textAlign: 'center',
        zIndex: 99999, boxShadow: '0 0 0 2px #fff' });
      document.body.appendChild(d);
      return n;
    }, [n, Math.max(bb.x - 32, pb.x - 36), bb.y + Math.min(bb.height, 26) / 2 - 13]));
  }
  await p.screenshot({ path: `${out}/${s.name}.png`,
    clip: { x: pb.x - 40, y: top, width: pb.width + 56, height: bottom - top } });
  await p.evaluate(() => document.querySelectorAll('.tour-mark').forEach((e) => e.remove()));
  console.log('captured', s.name);
}
await b.close();
