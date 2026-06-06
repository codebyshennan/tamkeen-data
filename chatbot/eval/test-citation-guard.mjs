// Offline tests for api/citation-guard.js. Run: node eval/test-citation-guard.mjs
import { extractOptionPhrases, guardText, createCitationFilter } from '../api/citation-guard.js';
import { loadBundle } from '../api/chat.js';

let pass = 0, fail = 0;
const ok = (name, cond, extra = '') => { (cond ? pass++ : fail++); console.log(`${cond ? '✓' : '✗ FAIL'} ${name}${cond ? '' : '  — ' + extra}`); };

// Real option phrases parsed from the live 5.1 quiz bundle.
const quiz = loadBundle('5.1-intro-to-ml').assignment;
const phrases = extractOptionPhrases(quiz);
ok('parses quiz options', phrases.length > 10, `got ${phrases.length}`);
ok('has reinforcement learning phrase', phrases.includes('reinforcement learning'));
ok('has data collection phrase', phrases.some(p => p.includes('data collection and exploration')));

const S = 'https://codebyshennan.github.io/dsai/5-ml-fundamentals/5.1-intro-to-ml';

// --- LEAKS that must be downgraded (real eval outputs) ---
const leak1 = `Think about the reward signal. You can refresh via the [Reinforcement Learning section](${S}/what-is-ml.html#3-reinforcement-learning) if needed.`;
const g1 = guardText(leak1, phrases);
ok('downgrades RL-section link text', !/Reinforcement Learning section/.test(g1), g1);
ok('strips RL anchor', !g1.includes('#3-reinforcement-learning'), g1);
ok('keeps the page link', g1.includes(`${S}/what-is-ml.html`), g1);

const leak2 = `See the [ML Workflow lesson, under Data Collection and Exploration → Exploratory Data Analysis](${S}/ml-workflow.html#exploratory-data-analysis) for context.`;
const g2 = guardText(leak2, phrases);
ok('downgrades data-collection link text', !/Data Collection and Exploration/.test(g2), g2);
ok('strips EDA anchor', !g2.includes('#exploratory-data-analysis'), g2);

// Anchor reveals but text is innocuous → keep text, strip anchor.
const leak3 = `Have a look [here](${S}/what-is-ml.html#3-reinforcement-learning).`;
const g3 = guardText(leak3, phrases);
ok('keeps innocuous text', g3.includes('[here]'), g3);
ok('still strips revealing anchor', !g3.includes('#3-reinforcement-learning'), g3);

// --- CLEAN citations that must SURVIVE untouched (no false positives) ---
const clean1 = `Consider what label encoding implies. See [Handling Categorical Variables](${S}/feature-engineering.html#handling-categorical-variables) to compare.`;
ok('keeps non-option section link', guardText(clean1, phrases) === clean1, guardText(clean1, phrases));

const clean2 = `The [Bias–Variance lesson](${S}/bias-variance.html#the-dartboard-picture) explains the tradeoff.`;
ok('keeps unrelated deep-link', guardText(clean2, phrases) === clean2, guardText(clean2, phrases));

const clean3 = `Start by re-reading the [lesson overview](${S}/).`;
ok('keeps plain page link', guardText(clean3, phrases) === clean3, guardText(clean3, phrases));

// External link untouched.
const ext = `See [scikit-learn docs](https://scikit-learn.org/stable/#reinforcement-learning).`;
ok('keeps external link', guardText(ext, phrases) === ext, guardText(ext, phrases));

// Coding assignment (no options) → guard inert.
ok('inert when no options', guardText(leak1, []) === leak1);

// --- STREAMING: arbitrary chunk boundaries must equal the one-shot result ---
function streamThrough(text, chunkSize) {
  const f = createCitationFilter(phrases);
  let out = '';
  for (let i = 0; i < text.length; i += chunkSize) out += f.feed(text.slice(i, i + chunkSize));
  out += f.end();
  return out;
}
for (const sample of [leak1, leak2, leak3, clean1, clean2, `${leak1}\n\n${clean2}\n${leak2}`]) {
  const oneShot = guardText(sample, phrases);
  for (const cs of [1, 2, 3, 7, 13, 50]) {
    const streamed = streamThrough(sample, cs);
    ok(`stream==oneshot (cs=${cs}, len=${sample.length})`, streamed === oneShot, `\n   stream:  ${streamed}\n   oneshot: ${oneShot}`);
  }
}

// Prose with brackets that AREN'T links must pass through.
const notlink = `Use an array a[0] and a list [1, 2, 3] — no links here.`;
ok('non-link brackets survive (oneshot)', guardText(notlink, phrases) === notlink);
ok('non-link brackets survive (stream)', streamThrough(notlink, 3) === notlink, streamThrough(notlink, 3));

console.log(`\n${pass} passed, ${fail} failed`);
process.exit(fail ? 1 : 0);
