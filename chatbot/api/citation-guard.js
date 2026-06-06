// Citation answer-leak guard.
//
// The tutor's PROSE reliably stays Socratic, but it sometimes appends a citation
// link to a lesson section whose TITLE (or anchor) literally names the correct
// answer — e.g. for "which learning paradigm is this?" it links
//   [the Reinforcement Learning section](…/what-is-ml.html#3-reinforcement-learning)
// which reveals the answer regardless of how careful the prose was. Prompt rules
// don't reliably stop this (see eval/FINDINGS.md), so we strip it deterministically
// server-side: any docsite link whose visible text or #anchor matches one of THIS
// assignment's answer options gets downgraded to a plain page-level link (anchor
// dropped; revealing link-text neutralised). Links that don't name an option — and
// all citations on coding assignments, which have no options — pass through intact,
// so the "exact location" feature is preserved everywhere it's safe.

const SITE = 'https://codebyshennan.github.io/dsai';
const NEUTRAL_TEXT = 'the relevant lesson section';
const LINK_RE = /\[([^\]]*)\]\(([^)]*)\)/; // [text](url) — no nested ] in text, no ) in url

const normalize = (s) => s.toLowerCase().replace(/[^a-z0-9]+/g, ' ').trim();

// Pull distinctive phrases from an assignment's multiple-choice options.
// Each `- [ ] <option>` line yields its full text plus the "core" before a
// qualifier ("… with …", "(…", ", …") so "One-hot encoding with pd.get_dummies"
// also contributes "one hot encoding". Phrases shorter than 2 words / 9 chars are
// dropped as too generic to match safely. Coding assignments have no such lines,
// so this returns [] and the guard is inert for them.
export function extractOptionPhrases(assignment) {
  const phrases = new Set();
  for (const line of String(assignment || '').split('\n')) {
    const m = line.match(/^\s*[-*]\s*\[[ xX]?\]\s*(.+?)\s*$/);
    if (!m) continue;
    const opt = m[1].replace(/[`*_]/g, '');
    for (const variant of [opt, opt.split(/\s+with\s+|\(/)[0], opt.split(/[,—:;]/)[0]]) {
      const n = normalize(variant);
      if (n.split(' ').length >= 2 && n.length >= 9) phrases.add(n);
    }
  }
  return [...phrases];
}

function revealsAnswer(haystack, phrases) {
  const hay = normalize(haystack);
  return phrases.some((p) => hay.includes(p));
}

// Rewrite a single [text](url). Non-docsite links are untouched. If the link
// names an option via its anchor, drop the anchor; via its text, neutralise the
// text. Otherwise return it unchanged.
export function rewriteLink(text, url, phrases) {
  if (!phrases.length || !url.startsWith(SITE)) return `[${text}](${url})`;
  const hashAt = url.indexOf('#');
  const anchor = hashAt >= 0 ? url.slice(hashAt + 1) : '';
  const base = hashAt >= 0 ? url.slice(0, hashAt) : url;
  const textReveals = revealsAnswer(text, phrases);
  const anchorReveals = anchor && revealsAnswer(anchor, phrases);
  if (!textReveals && !anchorReveals) return `[${text}](${url})`;
  return `[${textReveals ? NEUTRAL_TEXT : text}](${base})`;
}

// Non-streaming: rewrite every markdown link in a complete string. Used by tests
// and by the eval harness so its measured behaviour matches production.
export function guardText(text, phrases) {
  if (!phrases.length) return text;
  return text.replace(new RegExp(LINK_RE, 'g'), (_m, t, u) => rewriteLink(t, u, phrases));
}

// Streaming: a small state machine that passes prose straight through and only
// holds back bytes while inside a potential `[…](…)` link, so it can rewrite the
// completed link before emitting. Latency impact is limited to the span of a link.
// Usage: const f = createCitationFilter(phrases); res.write(f.feed(delta)); … res.write(f.end());
export function createCitationFilter(phrases) {
  let buf = '';
  if (!phrases.length) {
    // Fast path: nothing to guard — pure passthrough.
    return { feed: (s) => s, end: () => '' };
  }
  return {
    feed(chunk) {
      buf += chunk;
      let out = '';
      for (;;) {
        const open = buf.indexOf('[');
        if (open < 0) { out += buf; buf = ''; break; }      // no link possible — flush all
        out += buf.slice(0, open);                          // emit prose before '['
        buf = buf.slice(open);                              // buf now starts at '['
        const m = buf.match(/^\[([^\]]*)\]\(([^)]*)\)/);
        if (m) {                                            // complete link — rewrite + emit
          out += rewriteLink(m[1], m[2], phrases);
          buf = buf.slice(m[0].length);
          continue;
        }
        // Incomplete. If it can't become a link (newline inside, or grown too long),
        // release the '[' and rescan. Otherwise wait for more bytes.
        if (/\n/.test(buf) || buf.length > 600) {
          out += buf[0];
          buf = buf.slice(1);
          continue;
        }
        break;
      }
      return out;
    },
    end() { const rest = buf; buf = ''; return rest; },
  };
}
