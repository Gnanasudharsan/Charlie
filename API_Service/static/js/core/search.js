// Station and route search ranking. Pure functions.

const ABBREVIATIONS = [
  [/\bst\b/g, 'street'], [/\bsq\b/g, 'square'], [/\bctr\b/g, 'center'], [/\bave\b/g, 'avenue'],
  [/\bmt\b/g, 'mount'], [/\bjct\b/g, 'junction'], [/\bstn\b/g, 'station'],
];

export function normalize(value) {
  let s = String(value || '').toLowerCase()
    .normalize('NFD').replace(/[̀-ͯ]/g, '')
    .replace(/[’'`.]/g, '')
    .replace(/[^a-z0-9]+/g, ' ')
    .trim();
  for (const [re, full] of ABBREVIATIONS) s = s.replace(re, full);
  return s;
}

/** Characters of `query` appear in order inside `text` (loose fuzzy match). */
function isSubsequence(query, text) {
  let i = 0;
  for (const ch of text) if (ch === query[i]) i++;
  return i === query.length;
}

/** Lower is better; null means no match. */
export function scoreMatch(query, name) {
  const q = normalize(query), n = normalize(name);
  if (!q) return null;
  if (n === q) return 0;
  if (n.startsWith(q)) return 1;
  if (n.split(' ').some(w => w.startsWith(q))) return 2;
  if (n.includes(q)) return 3;
  const compact = q.replace(/ /g, '');
  if (compact.length >= 3 && isSubsequence(compact, n.replace(/ /g, ''))) return 4;
  return null;
}

/**
 * Rank items by how well `getText(item)` matches the query.
 * Ties keep shorter names first, then alphabetical.
 */
export function rank(items, query, getText = x => x.name, limit = 8) {
  const scored = [];
  for (const item of items) {
    const texts = [].concat(getText(item)).filter(Boolean);
    let best = null;
    for (const t of texts) {
      const s = scoreMatch(query, t);
      if (s !== null && (best === null || s < best)) best = s;
    }
    if (best !== null) scored.push({ item, score: best, name: texts[0] });
  }
  scored.sort((a, b) => a.score - b.score || a.name.length - b.name.length || a.name.localeCompare(b.name));
  return scored.slice(0, limit).map(s => s.item);
}
