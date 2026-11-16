// Trip Planner view: form + ranked journey options.

import { api } from '../core/api.js';
import { store } from '../core/store.js';
import { escapeHtml as e } from '../core/format.js';
import { alertsForRoutes } from '../core/alerts.js';
import { tripFormHtml, bindTripForm } from '../ui/trip-form.js';
import { tripOptionCard, emptyState, errorState, skeleton, sectionHeader } from '../ui/components.js';
import { icon } from '../ui/icons.js';

let controller = null;
let lastQuery = null;
let lastRoutes = null;

/** Map a trip step's display name to MBTA route ids, for alert matching. */
export function routeIdsForStep(step) {
  const name = String(step.name || '');
  const m = name.match(/Green Line ([BCDE])\b/);
  if (m) return [`Green-${m[1]}`];
  if (/^Green Line/.test(name)) return ['Green-B', 'Green-C', 'Green-D', 'Green-E'];
  for (const line of ['Red', 'Orange', 'Blue']) if (name.startsWith(`${line} Line`)) return [line];
  if (/Mattapan/i.test(name)) return ['Mattapan'];
  const bus = name.match(/^Bus (\w+)/);
  if (bus) return [bus[1]];
  return [];
}

/** Transparent labels: only the minimum-duration and fewest-transfer options. */
export function tripBadges(routes) {
  const badges = routes.map(() => []);
  if (routes.length < 2) return badges;
  const minDur = Math.min(...routes.map(r => r.duration));
  const fastest = routes.findIndex(r => r.duration === minDur);
  badges[fastest].push('Shortest trip');
  const minTr = Math.min(...routes.map(r => r.transfers));
  const fewest = routes.filter(r => r.transfers === minTr);
  if (fewest.length === 1) {
    const i = routes.indexOf(fewest[0]);
    if (i !== fastest) badges[i].push('Fewest transfers');
  }
  return badges;
}

function renderResults(el, routes) {
  const alerts = store.get('alerts') || [];
  const badges = tripBadges(routes);
  el.innerHTML = `
    ${sectionHeader(`${routes.length} option${routes.length === 1 ? '' : 's'}`)}
    <div class="stack">${routes.map((r, i) => {
      const affected = [];
      for (const step of r.steps) {
        const hits = alertsForRoutes(alerts, routeIdsForStep(step));
        if (hits.length) {
          const text = hits[0].header || hits[0].service_effect || 'Service alert';
          affected.push(`${step.name}: ${text.length > 110 ? `${text.slice(0, 107)}…` : text}`);
        }
      }
      return tripOptionCard(r, i, { badges: badges[i], alerts: affected });
    }).join('')}</div>
    <p class="fine-print">${icon('info', { size: 14 })}Journey times come from the routing service and are estimates. Fares are estimated. Check alerts before you travel.</p>`;
}

async function search(resultsEl, q) {
  lastQuery = q;
  controller?.abort();
  controller = new AbortController();
  resultsEl.innerHTML = `<div class="stack">${skeleton(3, { height: 112 })}</div>`;
  resultsEl.setAttribute('aria-busy', 'true');
  const body = { origin: q.origin.name, destination: q.destination.name };
  if (q.origin.lat != null) body.origin_coords = `${q.origin.lat},${q.origin.lon}`;
  if (q.destination.lat != null) body.dest_coords = `${q.destination.lat},${q.destination.lon}`;
  const { data, error, aborted } = await api.trip(body, { signal: controller.signal });
  if (aborted) return;
  resultsEl.removeAttribute('aria-busy');
  if (error) {
    resultsEl.innerHTML = errorState({ title: 'Couldn’t plan this trip', body: error });
    resultsEl.querySelector('[data-action="retry"]').addEventListener('click', () => search(resultsEl, q));
    return;
  }
  if (!data.routes || !data.routes.length) {
    resultsEl.innerHTML = emptyState({ icon: 'route', title: 'No transit options found', body: 'Try a nearby station or a more specific address. Late at night, service may have ended.' });
    return;
  }
  store.rememberTrip({ origin: q.origin, destination: q.destination });
  lastRoutes = data.routes;
  renderResults(resultsEl, data.routes);
}

export const tripView = {
  enter(el) {
    const pending = store.get('pendingTrip') || {};
    store.set('pendingTrip', null);
    const recents = store.get('recentTrips');
    el.innerHTML = `
      <div class="page page--split">
        <section class="page__aside">
          <header class="page-head"><h1>Trip Planner</h1><p class="muted">Door-to-door transit directions across subway, bus, Commuter Rail and ferry.</p></header>
          <div class="card card--pad">${tripFormHtml({ large: true })}</div>
          ${recents.length ? `<div class="card card--pad"><h2 class="subhead">Recent trips</h2><ul class="plain-list recent-trips">${recents.map((t, i) =>
            `<li><button class="link-row" data-recent="${i}">${icon('history', { size: 16 })}<span>${e(t.origin.name)} → ${e(t.destination.name)}</span></button></li>`).join('')}</ul></div>` : ''}
        </section>
        <section class="page__main" aria-live="polite" data-results>
          ${emptyState({ icon: 'route', title: 'Where to?', body: 'Pick a station from the suggestions or type an address. Options appear here with times, transfers and walking.' })}
        </section>
      </div>`;
    const results = el.querySelector('[data-results]');
    const form = el.querySelector('[data-trip-form]');
    const initial = { origin: pending.origin || lastQuery?.origin, destination: pending.destination || lastQuery?.destination };
    bindTripForm(form, { initial, onSubmit: q => search(results, q) });
    el.querySelectorAll('[data-recent]').forEach(btn => btn.addEventListener('click', () => {
      const t = recents[Number(btn.dataset.recent)];
      form.querySelector('[name="origin"]').value = t.origin.name;
      Object.assign(form.querySelector('[name="origin"]').dataset, t.origin.lat != null ? { lat: t.origin.lat, lon: t.origin.lon, id: t.origin.id || '' } : {});
      form.querySelector('[name="destination"]').value = t.destination.name;
      Object.assign(form.querySelector('[name="destination"]').dataset, t.destination.lat != null ? { lat: t.destination.lat, lon: t.destination.lon, id: t.destination.id || '' } : {});
      search(results, t);
    }));
    if (pending.origin && pending.destination) search(results, { origin: pending.origin, destination: pending.destination });
    else if (pending.origin || pending.destination) form.querySelector(pending.origin ? '[name="destination"]' : '[name="origin"]').focus();
    else if (lastQuery && lastRoutes) renderResults(results, lastRoutes);
  },
  leave() { controller?.abort(); },
};
