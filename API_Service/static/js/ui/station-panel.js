// Station details (shared by every view): served lines, accessibility, live
// arrivals grouped by line and direction, and active alerts.

import { api } from '../core/api.js';
import { store } from '../core/store.js';
import { navigate } from '../core/router.js';
import { escapeHtml as e, formatClockSeconds } from '../core/format.js';
import { openPanel, updatePanelBody, isPanelOpen } from './panel.js';
import { lineBadge, arrivalRow, alertCard, emptyState, errorState, skeleton } from './components.js';
import { icon } from './icons.js';

const REFRESH_MS = 20000;
let timer = null;
let inflight = null;

export function groupArrivals(predictions) {
  const groups = new Map();
  for (const p of predictions) {
    const key = `${p.route_id}|${p.direction_id}`;
    if (!groups.has(key)) groups.set(key, { route_id: p.route_id, route_name: p.route_name, direction_id: p.direction_id, platform: p.platform_name, headsigns: new Set(), items: [] });
    const g = groups.get(key);
    if (p.headsign) g.headsigns.add(p.headsign);
    g.items.push(p);
  }
  // Groups ordered by their soonest train (items are already time-sorted by the API).
  return [...groups.values()];
}

function accessibilityText(code) {
  if (code === 1) return `<span class="chip chip--ok">${icon('accessible', { size: 14 })}Accessible station</span>`;
  if (code === 2) return `<span class="chip chip--muted">${icon('accessible', { size: 14 })}Not accessible</span>`;
  return '';
}

function render(data) {
  const { station, predictions = [], alerts = [] } = data;
  const service = alerts.filter(a => a.category === 'service');
  const access = alerts.filter(a => a.category === 'accessibility');
  const groups = groupArrivals(predictions);

  const arrivals = groups.length
    ? groups.map(g => `<section class="board" aria-label="${e(g.route_name)} toward ${e(g.platform || [...g.headsigns].join(' / '))}">
        <header class="board__head">${lineBadge(g.route_id, { size: 'sm' })}<span class="board__dir">to ${e(g.platform || [...g.headsigns].join(' / ') || 'destination')}</span></header>
        <ul class="arrivals">${g.items.slice(0, 3).map(arrivalRow).join('')}</ul>
      </section>`).join('')
    : emptyState({ icon: 'clock', title: 'No predictions right now', body: 'The MBTA isn’t reporting upcoming trips for this station. Service may have ended for the night.' });

  return `
    <div class="station-detail">
      <div class="station-detail__lines">${(station.lines || []).map(l => lineBadge(l)).join('')}${accessibilityText(station.wheelchair_boarding)}</div>
      <div class="btn-row">
        ${document.body.dataset.route === 'map' ? '' : `<button class="btn btn--secondary btn--sm" data-act="map">${icon('map', { size: 16 })}View on map</button>`}
        <button class="btn btn--secondary btn--sm" data-act="from">${icon('route', { size: 16 })}From here</button>
        <button class="btn btn--secondary btn--sm" data-act="to">To here</button>
      </div>
      ${service.length ? `<div class="stack-sm">${service.slice(0, 3).map(a => alertCard(a, { compact: true })).join('')}</div>` : ''}
      <h3 class="subhead">Next trains</h3>
      <div class="boards">${arrivals}</div>
      ${access.length ? `<details class="disclosure"><summary>${icon('accessible', { size: 16 })}${access.length} accessibility alert${access.length > 1 ? 's' : ''}</summary>
        <ul class="plain-list">${access.map(a => `<li>${e(a.header)}</li>`).join('')}</ul></details>` : ''}
      <p class="fine-print">${icon('clock', { size: 14 })}Updated ${e(formatClockSeconds(data.updated_at || new Date()))} · refreshes every 20 s</p>
    </div>`;
}

function bind(body, station) {
  body.querySelector('[data-act="map"]')?.addEventListener('click', () => navigate('map', [], { station: station.id }));
  const plan = which => {
    const s = store.stationById(station.id) || { id: station.id, name: station.name, lat: station.latitude, lon: station.longitude };
    store.set('pendingTrip', { [which]: { name: s.name, id: s.id, lat: s.lat ?? station.latitude, lon: s.lon ?? station.longitude } });
    navigate('trip');
  };
  body.querySelector('[data-act="from"]')?.addEventListener('click', () => plan('origin'));
  body.querySelector('[data-act="to"]')?.addEventListener('click', () => plan('destination'));
  body.querySelector('[data-action="retry"]')?.addEventListener('click', () => load(station.id));
}

async function load(id, { quiet = false } = {}) {
  const key = `station:${id}`;
  inflight?.abort();
  inflight = new AbortController();
  if (!quiet) updatePanelBody(key, skeleton(5, { height: 52 }));
  const { data, error, aborted } = await api.station(id, { signal: inflight.signal });
  if (aborted || !isPanelOpen(key)) return;
  if (error) {
    if (quiet) return; // keep the last good board on a failed background refresh
    const body = updatePanelBody(key, errorState({ title: 'Couldn’t load arrivals', body: error }));
    body?.querySelector('[data-action="retry"]')?.addEventListener('click', () => load(id));
    return;
  }
  const body = updatePanelBody(key, render(data));
  if (body) bind(body, data.station);
}

export function openStation(id) {
  const meta = store.stationById(id);
  const key = `station:${id}`;
  store.rememberStation(id);
  openPanel({
    key,
    eyebrow: 'Station',
    title: meta ? meta.name : 'Station',
    html: skeleton(5, { height: 52 }),
    onClose: () => { clearInterval(timer); timer = null; inflight?.abort(); document.dispatchEvent(new CustomEvent('station-closed', { detail: id })); },
  });
  load(id);
  clearInterval(timer);
  timer = setInterval(() => { if (!document.hidden && isPanelOpen(key)) load(id, { quiet: true }); }, REFRESH_MS);
  document.dispatchEvent(new CustomEvent('station-opened', { detail: id }));
}
