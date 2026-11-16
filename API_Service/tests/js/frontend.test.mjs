// Frontend logic tests — run with: node --test tests/js/
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';

import { rank, normalize, scoreMatch } from '../../static/js/core/search.js';
import { countdown, escapeHtml, activePeriodText, renderRichText, vehicleStatusText } from '../../static/js/core/format.js';
import { filterAlerts, countByFilter, prioritize, alertsForRoutes, severityBand } from '../../static/js/core/alerts.js';
import { decodePolyline } from '../../static/js/core/polyline.js';
import { routeIdentity, LINES, SUBWAY_IDS } from '../../static/js/core/constants.js';
import { parseHash } from '../../static/js/core/router.js';
import { routeIdsForStep, tripBadges } from '../../static/js/views/trip.js';
import { mentions } from '../../static/js/views/assistant.js';
import { groupArrivals } from '../../static/js/ui/station-panel.js';

// The real station list the app serves (extracted from app.py's STATION_DATA).
const APP = readFileSync(new URL('../../app.py', import.meta.url), 'utf8');
const STATIONS = [...APP.matchAll(/"([^"]+)": \{"id": "(place-[a-z0-9]+)", "lat": ([\d.-]+), "lon": ([\d.-]+), "lines": \[([^\]]*)\]\}/g)]
  .map(m => ({ name: m[1], id: m[2], lines: m[5].replace(/"/g, '').split(', ').filter(Boolean) }))
  .filter((s, i, all) => all.findIndex(x => x.id === s.id) === i);

test('station list fixture covers all 126 stations', () => {
  assert.equal(STATIONS.length, 126);
});

test('search: every station is findable by its own name and ranks first', () => {
  for (const s of STATIONS) {
    const top = rank(STATIONS, s.name, x => x.name, 3);
    assert.ok(top.some(x => x.id === s.id), `${s.name} not found`);
  }
});

test('search: partial, abbreviated and fuzzy queries', () => {
  assert.equal(rank(STATIONS, 'harv', x => x.name, 1)[0].name, 'Harvard');
  assert.equal(rank(STATIONS, 'park st', x => x.name, 1)[0].name, 'Park Street');
  assert.equal(rank(STATIONS, 'oak', x => x.name, 1)[0].name, 'Oak Grove');
  assert.ok(rank(STATIONS, 'dwntn', x => x.name, 3).some(x => x.name === 'Downtown Crossing'));
  assert.equal(rank(STATIONS, 'zzzz', x => x.name).length, 0);
  assert.equal(normalize("St. Mary's St"), 'street marys street');
  assert.equal(scoreMatch('', 'Harvard'), null);
});

test('countdown: minutes, arriving/departing, boarding, hours, missing', () => {
  const now = new Date('2026-11-16T13:00:00Z');
  const at = m => new Date(now.getTime() + m * 60000).toISOString();
  assert.equal(countdown({ time: at(4.5), event: 'arrival' }, now).label, '4 min');
  assert.equal(countdown({ time: at(0.4), event: 'arrival' }, now).label, 'Arriving');
  assert.equal(countdown({ time: at(0.4), event: 'departure' }, now).label, 'Departing');
  assert.equal(countdown({ time: at(9), status: 'Boarding' }, now).label, 'Boarding');
  assert.equal(countdown({ time: at(75) }, now).label, '1 hr 15 min');
  assert.equal(countdown({}, now).label, 'No prediction');
  assert.equal(countdown({ time: at(2) }, now).state, 'soon');
});

test('format: escaping and safe rich text', () => {
  assert.equal(escapeHtml('<b>"x"</b>'), '&lt;b&gt;&quot;x&quot;&lt;/b&gt;');
  assert.equal(renderRichText('**Next** <script>'), '<strong>Next</strong> &lt;script&gt;');
  assert.equal(vehicleStatusText('STOPPED_AT', 'Park Street'), 'Stopped at Park Street');
  const now = new Date('2026-11-16T13:00:00Z');
  assert.match(activePeriodText([{ start: '2026-11-16T12:00:00Z', end: null }], now), /Ongoing/);
  assert.match(activePeriodText([{ start: '2026-11-20T12:00:00Z', end: '2026-11-22T12:00:00Z' }], now), /^Starts /);
});

const ALERTS = [
  { id: 'red', category: 'service', active_now: true, severity: 7, modes: ['subway'], affected_routes: ['Red'], updated_at: '2026-11-16T10:00:00Z' },
  { id: 'bus', category: 'service', active_now: true, severity: 3, modes: ['bus'], affected_routes: ['66'], updated_at: '2026-11-16T11:00:00Z' },
  { id: 'cr', category: 'service', active_now: false, severity: 8, modes: ['commuter_rail'], affected_routes: ['CR-Needham'] },
  { id: 'ferry', category: 'service', active_now: true, severity: 5, affected_routes: ['Boat-F1'] },
  { id: 'elev', category: 'accessibility', active_now: true, severity: 9, modes: ['subway'], affected_routes: ['Red'] },
];

test('alerts: filters by mode and keeps accessibility separate', () => {
  assert.deepEqual(filterAlerts(ALERTS, 'all').map(a => a.id), ['red', 'bus', 'cr', 'ferry']);
  assert.deepEqual(filterAlerts(ALERTS, 'subway').map(a => a.id), ['red']);
  assert.deepEqual(filterAlerts(ALERTS, 'ferry').map(a => a.id), ['ferry']);   // mode inferred from route id
  assert.deepEqual(filterAlerts(ALERTS, 'accessibility').map(a => a.id), ['elev']);
  assert.deepEqual(countByFilter(ALERTS), { all: 4, subway: 1, bus: 1, commuter_rail: 1, ferry: 1, accessibility: 1 });
});

test('alerts: priority and route matching', () => {
  assert.deepEqual(prioritize(ALERTS).map(a => a.id), ['red', 'ferry', 'bus', 'elev', 'cr']);
  assert.deepEqual(alertsForRoutes(ALERTS, ['Red']).map(a => a.id), ['red']);
  assert.equal(severityBand(8), 'major');
  assert.equal(severityBand(4), 'moderate');
  assert.equal(severityBand(1), 'minor');
});

test('polyline: decodes the reference example', () => {
  const pts = decodePolyline('_p~iF~ps|U_ulLnnqC_mqNvxq`@');
  assert.deepEqual(pts, [[38.5, -120.2], [40.7, -120.95], [43.252, -126.453]]);
});

test('constants: identities for subway, bus, rail and ferry', () => {
  assert.equal(SUBWAY_IDS.length, 8);
  for (const id of SUBWAY_IDS) assert.ok(LINES[id].name && LINES[id].color);
  assert.equal(routeIdentity('CR-Worcester', { name: 'Framingham/Worcester Line' }).mode, 'commuter_rail');
  assert.equal(routeIdentity('Boat-F1').mode, 'ferry');
  assert.equal(routeIdentity('66', { short_name: '66' }).name, 'Route 66');
  assert.equal(routeIdentity('741', { short_name: 'SL1', name: 'x' }).group, 'Silver');
});

test('router: parses views, params and queries', () => {
  assert.deepEqual(parseHash('#/lines/Green-B'), { name: 'lines', params: ['Green-B'], query: {} });
  assert.deepEqual(parseHash('#/map?route=66'), { name: 'map', params: [], query: { route: '66' } });
  assert.equal(parseHash('#/nope').name, 'overview');
  assert.equal(parseHash('').name, 'overview');
});

test('trip: step → route ids and transparent badges', () => {
  assert.deepEqual(routeIdsForStep({ name: 'Green Line D' }), ['Green-D']);
  assert.deepEqual(routeIdsForStep({ name: 'Red Line' }), ['Red']);
  assert.deepEqual(routeIdsForStep({ name: 'Bus 66' }), ['66']);
  assert.deepEqual(routeIdsForStep({ name: 'Framingham/Worcester Line' }), []);
  const badges = tripBadges([{ duration: 40, transfers: 3 }, { duration: 33, transfers: 2 }, { duration: 48, transfers: 1 }]);
  assert.deepEqual(badges, [[], ['Shortest trip'], ['Fewest transfers']]);
  assert.deepEqual(tripBadges([{ duration: 10, transfers: 0 }]), [[]]);
});

test('assistant: detects stations and lines mentioned in replies', () => {
  const found = mentions('🚇 From **Harvard** on the Red Line: next train to Alewife', STATIONS);
  assert.ok(found.some(m => m.type === 'station' && m.name === 'Harvard'));
  assert.ok(found.some(m => m.type === 'station' && m.name === 'Alewife'));
  assert.ok(found.some(m => m.type === 'line' && m.id === 'Red'));
});

test('station board: groups arrivals by line and direction, keeping order', () => {
  const groups = groupArrivals([
    { route_id: 'Red', direction_id: 1, headsign: 'Alewife', time: '1' },
    { route_id: 'Red', direction_id: 0, headsign: 'Ashmont', time: '2' },
    { route_id: 'Red', direction_id: 1, headsign: 'Alewife', time: '3' },
    { route_id: 'Red', direction_id: 0, headsign: 'Braintree', time: '4' },
  ]);
  assert.equal(groups.length, 2);
  assert.deepEqual(groups[0].items.map(i => i.time), ['1', '3']);
  assert.deepEqual([...groups[1].headsigns], ['Ashmont', 'Braintree']);
});

import { inkFor } from '../../static/js/core/constants.js';
test('badges: text color keeps WCAG contrast on every line color', () => {
  assert.equal(inkFor('#ED8B00'), '#1A1A1A');   // Orange Line
  assert.equal(inkFor('#FFC72C'), '#1A1A1A');   // bus yellow
  assert.equal(inkFor('#003DA5'), '#FFFFFF');   // Blue Line
  assert.equal(inkFor('#80276C'), '#FFFFFF');   // Commuter Rail
  assert.equal(inkFor('#00843D'), '#FFFFFF');   // Green Line
});
