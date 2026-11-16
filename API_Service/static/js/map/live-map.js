// Leaflet controller for the live map. Loaded lazily the first time the map opens.
// Vehicles are diffed by id and updated in place (no marker rebuilds → no flicker).

import { api } from '../core/api.js';
import { store } from '../core/store.js';
import { LINES, LINE_GROUPS, SUBWAY_IDS, BOSTON, routeIdentity } from '../core/constants.js';
import { decodePolyline } from '../core/polyline.js';
import { escapeHtml as e } from '../core/format.js';

const LEAFLET_VERSION = '1.9.4';
const REFRESH_MS = 10000;
const ESRI = 'https://server.arcgisonline.com/ArcGIS/rest/services/Canvas';

let leafletPromise = null;
export function loadLeaflet() {
  if (window.L) return Promise.resolve(window.L);
  if (!leafletPromise) {
    leafletPromise = new Promise((resolve, reject) => {
      const css = document.createElement('link');
      css.rel = 'stylesheet';
      css.href = `https://unpkg.com/leaflet@${LEAFLET_VERSION}/dist/leaflet.css`;
      document.head.appendChild(css);
      const js = document.createElement('script');
      js.src = `https://unpkg.com/leaflet@${LEAFLET_VERSION}/dist/leaflet.js`;
      js.onload = () => resolve(window.L);
      js.onerror = () => { leafletPromise = null; reject(new Error('Map library failed to load')); };
      document.head.appendChild(js);
    });
  }
  return leafletPromise;
}

const groupOf = routeId => (LINES[routeId] ? LINES[routeId].group : null);

const VEHICLE_SVG = '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M12 2.5 19.5 20 12 16.2 4.5 20z"/></svg>';

export class LiveMap {
  constructor(el, { onVehicleSelect, onStationSelect, onStatus } = {}) {
    this.el = el;
    this.onVehicleSelect = onVehicleSelect || (() => {});
    this.onStationSelect = onStationSelect || (() => {});
    this.onStatus = onStatus || (() => {});
    this.vehicles = new Map();      // id -> { marker, data }
    this.stationMarkers = new Map();
    this.shapeLayers = new Map();   // routeId -> [polylines]
    this.groups = new Set(LINE_GROUPS.map(g => g.id));
    this.layers = { vehicles: true, stations: true, lines: true };
    this.focusRoute = null;
    this.selectedVehicle = null;
    this.selectedStation = null;
    this.timer = null;
    this.seq = 0;
    this.controller = null;
    this.lastUpdate = null;
  }

  async init() {
    const L = await loadLeaflet();
    this.L = L;
    const map = L.map(this.el, { zoomControl: false, minZoom: 9, maxZoom: 16, worldCopyJump: false, keyboard: true });
    this.map = map;
    map.setView(BOSTON.center, BOSTON.zoom);
    L.control.zoom({ position: 'bottomright' }).addTo(map);
    L.tileLayer(`${ESRI}/World_Dark_Gray_Base/MapServer/tile/{z}/{y}/{x}`, {
      maxZoom: 16, maxNativeZoom: 16,
      attribution: 'Tiles &copy; <a href="https://www.esri.com" target="_blank" rel="noopener">Esri</a> &mdash; Esri, HERE, Garmin, FAO, NOAA, USGS, &copy; OpenStreetMap contributors · Data: <a href="https://www.mbta.com/developers" target="_blank" rel="noopener">MBTA</a>',
    }).addTo(map);
    L.tileLayer(`${ESRI}/World_Dark_Gray_Reference/MapServer/tile/{z}/{y}/{x}`, { maxZoom: 16, maxNativeZoom: 16, pane: 'overlayPane' }).addTo(map);

    map.createPane('routes').style.zIndex = 380;
    map.createPane('stations').style.zIndex = 600;
    map.createPane('vehicles').style.zIndex = 650;
    this.shapeGroup = L.layerGroup().addTo(map);
    this.stationGroup = L.layerGroup().addTo(map);
    this.vehicleGroup = L.layerGroup().addTo(map);
    this.userGroup = L.layerGroup().addTo(map);

    map.on('zoomend', () => this.applyZoomClass());
    this.applyZoomClass();
    await Promise.all([this.drawShapes(SUBWAY_IDS), this.drawStations()]);
  }

  applyZoomClass() {
    const z = this.map.getZoom();
    this.el.classList.toggle('z-far', z <= 11);
    this.el.classList.toggle('z-mid', z > 11 && z < 14);
    this.el.classList.toggle('z-near', z >= 14);
  }

  async drawShapes(routeIds, color = null) {
    const missing = routeIds.filter(r => !this.shapeLayers.has(r));
    if (!missing.length) return;
    const { data } = await api.shapes(missing.join(','));
    if (!data) return;
    for (const s of data.shapes) {
      const id = routeIdentity(s.route_id);
      const line = this.L.polyline(decodePolyline(s.polyline), {
        pane: 'routes', color: color || id.bright, weight: 4, opacity: 0.9, lineCap: 'round', lineJoin: 'round', interactive: false,
      });
      if (!this.shapeLayers.has(s.route_id)) this.shapeLayers.set(s.route_id, []);
      this.shapeLayers.get(s.route_id).push(line);
    }
    this.refreshShapes();
  }

  refreshShapes() {
    this.shapeGroup.clearLayers();
    if (!this.layers.lines) return;
    for (const [routeId, lines] of this.shapeLayers) {
      const visible = this.focusRoute ? routeId === this.focusRoute : (LINES[routeId] && this.groups.has(groupOf(routeId)));
      if (visible) lines.forEach(l => this.shapeGroup.addLayer(l));
    }
  }

  drawStations() {
    const L = this.L;
    for (const st of store.get('stations')) {
      const lines = st.lines || [];
      const multi = new Set(lines.map(groupOf)).size > 1;
      const color = multi ? '#F8FAFC' : routeIdentity(lines[0]).bright;
      const marker = L.marker([st.lat, st.lon], {
        pane: 'stations', keyboard: true, title: st.name, riseOnHover: true,
        icon: L.divIcon({ className: 'stop-icon', html: `<span class="stop${multi ? ' stop--hub' : ''}" style="--c:${color}"></span>`, iconSize: [16, 16], iconAnchor: [8, 8] }),
      });
      marker.on('click', () => this.onStationSelect(st.id));
      marker.on('add', () => { const el = marker.getElement(); if (el) { el.setAttribute('aria-label', `${st.name} station`); el.setAttribute('role', 'button'); } });
      this.stationMarkers.set(st.id, { marker, station: st });
    }
    this.refreshStations();
  }

  refreshStations() {
    this.stationGroup.clearLayers();
    if (!this.layers.stations) return;
    for (const { marker, station } of this.stationMarkers.values()) {
      const visible = this.focusRoute
        ? (station.lines || []).some(l => l === this.focusRoute || (this.focusRoute.startsWith('Green') && l.startsWith('Green')))
        : (station.lines || []).some(l => this.groups.has(groupOf(l)));
      if (visible) this.stationGroup.addLayer(marker);
    }
  }

  vehicleVisible(v) {
    if (!this.layers.vehicles) return false;
    if (this.focusRoute) return true;
    return this.groups.has(groupOf(v.route?.id));
  }

  vehicleLabel(v) {
    const name = routeIdentity(v.route?.id, { name: v.route?.name }).name;
    return `${name} vehicle ${v.label || v.id}${v.headsign ? ` to ${v.headsign}` : ''}`;
  }

  renderVehicles(list) {
    const L = this.L;
    const seen = new Set();
    for (const v of list) {
      if (v.latitude == null || v.longitude == null) continue;
      seen.add(v.id);
      const color = LINES[v.route?.id] ? LINES[v.route.id].bright : (v.route?.color || '#94A3B8');
      const existing = this.vehicles.get(v.id);
      if (existing) {
        existing.data = v;
        existing.marker.setLatLng([v.latitude, v.longitude]);
        const el = existing.marker.getElement();
        if (el) {
          el.querySelector('.veh')?.style.setProperty('--rot', `${Number(v.bearing) || 0}deg`);
          el.setAttribute('aria-label', this.vehicleLabel(v));
        }
      } else {
        const marker = L.marker([v.latitude, v.longitude], {
          pane: 'vehicles', keyboard: true, title: this.vehicleLabel(v),
          icon: L.divIcon({ className: 'veh-icon', html: `<span class="veh" style="--c:${e(color)};--rot:${Number(v.bearing) || 0}deg">${VEHICLE_SVG}</span>`, iconSize: [26, 26], iconAnchor: [13, 13] }),
        });
        marker.on('click', () => this.selectVehicle(v.id));
        marker.on('add', () => {
          const el = marker.getElement();
          if (!el) return;
          el.setAttribute('role', 'button');
          el.setAttribute('aria-label', this.vehicleLabel(this.vehicles.get(v.id)?.data || v));
          el.classList.toggle('is-selected', this.selectedVehicle === v.id);
        });
        this.vehicles.set(v.id, { marker, data: v });
      }
    }
    for (const [id, entry] of this.vehicles) {
      if (!seen.has(id)) { this.vehicleGroup.removeLayer(entry.marker); this.vehicles.delete(id); }
    }
    this.refreshVehicles();
  }

  refreshVehicles() {
    for (const { marker, data } of this.vehicles.values()) {
      const show = this.vehicleVisible(data);
      const on = this.vehicleGroup.hasLayer(marker);
      if (show && !on) this.vehicleGroup.addLayer(marker);
      if (!show && on) this.vehicleGroup.removeLayer(marker);
    }
  }

  visibleVehicleCount() {
    let n = 0;
    for (const { data } of this.vehicles.values()) if (this.vehicleVisible(data)) n++;
    return n;
  }

  selectVehicle(id, { pan = false } = {}) {
    if (this.selectedVehicle) this.vehicles.get(this.selectedVehicle)?.marker.getElement()?.classList.remove('is-selected');
    this.selectedVehicle = id;
    const entry = id && this.vehicles.get(id);
    if (entry) {
      entry.marker.getElement()?.classList.add('is-selected');
      if (pan) this.map.panTo(entry.marker.getLatLng());
      this.onVehicleSelect(entry.data);
    }
  }

  selectStation(id, { fly = false } = {}) {
    if (this.selectedStation) this.stationMarkers.get(this.selectedStation)?.marker.getElement()?.classList.remove('is-selected');
    this.selectedStation = id;
    const entry = id && this.stationMarkers.get(id);
    if (!entry) return;
    entry.marker.getElement()?.classList.add('is-selected');
    if (fly) {
      // On phones the bottom sheet covers the lower half, so aim the station above it.
      const zoom = Math.max(this.map.getZoom(), 15);
      const sheetOffset = window.matchMedia('(max-width: 767px)').matches ? this.el.clientHeight * 0.24 : 0;
      const point = this.map.project(entry.marker.getLatLng(), zoom).add([0, sheetOffset]);
      this.map.flyTo(this.map.unproject(point, zoom), zoom, { duration: 0.6 });
    }
  }

  getVehicle(id) { return this.vehicles.get(id)?.data || null; }

  routesParam() { return this.focusRoute || SUBWAY_IDS.join(','); }

  async tick() {
    if (document.hidden) return;
    const mySeq = ++this.seq;
    this.controller?.abort();
    this.controller = new AbortController();
    this.onStatus({ loading: true, updatedAt: this.lastUpdate, count: this.visibleVehicleCount() });
    const { data, error, aborted } = await api.vehicles(this.routesParam(), { signal: this.controller.signal });
    if (aborted || mySeq !== this.seq) return;     // a newer request superseded this one
    if (error) {
      this.onStatus({ error, updatedAt: this.lastUpdate, count: this.visibleVehicleCount() });
      return;                                        // keep the last good markers on screen
    }
    this.lastUpdate = data.updated_at || new Date().toISOString();
    this.renderVehicles(data.vehicles || []);
    this.onStatus({ updatedAt: this.lastUpdate, count: this.visibleVehicleCount() });
    if (this.selectedVehicle) {
      const v = this.vehicles.get(this.selectedVehicle);
      document.dispatchEvent(new CustomEvent('vehicle-updated', { detail: v ? v.data : null }));
    }
  }

  start() {
    if (this.timer) return;
    this.tick();
    this.timer = setInterval(() => this.tick(), REFRESH_MS);
  }

  stop() {
    clearInterval(this.timer);
    this.timer = null;
    this.controller?.abort();
  }

  setGroup(id, on) {
    if (on) this.groups.add(id); else this.groups.delete(id);
    this.refreshShapes(); this.refreshStations(); this.refreshVehicles();
    this.onStatus({ updatedAt: this.lastUpdate, count: this.visibleVehicleCount() });
  }

  setLayer(name, on) {
    this.layers[name] = on;
    this.refreshShapes(); this.refreshStations(); this.refreshVehicles();
    this.onStatus({ updatedAt: this.lastUpdate, count: this.visibleVehicleCount() });
  }

  async setFocus(routeId) {
    if (this.focusRoute === routeId) return;
    this.focusRoute = routeId || null;
    // Vehicles for a different route set: clear and refetch.
    for (const { marker } of this.vehicles.values()) this.vehicleGroup.removeLayer(marker);
    this.vehicles.clear();
    if (routeId && !this.shapeLayers.has(routeId)) await this.drawShapes([routeId]);
    this.refreshShapes(); this.refreshStations();
    if (routeId) this.fitRoute(routeId); else this.fitNetwork();
    if (this.timer) this.tick();
  }

  fitRoute(routeId) {
    const lines = this.shapeLayers.get(routeId);
    if (!lines || !lines.length) return;
    const bounds = lines.reduce((b, l) => b.extend(l.getBounds()), this.L.latLngBounds(lines[0].getBounds()));
    this.map.fitBounds(bounds, { padding: [40, 40] });
  }

  fitNetwork() {
    const all = [...this.shapeLayers.entries()].filter(([id]) => LINES[id]).flatMap(([, ls]) => ls);
    if (!all.length) { this.map.setView(BOSTON.center, BOSTON.zoom); return; }
    const bounds = all.reduce((b, l) => b.extend(l.getBounds()), this.L.latLngBounds(all[0].getBounds()));
    this.map.fitBounds(bounds, { padding: [30, 30] });
  }

  showUser(lat, lon, { fly = true } = {}) {
    this.userGroup.clearLayers();
    this.L.circle([lat, lon], { radius: 60, color: '#4C8DF6', weight: 1, fillOpacity: 0.12, interactive: false }).addTo(this.userGroup);
    this.L.marker([lat, lon], {
      keyboard: false, interactive: false,
      icon: this.L.divIcon({ className: 'me-icon', html: '<span class="me" aria-hidden="true"></span>', iconSize: [18, 18], iconAnchor: [9, 9] }),
    }).addTo(this.userGroup);
    if (fly) this.map.flyTo([lat, lon], 15, { duration: 0.6 });
  }

  invalidate() { this.map?.invalidateSize(); }
}
