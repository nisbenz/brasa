import { createGlobe } from './globe.js';
import { createRelief } from './relief.js';
const $ = (s, r = document) => r.querySelector(s);
const $$ = (s, r = document) => [...r.querySelectorAll(s)];
const MN = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'];
const MONTH = ['January','February','March','April','May','June','July','August','September','October','November','December'];
const fmt = n => n == null ? '–' : Math.round(n).toLocaleString('en-US');
const big = n => n >= 1e6 ? `${(n / 1e6).toFixed(n >= 1e7 ? 0 : 1)} million` : fmt(n);
const esc = s => String(s).replace(/[&<>"]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
const css = n => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
const store = { get(k) { try { return localStorage.getItem(k); } catch { return null; } }, set(k, v) { try { localStorage.setItem(k, v); } catch {} } };
const state = { meta: null, area: null, areaName: '', start: '2000-11', end: '2025-12', mode: 'count', scale: 'log', year: null, data: null, tab: 'draw', view: 'grid', gmode: 'season', gstep: 8, gcol: 'count', ctl: null, seq: 0, grown: false, preset: 0 };

const PRESETS = [
  { name: 'Amazon arc of deforestation', note: 'Brazil', area: { type: 'bbox', west: -63, south: -12, east: -54, north: -6 } },
  { name: 'Northern California', note: 'United States', area: { type: 'polygon', coordinates: [[-124.2,40.0],[-122.0,38.0],[-119.0,38.5],[-119.0,42.0],[-124.2,42.0],[-124.2,40.0]] } },
  { name: 'Australia', note: 'Whole country', area: { type: 'country', name: 'Australia' } },
  { name: 'Sahel belt', note: 'West and Central Africa', area: { type: 'bbox', west: -17, south: 8, east: 40, north: 15 } },
  { name: 'Lake Tahoe, 50 km', note: 'Caldor fire, 2021', area: { type: 'point', lat: 38.85, lon: -120.2, radius_km: 50 } },
];
const SRC_TXT = {
  MT: 'Estimated from MODIS Terra alone',
  MTA: 'Estimated from MODIS Terra and Aqua',
  NN: 'Estimated from MODIS Terra and Aqua by the U-Net ensemble',
  VS: 'Observed by VIIRS on Suomi NPP',
  VN: 'Observed by VIIRS on NOAA-20, standing in during a Suomi NPP outage',
};
const isHarm = s => s === 'MT' || s === 'MTA' || s === 'NN';
let lenis = null;  // smooth scroll, set up in initScroll
let stepWatch = null;  // method-step observer, see watchSteps

/* ---------- theme ---------- */
const theme = () => document.documentElement.dataset.theme;
function applyTheme(t, persist) {
  document.documentElement.dataset.theme = t;
  if (persist) store.set('brasa-theme', t);
  $('#themeBtn').setAttribute('aria-label', t === 'night' ? 'Switch to day theme' : 'Switch to night theme');
  setTiles(); if (state.data) renderAll(false);
  if (globe) { globe.setTheme(); if (state.area) globe.setArea(state.area, state.data && state.data.summary.bbox); }
}
$('#themeBtn').onclick = () => applyTheme(theme() === 'night' ? 'day' : 'night', true);

/* ---------- map ---------- */
let map, tiles, drawn;
const shapeStyle = () => ({ color: css('--ink'), weight: 2, fillColor: css('--scar'), fillOpacity: .18 });
function setTiles() {
  if (!map) return;
  const t = theme() === 'night' ? 'World_Dark_Gray_Base' : 'World_Light_Gray_Base';
  if (tiles) map.removeLayer(tiles);
  tiles = L.tileLayer(`https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/${t}/MapServer/tile/{z}/{y}/{x}`, { maxZoom: 16, attribution: 'Basemap © Esri, HERE, Garmin, © OpenStreetMap contributors' }).addTo(map);
  tiles.bringToBack();
  if (drawn) drawn.eachLayer(l => l.setStyle && l.setStyle(shapeStyle()));
}
function initMap() {
  map = L.map('map', { worldCopyJump: true, attributionControl: true }).setView([0, -30], 2);
  setTiles();
  L.control.scale({ position: 'bottomleft', imperial: false }).addTo(map);
  drawn = new L.FeatureGroup().addTo(map);
  const so = shapeStyle();
  map.addControl(new L.Control.Draw({ position: 'topleft', edit: false, draw: { rectangle: { shapeOptions: so, showArea: false }, polygon: { shapeOptions: so, allowIntersection: false, showArea: false }, polyline: false, circle: false, marker: false, circlemarker: false } }));
  map.on(L.Draw.Event.CREATED, e => {
    const l = e.layer;
    if (e.layerType === 'rectangle') {
      const b = l.getBounds();
      const bb = { type: 'bbox', west: +b.getWest().toFixed(3), south: +b.getSouth().toFixed(3), east: +b.getEast().toFixed(3), north: +b.getNorth().toFixed(3) };
      $('#bN').value = bb.north; $('#bS').value = bb.south; $('#bW').value = bb.west; $('#bE').value = bb.east;
      setTab('draw'); setArea(bb, 'Your rectangle');
    } else {
      const ring = l.getLatLngs()[0].map(p => [+p.lng.toFixed(3), +p.lat.toFixed(3)]); ring.push(ring[0]);
      setTab('draw'); setArea({ type: 'polygon', coordinates: ring }, 'Your polygon');
    }
  });
  map.on('click', e => { if (state.tab === 'point') { $('#pLat').value = e.latlng.lat.toFixed(3); $('#pLon').value = e.latlng.wrap().lng.toFixed(3); $('#pointForm').requestSubmit(); } });
  map.on('mousemove', e => { $('#coords').textContent = `${e.latlng.lat.toFixed(2)}°, ${e.latlng.wrap().lng.toFixed(2)}°`; });
}
function showArea(a, bbox) {
  if (!map) return;
  drawn.clearLayers();
  const o = shapeStyle();
  let l = null;
  if (a.type === 'bbox') l = L.rectangle([[a.south, a.west], [a.north, a.east]], o);
  else if (a.type === 'polygon') l = L.polygon(a.coordinates.map(c => [c[1], c[0]]), o);
  else if (a.type === 'point') l = L.circle([a.lat, a.lon], { ...o, radius: a.radius_km * 1000 });
  else if (a.type === 'country' && bbox) l = L.rectangle([[bbox[1], bbox[0]], [bbox[3], bbox[2]]], { ...o, dashArray: '6 5', fillOpacity: .06 });
  if (l) { drawn.addLayer(l); map.fitBounds(l.getBounds(), { padding: [24, 24], maxZoom: 8 }); }
}

/* ---------- area ---------- */
function setArea(a, name, presetIdx = -1) {
  state.area = a; state.areaName = name; state.year = null; state.preset = presetIdx;
  $$('#presets button').forEach((b, i) => b.setAttribute('aria-current', String(i === presetIdx)));
  $('#areaTitle').textContent = name;
  showArea(a);
  run();
}
function setTab(t) {
  state.tab = t;
  $$('#areaTabs button').forEach(b => b.setAttribute('aria-selected', String(b.dataset.tab === t)));
  $$('.pane').forEach(p => p.hidden = p.dataset.pane !== t);
}
$$('#areaTabs button').forEach(b => b.onclick = () => setTab(b.dataset.tab));
$('#areaTabs').addEventListener('keydown', e => {
  const bs = $$('#areaTabs button'), i = bs.indexOf(document.activeElement);
  if (i < 0 || !['ArrowRight', 'ArrowLeft'].includes(e.key)) return;
  const n = bs[(i + (e.key === 'ArrowRight' ? 1 : bs.length - 1)) % bs.length]; n.focus(); n.click();
});
const num = id => { const v = $(id).value.trim().replace('−', '-'); return v === '' ? NaN : Number(v); };
$('#bboxForm').onsubmit = e => {
  e.preventDefault();
  const n = num('#bN'), s = num('#bS'), w = num('#bW'), ea = num('#bE');
  let m = '';
  if ([n, s, w, ea].some(Number.isNaN)) m = 'Enter all four edges as numbers, in degrees.';
  else if (Math.abs(n) > 90 || Math.abs(s) > 90) m = 'Latitudes go from −90 to 90.';
  else if (Math.abs(w) > 180 || Math.abs(ea) > 180) m = 'Longitudes go from −180 to 180.';
  else if (n <= s) m = 'North has to be greater than south.';
  else if (ea <= w) m = 'East has to be greater than west.';
  $('#bboxErr').textContent = m; if (m) return;
  setArea({ type: 'bbox', west: w, south: s, east: ea, north: n }, `${Math.abs(s)}°${s < 0 ? 'S' : 'N'}–${Math.abs(n)}°${n < 0 ? 'S' : 'N'}, ${Math.abs(w)}°${w < 0 ? 'W' : 'E'}–${Math.abs(ea)}°${ea < 0 ? 'W' : 'E'}`);
};
$('#pointForm').onsubmit = e => {
  e.preventDefault();
  const la = num('#pLat'), lo = num('#pLon'), r = num('#pRad');
  let m = '';
  if ([la, lo, r].some(Number.isNaN)) m = 'Enter latitude, longitude and radius as numbers.';
  else if (Math.abs(la) > 90 || Math.abs(lo) > 180) m = 'Latitude must be within ±90 and longitude within ±180.';
  else if (r <= 0 || r > 2000) m = 'Choose a radius between 1 and 2,000 km.';
  $('#pointErr').textContent = m; if (m) return;
  setArea({ type: 'point', lat: la, lon: lo, radius_km: r }, `${r} km around ${la.toFixed(2)}°, ${lo.toFixed(2)}°`);
};
$('#countryIn').addEventListener('change', e => {
  const v = e.target.value.trim(), ok = (state.countries || []).find(c => c.toLowerCase() === v.toLowerCase());
  $('#countryErr').textContent = v && !ok ? 'No country by that name. Pick one from the list.' : '';
  if (ok) { e.target.value = ok; setArea({ type: 'country', name: ok }, ok); }
});
$('#presets').innerHTML = PRESETS.map((p, i) => `<li><button type="button" data-i="${i}">${esc(p.name)}<span>${esc(p.note)}</span></button></li>`).join('');
$('#presets').onclick = e => {
  const b = e.target.closest('button'); if (!b) return;
  const i = +b.dataset.i, p = PRESETS[i];
  if (p.area.type === 'country') { setTab('country'); $('#countryIn').value = p.area.name; }
  else if (p.area.type === 'bbox') { setTab('bbox'); $('#bN').value = p.area.north; $('#bS').value = p.area.south; $('#bW').value = p.area.west; $('#bE').value = p.area.east; }
  else if (p.area.type === 'point') { setTab('point'); $('#pLat').value = p.area.lat; $('#pLon').value = p.area.lon; $('#pRad').value = p.area.radius_km; }
  else setTab('draw');
  setArea(p.area, p.name, i);
};

/* ---------- period ---------- */
function setRange(a, b) { state.start = a; state.end = b; $('#mFrom').value = a; $('#mTo').value = b; $('#rangeErr').textContent = ''; if (state.area) run(); }
function onRange() {
  const a = $('#mFrom').value, b = $('#mTo').value;
  if (!a || !b) { $('#rangeErr').textContent = 'Choose both months.'; return; }
  if (a > b) { $('#rangeErr').textContent = 'The first month has to come before the last.'; return; }
  setRange(a, b);
}
$('#mFrom').onchange = $('#mTo').onchange = onRange;
$$('[data-range]').forEach(b => b.onclick = () => {
  const last = state.meta.last_month, v = b.dataset.range;
  if (v === 'all') return setRange(state.meta.first_month, last);
  if (v === '2012') return setRange('2012-01', last);
  let [y, m] = last.split('-').map(Number); m -= 11; if (m < 1) { m += 12; y--; }
  setRange(`${y}-${String(m).padStart(2, '0')}`, last);
});
const ymLabel = s => { const [y, m] = s.split('-').map(Number); return `${MN[m - 1]} ${y}`; };

/* ---------- display switches ---------- */
function seg(id, attr, key) {
  $$(`${id} button`).forEach(b => b.onclick = () => { state[key] = b.dataset[attr]; $$(`${id} button`).forEach(x => x.setAttribute('aria-checked', String(x === b))); if (state.data) { renderHeat(); renderRing(); renderRelief(); } });
}
seg('#modeSeg', 'mode', 'mode'); seg('#scaleSeg', 'scale', 'scale');

/* ---------- fetch ---------- */
async function api(path, body, opts = {}) {
  const r = await fetch(path, { method: body ? 'POST' : 'GET', headers: body ? { 'Content-Type': 'application/json' } : {}, body: body ? JSON.stringify(body) : undefined, ...opts });
  if (!r.ok) { let d = ''; try { d = (await r.json()).detail || ''; } catch {} throw new Error(`${r.status} ${typeof d === 'string' ? d : JSON.stringify(d)}`.trim()); }
  return r;
}
const reqBody = () => ({ area: state.area, start: state.start, end: state.end });
async function run() {
  if (!state.area) return;
  const my = ++state.seq;
  if (state.ctl) state.ctl.abort();
  state.ctl = new AbortController();
  $('#status').className = 'status'; $('#status').textContent = state.data ? '' : 'Reading the record…';
  document.body.classList.add('busy');
  try {
    const d = await (await api('/api/calendar', reqBody(), { signal: state.ctl.signal })).json();
    if (my !== state.seq) return;
    state.data = d;
    state.byK = {}; d.months.forEach(m => state.byK[m.year * 12 + m.month - 1] = m);
    if (state.year == null || !d.months.some(m => m.year === state.year)) state.year = d.summary.worst_year ?? (d.months.length ? d.months[d.months.length - 1].year : null);
    if (state.area.type === 'country') showArea(state.area, d.summary.bbox);
    renderAll(!state.grown); state.grown = true;
    if (globe) globe.setArea(state.area, d.summary.bbox);
    $('#status').className = d.summary.total ? 'status sr' : 'status';
    $('#status').textContent = d.summary.total ? `Updated: ${fmt(d.summary.total)} hotspots in ${state.areaName}.` : 'No fires were detected here in this period. Try a larger area or a longer period.';
  } catch (e) {
    if (e.name === 'AbortError') return;
    state.data = null;
    $('#status').className = 'status error';
    $('#status').textContent = `The record could not be loaded (${e.message}). Check the area and period, then try again.`;
  } finally { if (my === state.seq) document.body.classList.remove('busy'); }
}

/* ---------- reading (plain-language summary) ---------- */
function monthsText(ms) {
  if (!ms.length) return '';
  const s = [...ms].sort((a, b) => a - b);
  const contiguous = s.every((m, i) => i === 0 || m === s[i - 1] + 1);
  if (contiguous) return s.length === 1 ? MONTH[s[0] - 1] : `${MONTH[s[0] - 1]} to ${MONTH[s[s.length - 1] - 1]}`;
  // wrap-around season (e.g. Nov–Feb)
  for (let k = 1; k < s.length; k++) if (s[k] !== s[k - 1] + 1) {
    const w = [...s.slice(k), ...s.slice(0, k)];
    if (w.every((m, i) => i === 0 || m === (w[i - 1] % 12) + 1)) return `${MONTH[w[0] - 1]} to ${MONTH[w[w.length - 1] - 1]}`;
    break;
  }
  return s.map(m => MONTH[m - 1]).join(', ').replace(/, ([^,]*)$/, ' and $1');
}
function renderReading() {
  const d = state.data, s = d.summary, out = [];
  $('#areaKm').textContent = s.area_km2 ? `${fmt(s.area_km2)} km² of grid` : '';
  $('#periodTxt').textContent = `${ymLabel(state.start)} to ${ymLabel(state.end)}`;
  $('#cellsTxt').textContent = s.n_cells ? `${fmt(s.n_cells)} cells of 0.25° (about 25 km)` : '';
  renderTelemetry(s);
  if (!s.total) { $('#reading').innerHTML = '<p>No fires were detected here in this period.</p>'; return; }
  if (s.peak_month && s.season_months.length === 1) out.push(`Burning here is concentrated in <strong>${MONTH[s.peak_month - 1]}</strong>; no other month comes close to half of it.`);
  else if (s.peak_month) out.push(s.season_months.length >= 8
    ? `Fires burn here <strong>all year</strong>, with the most in <strong>${MONTH[s.peak_month - 1]}</strong>.`
    : `Fire season runs <strong>${monthsText(s.season_months)}</strong> and peaks in <strong>${MONTH[s.peak_month - 1]}</strong>.`);
  if (s.worst_year) out.push(`<strong>${s.worst_year}</strong> was the worst year in this record, with <strong>${big(s.worst_year_total)}</strong> hotspots${s.quietest_year && s.quietest_year !== s.worst_year ? `; ${s.quietest_year} was the quietest` : ''}.`);
  const l = s.latest;
  if (l && l.anomaly_pct != null && l.median > 0) {
    const p = l.anomaly_pct, cls = p >= 0 ? 'up' : 'dn';
    const how = Math.abs(p) < 10 ? 'close to' : `<strong class="${cls}">${Math.abs(Math.round(p))}% ${p >= 0 ? 'above' : 'below'}</strong>`;
    out.push(`<strong>${MONTH[l.month - 1]} ${l.year}</strong> came in ${how} a usual ${MONTH[l.month - 1]}, ranking ${ordinal(l.rank)} of ${l.of}.`);
  }
  const t = s.trend_pct_per_decade;
  if (t != null && Math.abs(t) >= 3) out.push(`Across the years shown, burning has <strong class="${t >= 0 ? 'up' : 'dn'}">${t >= 0 ? 'risen' : 'fallen'} ${Math.abs(Math.round(t))}%</strong> per decade.`);
  $('#reading').innerHTML = out.map(x => `<p>${x}</p>`).join('');
}
// KPI cards: a figure that counts up, a chip that puts it in context, and a small chart that draws in
function renderTelemetry(s) {
  const el = $('#telemetry'), d = state.data;
  if (!s.total || !d) { el.innerHTML = ''; return; }
  const n = {}; d.months.forEach(m => n[m.year] = (n[m.year] || 0) + 1);
  const ys = d.yearly.filter(r => n[r.year] === 12);
  const vals = ys.map(r => r.hotspots), med = vals.length ? [...vals].sort((a, b) => a - b)[(vals.length - 1) >> 1] : 0;
  const clim = d.climatology.map(c => c.median), climSum = clim.reduce((a, b) => a + b, 0) || 1;
  const t = s.trend_pct_per_decade, pk = s.peak_month;
  const cards = [
    { k: 'Hotspots in this period', v: `<span data-count="${Math.round(s.total)}">${fmt(s.total)}</span>`,
      chip: `${ys.length} full years`, sub: `about ${big(s.total / Math.max(ys.length, 1))} a year`,
      chart: kpiArea(vals) },
    { k: 'Peak month', v: pk ? MONTH[pk - 1] : '–',
      chip: pk ? `${Math.round(100 * clim[pk - 1] / climSum)}% of a typical year` : '', sub: 'median month, 2001–2025',
      chart: kpiBars(clim, pk ? pk - 1 : -1, MN.map(m => m[0])) },
    { k: 'Worst year', v: s.worst_year ? `<span>${s.worst_year}</span>` : '–',
      chip: s.worst_year && med ? `${(s.worst_year_total / med).toFixed(1)}× the median year` : '', sub: s.worst_year ? `${big(s.worst_year_total)} hotspots` : '',
      chart: kpiBars(vals, ys.findIndex(r => r.year === s.worst_year), null, s.worst_year ? big(s.worst_year_total) : '') },
    { k: 'Trend per decade', v: t == null ? '–' : `${t >= 0 ? '+' : '−'}<span data-count="${Math.abs(Math.round(t))}">${Math.abs(Math.round(t))}</span>%`,
      chip: t == null ? '' : Math.abs(t) < 3 ? 'about flat' : t >= 0 ? 'rising' : 'falling', cls: t == null || Math.abs(t) < 3 ? '' : t >= 0 ? 'up' : 'dn',
      sub: 'fitted on full years', chart: kpiTrend(ys) },
  ];
  el.innerHTML = cards.map(c => `<div class="kpi"><dt>${c.k}</dt><dd><span class="kv ${c.cls || ''}">${c.v}</span>${c.chip ? `<span class="chip-d ${c.cls || ''}">${c.chip}</span>` : ''}<span class="ks">${c.sub}</span>${c.chart}</dd></div>`).join('');
  play(el);
}
let uid = 0;
const KW = 280, KH = 72;
function kpiArea(vals) {
  if (vals.length < 2) return '';
  const max = Math.max(...vals, 1), id = 'ka' + (++uid);
  const x = i => 2 + (KW - 4) * i / (vals.length - 1), y = v => KH - 4 - (KH - 12) * v / max;
  const ln = vals.map((v, i) => `${i ? 'L' : 'M'}${x(i).toFixed(1)},${y(v).toFixed(1)}`).join('');
  const last = vals.length - 1;
  return `<svg class="kc" viewBox="0 0 ${KW} ${KH}" aria-hidden="true"><defs><linearGradient id="${id}" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="var(--scar)" stop-opacity=".35"/><stop offset="1" stop-color="var(--scar)" stop-opacity="0"/></linearGradient></defs>
    <g class="sweep"><path d="${ln}L${x(last)},${KH}L${x(0)},${KH}Z" fill="url(#${id})"/><path d="${ln}" fill="none" class="kl"/></g>
    <circle class="pop" cx="${x(last)}" cy="${y(vals[last])}" r="3.5"/></svg>`;
}
function kpiBars(vals, hi, labels, callout) {
  if (!vals.length) return '';
  const max = Math.max(...vals, 1), n = vals.length, gap = n > 20 ? 1.5 : 3, bw = (KW - gap * (n - 1)) / n, base = labels ? KH - 12 : KH - 2;
  const bars = vals.map((v, i) => { const h = Math.max(1.5, (base - 14) * v / max); return `<rect class="bar${i === hi ? ' hi' : ''}" style="--i:${i}" x="${(i * (bw + gap)).toFixed(1)}" y="${(base - h).toFixed(1)}" width="${bw.toFixed(1)}" height="${h.toFixed(1)}" rx="${Math.min(2, bw / 3)}"/>`; }).join('');
  const lab = labels ? labels.map((l, i) => `<text x="${(i * (bw + gap) + bw / 2).toFixed(1)}" y="${KH - 1}" text-anchor="middle"${i === hi ? ' class="hi"' : ''}>${l}</text>`).join('') : '';
  let co = '';
  if (callout && hi >= 0) { const cx = Math.min(KW - 30, Math.max(30, hi * (bw + gap) + bw / 2)); co = `<g class="pop"><rect class="co" x="${cx - 28}" y="0" width="56" height="13" rx="6.5"/><text class="cot" x="${cx}" y="9.5" text-anchor="middle">${callout}</text></g>`; }
  return `<svg class="kc" viewBox="0 0 ${KW} ${KH}" aria-hidden="true">${bars}${lab}${co}</svg>`;
}
function kpiTrend(ys) {
  if (ys.length < 5) return '';
  const xs = ys.map(r => r.year), lv = ys.map(r => Math.log1p(r.hotspots));
  const mx = xs.reduce((a, b) => a + b, 0) / xs.length, my = lv.reduce((a, b) => a + b, 0) / lv.length;
  const b = xs.reduce((a, x, i) => a + (x - mx) * (lv[i] - my), 0) / xs.reduce((a, x) => a + (x - mx) ** 2, 0);
  const fit = x => Math.expm1(my + b * (x - mx));
  const vals = ys.map(r => r.hotspots), max = Math.max(...vals, fit(xs[0]), fit(xs[xs.length - 1]), 1);
  const x = yr => 2 + (KW - 4) * (yr - xs[0]) / (xs[xs.length - 1] - xs[0]), y = v => KH - 4 - (KH - 12) * v / max;
  const ln = ys.map((r, i) => `${i ? 'L' : 'M'}${x(r.year).toFixed(1)},${y(r.hotspots).toFixed(1)}`).join('');
  const tr = xs.map((yr, i) => `${i ? 'L' : 'M'}${x(yr).toFixed(1)},${y(fit(yr)).toFixed(1)}`).join('');
  return `<svg class="kc" viewBox="0 0 ${KW} ${KH}" aria-hidden="true"><g class="sweep"><path d="${ln}" fill="none" class="kl soft"/><path d="${tr}" fill="none" class="kt ${b >= 0 ? 'up' : 'dn'}"/></g></svg>`;
}
// charts draw in when they come into view; re-rendered charts replay immediately if visible
const playIO = new IntersectionObserver(es => es.forEach(e => { if (e.isIntersecting) { e.target.classList.add('play'); playIO.unobserve(e.target); countIn(e.target); } }), { rootMargin: '0px 0px -8% 0px' });
function play(el) {
  el.classList.remove('play'); void el.offsetWidth;
  const r = el.getBoundingClientRect();
  if (r.top < innerHeight && r.bottom > 0) requestAnimationFrame(() => { el.classList.add('play'); countIn(el); });
  else playIO.observe(el);
}
function countIn(el) { $$('[data-count]', el).forEach(countUp); }
const reduceMotion = matchMedia('(prefers-reduced-motion: reduce)');
function countUp(el) {
  const to = +el.dataset.count; if (reduceMotion.matches || !to) return;
  const t0 = performance.now(), dur = 900;
  const tick = now => { const k = Math.min(1, (now - t0) / dur), e = 1 - Math.pow(1 - k, 4); el.textContent = fmt(to * e); if (k < 1) requestAnimationFrame(tick); };
  requestAnimationFrame(tick);
}
const ordinal = n => n + (n % 100 >= 11 && n % 100 <= 13 ? 'th' : ['th','st','nd','rd'][n % 10] || 'th');

/* ---------- colour scales ---------- */
function scaleOf(months) {
  const v = months.map(m => m.hotspots).filter(x => x > 0).sort((a, b) => a - b);
  const max = v.length ? v[v.length - 1] : 1, p99 = v.length ? v[Math.min(v.length - 1, Math.floor(.99 * v.length))] : 1;
  return { max, p99: Math.max(p99, 1) };
}
function seqIdx(v, sc) {
  if (v <= 0) return -1;
  if (state.scale === 'log') return Math.min(8, Math.floor(8.999 * Math.log1p(v) / Math.log1p(sc.max)));
  return Math.min(8, Math.floor(9 * v / sc.p99));
}
const divIdx = z => Math.max(0, Math.min(8, Math.round(z / Math.log(4) * 4 + 4)));
function colorOf(m, sc) {
  if (state.mode === 'anom') return (m.median === 0 && m.hotspots === 0) || m.z_log == null ? 'var(--nodata)' : `var(--div-${m.median === 0 ? 8 : divIdx(m.z_log)})`;
  const i = seqIdx(m.hotspots, sc);
  return i < 0 ? 'var(--nodata)' : `var(--seq-${i})`;
}
function compareText(m) {
  if (!m.median) return m.hotspots ? 'Usually there are no fires in this month here.' : '';
  const r = m.hotspots / m.median;
  if (r >= 1.15) return `${r >= 10 ? Math.round(r) : r.toFixed(1)}× a usual ${MONTH[m.month - 1]} (median ${fmt(m.median)})`;
  if (r <= 0.87) return `${Math.round((1 - r) * 100)}% below a usual ${MONTH[m.month - 1]} (median ${fmt(m.median)})`;
  return `About a usual ${MONTH[m.month - 1]} (median ${fmt(m.median)})`;
}
function srcText(m) { return SRC_TXT[m.src] + (m.rel_uncert ? `, typically within ±${Math.round(m.rel_uncert * 100)}%` : ''); }

/* ---------- the rings ---------- */
const CX = 300, CY = 300, R0 = 50, R1 = 262, GAP = 18;
const SPAN = (360 - GAP) / 12;
const pt = (r, a) => { const t = a * Math.PI / 180; return `${(CX + r * Math.cos(t)).toFixed(2)},${(CY + r * Math.sin(t)).toFixed(2)}`; };
const sector = (r1, r2, a1, a2) => `M${pt(r2, a1)}A${r2},${r2} 0 0 1 ${pt(r2, a2)}L${pt(r1, a2)}A${r1},${r1} 0 0 0 ${pt(r1, a1)}Z`;
const angle0 = m => -90 + GAP / 2 + m * SPAN;
function renderRing() {
  const d = state.data, el = $('#ring');
  const years = [...new Set(d.months.map(m => m.year))].sort((a, b) => a - b);
  if (!years.length) { el.innerHTML = ''; return; }
  const N = years.length, dr = (R1 - R0) / N, sc = scaleOf(d.months);
  let g = '';
  years.forEach((y, i) => {
    const r1 = R0 + i * dr, r2 = r1 + dr;
    let p = '';
    for (let mo = 0; mo < 12; mo++) {
      const m = state.byK[y * 12 + mo]; if (!m) continue;
      p += `<path class="cellp" d="${sector(r1, r2, angle0(mo), angle0(mo) + SPAN)}" style="fill:${colorOf(m, sc)}" data-k="${y * 12 + mo}"/>`;
    }
    g += `<g class="yrg" style="--i:${i}">${p}</g>`;
  });
  // MODIS-estimated rings carry a hatch; the 2012 boundary is drawn as a heavy ring
  const iEra = years.indexOf(2012), harmYears = years.filter(y => y < 2012).length;
  let over = '';
  if (harmYears) {
    const rH = R0 + harmYears * dr;
    over += `<path d="${sector(R0, rH, angle0(0), angle0(12) - .001)}" fill="url(#rh)" pointer-events="none"/>`;
  }
  if (iEra > 0) {
    const rE = R0 + iEra * dr;
    over += `<circle class="era" cx="${CX}" cy="${CY}" r="${rE}"/>`;
  }
  // selected year
  const iy = years.indexOf(state.year);
  if (iy >= 0) over += `<circle class="selring" cx="${CX}" cy="${CY}" r="${R0 + iy * dr}"/><circle class="selring" cx="${CX}" cy="${CY}" r="${R0 + (iy + 1) * dr}"/>`;
  // labels: months around the bark, years in the wedge
  const mons = MN.map((m, i) => { const a = (angle0(i) + SPAN / 2) * Math.PI / 180, r = R1 + 20; return `<text class="mon" x="${CX + r * Math.cos(a)}" y="${CY + r * Math.sin(a) + 6}" text-anchor="middle">${m}</text>`; }).join('');
  const step = N > 16 ? 5 : N > 8 ? 2 : 1;
  const yl = years.map((y, i) => (y % step === 0 || i === 0 || i === N - 1) && !(i === N - 2 && (years[N - 1] % step === 0)) ? `<text class="yr" x="${CX}" y="${CY - (R0 + (i + .5) * dr) + 4}" text-anchor="middle">${i === 0 || i === N - 1 || y % 10 === 0 ? y : "'" + String(y).slice(2)}</text>` : '').join('');
  el.innerHTML = `<svg viewBox="0 0 600 600" role="img" aria-label="Fire rings for ${esc(state.areaName)}: ${N} years from ${years[0]} at the centre to ${years[N - 1]} at the edge, twelve months around. The same data is in the month-by-month table below.">
    <defs><pattern id="rh" width="5" height="5" patternUnits="userSpaceOnUse" patternTransform="rotate(-45)"><rect width="1.1" height="5" style="fill:var(--hatch)"/></pattern></defs>
    <circle cx="${CX}" cy="${CY}" r="${R0 - 4}" fill="none" style="stroke:var(--rule)" stroke-dasharray="2 3"/>
    ${g}${over}${mons}${yl}</svg>`;
  el.classList.toggle('grow', !!state.growNext); state.growNext = false;
  const svg = $('svg', el);
  const pick = t => { const p = t.closest('.cellp'); return p ? state.byK[p.dataset.k] : null; };
  svg.addEventListener('pointerover', e => { const m = pick(e.target); if (m) { $$('.cellp.on', svg).forEach(x => x.classList.remove('on')); e.target.classList.add('on'); renderReadout(m); } });
  svg.addEventListener('click', e => { const m = pick(e.target); if (m) selectYear(m.year); });
  $('#ringLegend').innerHTML = legendHTML(sc, state.mode === 'anom');
  $('#ringKey').innerHTML = [
    `<li><i class="rk-c"></i>${years[0]} at the centre, ${years[N - 1]} at the edge</li>`,
    `<li><i class="rk-m"></i>Each spoke is a month, January at the top</li>`,
    iEra > 0 ? `<li><i class="rk-era"></i>Heavy ring: January 2012, when VIIRS takes over</li>` : '',
    harmYears ? `<li><i class="rk-h"></i>Hatched rings are estimated from MODIS</li>` : '',
    state.year ? `<li><i class="rk-sel"></i>Orange outline: ${state.year}, traced in the charts below</li>` : '',
    `<li class="tip-l">Hover a cell to read it. Click to trace that year.</li>`,
  ].join('');
}
function renderReadout(m) {
  if (!m) { const l = state.data.summary.latest; m = l ? state.byK[l.year * 12 + l.month - 1] : null; }
  if (!m) { $('#readout').innerHTML = ''; return; }
  $('#readout').innerHTML = `<div class="when">${MONTH[m.month - 1]} ${m.year}</div>
    <div class="n">${fmt(m.hotspots)}</div><div class="unit">hotspots, in VIIRS terms</div>
    <p>${compareText(m)}</p><p class="src">${srcText(m)}.</p>${yearBars(m)}`;
}
// the hovered month inside its year: twelve small bars, that month in ember
function yearBars(m) {
  const v = Array.from({ length: 12 }, (_, i) => { const x = state.byK[m.year * 12 + i]; return x ? x.hotspots : 0; });
  const max = Math.max(...v, 1), w = 240, h = 56, bw = (w - 11 * 3) / 12;
  return `<svg class="rb" viewBox="0 0 ${w} ${h}" aria-hidden="true">${v.map((x, i) => { const bh = Math.max(1.5, (h - 14) * x / max); return `<rect class="${i === m.month - 1 ? 'hi' : ''}" x="${(i * (bw + 3)).toFixed(1)}" y="${(h - 12 - bh).toFixed(1)}" width="${bw.toFixed(1)}" height="${bh.toFixed(1)}" rx="2"/><text class="${i === m.month - 1 ? 'hi' : ''}" x="${(i * (bw + 3) + bw / 2).toFixed(1)}" y="${h - 1}" text-anchor="middle">${MN[i][0]}</text>`; }).join('')}</svg>`;
}
function selectYear(y) {
  state.year = y;
  $$('.heat .yl').forEach(x => x.setAttribute('aria-pressed', String(+x.dataset.y === y)));
  $$('.heat .cell').forEach(c => c.classList.toggle('selyr', Math.floor(+c.dataset.k / 12) === y));
  renderRing(); renderClim();
}

/* ---------- calendar grid ---------- */
function cellInfo(m) {
  return `${MONTH[m.month - 1]} ${m.year}: ${fmt(m.hotspots)} hotspots. ${compareText(m)}. ${srcText(m)}.`;
}
function renderHeat() {
  const d = state.data, sc = scaleOf(d.months), anom = state.mode === 'anom';
  const y0 = +state.start.slice(0, 4), y1 = +state.end.slice(0, 4), lat = d.summary.latest;
  let html = '<div class="hh" aria-hidden="true"></div>' + MN.map(m => `<div class="hh" aria-hidden="true">${m}</div>`).join('');
  for (let y = y0; y <= y1; y++) {
    if (y === 2012 && y0 < 2012) html += `<div class="era-gap" aria-hidden="true"><span>VIIRS observes from 2012</span></div>`;
    html += `<button type="button" class="yl" data-y="${y}" aria-pressed="${y === state.year}" aria-label="Trace ${y} in the typical-year chart">${y}</button>`;
    for (let mo = 1; mo <= 12; mo++) {
      const k = y * 12 + mo - 1, m = state.byK[k];
      if (!m) { html += `<div class="cell empty" role="gridcell" aria-label="${MONTH[mo - 1]} ${y}: outside the period"></div>`; continue; }
      const cls = 'cell' + (isHarm(m.src) ? ' h' : m.src === 'VN' ? ' vn' : '') + (lat && lat.year === y && lat.month === mo ? ' cur' : '') + (y === state.year ? ' selyr' : '');
      html += `<div class="${cls}" role="gridcell" tabindex="-1" style="background:${colorOf(m, sc)}" data-k="${k}" aria-label="${esc(cellInfo(m))}"></div>`;
    }
  }
  const w = $('#heatWrap');
  w.innerHTML = `<div class="heat" role="grid" aria-label="Month by month, ${anom ? 'compared with the usual month' : 'hotspot counts'}">${html}</div>`;
  const g = $('.heat', w), first = $('.cell[data-k]', g); if (first) first.tabIndex = 0;
  g.onclick = e => { const b = e.target.closest('.yl'); if (b) selectYear(+b.dataset.y); const c = e.target.closest('.cell[data-k]'); if (c) selectYear(Math.floor(+c.dataset.k / 12)); };
  const tip = $('#tip');
  const show = el => {
    const m = state.byK[el.dataset.k]; if (!m) return;
    tip.innerHTML = `<div><b>${fmt(m.hotspots)}</b> hotspots, ${MN[m.month - 1]} ${m.year}</div><div>${compareText(m)}</div><div class="m">${srcText(m)}</div>`;
    tip.hidden = false; renderReadout(m);
    const r = el.getBoundingClientRect(), tw = tip.offsetWidth, th = tip.offsetHeight;
    tip.style.left = Math.max(8, Math.min(innerWidth - tw - 8, r.left + r.width / 2 - tw / 2)) + 'px';
    tip.style.top = (r.top - th - 8 < 8 ? r.bottom + 8 : r.top - th - 8) + 'px';
  };
  g.addEventListener('mouseover', e => { const c = e.target.closest('.cell[data-k]'); if (c) show(c); });
  g.addEventListener('mouseleave', () => tip.hidden = true);
  g.addEventListener('focusin', e => { const c = e.target.closest('.cell[data-k]'); if (c) show(c); });
  g.addEventListener('focusout', () => tip.hidden = true);
  g.addEventListener('keydown', e => {
    const c = e.target.closest('.cell[data-k]'); if (!c) return;
    if (e.key === 'Escape') { tip.hidden = true; return; }
    if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); selectYear(Math.floor(+c.dataset.k / 12)); return; }
    const dm = { ArrowLeft: -1, ArrowRight: 1, ArrowUp: -12, ArrowDown: 12 }[e.key]; if (dm == null) return;
    e.preventDefault();
    const n = $(`.cell[data-k="${+c.dataset.k + dm}"]`, g); if (n) { c.tabIndex = -1; n.tabIndex = 0; n.focus(); }
  });
  renderLegend(sc, anom);
}
function renderLegend(sc, anom) { $('#legend').innerHTML = legendHTML(sc, anom); }
function legendHTML(sc, anom) {
  const bar = Array.from({ length: 9 }, (_, i) => `<i style="background:var(--${anom ? 'div' : 'seq'}-${i})"></i>`).join('');
  let t;
  if (anom) t = ['a quarter of usual', 'usual', '4× usual'];
  else { const lg = state.scale === 'log', e = f => lg ? Math.expm1(Math.log1p(sc.max) * f) : sc.p99 * f; t = ['1', fmt(e(.5)), (lg ? '' : 'over ') + fmt(lg ? sc.max : sc.p99)]; }
  return `<div class="lg-bar" aria-hidden="true">${bar}</div><div class="lg-ticks"><span>${t[0]}</span><span>${t[1]}</span><span>${t[2]}</span></div><p class="hint" style="margin-top:4px">${anom ? 'Each month against the median of the same month, 2001–2025.' : `Hotspots per month${state.scale === 'log' ? ', log scale' : ''}. Grey means none detected.`}</p>`;
}

/* ---------- charts ---------- */
const W = 520, H = 280, P = { l: 46, r: 8, t: 14, b: 28 };
function niceMax(v) { if (v <= 0) return 1; const p = 10 ** Math.floor(Math.log10(v)); const f = v / p; return (f <= 1 ? 1 : f <= 2 ? 2 : f <= 5 ? 5 : 10) * p; }
const kfmt = v => v >= 1e6 ? +(v / 1e6).toFixed(1) + 'M' : v >= 1e3 ? +(v / 1e3).toFixed(1) + 'k' : String(Math.round(v));
function yAxis(max, w = W, h = H, p = P) {
  let s = '';
  for (let i = 0; i <= 4; i++) { const v = max * i / 4, y = p.t + (h - p.t - p.b) * (1 - i / 4); s += `<line class="${i ? 'gl' : 'gl0'}" x1="${p.l}" x2="${w - p.r}" y1="${y}" y2="${y}"/><text x="${p.l - 8}" y="${y + 4}" text-anchor="end">${kfmt(v)}</text>`; }
  return s;
}
const grad = (id, op = .32) => `<linearGradient id="${id}" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="var(--scar)" stop-opacity="${op}"/><stop offset="1" stop-color="var(--scar)" stop-opacity="0"/></linearGradient>`;
// hover: a vertical crosshair snaps to the nearest x and a tooltip reads that column
function crosshair(box, w, h, p, xs, html) {
  const svg = $('svg', box); if (!svg || !xs.length) return;
  svg.insertAdjacentHTML('beforeend', `<line class="xh" x1="0" x2="0" y1="${p.t}" y2="${h - p.b}" visibility="hidden"/>`);
  const xh = $('.xh', svg);
  svg.onpointermove = e => {
    const r = svg.getBoundingClientRect(), vx = (e.clientX - r.left) * w / r.width;
    let k = 0; xs.forEach((x, i) => { if (Math.abs(x - vx) < Math.abs(xs[k] - vx)) k = i; });
    xh.setAttribute('x1', xs[k]); xh.setAttribute('x2', xs[k]); xh.setAttribute('visibility', 'visible');
    tipAt(html(k), r.left + xs[k] * r.width / w, r.top + p.t * r.height / h);
  };
  svg.onpointerleave = () => { xh.setAttribute('visibility', 'hidden'); $('#tip').hidden = true; };
}
function renderClim() {
  const d = state.data, c = d.climatology, id = 'cg' + (++uid);
  const ym = {}; d.months.filter(m => m.year === state.year).forEach(m => ym[m.month] = m.hotspots);
  const max = niceMax(Math.max(...c.map(x => x.p90), ...Object.values(ym), 1));
  const x = i => P.l + (W - P.l - P.r) * (i + .5) / 12, y = v => P.t + (H - P.t - P.b) * (1 - v / max);
  const line = f => c.map((r, i) => `${i ? 'L' : 'M'}${x(i).toFixed(1)},${y(f(r)).toFixed(1)}`).join('');
  const band = c.map((r, i) => `${i ? 'L' : 'M'}${x(i)},${y(r.p90)}`).join('') + [...c].reverse().map((r, i) => `L${x(11 - i)},${y(r.p10)}`).join('') + 'Z';
  const yr = Object.keys(ym).map(Number).sort((a, b) => a - b);
  const yl = yr.map((m, i) => `${i ? 'L' : 'M'}${x(m - 1).toFixed(1)},${y(ym[m]).toFixed(1)}`).join('');
  const ya = yr.length > 1 ? `${yl}L${x(yr[yr.length - 1] - 1)},${H - P.b}L${x(yr[0] - 1)},${H - P.b}Z` : '';
  $('#clim').innerHTML = `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="Typical year: median month and the range holding 8 years in 10, with ${state.year} traced"><defs>${grad(id)}</defs>${yAxis(max)}
    ${MN.map((m, i) => `<text x="${x(i)}" y="${H - 8}" text-anchor="middle">${m}</text>`).join('')}
    <g class="sweep"><path d="${band}" class="band"/><path d="${line(r => r.median)}" fill="none" class="med"/>
    ${yl ? `<path d="${ya}" fill="url(#${id})"/><path d="${yl}" fill="none" class="sel"/>` : ''}</g>
    ${yr.map((m, i) => `<circle class="pop dot" style="--i:${i}" cx="${x(m - 1)}" cy="${y(ym[m])}" r="3.6"/>`).join('')}</svg>`;
  $('#yrKey').textContent = state.year ? `The orange line is ${state.year}.` : '';
  crosshair($('#clim'), W, H, P, c.map((_, i) => x(i)), k => `<div><b>${MONTH[k]}</b></div><div>Median ${fmt(c[k].median)}, 8 in 10 years ${fmt(c[k].p10)}–${fmt(c[k].p90)}</div>${ym[k + 1] != null ? `<div>${state.year}: <b>${fmt(ym[k + 1])}</b></div>` : ''}`);
  play($('#clim'));
}
function renderAnnual() {
  const d = state.data, ys = d.yearly; if (!ys.length) { $('#annual').innerHTML = ''; return; }
  const cnt = {}; d.months.forEach(m => { const o = cnt[m.year] ||= { h: 0, n: 0 }; o.n++; if (isHarm(m.src)) o.h++; });
  const max = niceMax(Math.max(...ys.map(r => r.hotspots), 1)), bw = (W - P.l - P.r) / ys.length, worst = d.summary.worst_year, id = 'ag' + (++uid);
  const cx = i => P.l + i * bw + bw / 2;
  const bars = ys.map((r, i) => {
    const est = cnt[r.year] && cnt[r.year].h > cnt[r.year].n / 2, partial = cnt[r.year] && cnt[r.year].n < 12;
    const hh = Math.max((H - P.t - P.b) * r.hotspots / max, 0), xx = P.l + i * bw + bw * .16, w = bw * .68, yy = H - P.b - hh;
    const cls = r.year === state.year ? 'bar sel' : r.year === worst ? 'bar worst' : 'bar';
    return `<g><rect class="${cls}" style="--i:${i}${partial ? ';opacity:.45' : ''}" x="${xx}" y="${yy}" width="${w}" height="${hh}" rx="${Math.min(3, w / 3)}"/>${est ? `<rect class="bar" style="--i:${i};fill:url(#bh)" x="${xx}" y="${yy}" width="${w}" height="${hh}" rx="${Math.min(3, w / 3)}"/>` : ''}${r.year === worst ? `<g class="pop"><rect class="co" x="${cx(i) - 34}" y="${yy - 22}" width="68" height="16" rx="8"/><text class="cot" x="${cx(i)}" y="${yy - 10.5}" text-anchor="middle">Worst, ${r.year}</text></g>` : ''}</g>`;
  }).join('');
  const step = ys.length > 14 ? 5 : ys.length > 7 ? 2 : 1;
  const ticks = ys.map((r, i) => r.year % step === 0 ? `<text x="${cx(i)}" y="${H - 8}" text-anchor="middle">${r.year}</text>` : '').join('');
  $('#annual').innerHTML = `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="Hotspots per year"><defs>${grad(id, .9)}<pattern id="bh" width="5" height="5" patternUnits="userSpaceOnUse" patternTransform="rotate(-45)"><rect width="1.3" height="5" style="fill:var(--hatch)"/></pattern></defs>${yAxis(max)}${bars}${ticks}</svg>`;
  crosshair($('#annual'), W, H, P, ys.map((_, i) => cx(i)), k => { const r = ys[k], c = cnt[r.year]; return `<div><b>${fmt(r.hotspots)}</b> hotspots in ${r.year}</div>${c && c.h > c.n / 2 ? '<div class="m">Estimated from MODIS</div>' : ''}${c && c.n < 12 ? `<div class="m">${c.n} months only</div>` : ''}`; });
  play($('#annual'));
}
function renderAll(grow) {
  state.growNext = grow;
  renderReading(); renderRing(); renderReadout(null); renderHeat(); renderClim(); renderAnnual(); renderRelief(); renderNow(); renderAgree();
}

/* ---------- exports ---------- */
async function download(path, body, name) {
  const el = $('#exportMsg'); el.textContent = 'Preparing the file…';
  try {
    const r = await api(path, body), b = await r.blob(), u = URL.createObjectURL(b), a = document.createElement('a');
    a.href = u; a.download = name; document.body.append(a); a.click(); a.remove(); setTimeout(() => URL.revokeObjectURL(u), 5000);
    el.textContent = `Downloaded ${name}, ${(b.size / 1048576).toFixed(b.size > 1048576 ? 1 : 2)} MB.`;
  } catch (e) { el.textContent = `The download did not finish (${e.message}). Try a smaller area or a shorter period.`; }
}
const slug = () => state.areaName.toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/^-|-$/g, '') || 'area';
$('#csvCal').onclick = () => state.area && download('/api/export/calendar.csv', reqBody(), `brasa-${slug()}-calendar.csv`);
$('#csvPts').onclick = () => {
  if (!state.area) return;
  const [a, b] = [state.start, state.end].map(s => +s.slice(0, 4) * 12 + +s.slice(5) - 1);
  if ((b - a > 24 || state.area.type === 'country') && !confirm('Every hotspot for this area and period could be a very large file (up to 500,000 rows). Download it anyway?')) return;
  download('/api/export/points.csv', { ...reqBody(), sensors: ['modis', 'viirs-snpp', 'viirs-jpss1'] }, `brasa-${slug()}-hotspots.csv`);
};

/* ---------- method dialog ---------- */
$$('[data-how]').forEach(b => b.onclick = () => scrollToEl($('#method')));
// When the U-Net ensemble is loaded (pipeline/08_merge_unet.py writes app/nn.json), the method section
// describes it; otherwise it keeps describing the ratio model the data actually comes from.
// The method, as the U-Net pipeline actually runs. Each step has a diagram; on wide screens the diagram
// sits in a sticky stage beside the text, on narrow ones it follows its step.
const VW = 560, VH = 400;
const svgV = (body, label) => `<svg viewBox="0 0 ${VW} ${VH}" role="img" aria-label="${label}">${body}</svg>`;
const cap = (lines, y = 356) => `<text class="vm" x="40" y="${y}">${lines.map((l, i) => `<tspan x="40" dy="${i ? 17 : 0}">${l}</tspan>`).join('')}</text>`;
function rng(seed) { return () => (seed = (seed * 16807) % 2147483647) / 2147483647; }
const VIZ = {
  sensors() {
    const fires = [[50, 50], [140, 140], [156, 46]], g = 40;
    const lit = fires.map(([fx, fy]) => [Math.floor(fx / g), Math.floor(fy / g)]);
    let grid = ''; for (let k = 1; k < 5; k++) grid += `<line class="gl2" x1="${320 + k * g}" y1="74" x2="${320 + k * g}" y2="274"/><line class="gl2" x1="320" y1="${74 + k * g}" x2="520" y2="${74 + k * g}"/>`;
    return svgV(`<text class="vt" x="40" y="40">MODIS</text><text class="vm" x="40" y="58">1 km pixels · since Nov 2000</text>
      <rect class="lit pop" style="--i:0" x="40" y="74" width="200" height="200" rx="6"/><rect class="px" x="40" y="74" width="200" height="200" rx="6"/>
      ${fires.map(([x, y]) => `<circle class="fire" cx="${40 + x}" cy="${74 + y}" r="5"/>`).join('')}
      <text class="vb pop" style="--i:2" x="40" y="318">1</text><text class="vm pop" style="--i:2" x="62" y="318">hotspot</text>
      <text class="vt" x="320" y="40">VIIRS</text><text class="vm" x="320" y="58">375 m pixels · since Jan 2012</text>
      ${lit.map(([c, r], k) => `<rect class="lit pop" style="--i:${4 + k * 3}" x="${320 + c * g}" y="${74 + r * g}" width="${g}" height="${g}"/>`).join('')}
      <rect class="px" x="320" y="74" width="200" height="200" rx="6"/>${grid}
      ${fires.map(([x, y]) => `<circle class="fire" cx="${320 + x}" cy="${74 + y}" r="5"/>`).join('')}
      <text class="vb pop" style="--i:14" x="320" y="318">3</text><text class="vm pop" style="--i:14" x="342" y="318">hotspots</text>
      ${cap(['The same three fires.', 'Over the whole planet VIIRS reports about 4× as many hotspots.'])}`, 'One coarse MODIS pixel flags one hotspot; the finer VIIRS grid flags three for the same fires');
  },
  grid() {
    const r = rng(7), cx0 = 56, cy0 = 76, c = 64, cols = 7, rows = 4, pts = [], cnt = {};
    const hubs = [[150, 140, 40], [330, 230, 34], [420, 120, 24]];
    for (let k = 0; k < 46; k++) { const h = hubs[k % 3], a = r() * 6.283, d = Math.sqrt(r()) * h[2] * 1.6; pts.push([h[0] + Math.cos(a) * d, h[1] + Math.sin(a) * d]); }
    pts.forEach(([x, y]) => { const key = Math.floor((x - cx0) / c) + ',' + Math.floor((y - cy0) / c); cnt[key] = (cnt[key] || 0) + 1; });
    const max = Math.max(...Object.values(cnt));
    let cells = '';
    for (let i = 0; i < cols; i++) for (let j = 0; j < rows; j++) { const n = cnt[i + ',' + j] || 0; cells += n ? `<rect class="pop" style="--i:${30 + (i + j) * 1.5};fill:var(--seq-${Math.min(8, 2 + Math.round(6 * n / max))})" x="${cx0 + i * c + 1}" y="${cy0 + j * c + 1}" width="${c - 2}" height="${c - 2}" rx="4"/><text class="vn pop" style="--i:${32 + (i + j) * 1.5}" x="${cx0 + i * c + c / 2}" y="${cy0 + j * c + c / 2 + 5}" text-anchor="middle">${n}</text>` : ''; }
    let lines = ''; for (let i = 0; i <= cols; i++) lines += `<line class="gl2" x1="${cx0 + i * c}" y1="${cy0}" x2="${cx0 + i * c}" y2="${cy0 + rows * c}"/>`; for (let j = 0; j <= rows; j++) lines += `<line class="gl2" x1="${cx0}" y1="${cy0 + j * c}" x2="${cx0 + cols * c}" y2="${cy0 + j * c}"/>`;
    const out = [[500, 300], [470, 96], [90, 300]];
    return svgV(`<text class="vt" x="56" y="40">Every detection → one cell, one month</text><text class="vm" x="56" y="58">0.25° cells, about 25 km across</text>
      ${cells}${lines}${pts.map(([x, y], k) => `<circle class="dotp pop" style="--i:${k * .5}" cx="${x.toFixed(1)}" cy="${y.toFixed(1)}" r="3"/>`).join('')}
      ${out.map(([x, y], k) => `<g class="pop" style="--i:${8 + k}"><path class="xo" d="M${x - 5},${y - 5}L${x + 5},${y + 5}M${x + 5},${y - 5}L${x - 5},${y + 5}"/></g>`).join('')}
      ${cap(['✕  volcanoes, gas flares and low-confidence', 'detections are removed'])}`, 'Fire detections fall into a grid of 25 km cells and are counted per cell');
  },
  inputs() {
    const L = ['Fire counts, all and confident', 'Terra and night counts', 'Fire power: histogram, total, peak', 'Fire days and the busiest day', 'Pixel size, confidence, brightness'];
    const layers = L.map((t, k) => { const y = 300 - k * 44; return `<g class="pop" style="--i:${k * 3}"><path class="layer${k % 2 ? ' alt' : ''}" d="M40,${y} l170,0 l50,-34 l-170,0 z"/><text class="vl" x="282" y="${y - 12}">${t}</text></g>`; }).join('');
    return svgV(`<text class="vt" x="40" y="40">15 MODIS signals per cell and month</text>
      <g class="pop" style="--i:16">${['t−1', 't', 't+1'].map((t, k) => `<rect class="tab${k === 1 ? ' on' : ''}" x="${40 + k * 58}" y="58" width="50" height="22" rx="11"/><text class="vm" x="${65 + k * 58}" y="73" text-anchor="middle">${t}</text>`).join('')}<text class="vm" x="220" y="73">the month before, itself and after</text></g>
      ${layers}
      <text class="vm pop" style="--i:18" x="40" y="340">+ latitude, longitude and a MODIS-only climatology</text>
      <g class="pop" style="--i:20"><rect class="tgt" x="40" y="356" width="480" height="30" rx="8"/><text class="vm" x="54" y="375">Target while training, 2012–2017 only: what VIIRS S-NPP counted</text></g>`, 'Five stacked layers of MODIS signals for the previous, current and next month');
  },
  unet() {
    const B = [[64, 170, 200], [124, 205, 130], [184, 235, 80], [250, 262, 44], [316, 235, 80], [376, 205, 130], [436, 170, 200]];
    const blk = B.map(([x, c, h], k) => `<rect class="blk${k === 3 ? ' mid' : ''} pop" style="--i:${k * 2}" x="${x}" y="${c - h / 2}" width="28" height="${h}" rx="5"/>`).join('');
    const skip = [0, 1, 2].map(k => { const a = B[k], b = B[6 - k], y = a[1] - a[2] / 2 + 14 + k * 6; return `<path class="skip" d="M${a[0] + 28},${y} L${b[0]},${y}"/>`; }).join('');
    const flow = B.slice(0, -1).map((a, k) => { const b = B[k + 1]; return `<path class="flow" d="M${a[0] + 28},${a[1]} L${b[0]},${b[1]}"/>`; }).join('');
    const tile = (x, cls) => { let t = ''; for (let i = 0; i < 4; i++) for (let j = 0; j < 4; j++) t += `<rect x="${x + i * 9}" y="${152 + j * 9}" width="8" height="8" rx="1.5" style="fill:var(--seq-${cls[(i * 4 + j) % cls.length]})"/>`; return t; };
    return svgV(`<text class="vt" x="40" y="40">A U-Net: 3 levels, 1.8 million parameters</text><text class="vm" x="40" y="58">down to see wide context, back up to 25 km detail</text>
      <g class="pop" style="--i:0">${tile(18, [2, 4, 3, 5, 6, 3, 2, 4])}</g>
      <g class="sweep">${flow}${skip}</g>${blk}
      <g class="pop" style="--i:16">${tile(482, [4, 6, 5, 7, 8, 5, 4, 6])}</g>
      <text class="vm" x="64" y="318">encoder: ~17° of context</text>
      <text class="vm" x="464" y="338" text-anchor="end">decoder: back to each cell</text>
      ${cap(['Output: a correction to the 4.1× ratio,', 'for every cell and every month'])}`, 'U-shaped network: encoder blocks shrink toward a bottleneck, decoder blocks grow back, with skip connections across');
  },
  ensemble() {
    return svgV(`<text class="vt" x="40" y="40">Two models, one answer</text>
      <g class="pop" style="--i:0"><rect class="box" x="40" y="72" width="220" height="96" rx="12"/><text class="vt" x="60" y="104">U-Net</text><text class="vm" x="60" y="124">reads about 17° of map</text><rect class="chipv" x="60" y="136" width="178" height="20" rx="10"/><text class="vm" x="70" y="150">under-counts tiny fires</text></g>
      <g class="pop" style="--i:3"><rect class="box" x="40" y="222" width="220" height="96" rx="12"/><text class="vt" x="60" y="254">Per-pixel MLP</text><text class="vm" x="60" y="274">cell plus neighbourhood sums</text><rect class="chipv ok" x="60" y="286" width="120" height="20" rx="10"/><text class="vm" x="70" y="300">nearly unbiased</text></g>
      <g class="sweep"><path class="flow" d="M260,120 C310,120 300,195 330,195"/><path class="flow" d="M260,270 C310,270 300,195 330,195"/><path class="flow" d="M390,195 L410,195"/></g>
      <g class="pop" style="--i:8"><circle class="mean" cx="360" cy="195" r="30"/><text class="vt" x="360" y="200" text-anchor="middle">½ + ½</text></g>
      <g class="pop" style="--i:11"><rect class="box hot" x="410" y="145" width="140" height="100" rx="12"/><text class="vt" x="426" y="177">Prediction</text><text class="vm" x="426" y="199">VIIRS-equivalent</text><text class="vm" x="426" y="217">hotspots per cell</text><text class="vm" x="426" y="233">and month</text></g>
      ${cap(['The 50/50 weights were fixed before', 'looking at the test years'])}`, 'The U-Net and a per-pixel model are averaged into one prediction');
  },
  holdout() {
    const H = (state.meta.nn && state.meta.nn.holdout) || [], short = { m3: 'Ratio model (M3)', mlp: 'Per-pixel MLP', unet: 'U-Net', ens: 'U-Net + MLP', ceil: 'Two VIIRS satellites' };
    const rows = ['m3', 'mlp', 'unet', 'ens', 'ceil'].map(k => H.find(r => r.key === k)).filter(Boolean);
    const sx = v => 300 * v / .3;
    return svgV(`<text class="vt" x="40" y="40">Error at 5° boxes, 2019–2024</text><text class="vm" x="40" y="58">median |log error| of monthly totals · lower is better</text>
      ${rows.map((r, k) => { const y = 88 + k * 52; return `<text class="vl${r.key === 'ens' ? ' hot' : ''}" x="40" y="${y + 18}">${short[r.key]}</text>
        <rect class="hb gx${r.key === 'ens' ? ' hot' : r.key === 'ceil' ? ' ceil' : ''}" style="--i:${k * 3}" x="200" y="${y}" width="${sx(r.e5).toFixed(1)}" height="26" rx="6"/>
        <text class="vv pop" style="--i:${8 + k * 2}" x="${(208 + sx(r.e5)).toFixed(1)}" y="${y + 18}">${r.e5.toFixed(3)}</text>`; }).join('')}
      ${cap(['0.20 is a typical miss of about 22%.', 'The last bar is the best any conversion could do.'])}`, 'Bar chart of 5-degree errors on the held-out years');
  },
  record() {
    const x = y => 172 + (y - 2000) / 26 * 360, seg = (a, b, y, cls, i) => `<rect class="seg gx ${cls}" style="--i:${i}" x="${x(a).toFixed(1)}" y="${y}" width="${(x(b) - x(a)).toFixed(1)}" height="30" rx="6"/>`;
    return svgV(`<text class="vt" x="40" y="40">Where each month of the calendar comes from</text>
      <text class="vl" x="40" y="119">Rest of the world</text><text class="vl" x="40" y="199">Brazil</text>
      ${seg(2000.83, 2002.5, 98, 'm3', 0)}${seg(2002.5, 2012.08, 98, 'nn', 2)}${seg(2012.08, 2026, 98, 'vo', 4)}
      ${seg(2000.83, 2012.08, 178, 'm3', 1)}${seg(2012.08, 2026, 178, 'vo', 5)}
      ${[2022.58, 2024.5].map((y, k) => `<g class="pop" style="--i:${10 + k}"><line class="out" x1="${x(y)}" y1="92" x2="${x(y)}" y2="214"/></g>`).join('')}
      ${[2000, 2006, 2012, 2018, 2025].map(y => `<text class="vm" x="${x(y)}" y="240" text-anchor="middle">${y}</text>`).join('')}
      <g class="pop" style="--i:12"><rect class="seg m3" x="40" y="276" width="16" height="12" rx="3"/><text class="vm" x="62" y="287">ratio model</text>
      <rect class="seg nn" x="160" y="276" width="16" height="12" rx="3"/><text class="vm" x="182" y="287">U-Net ensemble</text>
      <rect class="seg vo" x="300" y="276" width="16" height="12" rx="3"/><text class="vm" x="322" y="287">VIIRS observed</text>
      <line class="out" x1="48" y1="300" x2="48" y2="316"/><text class="vm" x="62" y="313">NOAA-20 fills a Suomi NPP outage</text></g>
      ${cap(['Brazil and the Terra-only months before July 2002', 'stay on the ratio model, which is more accurate there'])}`, 'Timeline of data sources from 2000 to 2025');
  },
};
// Diagrams for the ratio dataset shipped in the prebuilt release.
Object.assign(VIZ, {
  ratio() {
    return svgV(`<text class="vt" x="40" y="40">Learn the conversion during the overlap</text>
      <text class="vm" x="40" y="64">Illustrative counts for the same cell and month</text>
      <rect class="box" x="40" y="104" width="200" height="112" rx="12"/>
      <text class="vl" x="60" y="135">MODIS observed</text><text class="vb" x="60" y="183">10</text>
      <rect class="box hot" x="320" y="104" width="200" height="112" rx="12"/>
      <text class="vl" x="340" y="135">VIIRS observed</text><text class="vb" x="340" y="183">40</text>
      <path class="flow" d="M250,160H310M300,152L310,160L300,168"/>
      <text class="vl" x="40" y="267">40 / 10 = a conversion factor of 4</text>
      ${cap(['The fitted factor varies by location and season.', 'Sparse cells borrow information from larger neighbourhoods.'])}`,
      'Illustrative MODIS and VIIRS counts of 10 and 40 give a local conversion factor of four');
  },
  floor() {
    return svgV(`<text class="vt" x="40" y="40">Account for small fires MODIS misses</text>
      <text class="vm" x="40" y="64">Illustrative overlap observations</text>
      <rect class="box" x="40" y="104" width="200" height="112" rx="12"/>
      <text class="vl" x="60" y="135">MODIS observed</text><text class="vb" x="60" y="183">0</text>
      <rect class="box hot" x="320" y="104" width="200" height="112" rx="12"/>
      <text class="vl" x="340" y="135">VIIRS observed</text><text class="vb" x="340" y="183">3</text>
      <text class="vl" x="40" y="267">A multiplier alone would still give zero.</text>
      ${cap(['The small-fire floor learns a typical VIIRS count', 'for active cells and months where MODIS recorded zero.'])}`,
      'Illustrative zero MODIS detections and three VIIRS detections explain the estimated small-fire floor');
  },
  history() {
    return svgV(`<text class="vt" x="40" y="40">Keep the source of every monthly value</text>
      <rect class="box" x="40" y="90" width="480" height="64" rx="10"/>
      <text class="vl" x="58" y="117">Nov 2000 to Jun 2002</text><text class="vm" x="58" y="139">Terra-only conversion</text>
      <rect class="box hot" x="40" y="170" width="480" height="64" rx="10"/>
      <text class="vl" x="58" y="197">Jul 2002 to Jan 2012</text><text class="vm" x="58" y="219">Terra + Aqua converted to VIIRS-equivalent counts</text>
      <rect class="box" x="40" y="250" width="480" height="64" rx="10"/>
      <text class="vl" x="58" y="277">From Feb 2012</text><text class="vm" x="58" y="299">S-NPP observed; NOAA-20 substitutes in selected gaps</text>
      ${cap(['The calendar combines converted history', 'and observed VIIRS, with the source recorded each month.'])}`,
      'Timeline of the ratio calendar sources: Terra, Terra plus Aqua, then observed VIIRS');
  },
});
const NN_STEPS = [
  { t: 'Two satellites, two scales', viz: 'sensors', facts: ['MODIS · 1 km · 2000–', 'VIIRS · 375 m · 2012–', 'about 4× more hotspots'],
    p: 'MODIS has watched fires since 2000 with 1 km pixels. VIIRS, flying since 2012, has 375 m pixels and catches fires too small for MODIS, so it reports about four times as many hotspots. Put side by side, 2012 would look like a sudden explosion of fire that never happened.' },
  { t: 'Put every detection on one grid', viz: 'grid', facts: ['0.25° ≈ 25 km cells', 'monthly', 'vegetation fires only'],
    p: 'Every detection lands in a 0.25° cell, about 25 km across, and a month. Only presumed vegetation fires count: volcanoes, gas flares and other fixed heat sources are removed, and so are low-confidence detections.' },
  { t: 'Describe each month with 15 MODIS signals', viz: 'inputs', facts: ['15 channels', 't−1, t, t+1', 'no VIIRS input'],
    p: 'A count alone hides a lot. The model sees how many fires MODIS found, how many at night and by Terra, how much heat they gave off, on how many days and how large the pixels were, for the month before, the month itself and the month after. Location and a MODIS-only climatology complete the picture. Nothing comes from VIIRS, so it runs on years before VIIRS existed.' },
  { t: 'A U-Net reads the map around each cell', viz: 'unet', facts: ['3-level U-Net', '1.8 M parameters', 'trained 2012–2017'],
    p: 'The network starts from the global ratio of about 4.1 VIIRS hotspots per MODIS hotspot and learns where and when to depart from it. Seeing about 17° of map around every cell, it learns where VIIRS finds small fires MODIS missed entirely. It was trained on 2012–2017, when both satellites flew, with a loss built for overdispersed counts and a term that keeps regional totals right.' },
  { t: 'Average it with a second model', viz: 'ensemble', facts: ['mean of 2 models', 'weights fixed in advance'],
    p: 'A small per-pixel network sees the same signals plus neighbourhood sums, but no map. The U-Net under-predicts the very smallest fires, where the per-pixel model is nearly unbiased. Their mistakes fall in different places, so the plain average of the two beats both.' },
  { t: 'Test it on years it never saw', viz: 'holdout', facts: ['test 2019–2024', '0.203 vs 0.246', 'better every test year'],
    p: '2019 to 2024 was held out entirely. Asked to predict those years from MODIS alone, the ensemble lands closer to what VIIRS actually saw than Brasa\'s ratio model at every scale and in every test year. Two VIIRS satellites compared with each other set the floor no conversion can beat.' },
  { t: 'Assemble the 25-year record', viz: 'record', facts: ['2002–2012: U-Net', '2012–: VIIRS observed', 'Brazil: ratio model'],
    p: 'The calendar uses the ensemble from July 2002 to January 2012 and what VIIRS observed from then on, with NOAA-20 filling the two Suomi NPP outages. In Brazil, and in the Terra-only months before July 2002, the ratio model is more accurate, so it stays there.' },
];
function renderMethod() {
  const nn = state.meta.nn;
  if (!nn) {
    const diagrams = [VIZ.grid(), VIZ.ratio(), VIZ.floor(), VIZ.history()];
    $$('#howSteps > li').forEach((step, i) => {
      $('.vz-inline', step)?.remove();
      step.insertAdjacentHTML('beforeend', `<div class="vz vz-inline">${diagrams[i]}</div>`);
    });
    const stage = $('#howStage');
    stage.hidden = false;
    stage.innerHTML = '<div class="stage-prog"><i></i></div>'
      + diagrams.map((diagram, i) => `<div class="vz" data-step="${i}">${diagram}</div>`).join('');
    watchSteps();
    return;
  }
  $('#howLede').textContent = 'VIIRS sees about four times as many hotspots as MODIS. A neural network trained where both satellites flew rewrites the MODIS years in VIIRS terms.';
  $('#tlPanel').hidden = true;
  const n = NN_STEPS.length, pad = k => String(k).padStart(2, '0');
  $('#howSteps').innerHTML = NN_STEPS.map((x, i) => `<li data-step="${i}"><span class="sn">${pad(i + 1)} <i>/ ${pad(n)}</i></span><h3>${x.t}</h3><p>${x.p}</p><ul class="facts-v">${x.facts.map(f => `<li>${f}</li>`).join('')}</ul><div class="vz vz-inline">${VIZ[x.viz]()}</div></li>`).join('');
  const st = $('#howStage'); st.hidden = false;
  st.innerHTML = `<div class="stage-prog"><i></i></div>` + NN_STEPS.map((x, i) => `<div class="vz" data-step="${i}">${VIZ[x.viz]()}</div>`).join('');
  $('#valSub').textContent = 'Every model was trained on 2012–2018 at most, then asked to predict 2019–2024 from MODIS alone and compared with what VIIRS actually saw. Lower error is better; the sign column is how often a strong anomaly at 5° points the right way. The last row compares two VIIRS satellites with each other: no conversion can do better than that.';
  const f = x => x == null ? '–' : (+x).toFixed(3);
  $('#valTable').innerHTML = `<tr><th>Method</th><th>Error, 1° boxes</th><th>Error, 5° boxes</th><th>Error, countries</th><th>Anomaly sign, 5°</th></tr>`
    + nn.holdout.map(r => `<tr class="${r.key === 'ens' ? 'best' : r.key === 'ceil' ? 'ceil' : ''}"><td>${esc(r.method)}</td><td>${f(r.e1)}</td><td>${f(r.e5)}</td><td>${f(r.ec)}</td><td>${f(r.sign5)}</td></tr>`).join('');
  watchSteps();
}
function renderValidation() {
  const v = (state.meta.validation || []).filter(r => /^M0|M3 \+floor \(a=20\)|CEILING/.test(r.model) && r.scale !== '1deg');
  const name = m => /^M0/.test(m) ? 'One global ratio (about 4×)' : /CEILING/.test(m) ? 'Two VIIRS satellites, best possible' : 'Brasa';
  const scl = s => ({ '5deg': '5° boxes', country: 'Countries', '1deg': '1° boxes' }[s] || s);
  const f = (x, d) => x == null ? '–' : (+x).toFixed(d);
  $('#valTable').innerHTML = `<tr><th>Method</th><th>Areas</th><th>Bias</th><th>Typical error</th><th>Correlation</th><th>Anomaly correlation</th></tr>`
    + v.map(r => `<tr class="${/CEILING/.test(r.model) ? 'ceil' : /M3/.test(r.model) ? 'best' : ''}"><td>${name(r.model)}</td><td>${scl(r.scale)}</td><td>${(+r.bias_pct >= 0 ? '+' : '−') + Math.abs(+r.bias_pct).toFixed(1)}%</td><td>${f(r.med_abs_log_err, 2)}</td><td>${f(r.r_log, 3)}</td><td>${f(r.anomaly_r, 2)}</td></tr>`).join('');
}

/* ---------- start ---------- */
(async function init() {
  $('#themeBtn').setAttribute('aria-label', theme() === 'night' ? 'Switch to day theme' : 'Switch to night theme');
  initScroll();
  if (typeof L !== 'undefined') initMap(); else $('#map').innerHTML = '<p class="hint" style="padding:16px">The map could not load. Use Coordinates, Radius or Country instead.</p>';
  try {
    const [meta, countries] = await Promise.all([api('/api/meta').then(r => r.json()), api('/api/countries').then(r => r.json())]);
    state.meta = meta; state.countries = countries;
    state.start = meta.first_month; state.end = meta.last_month;
    for (const id of ['#mFrom', '#mTo']) { $(id).min = meta.first_month; $(id).max = meta.last_month; }
    $('#mFrom').value = state.start; $('#mTo').value = state.end;
    $('#countryList').innerHTML = countries.map(c => `<option value="${esc(c)}">`).join('');
    renderValidation(); renderMethod();
    $('#liveTag span').textContent = `Record to ${ymLabel(meta.last_month)}`; $('#liveTag').hidden = false;
    $('#presets button').click();
    init3d();
  } catch (e) {
    $('#status').className = 'status error';
    $('#status').textContent = `The dataset could not be reached (${e.message}). Make sure the Brasa server is running, then reload.`;
  }
})();

/* ---------- 3D: globe + calendar relief ---------- */
let globe = null, relief = null, gTimer = null;
const tipAt = (html, x, y) => { const t = $('#tip'); t.innerHTML = html; t.hidden = false; const tw = t.offsetWidth, th = t.offsetHeight; t.style.left = Math.max(8, Math.min(innerWidth - tw - 8, x - tw / 2)) + 'px'; t.style.top = (y - th - 14 < 8 ? y + 18 : y - th - 14) + 'px'; };
function globeStep() {
  if (!globe) return;
  const r = globe.setStep(state.gmode, state.gstep, state.gmode === 'year' ? state.gcol : 'count');
  $('#gWhen').textContent = r.label;
  $('#gTotal').innerHTML = state.gmode === 'season'
    ? `<strong>${big(r.total)}</strong> hotspots in a typical ${r.label}, worldwide`
    : `<strong>${big(r.total)}</strong> hotspots worldwide in ${r.label}${+r.label < 2012 ? ', converted from MODIS' : ''}`;
  if (state.gmode === 'year') $('#gNote').textContent = state.gcol === 'anom'
    ? 'Each column is one 1° cell, about 110 km across. Height shows that year\'s hotspots; colour compares them with the cell\'s own median year, 2001–2025: red above usual, blue below.'
    : 'Each column is one 1° cell, about 110 km across. Its height and colour show the hotspots of the whole year. Years before 2012 are converted from MODIS, so every year is on the same scale.';
  const anom = state.gmode === 'year' && state.gcol === 'anom', gs = $('.g-scale');
  gs.classList.toggle('anom', anom);
  $$('span', gs).forEach((x, i) => x.textContent = anom ? ['Below usual', 'Above usual'][i] : ['Fewer', 'More hotspots'][i]);
  $('#gRange').value = state.gstep;
  $('#gRange').setAttribute('aria-valuetext', r.label);
}
function setGMode(m) {
  state.gmode = m;
  $('#globeColour').hidden = m !== 'year';
  $$('#globeMode button').forEach(b => b.setAttribute('aria-checked', String(b.dataset.gmode === m)));
  const r = $('#gRange');
  if (m === 'season') {
    r.max = 11; r.setAttribute('aria-label', 'Month'); state.gstep = 8;
    $('#gNote').textContent = 'Each column is one 1° cell, about 110 km across. Its height and colour show the average hotspots in that month, 2001–2025, in VIIRS terms.';
  } else {
    r.max = globe.years.length - 1; r.setAttribute('aria-label', 'Year'); state.gstep = globe.years.length - 1;
    $('#gNote').textContent = 'Each column is one 1° cell, about 110 km across. Its height and colour show the hotspots of the whole year. Years before 2012 are converted from MODIS, so every year is on the same scale.';
  }
  globeStep();
}
function stopPlay() { clearInterval(gTimer); gTimer = null; $('#gPlay').textContent = 'Play'; $('#gPlay').setAttribute('aria-pressed', 'false'); }
function renderRelief() {
  if (!relief || state.view !== 'relief' || !state.data) return;
  relief.render(state.data, { mode: state.mode, scale: state.scale, byK: state.byK });
}
async function init3d() {
  try {
    globe = await createGlobe($('#globe'), {
      onPick: (lat, lon) => {
        setTab('point'); $('#pLat').value = lat; $('#pLon').value = lon; $('#pRad').value = 100;
        setArea({ type: 'point', lat, lon, radius_km: 100 }, `100 km around ${lat.toFixed(1)}°, ${lon.toFixed(1)}°`);
        scrollToEl($('#record'));
      },
      onHover: h => {
        if (!h) { $('#tip').hidden = true; return; }
        tipAt(`<div><b>${fmt(h.value)}</b> hotspots in ${h.label}</div>${h.usual != null ? `<div>${h.usual > 0 ? `${h.value >= h.usual ? '+' : '−'}${Math.abs(Math.round(100 * (h.value / h.usual - 1)))}% against its usual year (${fmt(h.usual)})` : 'Usually no fire here'}</div>` : ''}<div class="m">1° cell at ${h.lat.toFixed(1)}°, ${h.lon.toFixed(1)}°. Click to load it.</div>`, h.x, h.y);
      },
    });
    if (state.area && state.data) globe.setArea(state.area, state.data.summary.bbox);
    globeStep();
  } catch (e) {
    $('#globe').innerHTML = `<p class="hint" style="padding:24px">The 3D globe could not start (${esc(e.message)}). Everything else on the page still works.</p>`;
  }
  $$('#globeMode button').forEach(b => b.onclick = () => { stopPlay(); if (globe) setGMode(b.dataset.gmode); });
  $('#gIn').onclick = () => globe && globe.zoom(.8); $('#gOut').onclick = () => globe && globe.zoom(1.25);
  $$('#globeColour button').forEach(b => b.onclick = () => { state.gcol = b.dataset.gcol; $$('#globeColour button').forEach(x => x.setAttribute('aria-checked', String(x === b))); globeStep(); });
  $('#gRange').oninput = e => { stopPlay(); state.gstep = +e.target.value; globeStep(); };
  $('#gPlay').onclick = () => {
    if (gTimer) return stopPlay();
    $('#gPlay').textContent = 'Pause'; $('#gPlay').setAttribute('aria-pressed', 'true');
    gTimer = setInterval(() => { state.gstep = (state.gstep + 1) % (+$('#gRange').max + 1); globeStep(); }, state.gmode === 'season' ? 900 : 1100);
  };
  $$('#viewSeg button').forEach(b => b.onclick = () => {
    state.view = b.dataset.view;
    $$('#viewSeg button').forEach(x => x.setAttribute('aria-checked', String(x === b)));
    $('#heatWrap').hidden = state.view !== 'grid'; $('#relief').hidden = state.view !== 'relief';
    if (state.view === 'relief' && !relief) {
      try {
        relief = createRelief($('#relief'), {
          onHover: (k, e) => {
            if (k == null) { $('#tip').hidden = true; return; }
            const m = state.byK[k]; renderReadout(m);
            tipAt(`<div><b>${fmt(m.hotspots)}</b> hotspots, ${MN[m.month - 1]} ${m.year}</div><div>${compareText(m)}</div><div class="m">${srcText(m)}</div>`, e.clientX, e.clientY);
          },
          onPick: k => selectYear(Math.floor(k / 12)),
        });
      } catch (e) {
        $('#relief').innerHTML = `<p class="hint" style="padding:24px">The relief view could not start (${esc(e.message)}). Use the grid view.</p>`;
      }
    }
    renderRelief();
  });
}

/* ---------- right now: fire-danger-style dial + outlook ---------- */
const BANDS = [
  { name: 'Very low', c: 'var(--fd-1)' }, { name: 'Low', c: 'var(--fd-2)' }, { name: 'Normal', c: 'var(--fd-3)' },
  { name: 'High', c: 'var(--fd-4)' }, { name: 'Very high', c: 'var(--fd-5)' },
];
function renderNow() {
  const d = state.data, l = d.summary.latest, el = $('#now');
  if (!l || !d.summary.total || l.of < 3) { el.innerHTML = ''; return; }
  // rank 1 = the highest of all same-name months; turn it into a 0..1 position (1 = highest)
  const pos = l.of > 1 ? 1 - (l.rank - 1) / (l.of - 1) : .5;
  const band = BANDS[Math.min(4, Math.floor(pos * 5))];
  const cx = 110, cy = 104, r1 = 58, r2 = 92;
  const arc = (a0, a1) => { const p = (r, a) => `${(cx + r * Math.cos(a)).toFixed(1)},${(cy - r * Math.sin(a)).toFixed(1)}`;
    return `M${p(r2, a0)}A${r2},${r2} 0 0 1 ${p(r2, a1)}L${p(r1, a1)}A${r1},${r1} 0 0 0 ${p(r1, a0)}Z`; };
  const seg = BANDS.map((b, i) => { const a0 = Math.PI * (1 - i / 5), a1 = Math.PI * (1 - (i + 1) / 5) + .001; return `<path d="${arc(a0, a1)}" style="fill:${b.c}" style="stroke:var(--panel)" stroke-width="3"/>`; }).join('');
  const deg = (Math.min(Math.max(pos, .02), .98) - .5) * 180;
  // outlook: the three months after the latest one, from 2001-2025 medians
  const clim = {}; d.climatology.forEach(c => clim[c.month] = c);
  const peak = Math.max(...d.climatology.map(c => c.median), 1);
  const nxt = [1, 2, 3].map(k => ((l.month - 1 + k) % 12) + 1);
  const word = m => { const v = clim[m] ? clim[m].median : 0, f = v / peak; return f >= .5 ? 'fire season' : f >= .2 ? 'building' : 'quiet'; };
  el.innerHTML = `<div class="dial"><svg viewBox="0 0 220 122" role="img" aria-label="${MONTH[l.month - 1]} ${l.year} rates ${band.name}: ${ordinal(l.rank)} highest of ${l.of} ${MONTH[l.month - 1]}s since 2000.">
      ${seg}<g class="needle" style="--deg:${deg.toFixed(1)}deg"><path d="M${cx - 5},${cy} L${cx},${cy - 88} L${cx + 5},${cy} Z" style="fill:var(--ink)"/></g><circle cx="${cx}" cy="${cy}" r="8" style="fill:var(--ink)"/><circle cx="${cx}" cy="${cy}" r="3" style="fill:var(--panel)"/>
    </svg><div class="dial-t"><span class="dial-b">${band.name}</span><span>${MONTH[l.month - 1]} ${l.year} ranks ${ordinal(l.rank)} of ${l.of}</span></div></div>
    <div class="outlook"><h3>Coming up</h3><ol>${nxt.map(m => `<li><span class="om">${MONTH[m - 1]}</span><span class="ov">${fmt(clim[m] ? clim[m].median : 0)}</span><span class="ow w-${word(m).replace(' ', '-')}"><i aria-hidden="true"></i>${word(m)}</span></li>`).join('')}</ol>
    <p class="hint">Usual hotspots for each month, 2001–2025 median. The dial ranks the latest month against the same month in every other year.</p></div>`;
}

/* ---------- sensor agreement ---------- */
function renderAgree() {
  const d = state.data, by = {};
  d.months.forEach(m => { const o = by[m.year] ||= { n: 0, modis: 0, eq: 0, viirs: 0, vn: 0 }; o.n++; o.modis += m.modis; o.eq += m.modis_eq; o.viirs += m.src === 'VN' ? m.viirs_noaa20 : m.viirs_snpp; if (m.src === 'VN') o.vn++; });
  const years = Object.keys(by).map(Number).filter(y => by[y].n === 12).sort((a, b) => a - b);
  if (years.length < 2) { $('#agreeChart').innerHTML = '<p class="hint">Choose a period of at least two full years to compare the sensors.</p>'; $('#agreeSide').innerHTML = ''; return; }
  const narrow = $('#agreeChart').clientWidth < 560, W = narrow ? 420 : 760, H = narrow ? 300 : 320, P = { l: 44, r: 10, t: 18, b: 30 }, id = 'vg' + (++uid);
  const max = niceMax(Math.max(...years.map(y => Math.max(by[y].eq, by[y].viirs)), 1));
  const x = y => P.l + (W - P.l - P.r) * (years.length === 1 ? .5 : (y - years[0]) / (years[years.length - 1] - years[0]));
  const yv = v => P.t + (H - P.t - P.b) * (1 - v / max);
  const line = (ys, f) => ys.map((y, i) => `${i ? 'L' : 'M'}${x(y).toFixed(1)},${yv(f(by[y])).toFixed(1)}`).join('');
  const vy = years.filter(y => y >= 2012);
  const ov0 = Math.max(2012, years[0]), ov1 = years[years.length - 1];
  const band = ov1 >= 2012 ? `<rect x="${x(ov0) - 6}" y="${P.t}" width="${x(ov1) - x(ov0) + 12}" height="${H - P.t - P.b}" class="ovl" rx="6"/><text x="${x(ov0) + 6}" y="${P.t + 16}" class="ovt">Both satellites flying</text>` : '';
  const step = years.length > 14 ? (narrow ? 10 : 5) : years.length > 7 ? (narrow ? 4 : 2) : 1;
  const ticks = years.filter(y => y % step === 0).map(y => `<text x="${x(y)}" y="${H - 8}" text-anchor="middle">${y}</text>`).join('');
  const va = vy.length > 1 ? `${line(vy, o => o.viirs)}L${x(vy[vy.length - 1])},${H - P.b}L${x(vy[0])},${H - P.b}Z` : '';
  $('#agreeChart').innerHTML = `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="Yearly hotspots: raw MODIS, MODIS converted to VIIRS terms, and observed VIIRS"><defs>${grad(id, .28)}</defs>${band}${yAxis(max, W, H, P)}${ticks}
    <g class="sweep"><path d="${line(years, o => o.modis)}" fill="none" class="raw"/>
    ${va ? `<path d="${va}" fill="url(#${id})"/>` : ''}
    <path d="${line(years, o => o.eq)}" fill="none" class="eq"/>
    ${vy.length ? `<path d="${line(vy, o => o.viirs)}" fill="none" class="obs"/>` : ''}</g>
    ${vy.map((y, i) => `<circle class="pop dot" style="--i:${i}" cx="${x(y)}" cy="${yv(by[y].viirs)}" r="3.6"/>`).join('')}</svg>`;
  crosshair($('#agreeChart'), W, H, P, years.map(x), k => { const y = years[k], o = by[y]; return `<div><b>${y}</b></div>${y >= 2012 ? `<div><i class="sw-l obs"></i>VIIRS observed <b>${fmt(o.viirs)}</b></div>` : ''}<div><i class="sw-l eq"></i>MODIS converted <b>${fmt(o.eq)}</b></div><div class="m"><i class="sw-l raw"></i>MODIS raw ${fmt(o.modis)}</div>`; });
  play($('#agreeChart'));
  // agreement over the overlap years in this selection
  const ov = vy.filter(y => by[y].viirs > 0 && y <= 2024);
  let stat = '<p>The selected period has no full years where both satellites flew.</p>';
  if (ov.length) {
    const diffs = ov.map(y => Math.abs(by[y].eq / by[y].viirs - 1)).sort((a, b) => a - b);
    const medd = diffs[Math.floor((diffs.length - 1) / 2)], worst = diffs[diffs.length - 1];
    const raw = ov.map(y => by[y].viirs / Math.max(by[y].modis, 1)).sort((a, b) => a - b)[Math.floor((ov.length - 1) / 2)];
    const agree = Math.max(0, Math.round(100 * (1 - medd))), R = 52, C = 2 * Math.PI * R;
    stat = `<div class="gauge" style="--to:${(C * (1 - agree / 100)).toFixed(1)}px;--c:${C.toFixed(1)}px"><svg viewBox="0 0 128 128" aria-hidden="true"><circle cx="64" cy="64" r="${R}" class="rt"/><circle cx="64" cy="64" r="${R}" class="rv"/></svg>
        <div class="gauge-t"><strong><span data-count="${agree}">${agree}</span>%</strong><span>typical yearly agreement</span></div></div>
      <p>Across ${ov.length} overlap years, converted MODIS lands within ${Math.round(medd * 100)}% of what VIIRS saw in a typical year, ${Math.round(worst * 100)}% at worst.</p>
      <p>Raw MODIS counts here are about <strong>${raw.toFixed(1)}×</strong> smaller than VIIRS. That is the gap the conversion closes.</p>
      <p class="hint">${state.meta.nn ? 'The U-Net was trained on 2012–2017 and tuned on 2018; from 2019 these are years it never saw.' : 'These overlap years were also used to fit the conversion. For a fair score on years it never saw, see the Method section below.'}</p>`;
  }
  $('#agreeSide').innerHTML = `<ul class="agree-key"><li><i class="k-v"></i>VIIRS, observed</li><li><i class="k-e"></i>MODIS, converted to VIIRS terms</li><li><i class="k-m"></i>MODIS, raw counts</li></ul>${stat}`;
  play($('#agreeSide'));
}

/* ---------- motion: smooth scroll, reveals, the stage, method steps ---------- */
// method: the step in the middle of the screen lights up its label
function watchSteps() {
  if (stepWatch) stepWatch.disconnect();
  const steps = $$('#howSteps > li'), panes = $$('#howStage .vz'), prog = $('#howStage .stage-prog i');
  // Show a diagram immediately, even before a step crosses the observer's centre band.
  [...steps, ...panes].forEach(x => {
    const first = x.dataset.step === '0';
    x.classList.toggle('on', first);
    x.classList.toggle('play', first);
  });
  if (prog && steps.length) prog.style.transform = `scaleY(${1 / steps.length})`;
  stepWatch = new IntersectionObserver(es => es.forEach(e => {
    if (!e.isIntersecting) return;
    const k = e.target.dataset.step;
    // stage panes replay on every visit; inline diagrams (single-column layout) stay drawn once played
    panes.forEach(x => { const on = x.dataset.step === k; x.classList.toggle('on', on); x.classList.toggle('play', on); });
    steps.forEach(x => { const on = x.dataset.step === k; x.classList.toggle('on', on); if (on) x.classList.add('play'); });
    if (prog) prog.style.transform = `scaleY(${(+k + 1) / steps.length})`;
  }), { rootMargin: '-45% 0px -45% 0px' });
  steps.forEach(x => stepWatch.observe(x));
}

function scrollToEl(el) {
  if (!el) return;
  if (lenis) lenis.scrollTo(el, { offset: -72, duration: 1.4 });
  else el.scrollIntoView({ behavior: reduceMotion.matches ? 'auto' : 'smooth' });
}
// intro: rings draw in, the name writes itself, then flies into the nav logo and the page opens
function runIntro() {
  const root = document.documentElement, ov = $('#intro');
  if (!root.classList.contains('intro-on') || !ov) { if (ov) ov.remove(); root.classList.remove('intro-on'); return Promise.resolve(); }
  return new Promise(done => {
    requestAnimationFrame(() => ov.classList.add('go'));
    setTimeout(() => {
      // FLIP the word onto the nav wordmark
      const word = $('.intro-word', ov), mark = $('.mark span'), to = mark.getBoundingClientRect(), from = word.getBoundingClientRect();
      const k = parseFloat(getComputedStyle(mark).fontSize) / parseFloat(getComputedStyle(word).fontSize);
      word.style.transform = `translate(${to.left - from.left}px, ${to.top + to.height / 2 - from.top - from.height * k / 2}px) scale(${k})`;
      ov.classList.add('out');
      setTimeout(done, 380);
      setTimeout(() => { root.classList.remove('intro-on'); ov.remove(); }, 1000);
    }, 1550);
  });
}
function initScroll() {
  const reduced = reduceMotion.matches;
  // inertial page scroll; panels that scroll themselves opt out with data-lenis-prevent
  if (!reduced && window.Lenis) {
    lenis = new Lenis({ lerp: .09, wheelMultiplier: .9, smoothWheel: true });
    const raf = t => { lenis.raf(t); requestAnimationFrame(raf); };
    requestAnimationFrame(raf);
  }
  document.addEventListener('click', e => {
    const a = e.target.closest('a[href^="#"]'); if (!a) return;
    const el = $(a.getAttribute('href')); if (!el) return;
    e.preventDefault(); scrollToEl(el);
  });

  // headline: words rise out of a blur, one after another
  $$('.split').forEach(h => {
    let i = 0;
    $$(':scope > span', h).forEach(part => {
      part.innerHTML = part.textContent.trim().split(/\s+/).map(w => `<span class="w" style="--d:${i++ * 55}ms">${esc(w)}</span>`).join(' ');
    });
  });
  runIntro().then(() => document.documentElement.classList.add('loaded'));

  // sections fade up from a blur as they enter, children staggered
  const io = new IntersectionObserver(es => es.forEach(e => {
    if (!e.isIntersecting) return;
    e.target.classList.add('in'); io.unobserve(e.target);
  }), { rootMargin: '0px 0px -12% 0px', threshold: .01 });
  $$('[data-reveal]').forEach(el => io.observe(el));

  // the stage tilts up and settles as it scrolls into place
  const stage = $('#stage'), wrap = $('.stage-wrap');
  let queued = false;
  const tick = () => {
    queued = false;
    const r = wrap.getBoundingClientRect(), vh = innerHeight;
    const p = Math.min(1, Math.max(0, (vh - r.top) / (vh * .75)));
    stage.style.setProperty('--p', p.toFixed(3));
  };
  if (!reduced) { addEventListener('scroll', () => { if (!queued) { queued = true; requestAnimationFrame(tick); } }, { passive: true }); tick(); }

  watchSteps();

  // nav follows the section in view
  const links = $$('.nav-links a');
  const secs = links.map(a => $(a.getAttribute('href'))).filter(Boolean);
  const nio = new IntersectionObserver(es => {
    es.forEach(e => { if (e.isIntersecting) links.forEach(a => a.setAttribute('aria-current', String(a.getAttribute('href') === '#' + e.target.id))); });
  }, { rootMargin: '-45% 0px -50% 0px' });
  secs.forEach(s => nio.observe(s));
}
