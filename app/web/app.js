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
  VS: 'Observed by VIIRS on Suomi NPP',
  VN: 'Observed by VIIRS on NOAA-20, standing in during a Suomi NPP outage',
};
const isHarm = s => s === 'MT' || s === 'MTA';

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
  const ky = iEra > 0 ? ` The heavy ring marks January 2012, when VIIRS takes over; hatched rings inside it are estimated from MODIS.` : (harmYears ? ' Hatched rings are estimated from MODIS.' : '');
  $('#ringKey').textContent = `${years[0]} sits at the centre and ${years[N - 1]} at the edge; each spoke is a month.${ky} Hover a cell to read it, click to trace that year below.`;
}
function renderReadout(m) {
  if (!m) { const l = state.data.summary.latest; m = l ? state.byK[l.year * 12 + l.month - 1] : null; }
  if (!m) { $('#readout').innerHTML = ''; return; }
  $('#readout').innerHTML = `<div class="when">${MONTH[m.month - 1]} ${m.year}</div>
    <div class="n">${fmt(m.hotspots)}</div><div class="unit">hotspots, in VIIRS terms</div>
    <p>${compareText(m)}</p><p class="src">${srcText(m)}.</p>`;
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
function renderLegend(sc, anom) {
  const bar = Array.from({ length: 9 }, (_, i) => `<i style="background:var(--${anom ? 'div' : 'seq'}-${i})"></i>`).join('');
  let t;
  if (anom) t = ['a quarter of usual', 'usual', '4× usual'];
  else { const lg = state.scale === 'log', e = f => lg ? Math.expm1(Math.log1p(sc.max) * f) : sc.p99 * f; t = ['1', fmt(e(.5)), (lg ? '' : 'over ') + fmt(lg ? sc.max : sc.p99)]; }
  $('#legend').innerHTML = `<div class="lg-bar" aria-hidden="true">${bar}</div><div class="lg-ticks"><span>${t[0]}</span><span>${t[1]}</span><span>${t[2]}</span></div><p class="hint" style="margin-top:4px">${anom ? 'Each month against the median of the same month, 2001–2025.' : `Hotspots per month${state.scale === 'log' ? ', log scale' : ''}. Grey means none detected.`}</p>`;
}

/* ---------- charts ---------- */
const W = 520, H = 280, P = { l: 46, r: 8, t: 10, b: 28 };
function niceMax(v) { if (v <= 0) return 1; const p = 10 ** Math.floor(Math.log10(v)); const f = v / p; return (f <= 1 ? 1 : f <= 2 ? 2 : f <= 5 ? 5 : 10) * p; }
const kfmt = v => v >= 1e6 ? +(v / 1e6).toFixed(1) + 'M' : v >= 1e3 ? +(v / 1e3).toFixed(1) + 'k' : String(Math.round(v));
function yAxis(max) {
  let s = '';
  for (let i = 0; i <= 4; i++) { const v = max * i / 4, y = P.t + (H - P.t - P.b) * (1 - i / 4); s += `<line x1="${P.l}" x2="${W - P.r}" y1="${y}" y2="${y}" style="stroke:var(--rule)" stroke-width="${i ? .8 : 1.4}"/><text x="${P.l - 8}" y="${y + 4}" text-anchor="end">${kfmt(v)}</text>`; }
  return s;
}
function renderClim() {
  const d = state.data, c = d.climatology;
  const ym = {}; d.months.filter(m => m.year === state.year).forEach(m => ym[m.month] = m.hotspots);
  const max = niceMax(Math.max(...c.map(x => x.p90), ...Object.values(ym), 1));
  const x = i => P.l + (W - P.l - P.r) * (i + .5) / 12, y = v => P.t + (H - P.t - P.b) * (1 - v / max);
  const line = f => c.map((r, i) => `${i ? 'L' : 'M'}${x(i)},${y(f(r))}`).join('');
  const band = c.map((r, i) => `${i ? 'L' : 'M'}${x(i)},${y(r.p90)}`).join('') + [...c].reverse().map((r, i) => `L${x(11 - i)},${y(r.p10)}`).join('') + 'Z';
  const yr = Object.keys(ym).map(Number).sort((a, b) => a - b);
  const yl = yr.map((m, i) => `${i ? 'L' : 'M'}${x(m - 1)},${y(ym[m])}`).join('');
  $('#clim').innerHTML = `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="Typical year: median month and the range holding 8 years in 10, with ${state.year} traced">${yAxis(max)}
    ${MN.map((m, i) => `<text x="${x(i)}" y="${H - 8}" text-anchor="middle">${m}</text>`).join('')}
    <path d="${band}" style="fill:var(--rule)" opacity=".55"/><path d="${line(r => r.median)}" fill="none" style="stroke:var(--ink)" stroke-width="2"/>
    ${yl ? `<path d="${yl}" fill="none" style="stroke:var(--scar)" stroke-width="2.6"/>${yr.map(m => `<circle cx="${x(m - 1)}" cy="${y(ym[m])}" r="3.6" style="fill:var(--scar)"><title>${MONTH[m - 1]} ${state.year}: ${fmt(ym[m])}</title></circle>`).join('')}` : ''}</svg>`;
  $('#yrKey').textContent = state.year ? `The orange line is ${state.year}.` : '';
}
function renderAnnual() {
  const d = state.data, ys = d.yearly; if (!ys.length) { $('#annual').innerHTML = ''; return; }
  const cnt = {}; d.months.forEach(m => { const o = cnt[m.year] ||= { h: 0, n: 0 }; o.n++; if (isHarm(m.src)) o.h++; });
  const max = niceMax(Math.max(...ys.map(r => r.hotspots), 1)), bw = (W - P.l - P.r) / ys.length, worst = d.summary.worst_year;
  const bars = ys.map((r, i) => {
    const est = cnt[r.year] && cnt[r.year].h > cnt[r.year].n / 2, partial = cnt[r.year] && cnt[r.year].n < 12;
    const hh = Math.max((H - P.t - P.b) * r.hotspots / max, 0), xx = P.l + i * bw + bw * .14, w = bw * .72, yy = H - P.b - hh;
    const fill = r.year === worst ? 'var(--seq-7)' : r.year === state.year ? 'var(--scar)' : 'var(--seq-4)';
    return `<g><title>${r.year}: ${fmt(r.hotspots)} hotspots${est ? ', estimated from MODIS' : ''}${partial ? `, ${cnt[r.year].n} months only` : ''}</title><rect x="${xx}" y="${yy}" width="${w}" height="${hh}" style="fill:${fill}${partial ? ';opacity:.45' : ''}"/>${est ? `<rect x="${xx}" y="${yy}" width="${w}" height="${hh}" fill="url(#bh)"/>` : ''}${r.year === worst ? `<text x="${xx + w / 2}" y="${yy - 6}" text-anchor="middle" style="fill:var(--ink);font-weight:700">${r.year}</text>` : ''}</g>`;
  }).join('');
  const step = ys.length > 14 ? 5 : ys.length > 7 ? 2 : 1;
  const ticks = ys.map((r, i) => r.year % step === 0 ? `<text x="${P.l + (i + .5) * bw}" y="${H - 8}" text-anchor="middle">${r.year}</text>` : '').join('');
  $('#annual').innerHTML = `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="Hotspots per year"><defs><pattern id="bh" width="5" height="5" patternUnits="userSpaceOnUse" patternTransform="rotate(-45)"><rect width="1.3" height="5" style="fill:var(--hatch)"/></pattern></defs>${yAxis(max)}${bars}${ticks}</svg>`;
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
const how = $('#how');
$('#helpBtn').onclick = () => how.showModal();
$('#howClose').onclick = () => how.close();
how.addEventListener('click', e => { if (e.target === how) how.close(); });
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
  if (typeof L !== 'undefined') initMap(); else $('#map').innerHTML = '<p class="hint" style="padding:16px">The map could not load. Use Coordinates, Radius or Country instead.</p>';
  try {
    const [meta, countries] = await Promise.all([api('/api/meta').then(r => r.json()), api('/api/countries').then(r => r.json())]);
    state.meta = meta; state.countries = countries;
    state.start = meta.first_month; state.end = meta.last_month;
    for (const id of ['#mFrom', '#mTo']) { $(id).min = meta.first_month; $(id).max = meta.last_month; }
    $('#mFrom').value = state.start; $('#mTo').value = state.end;
    $('#countryList').innerHTML = countries.map(c => `<option value="${esc(c)}">`).join('');
    renderValidation();
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
        $('#record').scrollIntoView({ behavior: matchMedia('(prefers-reduced-motion: reduce)').matches ? 'auto' : 'smooth' });
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
  { name: 'Very low', c: 'var(--div-1)' }, { name: 'Low', c: 'var(--div-3)' }, { name: 'Normal', c: 'var(--rule)' },
  { name: 'High', c: 'var(--div-6)' }, { name: 'Very high', c: 'var(--div-8)' },
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
  const seg = BANDS.map((b, i) => { const a0 = Math.PI * (1 - i / 5), a1 = Math.PI * (1 - (i + 1) / 5) + .001; return `<path d="${arc(a0, a1)}" style="fill:${b.c}" stroke="var(--paper)" stroke-width="2"/>`; }).join('');
  const na = Math.PI * (1 - Math.min(Math.max(pos, .02), .98)), nx = cx + 86 * Math.cos(na), ny = cy - 86 * Math.sin(na);
  // outlook: the three months after the latest one, from 2001-2025 medians
  const clim = {}; d.climatology.forEach(c => clim[c.month] = c);
  const peak = Math.max(...d.climatology.map(c => c.median), 1);
  const nxt = [1, 2, 3].map(k => ((l.month - 1 + k) % 12) + 1);
  const word = m => { const v = clim[m] ? clim[m].median : 0, f = v / peak; return f >= .5 ? 'fire season' : f >= .2 ? 'building' : 'quiet'; };
  el.innerHTML = `<div class="dial"><svg viewBox="0 0 220 122" role="img" aria-label="${MONTH[l.month - 1]} ${l.year} rates ${band.name}: ${ordinal(l.rank)} highest of ${l.of} ${MONTH[l.month - 1]}s since 2000.">
      ${seg}<line x1="${cx}" y1="${cy}" x2="${nx.toFixed(1)}" y2="${ny.toFixed(1)}" style="stroke:var(--ink)" stroke-width="3.5" stroke-linecap="round"/><circle cx="${cx}" cy="${cy}" r="7" style="fill:var(--ink)"/>
    </svg><div class="dial-t"><span class="dial-b">${band.name}</span><span>${MONTH[l.month - 1]} ${l.year} ranks ${ordinal(l.rank)} of ${l.of}</span></div></div>
    <div class="outlook"><h3>Coming up</h3><ol>${nxt.map(m => `<li><span class="om">${MONTH[m - 1]}</span><span class="ov">${fmt(clim[m] ? clim[m].median : 0)}</span><span class="ow w-${word(m).replace(' ', '-')}">${word(m)}</span></li>`).join('')}</ol>
    <p class="hint">Usual hotspots for each month, 2001–2025 median. The dial ranks the latest month against the same month in every other year.</p></div>`;
}

/* ---------- sensor agreement ---------- */
function renderAgree() {
  const d = state.data, by = {};
  d.months.forEach(m => { const o = by[m.year] ||= { n: 0, modis: 0, eq: 0, viirs: 0, vn: 0 }; o.n++; o.modis += m.modis; o.eq += m.modis_eq; o.viirs += m.src === 'VN' ? m.viirs_noaa20 : m.viirs_snpp; if (m.src === 'VN') o.vn++; });
  const years = Object.keys(by).map(Number).filter(y => by[y].n === 12).sort((a, b) => a - b);
  if (years.length < 2) { $('#agreeChart').innerHTML = '<p class="hint">Choose a period of at least two full years to compare the sensors.</p>'; $('#agreeSide').innerHTML = ''; return; }
  const W = 760, H = 300, P = { l: 52, r: 16, t: 14, b: 30 };
  const max = niceMax(Math.max(...years.map(y => Math.max(by[y].eq, by[y].viirs)), 1));
  const x = y => P.l + (W - P.l - P.r) * (years.length === 1 ? .5 : (y - years[0]) / (years[years.length - 1] - years[0]));
  const yv = v => P.t + (H - P.t - P.b) * (1 - v / max);
  const line = (ys, f) => ys.map((y, i) => `${i ? 'L' : 'M'}${x(y).toFixed(1)},${yv(f(by[y])).toFixed(1)}`).join('');
  const vy = years.filter(y => y >= 2012);
  const ov0 = Math.max(2012, years[0]), ov1 = years[years.length - 1];
  const band = ov1 >= 2012 ? `<rect x="${x(ov0) - 6}" y="${P.t}" width="${x(ov1) - x(ov0) + 12}" height="${H - P.t - P.b}" style="fill:var(--paper-2)"/><text x="${x(ov0) + 4}" y="${P.t + 16}" style="fill:var(--ink);font-weight:600">Both satellites flying</text>` : '';
  let grid = '';
  for (let i = 0; i <= 4; i++) { const v = max * i / 4, y = yv(v); grid += `<line x1="${P.l}" x2="${W - P.r}" y1="${y}" y2="${y}" style="stroke:var(--rule)" stroke-width="${i ? .8 : 1.4}"/><text x="${P.l - 8}" y="${y + 4}" text-anchor="end">${kfmt(v)}</text>`; }
  const step = years.length > 14 ? 5 : years.length > 7 ? 2 : 1;
  const ticks = years.filter(y => y % step === 0).map(y => `<text x="${x(y)}" y="${H - 8}" text-anchor="middle">${y}</text>`).join('');
  $('#agreeChart').innerHTML = `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="Yearly hotspots: raw MODIS, MODIS converted to VIIRS terms, and observed VIIRS">${band}${grid}${ticks}
    <path d="${line(years, o => o.modis)}" fill="none" style="stroke:var(--ink-2)" stroke-width="1.6" stroke-dasharray="2 4"/>
    <path d="${line(years, o => o.eq)}" fill="none" style="stroke:var(--ink)" stroke-width="2.4" stroke-dasharray="7 5"/>
    ${vy.length ? `<path d="${line(vy, o => o.viirs)}" fill="none" style="stroke:var(--scar)" stroke-width="3.2"/>${vy.map(y => `<circle cx="${x(y)}" cy="${yv(by[y].viirs)}" r="3.4" style="fill:var(--scar)"><title>${y}: VIIRS ${fmt(by[y].viirs)}, converted MODIS ${fmt(by[y].eq)}, raw MODIS ${fmt(by[y].modis)}</title></circle>`).join('')}` : ''}</svg>`;
  // agreement over the overlap years in this selection
  const ov = vy.filter(y => by[y].viirs > 0 && y <= 2024);
  let stat = '<p>The selected period has no full years where both satellites flew.</p>';
  if (ov.length) {
    const diffs = ov.map(y => Math.abs(by[y].eq / by[y].viirs - 1)).sort((a, b) => a - b);
    const medd = diffs[Math.floor((diffs.length - 1) / 2)], worst = diffs[diffs.length - 1];
    const ratio = years.filter(y => by[y].modis > 0).map(y => by[y].viirs / by[y].modis).filter(r => isFinite(r) && r > 0);
    const raw = ov.map(y => by[y].viirs / Math.max(by[y].modis, 1)).sort((a, b) => a - b)[Math.floor((ov.length - 1) / 2)];
    stat = `<p class="agree-big"><strong>${Math.round(medd * 100)}%</strong> typical yearly gap</p>
      <p>Across ${ov.length} overlap years, converted MODIS lands within ${Math.round(medd * 100)}% of what VIIRS saw in a typical year, ${Math.round(worst * 100)}% at worst.</p>
      <p>Raw MODIS counts here are about <strong>${raw.toFixed(1)}×</strong> smaller than VIIRS. That is the gap the conversion closes.</p>
      <p class="hint">These overlap years were also used to fit the conversion. For a fair score on years it never saw, open "How the numbers are made".</p>`;
  }
  $('#agreeSide').innerHTML = `<ul class="agree-key"><li><i class="k-v"></i>VIIRS, observed</li><li><i class="k-e"></i>MODIS, converted to VIIRS terms</li><li><i class="k-m"></i>MODIS, raw counts</li></ul>${stat}`;
}
