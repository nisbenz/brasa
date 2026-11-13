// Fire globe: every burning 1° cell is a column rising from the Earth, height and colour = harmonized hotspots.
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { feature, mesh } from 'https://cdn.jsdelivr.net/npm/topojson-client@3.1.0/+esm';

const SEQ = ['#17092f','#3d0f6b','#6a176e','#932667','#bc3754','#dd513a','#f37819','#fca50a','#f9d83f'].map(c => new THREE.Color(c));
const DIV = ['#1b4f8f','#4a8bcb','#7ab0dc','#bcd7ee','#e9e6e1','#f8cfa0','#f2a15a','#d8662a','#a8330f'].map(c => new THREE.Color(c));
const MONTH = ['January','February','March','April','May','June','July','August','September','October','November','December'];
const DEG = Math.PI / 180;
const css = n => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
const reduced = matchMedia('(prefers-reduced-motion: reduce)').matches;

function ramp(t, out) {
  t = Math.min(1, Math.max(0, t)) * 8;
  const i = Math.min(7, Math.floor(t));
  return out.copy(SEQ[i]).lerp(SEQ[i + 1], t - i);
}
// lat/lon (degrees) -> unit vector; three.js: y up, lon 0 facing +z
const vec = (lat, lon, r = 1) => new THREE.Vector3(
  r * Math.cos(lat * DEG) * Math.sin(lon * DEG), r * Math.sin(lat * DEG), r * Math.cos(lat * DEG) * Math.cos(lon * DEG));

export async function createGlobe(el, { onPick, onHover } = {}) {
  const buf = await (await fetch('/api/globe.bin')).arrayBuffer();
  const [n, ny, year0] = new Uint32Array(buf, 0, 3);
  let o = 12;
  const lat = new Int16Array(buf, o, n); o += 2 * n;
  const lon = new Int16Array(buf, o, n); o += 2 * n;
  const clim = new Float32Array(buf, o, n * 12); o += 4 * n * 12;
  const annual = new Float32Array(buf, o, n * ny);
  const years = Array.from({ length: ny }, (_, i) => year0 + i);

  // ---- scene ----
  const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
  renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
  el.appendChild(renderer.domElement);
  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(32, 1, .1, 100);
  camera.position.copy(vec(-5, -55, 3.6));
  const controls = new OrbitControls(camera, renderer.domElement);
  Object.assign(controls, { enableDamping: true, dampingFactor: .08, enablePan: false, minDistance: 1.55, maxDistance: 5, rotateSpeed: .55, zoomSpeed: .7 });
  controls.autoRotate = !reduced; controls.autoRotateSpeed = .35;
  scene.add(new THREE.AmbientLight(0xffffff, 1.6));
  const sun = new THREE.DirectionalLight(0xffffff, 1.4); scene.add(sun);

  const ocean = new THREE.Mesh(new THREE.SphereGeometry(1, 96, 64), new THREE.MeshBasicMaterial());
  scene.add(ocean);
  const grat = new THREE.Group(); scene.add(grat);
  const lineMat = new THREE.LineBasicMaterial({ transparent: true });
  const gratMat = new THREE.LineBasicMaterial({ transparent: true });
  for (let la = -60; la <= 60; la += 30) { const p = []; for (let lo = -180; lo <= 180; lo += 3) p.push(vec(la, lo, 1.0004)); grat.add(new THREE.Line(new THREE.BufferGeometry().setFromPoints(p), gratMat)); }
  for (let lo = -180; lo < 180; lo += 30) { const p = []; for (let la = -90; la <= 90; la += 3) p.push(vec(la, lo, 1.0004)); grat.add(new THREE.Line(new THREE.BufferGeometry().setFromPoints(p), gratMat)); }
  // coastlines (Natural Earth 1:110m via world-atlas)
  fetch('https://cdn.jsdelivr.net/npm/world-atlas@2.0.2/land-110m.json').then(r => r.json()).then(topo => {
    const m = mesh(topo, topo.objects.land), pos = [];
    for (const line of m.coordinates) for (let i = 1; i < line.length; i++) {
      const a = vec(line[i - 1][1], line[i - 1][0], 1.0008), b = vec(line[i][1], line[i][0], 1.0008);
      pos.push(a.x, a.y, a.z, b.x, b.y, b.z);
    }
    const g = new THREE.BufferGeometry(); g.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3));
    scene.add(new THREE.LineSegments(g, lineMat));
  }).catch(() => {});

  // ---- fire columns ----
  const box = new THREE.BoxGeometry(1, 1, 1); box.translate(0, 0, .5);
  const colMat = new THREE.MeshLambertMaterial({ color: 0xffffff });
  const cols = new THREE.InstancedMesh(box, colMat, n);
  cols.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
  scene.add(cols);
  const base = [], quat = [], w = new Float32Array(n);
  const up = new THREE.Vector3(0, 0, 1), m4 = new THREE.Matrix4(), s3 = new THREE.Vector3(), c3 = new THREE.Color();
  for (let i = 0; i < n; i++) {
    const la = lat[i] - 90 + .5, lo = lon[i] - 180 + .5, p = vec(la, lo, 1.0006);
    base.push(p); quat.push(new THREE.Quaternion().setFromUnitVectors(up, p.clone().normalize()));
    w[i] = Math.max(Math.cos(la * DEG), .05);
  }
  const cur = new Float32Array(n), target = new Float32Array(n);
  let vmax = 1, mode = 'season', step = 8, colour = 'count', values = new Float32Array(n);
  const zc = new Float32Array(n), med = new Float32Array(n);
  for (let i = 0; i < n; i++) { const a = Array.from(annual.subarray(i * ny, i * ny + ny)).sort((x, y) => x - y); med[i] = (a[(ny - 1) >> 1] + a[ny >> 1]) / 2; }
  const divRamp = (t, out) => { t = Math.min(1, Math.max(0, t)) * 8; const k = Math.min(7, Math.floor(t)); return out.copy(DIV[k]).lerp(DIV[k + 1], t - k); };
  const W = 1 * DEG * .84, HMAX = .19, FLOOR = .045;
  function setValues() {
    for (let i = 0; i < n; i++) values[i] = mode === 'season' ? clim[i * 12 + step] : annual[i * ny + step];
    // scale to a robust max across the whole mode so steps are comparable
    target.forEach((_, i) => target[i] = values[i]);
    for (let i = 0; i < n; i++) { const t = Math.sqrt(values[i] / vmax); target[i] = t >= FLOOR ? t : 0; }
    for (let i = 0; i < n; i++) zc[i] = Math.log((values[i] + 1) / (med[i] + 1)) / Math.log(4);
  }
  function robustMax(arr) {
    const s = Float32Array.from(arr).filter(v => v > 0).sort();
    return s.length ? s[Math.floor(s.length * .997)] : 1;
  }
  const vmaxSeason = robustMax(clim), vmaxYear = robustMax(annual);
  function apply(force) {
    let moving = false;
    for (let i = 0; i < n; i++) {
      const d = target[i] - cur[i];
      if (Math.abs(d) > .0005) { cur[i] += force || reduced ? d : d * .16; moving = true; } else cur[i] = target[i];
      const h = Math.min(cur[i], 1.15) * HMAX;
      if (h > 0) s3.set(W * w[i], W, h); else s3.set(0, 0, 0);
      m4.compose(base[i], quat[i], s3);
      cols.setMatrixAt(i, m4);
      cols.setColorAt(i, colour === 'anom' && mode === 'year' ? divRamp(zc[i] / 2 + .5, c3) : ramp(cur[i] > 0 ? .22 + .78 * Math.min(cur[i], 1) : 0, c3));
    }
    cols.instanceMatrix.needsUpdate = true; cols.instanceColor.needsUpdate = true;
    return moving;
  }

  // ---- selected area outline ----
  let areaObj = null;
  function setArea(a, bbox) {
    if (areaObj) { scene.remove(areaObj); areaObj.geometry.dispose(); areaObj = null; }
    if (!a) return;
    let ring = [];
    const dens = (pts) => { const out = []; for (let i = 0; i < pts.length - 1; i++) { const [x0, y0] = pts[i], [x1, y1] = pts[i + 1], k = Math.max(2, Math.ceil(Math.hypot(x1 - x0, y1 - y0) / 1.5)); for (let j = 0; j < k; j++) out.push([x0 + (x1 - x0) * j / k, y0 + (y1 - y0) * j / k]); } return out; };
    if (a.type === 'bbox' || (a.type === 'country' && bbox)) {
      const [W_, S, E, N] = a.type === 'bbox' ? [a.west, a.south, a.east, a.north] : bbox;
      ring = dens([[W_, S], [E, S], [E, N], [W_, N], [W_, S]]);
    } else if (a.type === 'polygon') ring = dens(a.coordinates);
    else if (a.type === 'point') {
      const r = a.radius_km / 111;
      for (let t = 0; t < 64; t++) { const th = t / 64 * 2 * Math.PI; ring.push([a.lon + r * Math.sin(th) / Math.max(Math.cos(a.lat * DEG), .05), a.lat + r * Math.cos(th)]); }
    }
    if (ring.length < 3) return;
    const pts = ring.map(([x, y]) => vec(y, x, 1.004));
    areaObj = new THREE.Mesh(new THREE.TubeGeometry(new THREE.CatmullRomCurve3(pts, true), Math.max(64, pts.length * 2), .006, 6, true),
      new THREE.MeshBasicMaterial({ color: new THREE.Color(css('--ink')), depthTest: false, transparent: true }));
    areaObj.renderOrder = 10;
    scene.add(areaObj);
    // swing the camera to face the area
    const c = new THREE.Vector3(); pts.forEach(p => c.add(p)); c.normalize();
    flyTo = c.multiplyScalar(camera.position.length());
    controls.autoRotate = false;
  }
  let flyTo = null;

  // ---- theme ----
  function setTheme() {
    ocean.material.color.set(css('--paper-2'));
    lineMat.color.set(css('--ink')); lineMat.opacity = .75;
    gratMat.color.set(css('--rule')); gratMat.opacity = .55;
    if (areaObj) areaObj.material.color.set(css('--ink'));
  }

  // ---- interaction ----
  const ray = new THREE.Raycaster(), ndc = new THREE.Vector2();
  let down = null;
  const toNdc = e => { const r = renderer.domElement.getBoundingClientRect(); ndc.set(((e.clientX - r.left) / r.width) * 2 - 1, -((e.clientY - r.top) / r.height) * 2 + 1); };
  renderer.domElement.addEventListener('pointerdown', e => { down = [e.clientX, e.clientY]; controls.autoRotate = false; });
  renderer.domElement.addEventListener('pointerup', e => {
    if (!down || Math.hypot(e.clientX - down[0], e.clientY - down[1]) > 5) return;
    toNdc(e); ray.setFromCamera(ndc, camera);
    const hit = ray.intersectObject(ocean)[0]; if (!hit) return;
    const p = hit.point.clone().normalize();
    const la = Math.asin(p.y) / DEG, lo = Math.atan2(p.x, p.z) / DEG;
    onPick && onPick(+la.toFixed(2), +lo.toFixed(2));
  });
  renderer.domElement.addEventListener('pointermove', e => {
    if (!onHover) return;
    toNdc(e); ray.setFromCamera(ndc, camera);
    const hit = ray.intersectObject(cols)[0];
    if (hit && target[hit.instanceId] > 0) {
      const i = hit.instanceId;
      onHover({ x: e.clientX, y: e.clientY, lat: lat[i] - 90 + .5, lon: lon[i] - 180 + .5, value: values[i], usual: mode === 'year' ? med[i] : null, label: mode === 'season' ? `a typical ${MONTH[step]}` : String(years[step]) });
    } else onHover(null);
  });
  renderer.domElement.addEventListener('pointerleave', () => onHover && onHover(null));

  // ---- loop ----
  let visible = true, dirty = true;
  new IntersectionObserver(es => { visible = es[0].isIntersecting; }).observe(el);
  const ro = new ResizeObserver(() => { const r = el.getBoundingClientRect(); renderer.setSize(r.width, r.height, false); camera.aspect = r.width / Math.max(r.height, 1); camera.updateProjectionMatrix(); dirty = true; });
  ro.observe(el);
  controls.addEventListener('change', () => dirty = true);
  function frame() {
    requestAnimationFrame(frame);
    if (!visible) return;
    if (flyTo) { camera.position.lerp(flyTo, reduced ? 1 : .07); camera.position.setLength(flyTo.length()); if (camera.position.distanceTo(flyTo) < .01) flyTo = null; dirty = true; }
    controls.update();
    if (apply(false)) dirty = true;
    sun.position.copy(camera.position).add(new THREE.Vector3(1.5, 2, 0));
    if (dirty || controls.autoRotate) { renderer.render(scene, camera); dirty = false; }
  }

  function setStep(m, s, c = 'count') {
    mode = m; step = s; colour = c; vmax = mode === 'season' ? vmaxSeason : vmaxYear;
    setValues(); apply(true); dirty = true;
    let tot = 0; for (let i = 0; i < n; i++) tot += values[i];
    return { label: mode === 'season' ? MONTH[step] : String(years[step]), total: tot };
  }
  setTheme(); setStep('season', 8); apply(true); frame();
  return { setArea, setTheme, setStep, years, months: MONTH };
}
