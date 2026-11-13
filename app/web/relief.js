// Calendar relief: the years x months calendar of one area as a landscape. Height = hotspots, colour = count or anomaly.
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { CSS2DRenderer, CSS2DObject } from 'three/addons/renderers/CSS2DRenderer.js';

const SEQ = ['#17092f','#3d0f6b','#6a176e','#932667','#bc3754','#dd513a','#f37819','#fca50a','#f9d83f'].map(c => new THREE.Color(c));
const DIV = ['#1b4f8f','#4a8bcb','#7ab0dc','#bcd7ee','#e9e6e1','#f8cfa0','#f2a15a','#d8662a','#a8330f'].map(c => new THREE.Color(c));
const MN = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'];
const css = n => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
const lerpRamp = (R, t, out) => { t = Math.min(1, Math.max(0, t)) * (R.length - 1); const i = Math.min(R.length - 2, Math.floor(t)); return out.copy(R[i]).lerp(R[i + 1], t - i); };

export function createRelief(el, { onHover, onPick } = {}) {
  const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
  renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
  el.appendChild(renderer.domElement);
  const labels = new CSS2DRenderer(); labels.domElement.className = 'relief-labels'; el.appendChild(labels.domElement);
  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(30, 1, .1, 200);
  const controls = new OrbitControls(camera, renderer.domElement);
  Object.assign(controls, { enableDamping: true, dampingFactor: .1, enablePan: false, minDistance: 14, maxDistance: 70, maxPolarAngle: Math.PI * .47 });
  scene.add(new THREE.AmbientLight(0xffffff, 1.25));
  const sun = new THREE.DirectionalLight(0xffffff, 1.6); sun.position.set(-8, 20, 14); scene.add(sun);
  const box = new THREE.BoxGeometry(.9, 1, .9); box.translate(0, .5, 0);
  const mat = new THREE.MeshLambertMaterial({ color: 0xffffff });
  let mesh = null, keys = [], group = new THREE.Group(); scene.add(group);
  let dirty = true, visible = true;

  function clear() {
    while (group.children.length) { const c = group.children.pop(); c.geometry && c.geometry.dispose(); if (c.element) c.element.remove(); }
    if (mesh) { scene.remove(mesh); mesh.dispose(); mesh = null; }
  }
  const label = (txt, cls, x, y, z) => { const d = document.createElement('div'); d.className = 'rl ' + cls; d.textContent = txt; const o = new CSS2DObject(d); o.position.set(x, y, z); group.add(o); };

  function render(data, { mode, scale, byK }) {
    clear();
    const years = [...new Set(data.months.map(m => m.year))].sort((a, b) => a - b);
    const N = years.length;
    const vals = data.months.map(m => m.hotspots).filter(v => v > 0).sort((a, b) => a - b);
    const vmax = vals.length ? vals[vals.length - 1] : 1, p99 = vals.length ? vals[Math.floor(vals.length * .99)] : 1;
    const H = 7;
    // height always on a power scale so peaks read as peaks; colour follows the chosen scale
    const height = v => v <= 0 ? .04 : .08 + H * Math.min(Math.pow(v / Math.max(p99, 1), .6), 1.15);
    keys = [];
    mesh = new THREE.InstancedMesh(box, mat, N * 12);
    const m4 = new THREE.Matrix4(), q = new THREE.Quaternion(), s = new THREE.Vector3(), p = new THREE.Vector3(), c = new THREE.Color();
    let i = 0;
    const x0 = -(N - 1) / 2, z0 = -5.5;  // years run left to right, months recede (Jan at the back)
    years.forEach((y, iy) => {
      for (let mo = 0; mo < 12; mo++) {
        const m = byK[y * 12 + mo];
        const h = m ? height(m.hotspots) : .02;
        p.set(x0 + iy, 0, z0 + mo); s.set(1, h, 1);
        m4.compose(p, q, s); mesh.setMatrixAt(i, m4);
        if (!m) c.set(css('--nodata'));
        else if (mode === 'anom') lerpRamp(DIV, m.median === 0 ? (m.hotspots ? 1 : .5) : (m.z_log / Math.log(4)) / 2 + .5, c);
        else lerpRamp(SEQ, m.hotspots <= 0 ? 0 : scale === 'log' ? Math.log1p(m.hotspots) / Math.log1p(vmax) : Math.min(m.hotspots / p99, 1), c);
        mesh.setColorAt(i, c); keys.push(m ? y * 12 + mo : null); i++;
      }
    });
    scene.add(mesh);
    // ground plate, era line and labels
    const plate = new THREE.Mesh(new THREE.BoxGeometry(N + .6, .06, 12.6), new THREE.MeshLambertMaterial({ color: new THREE.Color(css('--paper-2')) }));
    plate.position.set(0, -.04, 0); group.add(plate);
    const iEra = years.indexOf(2012);
    if (iEra > 0) {
      const ln = new THREE.Mesh(new THREE.BoxGeometry(.08, .08, 12.8), new THREE.MeshBasicMaterial({ color: new THREE.Color(css('--ink')) }));
      ln.position.set(x0 + iEra - .5, .02, 0); group.add(ln);
      label('VIIRS from 2012', 'era', x0 + iEra - .5, 0, z0 - 1.6);
    }
    MN.forEach((mn, mo) => label(mn, 'mon', x0 + N + .6, 0, z0 + mo));
    const step = N > 16 ? 5 : N > 8 ? 2 : 1;
    years.forEach((y, iy) => { if (y % step === 0 || iy === 0 || iy === N - 1) label(String(y), 'yr', x0 + iy, 0, z0 + 12.6); });
    const dist = Math.max(24, N * 1.25);
    camera.position.set(-dist * .18, dist * .62, dist * .82); controls.target.set(0, 1.4, 0); controls.update();
    dirty = true;
  }
  function setTheme() { dirty = true; }

  const ray = new THREE.Raycaster(), ndc = new THREE.Vector2();
  let down = null;
  const pick = e => { if (!mesh) return null; const r = renderer.domElement.getBoundingClientRect(); ndc.set(((e.clientX - r.left) / r.width) * 2 - 1, -((e.clientY - r.top) / r.height) * 2 + 1); ray.setFromCamera(ndc, camera); const h = ray.intersectObject(mesh)[0]; return h ? keys[h.instanceId] : null; };
  renderer.domElement.addEventListener('pointermove', e => { const k = pick(e); onHover && onHover(k, e); });
  renderer.domElement.addEventListener('pointerleave', () => onHover && onHover(null));
  renderer.domElement.addEventListener('pointerdown', e => down = [e.clientX, e.clientY]);
  renderer.domElement.addEventListener('pointerup', e => { if (down && Math.hypot(e.clientX - down[0], e.clientY - down[1]) < 5) { const k = pick(e); if (k != null && onPick) onPick(k); } });
  controls.addEventListener('change', () => dirty = true);
  new IntersectionObserver(es => { visible = es[0].isIntersecting; dirty = true; }).observe(el);
  new ResizeObserver(() => { const r = el.getBoundingClientRect(); renderer.setSize(r.width, r.height, false); labels.setSize(r.width, r.height); camera.aspect = r.width / Math.max(r.height, 1); camera.updateProjectionMatrix(); dirty = true; }).observe(el);
  (function frame() { requestAnimationFrame(frame); if (!visible) return; controls.update(); if (dirty) { renderer.render(scene, camera); labels.render(scene, camera); dirty = false; } })();
  return { render, setTheme };
}
