/* SOLAROM — lógica de la app. El modelo es el Isolation Forest REAL entrenado en scikit-learn
   (pipeline/entrenar.py) exportado árbol por árbol a modelo_if.js; aquí solo se recorre. */
(function () {
  const M = window.MODELO_IF, D = window.DATOS;
  const NOM = 800, HORAS = D.horas;
  const $ = (s) => document.querySelector(s);
  const fmt = (x, d = 0) => (x == null || isNaN(x)) ? "—" : x.toLocaleString("es-CO", { minimumFractionDigits: d, maximumFractionDigits: d });
  const median = (a) => { const b = a.filter((v) => v != null && !isNaN(v)).sort((x, y) => x - y); if (!b.length) return NaN; const m = b.length >> 1; return b.length % 2 ? b[m] : (b[m - 1] + b[m]) / 2; };
  const mean = (a) => { const b = a.filter((v) => v != null && !isNaN(v)); return b.length ? b.reduce((s, v) => s + v, 0) / b.length : NaN; };

  // ---------------- Isolation Forest (idéntico a sklearn.score_samples, con comparación en float32)
  const cN = (n) => n <= 1 ? 0 : n === 2 ? 1 : 2 * (Math.log(n - 1) + 0.5772156649015329) - 2 * (n - 1) / n;
  function scoreForest(forest, maxS, x) {
    let tot = 0;
    for (const t of forest) {
      let node = 0, depth = 0;
      while (t.l[node] !== -1) { node = Math.fround(x[t.f[node]]) <= t.t[node] ? t.l[node] : t.r[node]; depth++; }
      tot += depth + cN(t.n[node]);
    }
    return Math.pow(2, -tot / (forest.length * cN(maxS)));
  }
  function scoreIF(x) {
    const s = M.bosques.map((f, i) => scoreForest(f, M.max_samples[i], x));
    const m = mean(s); const sd = Math.sqrt(mean(s.map((v) => (v - m) ** 2)));
    return { m, sd, todas: s };
  }

  // ---------------- variables diarias (espejo de solarom_ml.variables_diarias)
  function variables(dia) { // dia: {INV-01:{p:[],t:[],ghi:[]},...}
    const invs = Object.keys(dia).sort();
    const nh = HORAS.length;
    const medP = [], medT = [];
    for (let h = 0; h < nh; h++) { medP.push(median(invs.map((i) => dia[i].p[h]))); medT.push(median(invs.map((i) => dia[i].t[h]))); }
    const out = {};
    for (const i of invs) {
      const p = dia[i].p, t = dia[i].t;
      const fracNan = p.filter((v) => v == null || isNaN(v)).length / nh;
      const E = p.reduce((s, v) => s + (isNaN(v) || v == null ? 0 : v), 0);
      const Em = medP.reduce((s, v) => s + (isNaN(v) ? 0 : v), 0);
      const idxP = HORAS.map((h, k) => (h >= 11 && h <= 14 ? k : -1)).filter((k) => k >= 0);
      const idxT = HORAS.map((h, k) => (h >= 10 && h <= 15 ? k : -1)).filter((k) => k >= 0);
      const pico = mean(idxP.map((k) => p[k])), picoM = mean(idxP.map((k) => medP[k]));
      let cero = 0; for (let k = 0; k < nh; k++) if (p[k] < 0.02 * NOM && medP[k] > 0.10 * NOM) cero++;
      const dT = mean(idxT.map((k) => t[k] - medT[k]));
      let nPl = 0, nTot = 0;
      for (let k = 1; k < nh; k++) { if (Math.abs(medP[k] - medP[k - 1]) > 5) { nTot++; if (Math.abs(p[k] - p[k - 1]) < 0.05) nPl++; } }
      const rE = (E / Math.max(Em, 1)) / M.ref.r_energia_bruto[i];
      const rP = (pico / Math.max(picoM, 1)) / M.ref.r_pico_bruto[i];
      out[i] = { r_energia: rE, r_pico: rP, horas_cero: cero, dT_inv: dT, frac_plano: nTot ? nPl / nTot : 0,
        energia_kwh: E, energia_med: Em, excluido: fracNan > 0.30,
        perdida_est: Math.max(0, (1 - rE) * Em * M.ref.r_energia_bruto[i]) };
    }
    return { vars: out, medP };
  }

  // ---------------- limpieza de un CSV subido (espejo de solarom_ml.limpiar)
  function limpiarCSV(texto) {
    const filas = texto.trim().split(/\r?\n/); const cab = filas.shift().split(",").map((s) => s.trim());
    const ix = (n) => cab.indexOf(n);
    for (const c of ["fecha_hora", "inversor", "ghi_wm2", "p_ac_kw", "t_inv_c"]) if (ix(c) < 0) throw new Error(`Falta la columna "${c}". Usa el formato de la plantilla.`);
    const log = { registros: 0, fuera_de_rango: 0, interpolados: 0, excluidos: 0, rellenados_mediana: 0 };
    const dias = {};
    for (const f of filas) {
      const c = f.split(","); if (c.length < cab.length) continue;
      const fh = c[ix("fecha_hora")].replace("T", " "); const dia = fh.slice(0, 10); const hora = parseInt(fh.slice(11, 13));
      const k = HORAS.indexOf(hora); if (k < 0) continue;
      const inv = c[ix("inversor")].trim(); log.registros++;
      let p = parseFloat(c[ix("p_ac_kw")]); if (!(p >= 0 && p <= 1.1 * NOM)) { if (!isNaN(p)) log.fuera_de_rango++; p = NaN; }
      dias[dia] ??= {}; dias[dia][inv] ??= { p: Array(HORAS.length).fill(NaN), t: Array(HORAS.length).fill(NaN), ghi: Array(HORAS.length).fill(NaN) };
      dias[dia][inv].p[k] = p; dias[dia][inv].t[k] = parseFloat(c[ix("t_inv_c")]); dias[dia][inv].ghi[k] = parseFloat(c[ix("ghi_wm2")]);
    }
    for (const dia of Object.values(dias)) {
      for (const s of Object.values(dia)) { // interpolar huecos internos de hasta 2 h
        const p = s.p; let k = 0;
        while (k < p.length) {
          if (isNaN(p[k])) { let j = k; while (j < p.length && isNaN(p[j])) j++;
            if (k > 0 && j < p.length && j - k <= 2) { for (let q = k; q < j; q++) p[q] = p[k - 1] + (p[j] - p[k - 1]) * (q - k + 1) / (j - k + 1); log.interpolados += j - k; }
            k = j; } else k++;
        }
        s.excl = p.filter(isNaN).length / p.length > 0.30; if (s.excl) log.excluidos++;
      }
      for (let h = 0; h < HORAS.length; h++) {
        const m = median(Object.values(dia).map((s) => s.p[h]));
        for (const s of Object.values(dia)) if (!s.excl && isNaN(s.p[h])) { s.p[h] = m; log.rellenados_mediana++; }
      }
    }
    return { dias, log };
  }

  // ---------------- inyección de fallas (simulador del escenario)
  function inyectar(dia, inv, tipo, sev) {
    const c = JSON.parse(JSON.stringify(dia)); const s = c[inv]; if (!s) return c;
    if (tipo === "string_abierto" || tipo === "suciedad_salina") s.p = s.p.map((v) => v * (1 - sev));
    if (tipo === "derating_termico") { const lim = NOM * (1 - sev); s.p = s.p.map((v) => Math.min(v, lim)); s.t = s.t.map((v) => v + 11); }
    if (tipo === "disparos") { const n = Math.max(1, Math.round(sev * 10)); for (let k = 4; k < 4 + n && k < s.p.length; k++) s.p[k] = 0; }
    if (tipo === "sensor_congelado") s.p = s.p.map(() => s.p[3]);
    return c;
  }

  // ---------------- explicación (z robusto vs entrenamiento) y pistas
  const NOMBRES = { r_energia: "Energía relativa a la flota", r_pico: "Potencia de mediodía relativa", horas_cero: "Horas en cero con sol", dT_inv: "Exceso de temperatura del inversor (°C)", frac_plano: "Fracción de lecturas planas" };
  const z = (k, v) => (v - M.stats[k].med) / M.stats[k].mad;
  function pistas(v) {
    const p = [];
    if (v.frac_plano > 0.5) p.push(["datos", "Lectura plana o congelada: revisar datalogger y comunicación. La energía real puede ser normal; el SCADA está ciego en este equipo."]);
    if (v.horas_cero >= 1) p.push(["parada", `${v.horas_cero} h en cero con sol en la planta: revisar registro de alarmas, protecciones AC y reconexión del inversor.`]);
    if (v.dT_inv > 5) p.push(["termico", `Inversor ${fmt(v.dT_inv, 1)} °C más caliente que la flota: revisar ventiladores, filtros y ventilación del shelter (posible derating).`]);
    if (v.r_energia < 0.95 && v.frac_plano <= 0.5 && v.horas_cero === 0) p.push(["dc", `Produce ${fmt((1 - v.r_energia) * 100, 1)} % menos que su referencia: revisar strings y fusibles DC (termografía) o suciedad/sal en módulos (inspección visual).`]);
    if (!p.length) p.push(["otro", "Combinación poco común de variables sin patrón conocido: inspección general."]);
    return p;
  }

  // ---------------- estado
  const st = { fuente: "prueba", dia: D.dias.includes("2025-12-16") ? "2025-12-16" : D.dias[0], subidos: null, diaSub: null,
    iny: { inv: "", tipo: "string_abierto", sev: 0.15 }, verdad: false, invSel: null,
    precio: D.supuestos.precio, costo: D.supuestos.costo, k: D.supuestos.k };
  let contextoActual = null;
  let historialChat = [];

  function diaActual() {
    let dia, nombre;
    if (st.fuente === "csv" && st.subidos) { dia = {}; const src = st.subidos.dias[st.diaSub]; for (const i in src) dia[i] = { p: src[i].p, t: src[i].t, ghi: src[i].ghi }; nombre = st.diaSub + " (archivo)"; }
    else { dia = Object.fromEntries(D.inversores.map((i) => [i, { ...D.hora[i][st.dia] }])); nombre = st.dia; }
    if (st.iny.inv) dia = inyectar(dia, st.iny.inv, st.iny.tipo, st.iny.sev);
    return { dia, nombre };
  }

  function evaluar() {
    const { dia, nombre } = diaActual();
    const { vars, medP } = variables(dia);
    const filas = Object.entries(vars).map(([inv, v]) => {
      if (v.excluido) return { inv, v, excl: true };
      const x = M.variables.map((k) => v[k]);
      const s = scoreIF(x);
      const banderaIF = s.m > M.umbral, banderaPR = (1 - v.r_energia) > M.umbral_regla_pr;
      const estab = s.todas.filter((q) => q > M.umbral).length / s.todas.length;
      return { inv, v, s, banderaIF, banderaPR, alerta: banderaIF || banderaPR, estab };
    });
    return { filas, medP, dia, nombre };
  }

  // ---------------- gráficos SVG
  function barras(filas) {
    const ok = filas.filter((f) => !f.excl); const W = 640, fila = 26, H = ok.length * fila + 34, x0 = 70, x1 = W - 20;
    const vmax = Math.max(0.8, ...ok.map((f) => f.s.m + 2 * f.s.sd)), vmin = 0.3;
    const X = (v) => x0 + (x1 - x0) * (Math.min(Math.max(v, vmin), vmax) - vmin) / (vmax - vmin);
    let g = `<svg viewBox="0 0 ${W} ${H}" class="chart" role="img" aria-label="Puntaje de anomalía por inversor">`;
    for (let t = 0.3; t <= vmax + 1e-9; t += 0.1) g += `<line x1="${X(t)}" x2="${X(t)}" y1="8" y2="${H - 24}" class="grid"/><text x="${X(t)}" y="${H - 8}" class="tick" text-anchor="middle">${t.toFixed(1)}</text>`;
    const orden = [...ok].sort((a, b) => b.s.m - a.s.m);
    orden.forEach((f, k) => {
      const y = 12 + k * fila; const cls = f.banderaIF ? "b-alto" : f.banderaPR ? "b-medio" : "b-normal";
      g += `<g class="fila-bar" data-inv="${f.inv}"><text x="${x0 - 8}" y="${y + 12}" class="lbl" text-anchor="end">${f.inv}</text>
        <rect x="${x0}" y="${y + 2}" width="${Math.max(2, X(f.s.m) - x0)}" height="14" rx="2" class="${cls}"/>
        <line x1="${X(f.s.m - 2 * f.s.sd)}" x2="${X(f.s.m + 2 * f.s.sd)}" y1="${y + 9}" y2="${y + 9}" class="err"/>
        <line x1="${X(f.s.m - 2 * f.s.sd)}" x2="${X(f.s.m - 2 * f.s.sd)}" y1="${y + 5}" y2="${y + 13}" class="err"/><line x1="${X(f.s.m + 2 * f.s.sd)}" x2="${X(f.s.m + 2 * f.s.sd)}" y1="${y + 5}" y2="${y + 13}" class="err"/></g>`;
    });
    g += `<line x1="${X(M.umbral)}" x2="${X(M.umbral)}" y1="4" y2="${H - 24}" class="umbral"/><text x="${X(M.umbral) + 4}" y="${H - 26}" class="tick umbral-t">umbral ${M.umbral.toFixed(3)}</text></svg>`;
    return g;
  }
  function curvas(dia, medP, sel) {
    const W = 640, H = 230, x0 = 46, x1 = W - 14, y0 = H - 26, y1 = 12;
    const X = (k) => x0 + (x1 - x0) * k / (HORAS.length - 1), Y = (v) => y0 - (y0 - y1) * v / NOM;
    let g = `<svg viewBox="0 0 ${W} ${H}" class="chart" role="img" aria-label="Potencia horaria">`;
    for (let v = 0; v <= NOM; v += 200) g += `<line x1="${x0}" x2="${x1}" y1="${Y(v)}" y2="${Y(v)}" class="grid"/><text x="${x0 - 6}" y="${Y(v) + 4}" class="tick" text-anchor="end">${v}</text>`;
    HORAS.forEach((h, k) => { if (h % 2 === 0) g += `<text x="${X(k)}" y="${H - 8}" class="tick" text-anchor="middle">${h}h</text>`; });
    const path = (arr) => arr.map((v, k) => isNaN(v) || v == null ? "" : `${k && !isNaN(arr[k - 1]) ? "L" : "M"}${X(k).toFixed(1)},${Y(v).toFixed(1)}`).join("");
    for (const i in dia) if (i !== sel) g += `<path d="${path(dia[i].p)}" class="otra"/>`;
    g += `<path d="${path(medP)}" class="mediana"/>`;
    if (sel && dia[sel]) g += `<path d="${path(dia[sel].p)}" class="sel"/>`;
    g += `<text x="${x0 + 4}" y="${y1 + 10}" class="tick">kW AC</text></svg>`;
    return g;
  }

  // ---------------- render
  function render() {
    const R = evaluar(); const filas = R.filas.filter((f) => !f.excl);
    const alertas = filas.filter((f) => f.alerta).sort((a, b) => b.v.perdida_est - a.v.perdida_est || b.s.m - a.s.m);
    if (!st.invSel || !R.dia[st.invSel]) st.invSel = (alertas[0] || [...filas].sort((a, b) => b.s.m - a.s.m)[0]).inv;
    const hoy = alertas.slice(0, st.k), cola = alertas.slice(st.k);
    const kwh = hoy.reduce((s, f) => s + f.v.perdida_est, 0);
    contextoActual = {
      escenario: R.nombre,
      fuente: st.fuente === "csv" ? "archivo CSV del usuario" : "periodo de prueba sintético",
      capacidad_inspecciones: st.k,
      alertas_totales: alertas.length,
      inspeccionar_hoy: hoy.map((f) => ({
        inversor: f.inv,
        perdida_estimada_kwh_dia: Math.round(f.v.perdida_est),
        puntaje_anomalia: +f.s.m.toFixed(3),
        umbral_isolation_forest: +M.umbral.toFixed(3),
        detectado_por: [f.banderaIF ? "Isolation Forest" : "", f.banderaPR ? "regla de rendimiento" : ""].filter(Boolean).join(" y "),
        estabilidad_bosques_pct: Math.round(f.estab * 100),
        pista_tecnica: pistas(f.v)[0][1]
      })),
      en_cola: cola.map((f) => f.inv),
      perdida_atendida_kwh_dia: Math.round(kwh),
      valor_estimado_cop_dia: Math.round(kwh * st.precio),
      costo_total_inspecciones_cop: Math.round(hoy.length * st.costo),
      inversor_seleccionado: st.invSel,
      regla_decision: `alerta si puntaje IF > ${M.umbral.toFixed(3)} o déficit energético > ${fmt(M.umbral_regla_pr * 100, 1)} %`
    };
    const contextoEl = $("#chat-contexto");
    if (contextoEl) contextoEl.textContent = `${R.nombre}: ${alertas.length} alerta(s), ${hoy.length} inspección(es), ${fmt(kwh)} kWh/día priorizados.`;
    $("#escenario-nombre").textContent = R.nombre;
    $("#kpi-alertas").textContent = alertas.length;
    $("#kpi-insp").textContent = `${hoy.length} de ${st.k}`;
    $("#kpi-kwh").textContent = fmt(kwh);
    $("#kpi-cop").textContent = "$ " + fmt(kwh * st.precio);

    // salida
    $("#barras").innerHTML = barras(R.filas);
    document.querySelectorAll(".fila-bar").forEach((el) => el.addEventListener("click", () => { st.invSel = el.dataset.inv; render(); }));
    $("#curvas").innerHTML = curvas(R.dia, R.medP, st.invSel);
    $("#inv-sel").textContent = st.invSel;
    const fs = R.filas.find((f) => f.inv === st.invSel);
    if (fs && !fs.excl) {
      $("#detalle").innerHTML = `<div class="det-cab"><span class="pill ${fs.banderaIF ? "p-alto" : fs.banderaPR ? "p-medio" : "p-ok"}">${fs.banderaIF ? "Anomalía (Isolation Forest)" : fs.banderaPR ? "Déficit energético (regla PR)" : "Normal"}</span>
        <span class="mono">puntaje ${fs.s.m.toFixed(3)} ± ${(2 * fs.s.sd).toFixed(3)} · ${Math.round(fs.estab * 100)} % de los 5 bosques lo marcan</span></div>
        <table class="t-vars"><thead><tr><th>Variable</th><th class="n">Valor hoy</th><th class="n">Normal (mediana)</th><th class="n">Desvío z</th></tr></thead><tbody>
        ${M.variables.map((k) => { const zz = z(k, fs.v[k]); return `<tr><td>${NOMBRES[k]}</td><td class="n">${fmt(fs.v[k], k === "horas_cero" ? 0 : 3)}</td><td class="n">${fmt(M.stats[k].med, 3)}</td><td class="n ${Math.abs(zz) > 3 ? "z-alto" : ""}">${fmt(zz, 1)}</td></tr>`; }).join("")}
        </tbody></table>
        <p class="pista-t">Pista para el técnico <small>(orienta la inspección; no es un diagnóstico)</small></p>
        <ul class="pistas">${pistas(fs.v).map(([c, t]) => `<li class="pt-${c}">${t}</li>`).join("")}</ul>`;
    } else $("#detalle").innerHTML = `<p class="nota">Este inversor-día tiene más del 30 % de datos faltantes y no se puntúa. Acción: revisar comunicación SCADA.</p>`;
    $("#excluidos").textContent = R.filas.filter((f) => f.excl).map((f) => f.inv).join(", ") || "ninguno";

    // decisión
    const vd = (key) => { const v = D.verdad[`${st.dia}|${key}`]; return st.fuente === "prueba" && v ? v : null; };
    const tarjeta = (f, rol) => {
      const pi = pistas(f.v)[0]; const ver = vd(f.inv);
      return `<li class="orden ${rol}"><div class="o-cab"><b>${f.inv}</b><span class="pill ${rol === "hoy" ? "p-alto" : "p-medio"}">${rol === "hoy" ? "Inspeccionar hoy" : "En cola"}</span></div>
        <div class="o-num"><span><b>${fmt(f.v.perdida_est)}</b> kWh/día</span><span><b>$ ${fmt(f.v.perdida_est * st.precio)}</b>/día</span><span>${f.banderaIF ? "IF" : ""}${f.banderaIF && f.banderaPR ? " + " : ""}${f.banderaPR ? "regla PR" : ""}</span></div>
        <p>${pi[1]}</p>${st.verdad && ver ? `<p class="verdad">Verdad de terreno: ${ver[0] || "sin falla"} (${fmt(ver[1])} kWh perdidos)</p>` : ""}</li>`;
    };
    $("#lista-hoy").innerHTML = hoy.length ? hoy.map((f) => tarjeta(f, "hoy")).join("") : `<li class="orden vacio">Ningún inversor supera los umbrales. Decisión: <b>no enviar la cuadrilla</b> y ahorrar ${fmt(st.k)} inspecciones ($ ${fmt(st.k * st.costo)}).</li>`;
    $("#lista-cola").innerHTML = cola.map((f) => tarjeta(f, "cola")).join("") || `<li class="orden vacio">Sin equipos en cola.</li>`;
    const normales = filas.filter((f) => !f.alerta).map((f) => f.inv);
    $("#normales").textContent = normales.join(", ");
    const ahorro = st.k - hoy.length;
    $("#resumen-decision").innerHTML = `Con <b>${st.k}</b> inspecciones disponibles, SOLAROM asigna <b>${hoy.length}</b>${ahorro > 0 ? ` y libera <b>${ahorro}</b> ($ ${fmt(ahorro * st.costo)})` : ""}. Si las fallas se confirman y se corrigen hoy, se dejan de perder unos <b>${fmt(kwh)} kWh por día</b> ($ ${fmt(kwh * st.precio)} diarios) frente a un costo de inspección de $ ${fmt(hoy.length * st.costo)}. Una pérdida diaria de este tamaño paga la visita en <b>${kwh > 0 ? fmt(hoy.length * st.costo / (kwh * st.precio), 1) : "—"} días</b>.`;
    if (st.verdad && st.fuente === "prueba") {
      const reales = D.inversores.filter((i) => (D.verdad[`${st.dia}|${i}`] || [""])[0]);
      $("#verdad-dia").innerHTML = `Fallas inyectadas ese día: ${reales.length ? reales.map((i) => `${i} (${D.verdad[`${st.dia}|${i}`][0]})`).join(", ") : "ninguna"}.`;
    } else $("#verdad-dia").innerHTML = "";
  }

  // ---------------- evidencia (backtest)
  function evidencia() {
    const tb = D.backtest.filter((r) => r.periodo === "prueba");
    const mejor = Math.max(...tb.map((r) => r.valor_neto_cop));
    $("#t-backtest tbody").innerHTML = tb.map((r) => `<tr class="${r.metodo.startsWith("SOLAROM") ? "fila-sg" : ""}"><td>${r.metodo}</td><td class="n">${r.inspecciones}</td><td class="n">${r.utiles}</td><td class="n">${fmt(r.precision * 100)} %</td>
      <td class="n">${r.eventos_detectados}/${r.eventos}</td><td class="n">${fmt(r.retraso_medio_dias, 1)}</td><td class="n">${fmt(r.mwh_recuperados, 1)}</td><td class="n ${r.valor_neto_cop < 0 ? "neg" : ""}">${fmt(r.valor_neto_cop / 1e6, 1)}</td></tr>`).join("");
    const nombres = { persistencia: "Persistencia", regla_pr: "Regla PR vs flota", if_media: "Isolation Forest" };
    const tipos = Object.keys(D.auc_tipo);
    $("#t-auc").innerHTML = `<thead><tr><th>Tipo de falla</th>${Object.values(nombres).map((n) => `<th class="n">${n}</th>`).join("")}</tr></thead><tbody>${tipos.map((t) => `<tr><td>${t.replace("_", " ")}</td>${Object.keys(nombres).map((m) => `<td class="n ${D.auc_tipo[t][m] < 0.7 ? "neg" : ""}">${fmt(D.auc_tipo[t][m], 2)}</td>`).join("")}</tr>`).join("")}
      <tr class="fila-sg"><td>Todas (AUC-ROC)</td>${Object.keys(nombres).map((m) => `<td class="n">${fmt(D.auc[m].roc_auc, 3)}</td>`).join("")}</tr></tbody>`;
    const p = D.particion;
    $("#particion").innerHTML = Object.entries(p).map(([k, v]) => `<div class="seg seg-${k}" style="flex:${v[2]}"><b>${k}</b><span>${v[0]} → ${v[1]} · ${v[2]} días</span></div>`).join("");
    const b = D.bitacora;
    $("#bitacora").innerHTML = Object.entries(b).map(([k, v]) => `<tr><td>${k.replaceAll("_", " ")}</td><td class="n">${fmt(v)}</td></tr>`).join("");
    const sg = tb.find((r) => r.metodo.startsWith("SOLAROM")), rot = tb.find((r) => r.metodo.startsWith("Rotación")), pr = tb.find((r) => r.metodo.includes("regla PR vs"));
    $("#mensaje-jefe").innerHTML = `En los 73 días de prueba (oct–dic 2025, datos que el modelo nunca vio), SOLAROM detectó las <b>${sg.eventos_detectados} de ${sg.eventos}</b> fallas con <b>${sg.inspecciones}</b> inspecciones, contra <b>${rot.inspecciones}</b> de la rotación fija. Recuperó <b>${fmt(sg.mwh_recuperados, 1)} MWh</b> y dejó un valor neto de <b>$ ${fmt(sg.valor_neto_cop / 1e6, 1)} millones</b>, frente a <b>$ ${fmt(rot.valor_neto_cop / 1e6, 1)} millones</b> de la rotación. La regla simple de PR logra un valor neto parecido ($ ${fmt(pr.valor_neto_cop / 1e6, 1)} M) pero no ve ${sg.eventos_detectados - pr.eventos_detectados} de las fallas (sensor congelado y derating), que el Isolation Forest sí encuentra.`;
  }

  // ---------------- asistente IA contextual
  const PROMPT_SISTEMA = `Eres SOLAROM IA, asistente de apoyo para mantenimiento de una planta fotovoltaica.
Responde siempre en español claro y en máximo 180 palabras. Usa exclusivamente el contexto operativo incluido en la consulta y el conocimiento metodológico indicado aquí: SOLAROM combina Isolation Forest con una regla de rendimiento relativo para priorizar inspecciones. Una anomalía es rareza estadística, no una falla confirmada. Las pistas orientan la revisión, no son diagnósticos. Si el contexto no contiene el dato pedido, dilo sin inventar. Recomienda verificar SCADA, alarmas y condiciones en campo, y respetar los procedimientos de seguridad. No autorices maniobras eléctricas ni prometas ahorros como hechos confirmados.`;

  function textoIA(respuesta) {
    if (typeof respuesta === "string") return respuesta;
    const contenido = respuesta && (respuesta.message?.content ?? respuesta.content ?? respuesta.text);
    if (typeof contenido === "string") return contenido;
    if (Array.isArray(contenido)) return contenido.map((b) => typeof b === "string" ? b : (b.text || b.content || "")).join("\n").trim();
    return "No pude interpretar la respuesta del servicio de IA.";
  }

  function agregarMensaje(rol, texto) {
    const log = $("#chat-log");
    const msg = document.createElement("div");
    msg.className = `mensaje ${rol === "user" ? "usuario" : "asistente"}`;
    const etiqueta = document.createElement("small");
    etiqueta.textContent = rol === "user" ? "Tú" : "SOLAROM IA";
    msg.append(etiqueta, document.createTextNode(texto));
    log.appendChild(msg);
    log.scrollTop = log.scrollHeight;
  }

  function respuestaLocal(pregunta) {
    const c = contextoActual;
    if (!c) return "Aún no hay un escenario calculado para explicar.";
    const q = pregunta.toLowerCase();
    const primero = c.inspeccionar_hoy[0];
    if (q.includes("puntaje") || q.includes("anomal")) return `El puntaje indica qué tan diferente es el comportamiento del inversor frente a los patrones aprendidos. El umbral actual es ${M.umbral.toFixed(3)}. Superarlo aumenta la prioridad de revisión, pero no demuestra por sí solo una falla; el técnico debe comprobar alarmas, comunicación y condiciones físicas.`;
    if (q.includes("limit") || q.includes("verific")) return "El modelo fue validado con datos sintéticos y detecta rareza, no causas. Puede perder degradaciones comunes a toda la planta y la suciedad gradual. Antes de actuar, verifica calidad del SCADA, alarmas del inversor, irradiancia, protecciones y evidencia en campo siguiendo el procedimiento de seguridad.";
    if (primero) return `Para ${c.escenario}, SOLAROM prioriza ${primero.inversor}: estima ${fmt(primero.perdida_estimada_kwh_dia)} kWh/día en riesgo y lo marcó ${primero.detectado_por}. Pista: ${primero.pista_tecnica} Esta es una prioridad de inspección, no una falla confirmada.`;
    return `En ${c.escenario} ningún inversor supera los umbrales. La decisión sugerida es no enviar la cuadrilla por una alerta del modelo y continuar el monitoreo; cualquier alarma SCADA independiente debe revisarse por separado.`;
  }

  async function enviarChat(pregunta) {
    const enviar = $("#chat-enviar"), estado = $("#chat-estado");
    agregarMensaje("user", pregunta);
    enviar.disabled = true; estado.textContent = "Consultando el modelo de IA…";
    try {
      if (!window.puter?.ai?.chat) throw new Error("Servicio de IA no disponible");
      const mensajes = [
        { role: "system", content: PROMPT_SISTEMA },
        ...historialChat.slice(-8),
        { role: "user", content: `Contexto operativo actual (JSON):\n${JSON.stringify(contextoActual)}\n\nPregunta del usuario: ${pregunta}` }
      ];
      const limite = new Promise((_, reject) => setTimeout(() => reject(new Error("Tiempo de espera agotado")), 12000));
      const respuesta = textoIA(await Promise.race([window.puter.ai.chat(mensajes), limite]));
      agregarMensaje("assistant", respuesta);
      historialChat.push({ role: "user", content: pregunta }, { role: "assistant", content: respuesta });
      estado.textContent = "Respuesta generada con IA para el escenario activo.";
    } catch (err) {
      const respaldo = `Modo local de respaldo: ${respuestaLocal(pregunta)}`;
      agregarMensaje("assistant", respaldo);
      historialChat.push({ role: "user", content: pregunta }, { role: "assistant", content: respaldo });
      estado.textContent = "IA en línea no disponible; se mostró una explicación local identificada.";
    } finally { enviar.disabled = false; $("#chat-input").focus(); }
  }

  function limpiarChat() {
    historialChat = [];
    $("#chat-log").innerHTML = `<div class="mensaje asistente"><small>SOLAROM IA</small>Nueva conversación. Ya tengo el contexto del escenario activo. ¿Qué deseas entender?</div>`;
    $("#chat-estado").textContent = "Listo para consultar.";
  }

  // ---------------- controles
  function init() {
    const selDia = $("#sel-dia"); selDia.innerHTML = D.dias.map((d) => `<option ${d === st.dia ? "selected" : ""}>${d}</option>`).join("");
    selDia.addEventListener("change", () => { st.dia = selDia.value; st.invSel = null; render(); });
    const selInv = $("#iny-inv"); selInv.innerHTML = `<option value="">— sin falla simulada —</option>` + D.inversores.map((i) => `<option>${i}</option>`).join("");
    selInv.addEventListener("change", () => { st.iny.inv = selInv.value; st.invSel = selInv.value || null; render(); });
    $("#iny-tipo").addEventListener("change", (e) => { st.iny.tipo = e.target.value; render(); });
    $("#iny-sev").addEventListener("input", (e) => { st.iny.sev = +e.target.value / 100; $("#iny-sev-v").textContent = e.target.value + " %"; render(); });
    document.querySelectorAll("input[name=fuente]").forEach((r) => r.addEventListener("change", () => {
      st.fuente = r.value; $("#bloque-prueba").hidden = st.fuente !== "prueba"; $("#bloque-csv").hidden = st.fuente !== "csv";
      if (st.fuente === "csv" && !st.subidos) return; st.invSel = null; render(); }));
    $("#archivo").addEventListener("change", async (e) => {
      const f = e.target.files[0]; if (!f) return;
      try {
        const r = limpiarCSV(await f.text()); st.subidos = r; const ds = Object.keys(r.dias).sort(); st.diaSub = ds[ds.length - 1];
        $("#sel-dia-csv").innerHTML = ds.map((d) => `<option ${d === st.diaSub ? "selected" : ""}>${d}</option>`).join("");
        $("#log-csv").textContent = `Leídos ${r.log.registros} registros, ${ds.length} día(s). Limpieza: ${r.log.fuera_de_rango} fuera de rango → vacío; ${r.log.interpolados} interpolados (huecos ≤ 2 h); ${r.log.rellenados_mediana} rellenados con mediana de flota; ${r.log.excluidos} inversor-día excluidos (> 30 % faltante).`;
        st.invSel = null; render();
      } catch (err) { $("#log-csv").textContent = "No se pudo leer el archivo: " + err.message; }
    });
    $("#sel-dia-csv").addEventListener("change", (e) => { st.diaSub = e.target.value; st.invSel = null; render(); });
    $("#plantilla").addEventListener("click", () => {
      const lin = ["fecha_hora,inversor,ghi_wm2,t_amb_c,p_ac_kw,t_inv_c"];
      for (const i of D.inversores) HORAS.forEach((h, k) => { const s = D.hora[i][st.dia]; lin.push(`${st.dia} ${String(h).padStart(2, "0")}:00:00,${i},${s.ghi[k]},,${s.p[k]},${s.t[k]}`); });
      $("#plantilla-txt").value = lin.join("\n"); $("#plantilla-box").hidden = false;
    });
    $("#ver-verdad").addEventListener("change", (e) => { st.verdad = e.target.checked; render(); });
    for (const [id, key] of [["#in-k", "k"], ["#in-precio", "precio"], ["#in-costo", "costo"]]) {
      const el = $(id); el.value = st[key]; el.addEventListener("input", () => { const v = parseFloat(el.value); if (v >= 0) { st[key] = v; render(); } });
    }
    document.querySelectorAll(".tab").forEach((t) => t.addEventListener("click", () => mostrar(t.dataset.p)));
    $("#chat-form").addEventListener("submit", async (e) => {
      e.preventDefault(); const input = $("#chat-input"), pregunta = input.value.trim(); if (!pregunta) return;
      input.value = ""; await enviarChat(pregunta);
    });
    document.querySelectorAll(".pregunta-rapida").forEach((b) => b.addEventListener("click", () => {
      $("#chat-input").value = b.dataset.pregunta; $("#chat-input").focus();
    }));
    $("#chat-limpiar").addEventListener("click", limpiarChat);
    $("#umbral-if").textContent = M.umbral.toFixed(3); $("#umbral-pr").textContent = fmt(M.umbral_regla_pr * 100, 1) + " %";
    // verificación de paridad con scikit-learn
    const err = Math.max(...M.paridad.X.map((x, k) => Math.abs(scoreIF(x).m - M.paridad.s[k])));
    $("#paridad").textContent = err < 1e-6 ? `Verificado: el puntaje calculado en el navegador coincide con scikit-learn (error máximo ${err.toExponential(1)} en 50 casos).` : `Atención: diferencia con scikit-learn de ${err.toExponential(2)}.`;
    evidencia(); render();
    const h = location.hash.slice(1); if (h) mostrar(h);
  }
  function mostrar(p) {
    if (!document.getElementById(p)) p = "escenario";
    document.querySelectorAll(".tab").forEach((t) => t.setAttribute("aria-selected", t.dataset.p === p));
    document.querySelectorAll(".panel").forEach((el) => (el.hidden = el.id !== p));
    history.replaceState(null, "", `#${p}`);
  }
  window.SG = { scoreIF, variables, limpiarCSV };
  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", init); else init();
})();
