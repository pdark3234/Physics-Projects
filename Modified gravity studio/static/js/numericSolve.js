/**
 * Numeric solve and plotting UI for non-linear residual systems.
 * Loaded after app.js; uses shared UI helpers from the main script.
 */

let symbolicPlotCache = new Map();
let latestEnergyPlot = null;

function setPlotExportButtonsEnabled(enabled) {
    document.querySelectorAll('[data-export-plot]').forEach((btn) => {
        btn.disabled = !enabled;
        btn.onclick = exportNumericPlotPng;
    });
}

function renderNumericInput(container, name, value = '', placeholder = '1') {
    const wrap = document.createElement('div');
    wrap.className = 'param-input';
    const label = document.createElement('label');
    label.textContent = name;
    wrap.appendChild(label);
    const input = document.createElement('input');
    input.type = 'number';
    input.step = 'any';
    input.className = 'input-full';
    input.dataset.numericName = name;
    input.placeholder = placeholder;
    input.value = value;
    wrap.appendChild(input);
    container.appendChild(wrap);
    return input;
}

function setupSymbolicPlotPanel(results) {
    const panel = $('symbolic-plot-panel');
    const controls = $('symbolic-plot-controls');
    const buildBtn = $('build-symbolic-plot-btn');
    const runBtn = $('run-symbolic-plot-btn');
    const exportBtn = $('export-symbolic-plot-btn');
    const status = $('symbolic-plot-status');
    if (!panel || !controls || !buildBtn || !runBtn) return;

    const spec = results?.plot_data || {};
    if (results?.early_exit || !spec.available) {
        hide(panel);
        hide(controls);
        return;
    }

    show(panel);
    hide(controls);
    if (status) status.textContent = '';
    setPlotExportButtonsEnabled(false);

    buildBtn.onclick = () => {
        const domainEl = $('symbolic-plot-domain-controls');
        const paramEl = $('symbolic-plot-param-controls');
        if (!domainEl || !paramEl) return;
        clearNumericTabPlots();
        domainEl.innerHTML = '';
        paramEl.innerHTML = '';
        renderNumericInput(domainEl, `${spec.variable}_min`, '0.1');
        renderNumericInput(domainEl, `${spec.variable}_max`, '10');
        renderNumericInput(domainEl, 'points', '300', '300');

        const defaults = spec.parameter_defaults || {};
        (spec.parameters || []).forEach((name) => {
            renderNumericInput(paramEl, name, defaults[name] || '', 'required');
        });
        if (!(spec.parameters || []).length) {
            paramEl.innerHTML = '<p class="note">No free numeric parameters detected.</p>';
        }
        show(controls);
    };

    runBtn.onclick = async () => {
        const domainEl = $('symbolic-plot-domain-controls');
        const paramEl = $('symbolic-plot-param-controls');
        if (!domainEl || !paramEl) return;
        if (status) status.textContent = 'Evaluating solved expressions...';

        const readNamedInputs = (container) => {
            const values = {};
            container.querySelectorAll('input[data-numeric-name]').forEach((input) => {
                const name = input.dataset.numericName;
                if (input.value.trim() !== '') values[name] = input.value.trim();
            });
            return values;
        };
        const domainValues = readNamedInputs(domainEl);
        const payload = {
            variable: spec.variable,
            groups: spec.groups || {},
            parameters: readNamedInputs(paramEl),
            domain: {
                min: domainValues[`${spec.variable}_min`] || 0.1,
                max: domainValues[`${spec.variable}_max`] || 10,
                points: domainValues.points || 300,
            },
        };

        try {
            const cacheKey = JSON.stringify(payload);
            let data = symbolicPlotCache.get(cacheKey);
            const usedCache = Boolean(data);
            if (!data) {
                const resp = await fetch('/api/plot/evaluate', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(payload),
                });
                data = await resp.json();
                if (!resp.ok) throw new Error(data.error || 'Plot evaluation failed');
                symbolicPlotCache.set(cacheKey, data);
            }
            renderSymbolicPlotData(data);
            if (status) {
                const points = data.metadata?.points || 0;
                status.textContent = `Plotted ${points} samples.${usedCache ? ' Loaded from plot cache.' : ''}`;
            }
            setPlotExportButtonsEnabled(true);
        } catch (err) {
            if (status) status.textContent = `Error: ${err.message}`;
        }
    };
}

function setupNumericSolvePanel(results) {
    const panel = $('numeric-solve-panel');
    const controls = $('numeric-solve-controls');
    const buildBtn = $('build-numeric-solve-btn');
    const runBtn = $('run-numeric-solve-btn');
    const exportBtn = $('export-numeric-plot-btn');
    const status = $('numeric-solve-status');
    const plotContainer = $('numeric-plot-container');
    if (!panel || !controls || !buildBtn || !runBtn) return;

    const spec = results?.numeric_solve || {};
    if (!results?.early_exit || !spec.available) {
        hide(panel);
        hide(controls);
        return;
    }

    show(panel);
    hide(controls);
    if (status) status.textContent = '';
    if (plotContainer) {
        plotContainer.innerHTML = '';
        hide(plotContainer);
    }
    clearNumericTabPlots();
    setPlotExportButtonsEnabled(false);

    buildBtn.onclick = () => {
        const domainEl = $('numeric-domain-controls');
        const paramEl = $('numeric-param-controls');
        const guessEl = $('numeric-guess-controls');
        if (!domainEl || !paramEl || !guessEl) return;

        domainEl.innerHTML = '';
        paramEl.innerHTML = '';
        guessEl.innerHTML = '';

        renderNumericInput(domainEl, `${spec.variable}_min`, '0.1');
        renderNumericInput(domainEl, `${spec.variable}_max`, '10');
        renderNumericInput(domainEl, 'points', '120', '120');

        const defaults = spec.parameter_defaults || {};
        (spec.parameters || []).forEach((name) => {
            renderNumericInput(paramEl, name, defaults[name] || '', 'required');
        });
        if (!(spec.parameters || []).length) {
            paramEl.innerHTML = '<p class="note">No free numeric parameters detected.</p>';
        }

        (spec.unknowns || []).forEach((name) => {
            renderNumericInput(guessEl, name, name === 'rho' ? '1' : '0.1');
        });

        show(controls);
    };

    runBtn.onclick = async () => {
        const domainEl = $('numeric-domain-controls');
        const paramEl = $('numeric-param-controls');
        const guessEl = $('numeric-guess-controls');
        if (!domainEl || !paramEl || !guessEl) return;
        if (status) status.textContent = 'Solving numerical residual system...';

        const readNamedInputs = (container) => {
            const values = {};
            container.querySelectorAll('input[data-numeric-name]').forEach((input) => {
                const name = input.dataset.numericName;
                if (input.value.trim() !== '') values[name] = input.value.trim();
            });
            return values;
        };
        const domainValues = readNamedInputs(domainEl);
        const payload = {
            residuals: spec.residuals || [],
            variable: spec.variable,
            unknowns: spec.unknowns || [],
            background_id: spec.background_id,
            stress_tensor: spec.stress_tensor,
            metric_functions: spec.metric_functions || {},
            parameters: readNamedInputs(paramEl),
            initial_guesses: readNamedInputs(guessEl),
            domain: {
                min: domainValues[`${spec.variable}_min`] || 0.1,
                max: domainValues[`${spec.variable}_max`] || 10,
                points: domainValues.points || 120,
            },
        };

        try {
            const cacheKey = JSON.stringify(payload);
            let data = numericSolveCache.get(cacheKey);
            const usedCache = Boolean(data);
            if (data) {
                if (status) status.textContent = 'Loaded cached numerical curve for this parameter set.';
            } else {
                const resp = await fetch('/api/numeric/solve', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(payload),
                });
                data = await resp.json();
                if (!resp.ok) throw new Error(data.error || 'Numerical solve failed');
                numericSolveCache.set(cacheKey, data);
            }
            if (status) {
                const meta = data.metadata || {};
                const cached = usedCache ? ' Loaded from numeric cache.' : ' Symbolic equations were not recomputed.';
                status.textContent = `Converged ${meta.converged_points || 0}/${meta.points || 0} points.${cached}`;
            }
            renderNumericPlot(data);
            setPlotExportButtonsEnabled(true);
        } catch (err) {
            if (status) status.textContent = `Error: ${err.message}`;
        }
    };
}

function renderNumericPlot(data) {
    const x = data.x || [];
    if (!x.length) {
        const container = $('numeric-plot-container');
        if (!container) return;
        container.innerHTML = '<p class="note">No numeric solution points returned.</p>';
        show(container);
        return;
    }

    lastNumericPlot = data;
    clearNumericTabPlots();
    renderNumericPlotInto('numeric-plot-container', 'Matter variables', x, data.solutions || {}, data.variable || 'x', data.warnings);

    const diagnostics = data.diagnostics || {};
    renderEnergyConditionPlot(
        'Numerical energy conditions',
        x,
        labelSeries(
            pickSeries(diagnostics, ['NEC', 'NEC_r', 'NEC_t', 'WEC', 'SEC', 'DEC', 'DEC_r', 'DEC_t']),
            {
                NEC: 'NEC',
                NEC_r: 'NEC_r',
                NEC_t: 'NEC_t',
                WEC: 'WEC',
                SEC: 'SEC',
                DEC: 'DEC',
                DEC_r: 'DEC_r',
                DEC_t: 'DEC_t',
            }
        ),
        data.variable || 'x'
    );
    renderNumericPlotInto(
        'numeric-eos-plot-container',
        'Numerical equation of state',
        x,
        labelSeries(
            pickSeries(diagnostics, ['omega', 'omega_r', 'omega_t', 'omega_eff']),
            { omega: 'omega', omega_r: 'omega_r', omega_t: 'omega_t', omega_eff: 'omega_eff' }
        ),
        data.variable || 'x'
    );
    renderNumericPlotInto(
        'numeric-stability-plot-container',
        'Numerical stability',
        x,
        labelSeries(
            pickSeries(diagnostics, ['cs2', 'cs2_r', 'cs2_t']),
            { cs2: 'c_s^2', cs2_r: 'c_s^2_r', cs2_t: 'c_s^2_t' }
        ),
        data.variable || 'x'
    );
    renderNumericPlotInto(
        'numeric-tov-plot-container',
        'Numerical TOV equilibrium',
        x,
        labelSeries(
            pickSeries(data.tov || {}, ['hydrostatic_force', 'gravitational_force', 'anisotropic_force', 'residual']),
            {
                hydrostatic_force: 'F_h',
                gravitational_force: 'F_g',
                anisotropic_force: 'F_a',
                residual: 'F_h+F_g+F_a',
            }
        ),
        data.variable || 'x'
    );
}

function renderSymbolicPlotData(data) {
    const x = data.x || [];
    if (!x.length) return;
    const groups = data.groups || {};
    clearNumericTabPlots();
    renderNumericPlotInto('symbolic-matter-plot-container', 'Matter variables', x, groups.matter || {}, data.variable || 'x', data.warnings);
    renderEnergyConditionPlot(
        'Energy conditions',
        x,
        groups.energy_conditions || {},
        data.variable || 'x'
    );
    renderNumericPlotInto(
        'numeric-eos-plot-container',
        'Equation of state',
        x,
        groups.eos || {},
        data.variable || 'x'
    );
    renderNumericPlotInto(
        'numeric-stability-plot-container',
        'Stability',
        x,
        labelSeries(groups.stability || {}, { cs2: 'c_s^2', cs2_r: 'c_s^2_r', cs2_t: 'c_s^2_t' }),
        data.variable || 'x'
    );
    renderNumericPlotInto(
        'numeric-tov-plot-container',
        'TOV equilibrium',
        x,
        labelSeries(groups.tov || {}, {
            hydrostatic_force: 'F_h',
            gravitational_force: 'F_g',
            anisotropic_force: 'F_a',
            residual: 'F_h+F_g+F_a',
        }),
        data.variable || 'x'
    );
}

function clearNumericTabPlots() {
    latestEnergyPlot = null;
    [
        'numeric-plot-container',
        'symbolic-matter-plot-container',
        'numeric-ec-plot-container',
        'numeric-eos-plot-container',
        'numeric-stability-plot-container',
        'numeric-tov-plot-container',
    ].forEach((id) => {
        const el = $(id);
        if (el) {
            el.innerHTML = '';
            hide(el);
        }
    });
    const ecOptions = $('ec-plot-options');
    if (ecOptions) {
        ecOptions.innerHTML = '';
        hide(ecOptions);
    }
}

function renderEnergyConditionPlot(title, x, series, variableLabel) {
    latestEnergyPlot = { title, x, series: series || {}, variableLabel };
    setupEnergyConditionSelector(series || {});
    renderSelectedEnergyConditionPlot();
}

function setupEnergyConditionSelector(series) {
    const options = $('ec-plot-options');
    if (!options) return;
    const names = Object.keys(series || {}).filter((name) =>
        (series[name] || []).some((v) => typeof v === 'number' && Number.isFinite(v))
    );
    if (names.length <= 1) {
        options.innerHTML = '';
        hide(options);
        return;
    }
    const existing = new Set(
        Array.from(options.querySelectorAll('input[data-ec-component]:checked')).map((input) => input.dataset.ecComponent)
    );
    const hasExisting = existing.size > 0;
    options.innerHTML = `
        <div class="plot-component-options-title">Plot components</div>
        <div class="plot-component-list">
            ${names.map((name) => {
                const checked = !hasExisting || existing.has(name) ? 'checked' : '';
                return `
                    <label class="plot-component-choice">
                        <input type="checkbox" data-ec-component="${escapeHtml(name)}" ${checked}>
                        <span>${escapeHtml(name)}</span>
                    </label>
                `;
            }).join('')}
        </div>
    `;
    options.querySelectorAll('input[data-ec-component]').forEach((input) => {
        input.addEventListener('change', renderSelectedEnergyConditionPlot);
    });
    show(options);
}

function renderSelectedEnergyConditionPlot() {
    if (!latestEnergyPlot) return;
    const options = $('ec-plot-options');
    const selected = new Set(
        options
            ? Array.from(options.querySelectorAll('input[data-ec-component]:checked')).map((input) => input.dataset.ecComponent)
            : Object.keys(latestEnergyPlot.series || {})
    );
    const filtered = {};
    Object.entries(latestEnergyPlot.series || {}).forEach(([name, values]) => {
        if (selected.has(name)) filtered[name] = values;
    });
    renderNumericPlotInto(
        'numeric-ec-plot-container',
        latestEnergyPlot.title,
        latestEnergyPlot.x,
        filtered,
        latestEnergyPlot.variableLabel
    );
}

function pickSeries(source, keys) {
    const out = {};
    keys.forEach((key) => {
        if (source && source[key]) out[key] = source[key];
    });
    return out;
}

function labelSeries(series, labels) {
    const out = {};
    Object.entries(series || {}).forEach(([key, values]) => {
        out[labels[key] || key] = values;
    });
    return out;
}

function renderNumericPlotInto(containerId, title, x, series, variableLabel, warnings = []) {
    const container = $(containerId);
    if (!container) return;
    const svg = buildNumericPlotSvg(title, x, series, variableLabel);
    if (!svg) {
        hide(container);
        return;
    }
    const warnHtml = (warnings || []).length
        ? `<p class="note">${escapeHtml((warnings || []).slice(0, 2).join(' '))}</p>`
        : '';
    container.innerHTML = `${svg}${warnHtml}`;
    show(container);
}

function buildNumericPlotSvg(title, x, series, variableLabel) {
    const names = Object.keys(series || {}).filter((name) =>
        (series[name] || []).some((v) => typeof v === 'number' && Number.isFinite(v))
    );
    if (!names.length) return '';

    const width = 860;
    const height = 420;
    const padL = 78;
    const padT = 34;
    const padB = 64;
    const charWidth = 8.4;
    const legendW = Math.min(220, Math.max(118, 66 + Math.max(...names.map((name) => name.length)) * charWidth));
    const padR = legendW + 54;
    const plotW = width - padL - padR;
    const plotH = height - padT - padB;
    const colors = ['#111111', '#ff7f0e', '#1d4fff', '#15a55b', '#9b4dca', '#d62728', '#008b8b'];
    const allY = [];
    names.forEach((name) => (series[name] || []).forEach((v) => {
        if (typeof v === 'number' && Number.isFinite(v)) allY.push(v);
    }));
    if (!allY.length) return '';

    const minX = Math.min(...x);
    const maxX = Math.max(...x);
    const sortedY = [...allY].sort((a, b) => a - b);
    const percentile = (arr, p) => {
        if (!arr.length) return 0;
        const idx = Math.min(arr.length - 1, Math.max(0, Math.floor((arr.length - 1) * p)));
        return arr[idx];
    };
    const rawMinY = Math.min(...allY);
    const rawMaxY = Math.max(...allY);
    let minY = percentile(sortedY, 0.02);
    let maxY = percentile(sortedY, 0.98);
    if (!Number.isFinite(minY) || !Number.isFinite(maxY) || minY === maxY) {
        minY = rawMinY;
        maxY = rawMaxY;
    }
    const centerY = (minY + maxY) / 2;
    const spanY = maxY - minY;
    const minReadableSpan = Math.max(1e-6, Math.abs(centerY) * 0.12);
    if (!Number.isFinite(spanY) || spanY < minReadableSpan) {
        minY = centerY - minReadableSpan / 2;
        maxY = centerY + minReadableSpan / 2;
    }
    const padY = Math.max(1e-9, (maxY - minY) * 0.06);
    minY -= padY;
    maxY += padY;
    if (minY === maxY) {
        minY -= 1;
        maxY += 1;
    }
    const sx = (v) => padL + ((v - minX) / (maxX - minX || 1)) * plotW;
    const sy = (v) => padT + plotH - ((v - minY) / (maxY - minY || 1)) * plotH;
    const fmt = (v, step = null) => {
        if (!Number.isFinite(v)) return '';
        if (Math.abs(v) >= 1000 || (Math.abs(v) > 0 && Math.abs(v) < 0.01)) return v.toExponential(1);
        if (step && Number.isFinite(step) && step > 0) {
            const decimals = Math.min(8, Math.max(0, Math.ceil(-Math.log10(step)) + 1));
            return Number(v.toFixed(decimals)).toString();
        }
        return Number(v.toPrecision(4)).toString();
    };
    const tickValues = (min, max, count = 6) => {
        const values = [];
        for (let i = 0; i < count; i++) values.push(min + (i * (max - min)) / (count - 1));
        return values;
    };
    const xTicks = tickValues(minX, maxX, 6);
    const yTicks = tickValues(minY, maxY, 6);
    const xStep = xTicks.length > 1 ? Math.abs(xTicks[1] - xTicks[0]) : null;
    const yStep = yTicks.length > 1 ? Math.abs(yTicks[1] - yTicks[0]) : null;
    const xTickSvg = xTicks.map((v) => `
        <line x1="${sx(v).toFixed(2)}" y1="${padT + plotH}" x2="${sx(v).toFixed(2)}" y2="${padT + plotH + 5}" class="plot-tick"/>
        <text x="${sx(v).toFixed(2)}" y="${padT + plotH + 22}" class="plot-tick-label" text-anchor="middle">${fmt(v, xStep)}</text>
    `).join('');
    const yTickSvg = yTicks.map((v) => `
        <line x1="${padL - 5}" y1="${sy(v).toFixed(2)}" x2="${padL}" y2="${sy(v).toFixed(2)}" class="plot-tick"/>
        <text x="${padL - 10}" y="${(sy(v) + 4).toFixed(2)}" class="plot-tick-label" text-anchor="end">${fmt(v, yStep)}</text>
    `).join('');
    const zeroLine = (minY < 0 && maxY > 0)
        ? `<line x1="${padL}" y1="${sy(0).toFixed(2)}" x2="${padL + plotW}" y2="${sy(0).toFixed(2)}" class="plot-zero-line"/>`
        : '';

    const paths = names.map((name, idx) => {
        const y = series[name] || [];
        let d = '';
        let penUp = true;
        y.forEach((value, i) => {
            if (typeof value !== 'number' || !Number.isFinite(value)) {
                penUp = true;
                return;
            }
            if (value < minY || value > maxY) {
                penUp = true;
                return;
            }
            d += `${penUp ? ' M' : ' L'} ${sx(x[i]).toFixed(2)} ${sy(value).toFixed(2)}`;
            penUp = false;
        });
        return `<path d="${d}" fill="none" stroke="${colors[idx % colors.length]}" stroke-width="2.5"/>`;
    }).join('');

    const legendX = padL + plotW + 20;
    const legendY = padT + 18;
    const legendH = 24 + names.length * 24;
    const legendSvg = `
        <g class="plot-legend">
            <rect x="${legendX}" y="${legendY}" width="${legendW}" height="${legendH}" rx="6" class="plot-legend-box"/>
            ${names.map((name, idx) => {
                const y = legendY + 24 + idx * 24;
                return `
                    <line x1="${legendX + 12}" y1="${y}" x2="${legendX + 40}" y2="${y}" stroke="${colors[idx % colors.length]}" stroke-width="3"/>
                    <text x="${legendX + 50}" y="${y + 5}" class="plot-legend-label">${escapeHtml(name)}</text>
                `;
            }).join('')}
        </g>
    `;
    return `
        <div class="numeric-plot-title">${escapeHtml(title)}</div>
        <svg viewBox="0 0 ${width} ${height}" class="numeric-plot" role="img">
            <style>
                .plot-frame{fill:#fff;stroke:#000;stroke-width:1}
                .plot-tick{stroke:#000;stroke-width:1}
                .plot-tick-label{fill:#000;font:13px monospace}
                .plot-zero-line{stroke:#000;stroke-width:1.2}
                .plot-label{fill:#000;font:italic 16px Georgia,serif}
                .plot-legend-box{fill:#fff;stroke:#000;stroke-width:1.2}
                .plot-legend-label{fill:#000;font:14px monospace}
            </style>
            <rect x="${padL}" y="${padT}" width="${plotW}" height="${plotH}" class="plot-frame"/>
            ${zeroLine}
            ${xTickSvg}
            ${yTickSvg}
            <text x="${padL + plotW / 2}" y="${height - 18}" class="plot-label" text-anchor="middle">${escapeHtml(variableLabel)}</text>
            <text x="22" y="${padT + plotH / 2}" class="plot-label" text-anchor="middle" transform="rotate(-90 22 ${padT + plotH / 2})">Values</text>
            ${paths}
            ${legendSvg}
        </svg>
    `;
}

function exportNumericPlotPng() {
    const svgs = Array.from(document.querySelectorAll(
        '#symbolic-matter-plot-container svg, #numeric-plot-container svg, #numeric-ec-plot-container svg, #numeric-eos-plot-container svg, #numeric-stability-plot-container svg, #numeric-tov-plot-container svg'
    ));
    if (!svgs.length) return;
    const serializer = new XMLSerializer();
    const scale = 2;
    const boxes = svgs.map((svg) => svg.viewBox.baseVal);
    const width = Math.max(...boxes.map((box) => box.width));
    const height = boxes.reduce((sum, box) => sum + box.height + 24, 0);
    const canvas = document.createElement('canvas');
    canvas.width = width * scale;
    canvas.height = height * scale;
    const ctx = canvas.getContext('2d');
    ctx.fillStyle = '#ffffff';
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    const drawNext = (idx, yOffset) => {
        if (idx >= svgs.length) {
            const link = document.createElement('a');
            link.download = `modified-gravity-plot-${Date.now()}.png`;
            link.href = canvas.toDataURL('image/png');
            link.click();
            return;
        }
        const svg = svgs[idx];
        const source = serializer.serializeToString(svg);
        const img = new Image();
        const blob = new Blob([source], { type: 'image/svg+xml;charset=utf-8' });
        const url = URL.createObjectURL(blob);
        img.onload = () => {
            const box = svg.viewBox.baseVal;
            ctx.drawImage(img, 0, yOffset * scale, box.width * scale, box.height * scale);
            URL.revokeObjectURL(url);
            drawNext(idx + 1, yOffset + box.height + 24);
        };
        img.src = url;
    };
    drawNext(0, 0);
}

