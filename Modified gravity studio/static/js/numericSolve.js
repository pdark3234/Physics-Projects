/**
 * Numeric solve and plotting UI for non-linear residual systems.
 * Loaded after app.js; uses shared UI helpers from the main script.
 */

let symbolicPlotCache = new Map();
let parameterScanCache = new Map();
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

function readNamedInputs(container) {
    const values = {};
    if (!container) return values;
    container.querySelectorAll('input[data-numeric-name]').forEach((input) => {
        const name = input.dataset.numericName;
        if (input.value.trim() !== '') values[name] = input.value.trim();
    });
    return values;
}

function readPlotAxisOptions(container) {
    const values = readNamedInputs(container);
    return {
        y_min: values.y_min || '',
        y_max: values.y_max || '',
    };
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
        renderNumericInput(domainEl, 'y_min', '', 'auto');
        renderNumericInput(domainEl, 'y_max', '', 'auto');
        renderNumericInput(domainEl, 'points', '300', '300');

        const defaults = spec.parameter_defaults || {};
        (spec.parameters || []).forEach((name) => {
            renderNumericInput(paramEl, name, defaults[name] || '', 'required');
        });
        if (!(spec.parameters || []).length) {
            paramEl.innerHTML = '<p class="note">No free numeric parameters detected.</p>';
        }
        show(controls);
        setupParameterScanPanel(spec);
    };

    runBtn.onclick = async () => {
        const domainEl = $('symbolic-plot-domain-controls');
        const paramEl = $('symbolic-plot-param-controls');
        if (!domainEl || !paramEl) return;
        if (status) status.textContent = 'Evaluating solved expressions...';

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
            renderSymbolicPlotData(data, readPlotAxisOptions(domainEl));
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

function setupParameterScanPanel(spec) {
    const panel = $('parameter-scan-panel');
    const controls = $('parameter-scan-controls');
    const buildBtn = $('build-parameter-scan-btn');
    const runBtn = $('run-parameter-scan-btn');
    const rangeEl = $('parameter-scan-range-controls');
    const constraintEl = $('parameter-scan-constraints');
    const status = $('parameter-scan-status');
    const resultsEl = $('parameter-scan-results');
    if (!panel || !controls || !buildBtn || !runBtn || !rangeEl || !constraintEl) return;

    const params = spec.parameters || [];
    if (!spec.available || !params.length) {
        hide(panel);
        hide(controls);
        return;
    }

    show(panel);
    hide(controls);
    if (status) status.textContent = '';
    if (resultsEl) {
        resultsEl.innerHTML = '';
        hide(resultsEl);
    }

    buildBtn.onclick = () => {
        rangeEl.innerHTML = '';
        constraintEl.innerHTML = '';
        const defaults = spec.parameter_defaults || {};
        params.forEach((name) => {
            const seed = Number(defaults[name] || 1);
            const lo = Number.isFinite(seed) ? seed - Math.max(1, Math.abs(seed)) : -1;
            const hi = Number.isFinite(seed) ? seed + Math.max(1, Math.abs(seed)) : 1;
            const row = document.createElement('div');
            row.className = 'scan-range-row';
            row.dataset.scanParam = name;
            row.innerHTML = `
                <div class="scan-param-name">${escapeHtml(name)}</div>
                <label>min<input type="number" step="any" data-scan-field="min" value="${lo}"></label>
                <label>max<input type="number" step="any" data-scan-field="max" value="${hi}"></label>
                <label>steps<input type="number" step="1" min="1" max="41" data-scan-field="steps" value="7"></label>
            `;
            rangeEl.appendChild(row);
        });

        const constraints = [
            ['finite', 'Finite curves', true, 'Reject samples with too many NaN or complex values.'],
            ['rho_positive', 'rho >= 0', true, 'Require non-negative energy density over the sampled domain.'],
            ['energy_conditions', 'Energy conditions >= 0', true, 'Require returned energy-condition expressions to stay non-negative.'],
            ['stability', '0 <= sound speed <= 1', true, 'Require plotted stability curves to remain causal.'],
            ['tov_residual', 'Small TOV residual', false, 'Require |F_h + F_g + F_a| below the tolerance.'],
        ];
        constraintEl.innerHTML = constraints.map(([key, label, checked, desc]) => `
            <label class="scan-constraint-choice" title="${escapeHtml(desc)}">
                <input type="checkbox" data-scan-constraint="${key}" ${checked ? 'checked' : ''}>
                <span>${escapeHtml(label)}</span>
            </label>
        `).join('') + `
            <div class="scan-tolerance-row">
                <label>Tolerance<input type="number" step="any" data-scan-option="tolerance" value="1e-8"></label>
                <label>TOV tolerance<input type="number" step="any" data-scan-option="tov_tolerance" value="1"></label>
                <label>Finite fraction<input type="number" step="any" min="0" max="1" data-scan-option="min_finite_fraction" value="0.95"></label>
            </div>
        `;
        show(controls);
    };

    runBtn.onclick = async () => {
        const domainEl = $('symbolic-plot-domain-controls');
        if (!domainEl) return;
        const domainValues = readNamedInputs(domainEl);
        const parameterRanges = {};
        rangeEl.querySelectorAll('[data-scan-param]').forEach((row) => {
            const name = row.dataset.scanParam;
            parameterRanges[name] = {};
            row.querySelectorAll('[data-scan-field]').forEach((input) => {
                parameterRanges[name][input.dataset.scanField] = input.value.trim();
            });
        });
        const constraints = {};
        constraintEl.querySelectorAll('[data-scan-constraint]').forEach((input) => {
            constraints[input.dataset.scanConstraint] = input.checked;
        });
        constraintEl.querySelectorAll('[data-scan-option]').forEach((input) => {
            constraints[input.dataset.scanOption] = input.value.trim();
        });

        const payload = {
            variable: spec.variable,
            groups: spec.groups || {},
            parameter_ranges: parameterRanges,
            constraints,
            domain: {
                min: domainValues[`${spec.variable}_min`] || 0.1,
                max: domainValues[`${spec.variable}_max`] || 10,
                points: domainValues.points || 160,
            },
        };
        if (status) status.textContent = 'Scanning parameter grid...';
        try {
            const cacheKey = JSON.stringify(payload);
            let data = parameterScanCache.get(cacheKey);
            const usedCache = Boolean(data);
            if (!data) {
                const resp = await fetch('/api/plot/scan', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(payload),
                });
                data = await resp.json();
                if (!resp.ok) throw new Error(data.error || 'Parameter scan failed');
                parameterScanCache.set(cacheKey, data);
            }
            renderParameterScanResults(data);
            if (status) {
                status.textContent = `Accepted ${data.accepted || 0}/${data.total || 0} samples.${usedCache ? ' Loaded from scan cache.' : ''}`;
            }
        } catch (err) {
            if (status) status.textContent = `Error: ${err.message}`;
        }
    };
}

function renderParameterScanResults(data) {
    const resultsEl = $('parameter-scan-results');
    if (!resultsEl) return;
    const ranges = data.accepted_ranges || {};
    const best = data.best || null;
    const rangeRows = Object.entries(ranges).map(([name, range]) => `
        <tr>
            <td>${escapeHtml(name)}</td>
            <td>${range.min === null || range.min === undefined ? 'none' : formatScanNumber(range.min)}</td>
            <td>${range.max === null || range.max === undefined ? 'none' : formatScanNumber(range.max)}</td>
        </tr>
    `).join('');
    const bestParams = best?.parameters
        ? Object.entries(best.parameters).map(([k, v]) => `${escapeHtml(k)}=${formatScanNumber(v)}`).join(', ')
        : 'none';
    const topSamples = (data.top_samples || []).slice(0, 8);
    const topRows = topSamples.map((sample, idx) => `
        <tr>
            <td>${sample.passed ? 'pass' : 'fail'}</td>
            <td>${formatScanNumber(sample.score)}</td>
            <td>${Object.entries(sample.parameters || {}).map(([k, v]) => `${escapeHtml(k)}=${formatScanNumber(v)}`).join(', ')}</td>
            <td><button class="copy-btn scan-plot-btn" data-scan-sample="${idx}">Plot</button></td>
        </tr>
    `).join('');
    resultsEl.innerHTML = `
        <div class="scan-summary">
            <div><strong>${data.accepted || 0}</strong><span>accepted</span></div>
            <div><strong>${data.total || 0}</strong><span>tested</span></div>
            <div><strong>${formatScanNumber(100 * (data.acceptance_fraction || 0))}%</strong><span>acceptance</span></div>
        </div>
        <div class="scan-best">Best sample: <span>${bestParams}</span> · score ${best ? formatScanNumber(best.score) : 'n/a'}</div>
        <h4 class="mini-hdr">Accepted Ranges</h4>
        <table class="scan-table"><thead><tr><th>parameter</th><th>min</th><th>max</th></tr></thead><tbody>${rangeRows}</tbody></table>
        ${buildScanHeatmapSvg(data.heatmap)}
        <h4 class="mini-hdr">Top Samples</h4>
        <table class="scan-table"><thead><tr><th>status</th><th>score</th><th>parameters</th><th>plot</th></tr></thead><tbody>${topRows}</tbody></table>
    `;
    const bestRow = resultsEl.querySelector('.scan-best');
    if (bestRow && best?.parameters) {
        const button = document.createElement('button');
        button.className = 'copy-btn scan-plot-best-btn';
        button.style.marginLeft = '10px';
        button.textContent = 'Plot best point';
        button.addEventListener('click', () => applyScanSampleToPlot(best.parameters, true));
        bestRow.appendChild(button);
    }
    resultsEl.querySelectorAll('[data-scan-sample]').forEach((btn) => {
        btn.addEventListener('click', () => {
            const sample = topSamples[Number(btn.dataset.scanSample)];
            if (sample?.parameters) applyScanSampleToPlot(sample.parameters, true);
        });
    });
    show(resultsEl);
}

function applyScanSampleToPlot(parameters, runAfter = false) {
    const paramEl = $('symbolic-plot-param-controls');
    const runBtn = $('run-symbolic-plot-btn');
    const status = $('symbolic-plot-status');
    if (!paramEl) return;
    Object.entries(parameters || {}).forEach(([name, value]) => {
        const input = paramEl.querySelector(`input[data-numeric-name="${cssEscape(String(name))}"]`);
        if (input) input.value = String(value);
    });
    if (status) status.textContent = 'Loaded scan sample into plot parameters.';
    if (runAfter && runBtn) runBtn.click();
}

function cssEscape(value) {
    if (window.CSS && typeof window.CSS.escape === 'function') return window.CSS.escape(value);
    return value.replace(/["\\]/g, '\\$&');
}

function formatScanNumber(value) {
    const num = Number(value);
    if (!Number.isFinite(num)) return 'n/a';
    if (Math.abs(num) >= 1000 || (Math.abs(num) > 0 && Math.abs(num) < 0.001)) return num.toExponential(2);
    return Number(num.toPrecision(4)).toString();
}

function formatHeatmapTick(value) {
    const num = Number(value);
    if (!Number.isFinite(num)) return '';
    if (num === 0) return '0';
    const abs = Math.abs(num);
    if (abs >= 100 || abs < 0.01) return num.toExponential(1);
    return Number(num.toPrecision(3)).toString();
}

function sparseTickIndices(length, maxLabels = 6) {
    if (length <= maxLabels) return Array.from({ length }, (_, i) => i);
    const out = new Set([0, length - 1]);
    const denom = maxLabels - 1;
    for (let i = 1; i < denom; i++) {
        out.add(Math.round((i * (length - 1)) / denom));
    }
    return Array.from(out).sort((a, b) => a - b);
}

function buildScanHeatmapSvg(heatmap) {
    if (!heatmap || !Array.isArray(heatmap.cells) || !heatmap.cells.length) return '';
    const xValues = heatmap.x_values || [];
    const yValues = heatmap.y_values || [];
    if (!xValues.length || !yValues.length) return '';
    const cell = Math.max(30, Math.min(42, Math.floor(420 / Math.max(xValues.length, yValues.length))));
    const padL = 96;
    const padT = 36;
    const padB = 86;
    const padR = 34;
    const width = padL + xValues.length * cell + padR;
    const height = padT + yValues.length * cell + padB;
    const scoreColor = (score, passed) => {
        const s = Math.max(0, Math.min(100, Number(score) || 0)) / 100;
        if (!passed) {
            const v = Math.round(235 - s * 70);
            return `rgb(${v},${Math.round(v * 0.86)},${Math.round(v * 0.86)})`;
        }
        const r = Math.round(230 - s * 150);
        const g = Math.round(236 - s * 36);
        const b = Math.round(220 - s * 150);
        return `rgb(${r},${g},${b})`;
    };
    const cells = heatmap.cells.map((item) => {
        const xi = xValues.findIndex((v) => Number(v) === Number(item.x));
        const yi = yValues.findIndex((v) => Number(v) === Number(item.y));
        if (xi < 0 || yi < 0) return '';
        const x = padL + xi * cell;
        const y = padT + (yValues.length - 1 - yi) * cell;
        return `<rect x="${x}" y="${y}" width="${cell - 2}" height="${cell - 2}" fill="${scoreColor(item.score, item.passed)}"><title>${heatmap.x_param}=${formatScanNumber(item.x)}, ${heatmap.y_param}=${formatScanNumber(item.y)}, score=${formatScanNumber(item.score)}</title></rect>`;
    }).join('');
    const xTickSet = new Set(sparseTickIndices(xValues.length, 6));
    const yTickSet = new Set(sparseTickIndices(yValues.length, 7));
    const xLabels = xValues.map((v, i) => {
        if (!xTickSet.has(i)) return '';
        const x = padL + i * cell + cell / 2;
        return `
            <line x1="${x}" y1="${padT + yValues.length * cell}" x2="${x}" y2="${padT + yValues.length * cell + 6}" class="scan-heatmap-tick"/>
            <text x="${x}" y="${height - 38}" text-anchor="middle" class="scan-heatmap-label scan-heatmap-x-tick">${formatHeatmapTick(v)}</text>
        `;
    }).join('');
    const yLabels = yValues.map((v, i) => {
        if (!yTickSet.has(i)) return '';
        const y = padT + (yValues.length - 1 - i) * cell + cell / 2;
        return `
            <line x1="${padL - 6}" y1="${y}" x2="${padL}" y2="${y}" class="scan-heatmap-tick"/>
            <text x="${padL - 12}" y="${y + 4}" text-anchor="end" class="scan-heatmap-label">${formatHeatmapTick(v)}</text>
        `;
    }).join('');
    return `
        <h4 class="mini-hdr">Score Heatmap</h4>
        <svg class="scan-heatmap" viewBox="0 0 ${width} ${height}" role="img">
            <text x="${padL}" y="16" class="scan-heatmap-title">${escapeHtml(heatmap.y_param)} vs ${escapeHtml(heatmap.x_param)}</text>
            ${cells}
            ${xLabels}
            ${yLabels}
            <text x="${padL + (xValues.length * cell) / 2}" y="${height - 10}" text-anchor="middle" class="scan-heatmap-label scan-heatmap-axis-title">${escapeHtml(heatmap.x_param)}</text>
            <text x="14" y="${padT + (yValues.length * cell) / 2}" class="scan-heatmap-label scan-heatmap-y">${escapeHtml(heatmap.y_param)}</text>
        </svg>
    `;
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
        renderNumericInput(domainEl, 'y_min', '', 'auto');
        renderNumericInput(domainEl, 'y_max', '', 'auto');
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
            renderNumericPlot(data, readPlotAxisOptions(domainEl));
            setPlotExportButtonsEnabled(true);
        } catch (err) {
            if (status) status.textContent = `Error: ${err.message}`;
        }
    };
}

function renderNumericPlot(data, axisOptions = {}) {
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
    renderNumericPlotInto('numeric-plot-container', 'Matter variables', x, data.solutions || {}, data.variable || 'x', data.warnings, axisOptions);

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
        data.variable || 'x',
        [],
        axisOptions
    );
    renderNumericPlotInto(
        'numeric-eos-plot-container',
        'Numerical equation of state',
        x,
        labelSeries(
            pickSeries(diagnostics, ['omega', 'omega_r', 'omega_t', 'omega_eff']),
            { omega: 'omega', omega_r: 'omega_r', omega_t: 'omega_t', omega_eff: 'omega_eff' }
        ),
        data.variable || 'x',
        [],
        axisOptions
    );
    renderNumericPlotInto(
        'numeric-stability-plot-container',
        'Numerical stability',
        x,
        labelSeries(
            pickSeries(diagnostics, ['cs2', 'cs2_r', 'cs2_t']),
            { cs2: 'c_s^2', cs2_r: 'c_s^2_r', cs2_t: 'c_s^2_t' }
        ),
        data.variable || 'x',
        [],
        axisOptions
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
        data.variable || 'x',
        [],
        axisOptions
    );
}

function renderSymbolicPlotData(data, axisOptions = {}) {
    const x = data.x || [];
    if (!x.length) return;
    const groups = data.groups || {};
    clearNumericTabPlots();
    renderNumericPlotInto('symbolic-matter-plot-container', 'Matter variables', x, groups.matter || {}, data.variable || 'x', data.warnings, axisOptions);
    renderEnergyConditionPlot(
        'Energy conditions',
        x,
        groups.energy_conditions || {},
        data.variable || 'x',
        [],
        axisOptions
    );
    renderNumericPlotInto(
        'numeric-eos-plot-container',
        'Equation of state',
        x,
        groups.eos || {},
        data.variable || 'x',
        [],
        axisOptions
    );
    renderNumericPlotInto(
        'numeric-stability-plot-container',
        'Stability',
        x,
        labelSeries(groups.stability || {}, { cs2: 'c_s^2', cs2_r: 'c_s^2_r', cs2_t: 'c_s^2_t' }),
        data.variable || 'x',
        [],
        axisOptions
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
        data.variable || 'x',
        [],
        axisOptions
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

function renderEnergyConditionPlot(title, x, series, variableLabel, warnings = [], axisOptions = {}) {
    latestEnergyPlot = { title, x, series: series || {}, variableLabel, warnings, axisOptions };
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
        latestEnergyPlot.variableLabel,
        latestEnergyPlot.warnings || [],
        latestEnergyPlot.axisOptions || {}
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

function renderNumericPlotInto(containerId, title, x, series, variableLabel, warnings = [], axisOptions = {}) {
    const container = $(containerId);
    if (!container) return;
    const svg = buildNumericPlotSvg(title, x, series, variableLabel, axisOptions);
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

function buildNumericPlotSvg(title, x, series, variableLabel, axisOptions = {}) {
    const names = Object.keys(series || {}).filter((name) =>
        (series[name] || []).some((v) => typeof v === 'number' && Number.isFinite(v))
    );
    if (!names.length) return '';

    const width = 980;
    const height = 500;
    const padL = 94;
    const padT = 54;
    const padB = 76;
    const charWidth = 9.2;
    const legendW = Math.min(250, Math.max(130, 72 + Math.max(...names.map((name) => name.length)) * charWidth));
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
    const manualMinY = axisOptions && axisOptions.y_min !== '' ? Number(axisOptions.y_min) : NaN;
    const manualMaxY = axisOptions && axisOptions.y_max !== '' ? Number(axisOptions.y_max) : NaN;
    let minY = Number.isFinite(manualMinY) ? manualMinY : percentile(sortedY, 0.02);
    let maxY = Number.isFinite(manualMaxY) ? manualMaxY : percentile(sortedY, 0.98);
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
    if (Number.isFinite(manualMinY) && Number.isFinite(manualMaxY) && manualMinY > manualMaxY) {
        minY = manualMaxY;
        maxY = manualMinY;
    }
    const hasManualY = Number.isFinite(manualMinY) || Number.isFinite(manualMaxY);
    const padY = Math.max(1e-9, (maxY - minY) * 0.06);
    if (!hasManualY) {
        minY -= padY;
        maxY += padY;
    }
    if (minY === maxY) {
        minY -= 1;
        maxY += 1;
    }
    const sx = (v) => padL + ((v - minX) / (maxX - minX || 1)) * plotW;
    const sy = (v) => padT + plotH - ((v - minY) / (maxY - minY || 1)) * plotH;
    const fmt = (v, step = null) => {
        if (!Number.isFinite(v)) return '';
        if (Math.abs(v) >= 1000 || (Math.abs(v) > 0 && Math.abs(v) < 0.01)) return v.toExponential(2);
        if (step && Number.isFinite(step) && step > 0) {
            const decimals = Math.min(5, Math.max(0, Math.ceil(-Math.log10(step)) + 1));
            return Number(v.toFixed(decimals)).toString();
        }
        return Number(v.toPrecision(5)).toString();
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
        <text x="${sx(v).toFixed(2)}" y="${padT + plotH + 26}" class="plot-tick-label" text-anchor="middle">${fmt(v, xStep)}</text>
    `).join('');
    const yTickSvg = yTicks.map((v) => `
        <line x1="${padL - 5}" y1="${sy(v).toFixed(2)}" x2="${padL}" y2="${sy(v).toFixed(2)}" class="plot-tick"/>
        <text x="${padL - 12}" y="${(sy(v) + 4).toFixed(2)}" class="plot-tick-label" text-anchor="end">${fmt(v, yStep)}</text>
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
        <svg viewBox="0 0 ${width} ${height}" class="numeric-plot" role="img">
            <style>
                .plot-frame{fill:#fff;stroke:#000;stroke-width:1}
                .plot-tick{stroke:#000;stroke-width:1}
                .plot-tick-label{fill:#000;font:14px 'Cascadia Mono','Consolas',monospace}
                .plot-zero-line{stroke:#000;stroke-width:1.2}
                .plot-label{fill:#000;font:italic 19px 'Cambria Math',Georgia,serif}
                .plot-title{fill:#000;font:700 18px 'Cascadia Mono','Consolas',monospace}
                .plot-legend-box{fill:#fff;stroke:#000;stroke-width:1.2}
                .plot-legend-label{fill:#000;font:15px 'Cascadia Mono','Consolas',monospace}
            </style>
            <text x="${width / 2}" y="28" class="plot-title" text-anchor="middle">${escapeHtml(title)}</text>
            <rect x="${padL}" y="${padT}" width="${plotW}" height="${plotH}" class="plot-frame"/>
            ${zeroLine}
            ${xTickSvg}
            ${yTickSvg}
            <text x="${padL + plotW / 2}" y="${height - 20}" class="plot-label" text-anchor="middle">${escapeHtml(variableLabel)}</text>
            <text x="30" y="${padT + plotH / 2}" class="plot-label" text-anchor="middle" transform="rotate(-90 30 ${padT + plotH / 2})">Values</text>
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

