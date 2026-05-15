/**
 * Modified Gravity Studio — Frontend
 * Handles: all 4 theories, supported backgrounds, anisotropic gating,
 *          multi-ansatz fields, Lm choice, complete result display.
 */

'use strict';

// ── State ─────────────────────────────────────────────────────────────────────
let currentTaskId = null;
let eventSource   = null;
let currentResults = null;
let numericSolveCache = new Map();
let lastNumericPlot = null;

// Registry data loaded at boot
let backgroundRegistry = [];
let theoryRegistry     = [];

// Lm compatibility cache for fRTLm theory
let _lmCompatCache = {};

// Scalar name map cache for fRTLm theory
let _scalarMapCache = {};

// ── Dynamic Math Rendering Helper ─────────────────────────────────────────────
function renderDynamicMath(element) {
    if (!element) return;
    if (typeof renderMathInElement === 'undefined') return;
    renderMathInElement(element, {
        delimiters: [
            { left: '\\(',  right: '\\)',  display: false },
            { left: '\\[',  right: '\\]',  display: true  },
        ],
        throwOnError: false,
        ignoredTags: ['script', 'noscript', 'style', 'textarea', 'pre'],
    });
}

// ── Theory field equation LaTeX ───────────────────────────────────────────────
const THEORY_FIELD_EQ = {
    fR:    'f_R R_{\\mu\\nu} - \\tfrac{1}{2}f g_{\\mu\\nu} - \\nabla_\\mu\\nabla_\\nu f_R + g_{\\mu\\nu}\\Box f_R = 8\\pi T_{\\mu\\nu}',
    fT:    'e^{-1} e^i_{\\ \\mu}\\partial_\\rho(e\\,e_i^{\\ \\alpha} S_\\alpha^{\\ \\nu\\rho}) f_T + S^{\\nu\\lambda}_{\\ \\ \\alpha} T^\\alpha_{\\ \\lambda\\mu} f_T - S^{\\nu\\rho}_{\\ \\ \\mu}\\partial_\\rho T\\,f_{TT} + \\tfrac{1}{4}\\delta^\\nu_\\mu f = 4\\pi T^\\nu_{\\ \\mu}',
    fTB:   '2e\\,\\Box f_B\\,\\delta^\\lambda_\\nu - 2e\\nabla^\\lambda\\nabla_\\nu f_B + eB f_B \\delta^\\lambda_\\nu + 4e(\\partial_\\mu f_B + \\partial_\\mu f_T)S_\\nu^{\\ \\mu\\lambda} + 4e^a_{\\ \\nu}\\partial_\\mu(e S_a^{\\ \\mu\\lambda})f_T - 4e f_T T^\\sigma_{\\ \\mu\\nu}S_\\sigma^{\\ \\lambda\\mu} - ef\\delta^\\lambda_\\nu = 16\\pi e T^\\lambda_{\\ \\nu}',
    fRTLm: 'f_R R_{\\mu\\nu} + g_{\\mu\\nu}\\Box f_R - \\nabla_\\mu\\nabla_\\nu f_R - \\tfrac{1}{2}f g_{\\mu\\nu} - \\tfrac{1}{2}(f_L + 2f_T)(T_{\\mu\\nu} - \\mathcal{L}_m g_{\\mu\\nu}) = 8\\pi T_{\\mu\\nu}',
    fQ:    '-\\tfrac{2}{\\sqrt{-g}}\\nabla_\\alpha(\\sqrt{-g}\\,f_Q P^\\alpha{}_{\\mu\\nu}) - \\tfrac{1}{2}f g_{\\mu\\nu} + f_Q(P_{\\mu\\alpha\\beta}Q_\\nu{}^{\\alpha\\beta} - 2Q^{\\alpha\\beta}{}_{\\mu} P_{\\alpha\\beta\\nu}) = 8\\pi T_{\\mu\\nu}',
    fQC:   'f(Q,C),\\quad C=\\mathcal{R}-Q,\\quad F_R=f_C,\\quad F_Q=f_Q-f_C',
};

const THEORY_MODEL_LABEL = {
    fR:    'f(R) =',
    fT:    'f(T) =',
    fTB:   'f(T,B) =',
    fRTLm: 'f(R,T,Lm) =',
    fQ:    'f(Q) =',
    fQC:   'f(Q,C) =',
};

const SET_LATEX = {
    perfect_fluid: 'T_{\\mu\\nu} = (\\rho+p)u_\\mu u_\\nu + p g_{\\mu\\nu}',
    anisotropic:   'T^{\\mu\\nu} = (\\rho+P_t)u^\\mu u^\\nu + P_t g^{\\mu\\nu} + (P_r-P_t)x^\\mu x^\\nu',
    dust:          'T_{\\mu\\nu} = \\rho\\,u_\\mu u_\\nu',
    radiation:     'T_{\\mu\\nu} = \\tfrac{\\rho}{3}g_{\\mu\\nu}',
    vacuum:        'T_{\\mu\\nu} = -\\Lambda g_{\\mu\\nu}',
};

const BACKGROUND_ANSATZ = {
    FRW:             [{ fn: 'a',   category: 'cosmological' }],
    Bianchi_I:       [{ fn: 'A',   category: 'cosmological' }, { fn: 'B', category: 'cosmological' }],
    Bianchi_III:     [{ fn: 'A',   category: 'cosmological' }, { fn: 'B', category: 'cosmological' }],
    Kantowski_Sachs: [{ fn: 'A',   category: 'cosmological' }, { fn: 'B', category: 'cosmological' }],
    SS_wormhole:     [{ fn: 'b',   category: 'wormhole_b'  }, { fn: 'Phi', category: 'wormhole_Phi' }],
    SS_blackhole:    [{ fn: 'nu_bh',   category: 'blackhole_nu' }, { fn: 'lam_bh', category: 'blackhole_lam' }],
};

const ANSATZ_PRESETS = {
    cosmological: [
        { id: 'power_law', name: 'Power-law',         expr: 'a0*t**h'              },
        { id: 'de_sitter', name: 'de Sitter',         expr: 'a0*exp(H0*t)'         },
        { id: 'sinh',      name: 'Hyperbolic sine',   expr: 'a0*sinh(H0*t)**h'     },
        { id: 'pow_exp',   name: 'Power-exponential', expr: 'a0*t**h*exp(beta*t)'  },
    ],
    wormhole_b: [
        { id: 'exp',   name: 'Exponential',   expr: 'r/exp(r - r0)'     },
        { id: 'inv',   name: 'Inverse Power', expr: 'r0**2/r'           },
        { id: 'pow',   name: 'Power-law',     expr: 'r0*(r0/r)**(h-1)'  },
    ],
    wormhole_Phi: [
        { id: 'zero',  name: 'Zero Tidal Force', expr: '0'     },
        { id: 'const', name: 'Constant',         expr: 'Phi0'  },
    ],
    blackhole_nu: [
        { id: 'schwarzschild',      name: 'Schwarzschild',      expr: '1 - 2*M/r'                  },
        { id: 'reissner_nordstrom', name: 'Reissner-Nordstrom', expr: '1 - 2*M/r + Q**2/r**2'     },
    ],
    blackhole_lam: [
        { id: 'schwarzschild',      name: 'Schwarzschild',      expr: '1/(1 - 2*M/r)'              },
        { id: 'reissner_nordstrom', name: 'Reissner-Nordstrom', expr: '1/(1 - 2*M/r + Q**2/r**2)' },
    ],
};

const BACKGROUND_ANSATZ_GROUPS = {
    Bianchi_I: [
        { id: 'power_law', name: 'Directional power-law', functions: { A: 'A0*t**pA', B: 'B0*t**pB' }, params: ['A0', 'B0', 'pA', 'pB'] },
        { id: 'de_sitter', name: 'Directional de Sitter', functions: { A: 'A0*exp(Hx*t)', B: 'B0*exp(Hy*t)' }, params: ['A0', 'B0', 'Hx', 'Hy'] },
        { id: 'sinh', name: 'Directional hyperbolic sine', functions: { A: 'A0*sinh(H0*t)**pA', B: 'B0*sinh(H0*t)**pB' }, params: ['A0', 'B0', 'H0', 'pA', 'pB'] },
        { id: 'power_exp', name: 'Directional power-exponential', functions: { A: 'A0*t**pA*exp(Hx*t)', B: 'B0*t**pB*exp(Hy*t)' }, params: ['A0', 'B0', 'pA', 'pB', 'Hx', 'Hy'] },
    ],
    Bianchi_III: [
        { id: 'power_law', name: 'Directional power-law', functions: { A: 'A0*t**pA', B: 'B0*t**pB' }, params: ['A0', 'B0', 'pA', 'pB'] },
        { id: 'de_sitter', name: 'Directional de Sitter', functions: { A: 'A0*exp(Hx*t)', B: 'B0*exp(Hy*t)' }, params: ['A0', 'B0', 'Hx', 'Hy'] },
        { id: 'sinh', name: 'Directional hyperbolic sine', functions: { A: 'A0*sinh(H0*t)**pA', B: 'B0*sinh(H0*t)**pB' }, params: ['A0', 'B0', 'H0', 'pA', 'pB'] },
        { id: 'power_exp', name: 'Directional power-exponential', functions: { A: 'A0*t**pA*exp(Hx*t)', B: 'B0*t**pB*exp(Hy*t)' }, params: ['A0', 'B0', 'pA', 'pB', 'Hx', 'Hy'] },
    ],
    Kantowski_Sachs: [
        { id: 'power_law', name: 'Directional power-law', functions: { A: 'A0*t**pA', B: 'B0*t**pB' }, params: ['A0', 'B0', 'pA', 'pB'] },
        { id: 'de_sitter', name: 'Directional de Sitter', functions: { A: 'A0*exp(Hx*t)', B: 'B0*exp(Hy*t)' }, params: ['A0', 'B0', 'Hx', 'Hy'] },
        { id: 'sinh', name: 'Directional hyperbolic sine', functions: { A: 'A0*sinh(H0*t)**pA', B: 'B0*sinh(H0*t)**pB' }, params: ['A0', 'B0', 'H0', 'pA', 'pB'] },
        { id: 'power_exp', name: 'Directional power-exponential', functions: { A: 'A0*t**pA*exp(Hx*t)', B: 'B0*t**pB*exp(Hy*t)' }, params: ['A0', 'B0', 'pA', 'pB', 'Hx', 'Hy'] },
    ],
    SS_blackhole: [
        { id: 'schwarzschild', name: 'Schwarzschild', functions: { nu_bh: '1 - 2*M/r', lam_bh: '1/(1 - 2*M/r)' } },
        { id: 'reissner_nordstrom', name: 'Reissner-Nordstrom', functions: { nu_bh: '1 - 2*M/r + Q**2/r**2', lam_bh: '1/(1 - 2*M/r + Q**2/r**2)' } },
    ],
};

const MODEL_PRESETS = {
    fR:    [
        { id: 'GR',          expr: 'R',                                name: 'General Relativity (f = R)'      },
        { id: 'star',        expr: 'R + alpha*R**2',                  name: 'Starobinsky (f = R + αR²)'       },
        { id: 'exp',         expr: 'R + alpha*(1 - exp(-R/beta))',    name: 'Exponential'                      },
        { id: 'log',         expr: 'R + alpha*log(R)',                name: 'Logarithmic'                     },
        { id: 'pow_corr',    expr: 'R + alpha*R**n',                  name: 'R + αRⁿ'                         },
        { id: 'custom',      expr: '',                                name: 'Custom…'                         },
    ],
    fT:    [
        { id: 'TEGR',   expr: 'T',                          name: 'TEGR (GR limit, f = T)'          },
        { id: 'pow',    expr: 'T + alpha*T**n',             name: 'Power-law'                        },
        { id: 'exp',    expr: 'T*(1 + alpha*exp(beta*T))',  name: 'Exponential'                      },
        { id: 'log',    expr: 'T + alpha*log(T/T0)',        name: 'Logarithmic'                      },
        { id: 'custom', expr: '',                           name: 'Custom…'                          },
    ],
    fTB:   [
        { id: 'GR',     expr: 'T - B',                         name: 'GR Recovery (f = T−B)'    },
        { id: 'fR_lim', expr: '-(T - B) + alpha*(T - B)**2',   name: 'f(R) Limit'               },
        { id: 'lin',    expr: 'alpha*T + beta*B',              name: 'Linear Combination'        },
        { id: 'pow',    expr: 'T + alpha*B**n',                name: 'Power-law Mixed'           },
        { id: 'exp_T',  expr: 'T*exp(alpha*T) - B',           name: 'Exponential-T'             },
        { id: 'custom', expr: '',                              name: 'Custom…'                   },
    ],
    fRTLm: [
        { id: 'GR_mat', expr: 'R + lam*L',                         name: 'GR + Matter Coupling'          },
        { id: 'star_T', expr: 'R + alpha*R**2 + beta*T_scalar',    name: 'Starobinsky + T-coupling'      },
        { id: 'full',   expr: 'R + beta*T_scalar + lam*L',             name: 'Full Linear'             },
        { id: 'mixed_TL', expr: 'R + gamma*T_scalar*L',             name: 'Mixed T-L Coupling'            },
        { id: 'mixed_full', expr: 'R + alpha*R**2 + beta*T_scalar + gamma*L + delta*T_scalar*L', name: 'Full Mixed T-L' },
        { id: 'nonmin', expr: 'R*(1 + gamma*L)',                    name: 'Non-minimal Matter'            },
        { id: 'custom', expr: '',                                   name: 'Custom…'                       },
    ],
    fQ: [
        { id: 'STEGR',  expr: 'Q',                                 name: 'STEGR (GR limit, f = Q)'       },
        { id: 'pow',    expr: 'Q + alpha*Q**n',                    name: 'Power-law'                      },
        { id: 'exp',    expr: 'Q + alpha*(1 - exp(-Q/beta))',       name: 'Exponential'                   },
        { id: 'log',    expr: 'Q + alpha*log(Q/Q0)',               name: 'Logarithmic'                    },
        { id: 'sqrt',   expr: 'Q + alpha*sqrt(Q)',                  name: 'Square-root'                   },
        { id: 'quad',   expr: 'Q + alpha*Q**2',                    name: 'Quadratic'                      },
        { id: 'custom', expr: '',                                   name: 'Custom…'                        },
    ],
    fQC: [
        { id: 'STEGR',  expr: 'Q',                                  name: 'STEGR (f = Q)'                  },
        { id: 'linC',   expr: 'Q + alpha*C',                        name: 'Linear boundary'                },
        { id: 'curv',   expr: 'Q + C',                              name: 'Curvature equivalent'           },
        { id: 'quad',   expr: 'Q + alpha*Q**2 + beta*C',            name: 'Quadratic Q + C'                },
        { id: 'logC',   expr: 'Q + alpha*C*log(C/C0)',              name: 'Boundary logarithmic'           },
        { id: 'mixed',  expr: 'Q + alpha*Q*C',                      name: 'Mixed Q-C coupling'             },
        { id: 'custom', expr: '',                                   name: 'Custom'                         },
    ],
};

const FRW_BACKGROUNDS = new Set(['FRW']);
const COSMOLOGY_BACKGROUNDS = new Set([
    'FRW',
    'Bianchi_I',
    'Bianchi_III',
    'Kantowski_Sachs',
]);
const TOV_BACKGROUNDS = new Set(['SS_wormhole', 'SS_blackhole']);

const MATH_FUNCTION_TOKENS = new Set([
    'exp', 'log', 'sin', 'cos', 'tan', 'asin', 'acos', 'atan',
    'sinh', 'cosh', 'tanh', 'sqrt', 'Abs', 'pi', 'E'
]);

const THEORY_BASE_SYMBOLS = {
    fR: new Set(['R']),
    fT: new Set(['T', 'T0']),
    fTB: new Set(['T', 'B']),
    fRTLm: new Set(['R', 'T_scalar', 'T_mat', 'L', 'Lm']),
    fQ: new Set(['Q', 'Q0']),
    fQC: new Set(['Q', 'C', 'C0']),
};

function collectExprParameters(expr, reserved = new Set()) {
    const tokens = new Set((expr || '').match(/[a-zA-Z_][a-zA-Z0-9_]*/g) || []);
    const names = [];
    tokens.forEach((tok) => {
        if (reserved.has(tok) || MATH_FUNCTION_TOKENS.has(tok)) return;
        if (/^[A-Z]$/.test(tok)) return;
        names.push(tok);
    });
    return names.sort((a, b) => a.localeCompare(b, undefined, { numeric: true }));
}

function preserveParamValues(containerId) {
    const container = $(containerId);
    const values = {};
    if (!container) return values;
    container.querySelectorAll('input[data-param-name]').forEach((input) => {
        const value = input.value.trim();
        if (value !== '') values[input.dataset.paramName] = value;
    });
    return values;
}

function renderParamInputs(containerId, paramNames, existingValues = {}, helpText = '') {
    const container = $(containerId);
    if (!container) return;
    container.innerHTML = '';
    if (!paramNames.length) {
        hide(container);
        return;
    }
    paramNames.forEach((name) => {
        const wrap = document.createElement('div');
        wrap.className = 'param-input';

        const label = document.createElement('label');
        label.setAttribute('for', `${containerId}-${name}`);
        label.textContent = name;
        wrap.appendChild(label);

        const input = document.createElement('input');
        input.type = 'text';
        input.className = 'input-full';
        input.id = `${containerId}-${name}`;
        input.dataset.paramName = name;
        input.placeholder = 'optional';
        input.value = existingValues[name] || '';
        wrap.appendChild(input);

        container.appendChild(wrap);
    });
    if (helpText) {
        const help = document.createElement('div');
        help.className = 'param-help';
        help.style.gridColumn = '1 / -1';
        help.textContent = helpText;
        container.appendChild(help);
    }
    show(container);
}

function updateModelParamInputs() {
    const theory = getTheory();
    const modelSel = $('model-preset');
    const customInput = $('custom-model');
    const expr = (modelSel && modelSel.value === '__custom__')
        ? (customInput ? customInput.value.trim() : '')
        : (modelSel ? modelSel.value : '');
    const reserved = new Set([...(THEORY_BASE_SYMBOLS[theory] || new Set()), 'alpha', 'beta', 'gamma', 'delta', 'lam']);
    reserved.delete('Q0');
    reserved.delete('T0');
    reserved.delete('C0');
    const current = preserveParamValues('model-params-container');
    const params = collectExprParameters(expr, reserved);
    renderParamInputs('model-params-container', params, current, 'Optional numeric values are substituted before solving.');
    const hint = $('model-params-hint');
    if (hint) {
        hint.textContent = params.length
            ? `Detected model parameters: ${params.join(', ')}`
            : 'No extra model parameters detected.';
    }
}

function updateAnsatzParamInputs() {
    const bgId = $('background').value;
    const fields = BACKGROUND_ANSATZ[bgId] || [{ fn: 'a', category: 'cosmological' }];
    const grouped = BACKGROUND_ANSATZ_GROUPS[bgId] || [];
    const reserved = new Set(['t', 'r', 'theta', 'phi', 'x', 'y', 'z', 'pi', 'E']);
    fields.forEach(({ fn }) => reserved.add(fn));

    const current = preserveParamValues('ansatz-params-container');
    const params = new Set();

    if (grouped.length) {
        const groupSel = $('ansatz-group-sel');
        const isCustom = groupSel && groupSel.value === '__custom__';
        if (isCustom) {
            fields.forEach(({ fn }) => {
                const custom = $(`ansatz-custom-${fn}`);
                const expr = custom ? custom.value.trim() : '';
                collectExprParameters(expr, reserved).forEach((name) => params.add(name));
            });
        } else {
            const preset = grouped.find((p) => p.id === (groupSel ? groupSel.value : grouped[0].id)) || grouped[0];
            if (Array.isArray(preset.params)) {
                preset.params.forEach((name) => params.add(name));
            } else {
                Object.values(preset.functions || {}).forEach((expr) => {
                    collectExprParameters(expr, reserved).forEach((name) => params.add(name));
                });
            }
        }
    } else {
        fields.forEach(({ fn }) => {
            const sel = $(`ansatz-sel-${fn}`);
            const custom = $(`ansatz-custom-${fn}`);
            const expr = (sel && sel.value === '__custom__')
                ? (custom ? custom.value.trim() : '')
                : (sel ? sel.value : '');
            collectExprParameters(expr, reserved).forEach((name) => params.add(name));
        });
    }

    renderParamInputs(
        'ansatz-params-container',
        Array.from(params).sort((a, b) => a.localeCompare(b, undefined, { numeric: true })),
        current,
        'Optional ansatz constants are substituted directly into the metric functions.'
    );
}

function collectParamValues(containerId) {
    const container = $(containerId);
    const result = {};
    if (!container) return result;
    container.querySelectorAll('input[data-param-name]').forEach((input) => {
        const value = input.value.trim();
        if (value !== '') result[input.dataset.paramName] = value;
    });
    return result;
}

// ── Utilities ─────────────────────────────────────────────────────────────────
const $ = id => document.getElementById(id);
const show  = el => el && el.classList.remove('hidden');
const hide  = el => el && el.classList.add('hidden');

function escapeHtml(value) {
    return String(value ?? '')
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;')
        .replace(/'/g, '&#39;');
}

function renderKaTeX(el, latex, displayMode = true) {
    if (!el || !latex) return;
    try {
        katex.render(latex, el, { displayMode, throwOnError: false });
    } catch (e) {
        el.textContent = latex;
    }
}

function latexToMathematica(latex) {
    return latex
        .replace(/\\frac\{([^}]+)\}\{([^}]+)\}/g, '($1)/($2)')
        .replace(/\\sqrt\{([^}]+)\}/g, 'Sqrt[$1]')
        .replace(/\\pi/g, 'Pi')
        .replace(/\\alpha/g, 'alpha').replace(/\\beta/g, 'beta')
        .replace(/\\gamma/g, 'gamma').replace(/\\delta/g, 'delta')
        .replace(/\\rho/g, 'rho').replace(/\\omega/g, 'omega')
        .replace(/\\Lambda/g, 'Lambda').replace(/\\cdot/g, '*')
        .replace(/\\times/g, '*').replace(/\\left\(/g, '(').replace(/\\right\)/g, ')')
        .replace(/\^\{([^}]+)\}/g, '^($1)').replace(/_\{([^}]+)\}/g, '')
        .replace(/\\partial/g, 'D').replace(/\\infty/g, 'Infinity');
}

function copyToClipboard(btn, text) {
    navigator.clipboard.writeText(text).then(() => {
        btn.classList.add('copied');
        const orig = btn.textContent;
        btn.textContent = 'Copied!';
        setTimeout(() => { btn.textContent = orig; btn.classList.remove('copied'); }, 1500);
    });
}

// ── Build a result card ───────────────────────────────────────────────────────

function getRenderPayload(value) {
    if (value && typeof value === 'object' && !Array.isArray(value) && ('latex' in value || 'defs' in value)) {
        return value;
    }
    return null;
}

function getPrimaryLatex(value) {
    const payload = getRenderPayload(value);
    return payload ? (payload.latex || '') : (value || '');
}

function getCopyLatex(value) {
    const payload = getRenderPayload(value);
    if (!payload) return value || '';
    return payload.copy_latex || payload.latex || '';
}


function getCopyMathematica(value) {
    const payload = getRenderPayload(value);
    if (!payload) {
        return latexToMathematica(value || '');
    }
    return payload.copy_mathematica || payload.mathematica || latexToMathematica(payload.copy_latex || payload.latex || '');
}


function makeCard(labelHTML, latexStr) {
    const card = document.createElement('div');
    card.className = 'result-card';

    const lbl = document.createElement('span');
    lbl.className = 'card-label';
    lbl.innerHTML = labelHTML;
    card.appendChild(lbl);

    const disp = document.createElement('div');
    disp.className = 'latex-display';
    card.appendChild(disp);

    const payload = getRenderPayload(latexStr);
    const primaryLatex = getPrimaryLatex(latexStr);
    const copyExpr = getCopyLatex(latexStr);
    const copyMathematica = getCopyMathematica(latexStr);
    const isDeferred = latexStr && typeof latexStr === 'object' && latexStr.deferred;

    if (isDeferred) {
        const opsText = latexStr.ops ? ` (${latexStr.ops} ops)` : '';
        disp.textContent = opsText;
        disp.classList.add('note');
    } else if (primaryLatex) {
        if (payload && Array.isArray(payload.defs) && payload.defs.length) {
            const defsWrap = document.createElement('details');
            defsWrap.className = 'result-defs';
            const summary = document.createElement('summary');
            summary.textContent = 'Definitions';
            defsWrap.appendChild(summary);

            payload.defs.forEach((item) => {
                const row = document.createElement('div');
                row.className = 'latex-display';
                renderKaTeX(row, `${item.latex_name} = ${item.latex}`, true);
                defsWrap.appendChild(row);
            });
            card.appendChild(defsWrap);
        }
        renderKaTeX(disp, primaryLatex, true);
    } else {
        disp.textContent = '—';
    }

    const btns = document.createElement('div');
    btns.className = 'copy-buttons';

    const latexBtn = document.createElement('button');
    latexBtn.className = 'copy-btn';
    latexBtn.textContent = '📋 LaTeX';
    latexBtn.disabled = isDeferred;
    latexBtn.onclick = () => copyToClipboard(latexBtn, copyExpr);

    const mathBtn = document.createElement('button');
    mathBtn.className = 'copy-btn math-btn';
    mathBtn.textContent = '🔢 Mathematica';
    mathBtn.disabled = isDeferred;
    mathBtn.onclick = () => copyToClipboard(mathBtn, copyMathematica);

    btns.appendChild(latexBtn);
    btns.appendChild(mathBtn);
    card.appendChild(btns);

    renderDynamicMath(lbl);
    return card;
}

// ── Theory selection ──────────────────────────────────────────────────────────
function getTheory() {
    return document.querySelector('input[name="theory"]:checked')?.value || 'fR';
}

async function loadTheoryRegistry() {
    try {
        const response = await fetch('/api/theories');
        const data = await response.json();
        theoryRegistry = data.theories || [];
        buildTheoryControls(theoryRegistry);
    } catch (err) {
        console.warn('[THEORY] Failed to load registry, using static controls:', err);
    }
}

function buildTheoryControls(theories) {
    const container = $('theory-group');
    if (!container || !theories.length) return;

    const current = getTheory();
    const families = [];
    theories.forEach(spec => {
        let family = families.find(f => f.key === spec.geometry_class);
        if (!family) {
            family = {
                key: spec.geometry_class,
                label: spec.geometry_label || spec.geometry_class,
                theories: [],
            };
            families.push(family);
        }
        family.theories.push(spec);
    });

    container.innerHTML = '';
    families.forEach(family => {
        const familyEl = document.createElement('div');
        familyEl.className = 'theory-family';

        const label = document.createElement('div');
        label.className = 'theory-family-label';
        label.textContent = family.label;
        familyEl.appendChild(label);

        const group = document.createElement('div');
        group.className = 'radio-group';
        family.theories.forEach(spec => {
            const pill = document.createElement('label');
            pill.className = 'radio-pill';
            pill.title = spec.notes || spec.name;

            const input = document.createElement('input');
            input.type = 'radio';
            input.name = 'theory';
            input.value = spec.id;
            input.checked = spec.id === current || (!current && spec.id === 'fR');
            input.addEventListener('change', onTheoryChange);

            pill.appendChild(input);
            pill.insertAdjacentHTML('beforeend', theoryDisplayName(spec.id));
            pill.addEventListener('click', () => {
                input.checked = true;
                onTheoryChange();
            });
            group.appendChild(pill);
        });
        familyEl.appendChild(group);
        container.appendChild(familyEl);
    });
}

function theoryDisplayName(theoryId) {
    const labels = {
        fR: ' f(R)',
        fT: ' f(T)',
        fTB: ' f(T,B)',
        fRTLm: ' f(R,T,L<sub>m</sub>)',
        fQ: ' f(Q)',
        fQC: ' f(Q,C)',
    };
    return labels[theoryId] || ` ${theoryId}`;
}

function onTheoryChange() {
    const theory = getTheory();

    // Update radio pill active state
    document.querySelectorAll('.radio-pill').forEach(pill => {
        const inp = pill.querySelector('input[name="theory"]');
        if (inp) pill.classList.toggle('active', inp.checked);
    });

    // Update model label
    const lbl = $('model-var-label');
    if (lbl) lbl.textContent = THEORY_MODEL_LABEL[theory] || 'f =';

    // Update field equation display
    const eqEl = $('theory-eq-latex');
    if (eqEl) renderKaTeX(eqEl, THEORY_FIELD_EQ[theory] || '', true);

    // Update model preset dropdown
    populateModelPresets(theory);

    // Show/hide Lm choice
    const lmRow = $('lm-choice-row');
    lmRow && (theory === 'fRTLm' ? show(lmRow) : hide(lmRow));

    // Clear Lm compatibility cache when theory changes
    _lmCompatCache = {};

    // Update generic field eq in Tab 1
    const genEq = $('generic-field-eq');
    if (genEq) renderKaTeX(genEq, THEORY_FIELD_EQ[theory] || '', true);

    // Update anisotropic availability
    updateSetOptions();
}

// ── Model presets ─────────────────────────────────────────────────────────────
function populateModelPresets(theory) {
    const sel = $('model-preset');
    if (!sel) return;
    sel.innerHTML = '';
    const presets = MODEL_PRESETS[theory] || [];
    presets.forEach(p => {
        const opt = document.createElement('option');
        opt.value = p.expr;
        opt.textContent = p.name;
        if (p.id === 'custom') opt.value = '__custom__';
        sel.appendChild(opt);
    });
    // Select second option by default (Starobinsky / TEGR / GR / GR+mat)
    if (sel.options.length > 1) sel.selectedIndex = 1;
    onModelChange();
    updateModelParamInputs();
}

function updateModelDisplay(expr) {
    const display = $('model-display');
    if (!display) return;

    const theory = getTheory();
    const varMap = {
        'fR': 'R',
        'fT': 'T',
        'fTB': 'T, B',
        'fRTLm': 'R, T, L_m',
        'fQ': 'Q',
        'fQC': 'Q, C'
    };
    const vars = varMap[theory] || 'R';
    const latex = `f(${vars}) = ${expr}`;

    display.innerHTML = '\\[' + latex + '\\]';
    if (window.renderMathInElement) {
        renderMathInElement(display, {
            delimiters: [{left: '\\[', right: '\\]', display: true}],
            throwOnError: false
        });
    }
}

function onModelChange() {
    const sel     = $('model-preset');
    const custom  = $('custom-model-row');
    const isCustom = sel && sel.value === '__custom__';
    isCustom ? show(custom) : hide(custom);

    // Update functional form display
    const customInput = $('custom-model');
    const expr = isCustom
        ? (customInput ? customInput.value.trim() : '')
        : (sel ? sel.value : 'R');
    updateModelDisplay(expr);

    updateModelParamInputs();

    // Show/hide scalar name hint panel for fRTLm custom models
    updateScalarHintPanel();
}

function updateScalarHintPanel() {
    const theory = getTheory();
    const sel = $('model-preset');
    const isCustom = sel && sel.value === '__custom__';
    const hintContainer = $('scalar-hint-container');
    
    if (!hintContainer) return;
    
    if (theory === 'fRTLm' && isCustom) {
        // Fetch scalar map if not cached
        if (!_scalarMapCache['fRTLm']) {
            fetch('/api/scalar_map/fRTLm')
                .then(response => response.json())
                .then(data => {
                    _scalarMapCache['fRTLm'] = data;
                    renderScalarHintPanel(data);
                })
                .catch(error => {
                    console.error('Failed to load scalar map:', error);
                });
        } else {
            renderScalarHintPanel(_scalarMapCache['fRTLm']);
        }
    } else {
        hintContainer.innerHTML = '';
    }
}

function renderScalarHintPanel(scalarMap) {
    const hintContainer = $('scalar-hint-container');
    if (!hintContainer) return;
    
    let html = '<div class="section-label">Available symbols</div>';
    
    // Add physics scalars
    const physicsScalars = ['R', 'T_mat', 'T', 'L', 'Lm'];
    physicsScalars.forEach(symbol => {
        const info = scalarMap[symbol];
        if (info && info[0] !== null) {
            const [internalSymbol, displayName, latex] = info;
            html += `
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 4px;">
                    <span style="font-family: var(--mono); color: var(--teal);">${symbol}</span>
                    <span style="color: var(--text-dim);">— ${displayName}</span>
                    <span style="font-family: var(--mono); color: var(--gold);">\\(${latex}\\)</span>
                </div>`;
        }
    });
    
    // Add separator
    html += '<div style="border-bottom: 1px solid var(--border); margin: 8px 0;"></div>';
    
    // Add parameter names
    const paramSymbols = ['lam', 'alpha', 'beta', 'gamma', 'n'];
    paramSymbols.forEach(symbol => {
        const info = scalarMap[symbol];
        if (info && info[0] === null) {
            const [internalSymbol, displayName, latex] = info;
            html += `
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 4px;">
                    <span style="font-family: var(--mono); color: var(--muted);">${symbol}</span>
                    <span style="color: var(--text-dim);">— ${displayName}</span>
                    <span style="font-family: var(--mono); color: var(--warn);">\\(${latex}\\)</span>
                </div>`;
        }
    });
    
    hintContainer.innerHTML = html;
}

// ── Background selection ──────────────────────────────────────────────────────
function onBackgroundChange() {
    const bgId = $('background').value;
    updateCurvatureKVisibility(bgId);

    // Update metric display
    updateMetricDisplay(bgId);

    // Rebuild ansatz fields
    buildAnsatzFields(bgId);
    updateAnsatzParamInputs();

    // Update SET restrictions (anisotropic gating)
    updateSetOptions();
    updateTovOptionVisibility();

    // Fetch and display symmetry group / geometry info
    updateSymmetryPanel(bgId);
}

function updateCurvatureKVisibility(bgId) {
    const row = $('curvature-k-row');
    const select = $('curvature-k');
    if (!row || !select) return;
    if (bgId === 'FRW') {
        show(row);
        select.disabled = false;
    } else {
        hide(row);
        select.disabled = true;
        select.value = '0';
    }
}

function updateTovOptionVisibility() {
    const bgId = $('background')?.value || '';
    const setType = $('stress-energy')?.value || 'perfect_fluid';
    const row = $('compute-tov-row');
    const input = $('compute-tov');
    if (!row || !input) return;

    const supportsTov = TOV_BACKGROUNDS.has(bgId) &&
        (setType === 'perfect_fluid' || setType === 'anisotropic');
    const isCosmologyBackground = COSMOLOGY_BACKGROUNDS.has(bgId);

    if (supportsTov && !isCosmologyBackground) {
        show(row);
        input.disabled = false;
        input.checked = true;
    } else {
        hide(row);
        input.checked = false;
        input.disabled = true;
    }
}

function updateSymmetryPanel(bgId) {
    const panel = $('symmetry-panel');
    if (!panel) return;
    fetch(`/api/symmetry_group/${bgId}`)
        .then(r => r.json())
        .then(data => {
            const sg = data.symmetry_group || {};
            const gi = data.geometry_info  || {};
            let html = '';
            if (sg.geometry_class) {
                html += `<div class="sym-row"><span class="sym-label">Geometry class</span><span class="sym-val">${sg.geometry_class}</span></div>`;
            }
            if (sg.isometry_group) {
                html += `<div class="sym-row"><span class="sym-label">Isometry group</span><span class="sym-val sym-math">\\(${sg.isometry_group}\\)</span></div>`;
            }
            if (sg.Killing_vectors != null) {
                html += `<div class="sym-row"><span class="sym-label">Killing vectors</span><span class="sym-val">${sg.Killing_vectors}</span></div>`;
            }
            if (sg.lie_algebra) {
                html += `<div class="sym-row"><span class="sym-label">Lie algebra</span><span class="sym-val sym-math">\\(\\mathfrak{${sg.lie_algebra.replace(/[()]/g,'')}}\\)</span> <code style="font-size:0.75rem;opacity:0.7">${sg.lie_algebra}</code></div>`;
            }
            if (sg.spatial_symmetry) {
                html += `<div class="sym-row"><span class="sym-label">Spatial symmetry</span><span class="sym-val sym-math">\\(${sg.spatial_symmetry}\\)</span></div>`;
            }
            // Cosmological equations
            if (gi.Friedmann_1) {
                html += `<div class="sym-row"><span class="sym-label">Friedmann 1</span><span class="sym-val sym-math">\\(${gi.Friedmann_1}\\)</span></div>`;
            }
            if (gi.Friedmann_2) {
                html += `<div class="sym-row"><span class="sym-label">Friedmann 2</span><span class="sym-val sym-math">\\(${gi.Friedmann_2}\\)</span></div>`;
            }
            // Other geometry constraints
            if (gi.TOV) {
                html += `<div class="sym-row"><span class="sym-label">TOV equation</span><span class="sym-val sym-math">\\(${gi.TOV}\\)</span></div>`;
            }
            if (gi.throat) {
                html += `<div class="sym-row"><span class="sym-label">Throat condition</span><span class="sym-val sym-math">\\(${gi.throat}\\)</span></div>`;
            }
            if (gi.NEC) {
                html += `<div class="sym-row"><span class="sym-label">NEC requirement</span><span class="sym-val sym-math">\\(${gi.NEC}\\)</span></div>`;
            }
            if (sg.notes) {
                html += `<div class="sym-notes">${sg.notes}</div>`;
            }
            panel.innerHTML = html || '<span style="opacity:0.5;font-size:0.8rem">No symmetry data available.</span>';
            renderDynamicMath(panel);
        })
        .catch(() => {
            if (panel) panel.innerHTML = '';
        });
}

function updateMetricDisplay(bgId) {
    fetch(`/api/background_info/${bgId}`)
        .then(r => r.json())
        .then(info => {
            const el = $('metric-display');
            if (el && info.latex) renderKaTeX(el, info.latex, true);
        })
        .catch(() => {});
}

// ── Anisotropic gating ────────────────────────────────────────────────────────
function updateSetOptions() {
    const bgId = $('background').value;
    const isAnisotropicAllowed = !FRW_BACKGROUNDS.has(bgId);
    const isCosmologyBackground = FRW_BACKGROUNDS.has(bgId);
    const sel = $('stress-energy');
    if (!sel) return;
    
    Array.from(sel.options).forEach(opt => {
        // Anisotropic gating for FRW backgrounds
        if (opt.value === 'anisotropic') {
            opt.disabled = !isAnisotropicAllowed;
            opt.textContent = isAnisotropicAllowed
                ? 'Anisotropic Fluid'
                : 'Anisotropic Fluid (not available for FRW)';
        }
        
        // Disable vacuum, dust, radiation for non-cosmology backgrounds
        if (!isCosmologyBackground && ['vacuum', 'dust', 'radiation'].includes(opt.value)) {
            opt.disabled = true;
            opt.textContent = `${opt.textContent} (not available for ${bgId.replace('_', ' ')})`;
        } else if (opt.disabled && ['vacuum', 'dust', 'radiation'].includes(opt.value)) {
            // Re-enable if switching back to cosmology background
            opt.disabled = false;
            opt.textContent = opt.textContent.replace(/\s*\(not available for.*\)/, '');
        }
    });
    
    // If currently selected option is now disabled, fall back to perfect_fluid
    if (sel.options[sel.selectedIndex]?.disabled) {
        sel.value = 'perfect_fluid';
    }
    onSetChange();
}

// ── Lm compatibility for fRTLm theory ─────────────────────────────────────────────
async function updateLmOptions(setType) {
    // Only run when current theory is fRTLm
    if (getTheory() !== 'fRTLm') return;
    
    const lmSelect = $('lm-choice');
    if (!lmSelect) return;
    
    // Check cache first
    if (_lmCompatCache[setType]) {
        applyLmCompatibility(_lmCompatCache[setType], setType);
        return;
    }
    
    try {
        const response = await fetch(`/api/lm_compatibility/${setType}`);
        const compatibility = await response.json();
        
        // Cache the result
        _lmCompatCache[setType] = compatibility;
        
        applyLmCompatibility(compatibility, setType);
    } catch (error) {
        console.error('[LM] Failed to fetch Lm compatibility:', error);
    }
}

function applyLmCompatibility(compatibility, setType) {
    const lmSelect = $('lm-choice');
    if (!lmSelect) return;
    
    let currentSelectionChanged = false;
    const currentSelection = lmSelect.value;
    
    // Update options and track if current selection becomes invalid
    Array.from(lmSelect.options).forEach(option => {
        const matterLag = option.value;
        const compat = compatibility[matterLag];
        
        if (compat) {
            option.disabled = !compat.allowed;
            
            // Update option text for disallowed choices
            if (!compat.allowed) {
                const warningText = matterLag === 'p' ? ' ⚠ not valid for anisotropic' : ' ⚠ not supported';
                if (!option.textContent.includes('⚠')) {
                    option.textContent += warningText;
                }
            } else {
                // Remove warning text if it exists
                option.textContent = option.textContent.replace(/ ⚠.*$/, '');
            }
            
            // Check if current selection is now disallowed
            if (option.value === currentSelection && !compat.allowed) {
                currentSelectionChanged = true;
            }
        }
    });
    
    // Auto-switch to neg_rho if current selection became invalid
    if (currentSelectionChanged) {
        lmSelect.value = 'neg_rho';
        
        // Flash warning on Lm dropdown
        lmSelect.classList.add('input-warn');
        setTimeout(() => lmSelect.classList.remove('input-warn'), 1500);
        
        console.log('[LM] Auto-selected neg_rho due to compatibility change');
    }
    
    // Show/hide warning banner below Lm dropdown
    showLmWarning(compatibility, currentSelection);
}

function showLmWarning(compatibility, currentSelection) {
    const currentCompat = compatibility[currentSelection];
    const warningContainer = $('lm-warning-container');
    
    if (!warningContainer) return;
    
    if (currentCompat && !currentCompat.allowed && currentCompat.reason) {
        warningContainer.innerHTML = `
            <div class="error-banner" style="background: var(--warn-banner-bg); border: 1px solid var(--warn-banner-border); color: var(--warn);">
                <strong>⚠ Incompatible Lm choice:</strong> ${currentCompat.reason}
                <button type="button" onclick="this.parentElement.style.display='none'" style="float:right;background:none;border:none;color:inherit;cursor:pointer;">✕</button>
            </div>
        `;
        warningContainer.style.display = 'block';
    } else {
        warningContainer.style.display = 'none';
    }
}

// ── SET display ───────────────────────────────────────────────────────────────
function onSetChange() {
    const setType = $('stress-energy').value;
    const el = $('set-display');
    if (el) renderKaTeX(el, SET_LATEX[setType] || '', true);
    if (el) renderDynamicMath($('set-display'));
    
    // Default Lm selection for anisotropic SET in fRTLm theory
    if (getTheory() === 'fRTLm' && setType === 'anisotropic') {
        const lmSelect = $('lm-choice');
        if (lmSelect && lmSelect.value !== 'neg_rho') {
            lmSelect.value = 'neg_rho';
            console.log('[LM] Auto-selected neg_rho for anisotropic SET');
        }
    }
    
    // Update Lm options based on compatibility
    updateLmOptions(setType);
    updateTovOptionVisibility();
}

// ── Ansatz fields ─────────────────────────────────────────────────────────────
function buildAnsatzFields(bgId) {
    const container = $('ansatz-fields');
    if (!container) return;
    container.innerHTML = '';

    const fields = BACKGROUND_ANSATZ[bgId] || [{ fn: 'a', category: 'cosmological' }];
    const grouped = BACKGROUND_ANSATZ_GROUPS[bgId] || [];

    if (grouped.length) {
        const wrapper = document.createElement('div');
        wrapper.className = 'ansatz-field';

        const groupLabel = document.createElement('label');
        groupLabel.textContent = 'Metric preset =';
        wrapper.appendChild(groupLabel);

        const groupSel = document.createElement('select');
        groupSel.className = 'select-full';
        groupSel.id = 'ansatz-group-sel';
        grouped.forEach((preset) => {
            const opt = document.createElement('option');
            opt.value = preset.id;
            opt.textContent = preset.name;
            groupSel.appendChild(opt);
        });
        const customOpt = document.createElement('option');
        customOpt.value = '__custom__';
        customOpt.textContent = 'Custom pair…';
        groupSel.appendChild(customOpt);
        wrapper.appendChild(groupSel);

        const preview = document.createElement('div');
        preview.id = 'ansatz-group-preview';
        preview.className = 'note';
        preview.style.marginTop = '8px';
        wrapper.appendChild(preview);
        container.appendChild(wrapper);

        fields.forEach(({ fn }) => {
            const div = document.createElement('div');
            div.className = 'ansatz-field hidden';
            div.id = `ansatz-custom-row-${fn}`;

            const lbl = document.createElement('label');
            lbl.textContent = `${fn}(…) =`;
            div.appendChild(lbl);

            const inp = document.createElement('input');
            inp.type = 'text';
            inp.id = `ansatz-custom-${fn}`;
            inp.className = 'input-full';
            inp.placeholder = `Enter ${fn}(…) expression`;
            inp.addEventListener('input', updateAnsatzParamInputs);
            div.appendChild(inp);

            container.appendChild(div);
        });

        const refreshGroupedUI = () => {
            const isCustom = groupSel.value === '__custom__';
            fields.forEach(({ fn }) => {
                const row = $(`ansatz-custom-row-${fn}`);
                if (row) {
                    isCustom ? show(row) : hide(row);
                }
            });
            if (isCustom) {
                preview.textContent = 'Custom coupled metric functions.';
            } else {
                const preset = grouped.find((p) => p.id === groupSel.value) || grouped[0];
                preview.innerHTML = Object.entries(preset.functions || {})
                    .map(([fn, expr]) => `<div><strong>${fn}</strong> = ${expr}</div>`)
                    .join('');
            }
            updateAnsatzParamInputs();
        };

        groupSel.addEventListener('change', refreshGroupedUI);
        refreshGroupedUI();
        return;
    }

    fields.forEach(({ fn, category }) => {
        const presets = ANSATZ_PRESETS[category] || [];
        const div = document.createElement('div');
        div.className = 'ansatz-field';

        const lbl = document.createElement('label');
        lbl.textContent = `${fn}(…) =`;
        div.appendChild(lbl);

        const sel = document.createElement('select');
        sel.className = 'select-full';
        sel.id = `ansatz-sel-${fn}`;
        presets.forEach(p => {
            const opt = document.createElement('option');
            opt.value = p.expr;
            opt.textContent = `${p.name}  [${p.expr}]`;
            sel.appendChild(opt);
        });

        const customOpt = document.createElement('option');
        customOpt.value = '__custom__';
        customOpt.textContent = 'Custom…';
        sel.appendChild(customOpt);
        div.appendChild(sel);

        const customRow = document.createElement('div');
        customRow.id = `ansatz-custom-row-${fn}`;
        customRow.className = 'hidden';
        customRow.style.marginTop = '6px';
        const inp = document.createElement('input');
        inp.type = 'text';
        inp.id = `ansatz-custom-${fn}`;
        inp.className = 'input-full';
        inp.placeholder = `Enter ${fn}(…) expression`;
        customRow.appendChild(inp);
        div.appendChild(customRow);

        sel.addEventListener('change', () => {
            sel.value === '__custom__' ? show(customRow) : hide(customRow);
            updateAnsatzParamInputs();
        });
        inp.addEventListener('input', updateAnsatzParamInputs);

        container.appendChild(div);
    });
}

function gatherAnsatz() {
    const bgId  = $('background').value;
    const fields = BACKGROUND_ANSATZ[bgId] || [{ fn: 'a', category: 'cosmological' }];
    const grouped = BACKGROUND_ANSATZ_GROUPS[bgId] || [];
    const result = {};

    if (grouped.length) {
        const groupSel = $('ansatz-group-sel');
        const isCustom = groupSel && groupSel.value === '__custom__';
        if (isCustom) {
            fields.forEach(({ fn }) => {
                const inp = $(`ansatz-custom-${fn}`);
                result[fn] = inp ? (inp.value.trim() || '0') : '0';
            });
            return result;
        }
        const preset = grouped.find((p) => p.id === (groupSel ? groupSel.value : grouped[0].id)) || grouped[0];
        return { ...(preset.functions || {}) };
    }

    fields.forEach(({ fn }) => {
        const sel = $(`ansatz-sel-${fn}`);
        const inp = $(`ansatz-custom-${fn}`);
        result[fn] = (sel && sel.value === '__custom__' && inp)
            ? (inp.value.trim() || '0')
            : (sel ? sel.value : '0');
    });
    return result;
}

// ── Gather full input ─────────────────────────────────────────────────────────
function gatherInput() {
    const theory  = getTheory();
    const modelSel = $('model-preset');
    const modelExpr = (modelSel && modelSel.value === '__custom__')
        ? ($('custom-model').value.trim() || 'R')
        : (modelSel ? modelSel.value : 'R');

    return {
        background_id: $('background').value,
        theory,
        model_expr:    modelExpr,
        model_params:  collectParamValues('model-params-container'),
        stress_tensor: $('stress-energy').value,
        ansatz:        gatherAnsatz(),
        ansatz_params: collectParamValues('ansatz-params-container'),
        curvature_k:   parseInt($('curvature-k')?.value || '0', 10),
        matter_lag:    $('lm-choice')?.value || 'rho',
        diagnostics: {
            energy_conditions: $('compute-energy')?.checked ?? true,
            eos: $('compute-eos')?.checked ?? true,
            stability: $('compute-stability')?.checked ?? true,
            tov: (!$('compute-tov')?.disabled && $('compute-tov')?.checked) || false,
        },
        simplify_mode: $('simplify-mode')?.value || 'fast',
    };
}

// ── Compute ───────────────────────────────────────────────────────────────────
$('compute-btn').addEventListener('click', async () => {
    const config = gatherInput();

    hide($('idle-state'));
    hide($('results-panel'));
    hide($('error-banner'));
    show($('progress-panel'));
    updateProgress(3, 'Starting computation…');

    $('compute-btn').disabled = true;
    $('cancel-btn').disabled  = false;

    try {
        const resp = await fetch('/api/compute', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(config),
        });
        const { task_id } = await resp.json();
        currentTaskId = task_id;
        connectToStream(task_id);
    } catch (err) {
        showError(`Network error: ${err.message}`);
        resetButtons();
    }
});

$('cancel-btn').addEventListener('click', async () => {
    if (!currentTaskId) return;
    await fetch(`/api/cancel/${currentTaskId}`, { method: 'POST' }).catch(() => {});
});

$('quit-btn').addEventListener('click', async () => {
    if (!confirm('Shut down the local server? This will close the studio.')) return;
    await fetch('/api/quit', { method: 'POST' }).catch(() => {});
    document.body.innerHTML = '<div style="display:flex;align-items:center;justify-content:center;height:100vh;font-family:sans-serif;color:#6b7492;background:#0d0f14"><p>Server stopped. You may close this tab.</p></div>';
});

// ── SSE stream ────────────────────────────────────────────────────────────────
function connectToStream(taskId) {
    if (eventSource) eventSource.close();
    eventSource = new EventSource(`/api/stream/${taskId}`);
    eventSource.onmessage = e => handleEvent(JSON.parse(e.data));
    eventSource.onerror   = () => { eventSource.close(); resetButtons(); };
}

function handleEvent(event) {
    switch (event.type) {
        case 'progress':
            updateProgress(event.pct, event.label);
            break;
        case 'complete':
            updateProgress(100, 'Complete ✓');
            displayResults(event.results);
            resetButtons();
            eventSource?.close();
            break;
        case 'error':
            showError(event.message);
            resetButtons();
            eventSource?.close();
            break;
        case 'cancelled':
            updateProgress(0, 'Cancelled');
            resetButtons();
            eventSource?.close();
            break;
    }
}

function updateProgress(pct, label) {
    const fill  = $('progress-fill');
    const lblEl = $('progress-label');
    if (fill)  fill.style.width = `${pct}%`;
    if (lblEl) lblEl.textContent = label;
}

function showError(msg) {
    const banner = $('error-banner');
    if (banner) { banner.textContent = `Error: ${msg}`; show(banner); }
    hide($('results-panel'));
}

function resetButtons() {
    $('compute-btn').disabled = false;
    $('cancel-btn').disabled  = true;
}


// Numeric solve and plotting helpers live in static/js/numericSolve.js
// ── Display results ───────────────────────────────────────────────────────────
function displayResults(results) {
    currentResults = results;
    const theory = getTheory();
    const currentStressTensor = $('stress-energy')?.value || 'perfect_fluid';
    const isSelectedAnisotropic = currentStressTensor === 'anisotropic';

    // Handle early-exit (non-linear fRTLm models)
    const isEarlyExit = results.early_exit === true;
    const hasNumericSolve = !!results.numeric_solve?.available;
    const requestedDiagnostics = results.diagnostics_requested || {};
    
    // Show/hide early-exit warning banner
    const earlyExitBanner = $('early-exit-banner');
    const earlyExitMessage = $('early-exit-message');
    if (earlyExitBanner) {
        if (isEarlyExit) {
            earlyExitMessage.textContent = `Non-linear model - analytical matter solution not available: ${results.early_exit_reason || ''}`;
            show(earlyExitBanner);
        } else {
            hide(earlyExitBanner);
        }
    }


    const resultWarningBanner = $('result-warning-banner');
    const warnings = Array.isArray(results.warnings) ? results.warnings : [];
    if (resultWarningBanner) {
        if (warnings.length) {
            resultWarningBanner.innerHTML = `
                <div class="result-warning-header">
                    <span class="warning-icon">⚠</span>
                    <strong>Model warnings</strong>
                </div>
                <ul class="result-warning-list">
                    ${warnings.map(msg => `<li>${escapeHtml(msg)}</li>`).join('')}
                </ul>
            `;
            show(resultWarningBanner);
        } else {
            resultWarningBanner.innerHTML = '';
            hide(resultWarningBanner);
        }
    }

    // Update tab visibility for early-exit mode
    const tabBtns = document.querySelectorAll('.tab-btn');
    tabBtns.forEach(btn => {
        const tabId = btn.getAttribute('data-tab');
        if (isEarlyExit && !hasNumericSolve && (tabId === 'tab-energy' || tabId === 'tab-eos' || tabId === 'tab-stability' || tabId === 'tab-tov')) {
            btn.style.display = 'none';
        } else {
            btn.style.display = '';
        }
    });

    // Update Tab 1 header based on mode
    const matterTabBtn = document.querySelector('.tab-btn[data-tab="tab-matter"]');
    if (matterTabBtn) {
        matterTabBtn.textContent = isEarlyExit ? 'Reduced Field Equations' : 'Field Equations & Solutions';
    }

    // Generic field equation (Tab 1)
    const genEq = $('generic-field-eq');
    if (genEq) renderKaTeX(genEq, THEORY_FIELD_EQ[theory] || '', true);

    // ── Theory Scalars ──
    const scalarsEl = $('scalars-container');
    if (scalarsEl) {
        scalarsEl.innerHTML = '';
        const scalarDefs = {
            fR:    [['R', '\\mathcal{R}']],
            fT:    [['T', 'T']],
            fTB:   [['T', 'T'], ['B', 'B'], ['T_scalar', 'T - B']],
            fRTLm: [['R', '\\mathcal{R}'], ['T_scalar', 'T_{\\rm scalar}'], ['Lm', '\\mathcal{L}_m']],
            fQ:    [['T', 'Q']],   // Q is stored in the 'T' slot for fQ
            fQC:   [['T', 'Q'], ['B', 'C']],
        };
        (scalarDefs[theory] || []).forEach(([key, label]) => {
            const val = results.scalars?.[key];
            scalarsEl.appendChild(makeCard(
                `<span class="label-math">\\(${label}\\)</span>`,
                val || null
            ));
        });
    }

    // ── Matter solutions (hidden in early-exit mode) ──
    const solutionsSection = $('solutions-section');
    if (solutionsSection) {
        if (isEarlyExit) {
            hide(solutionsSection);
        } else {
            show(solutionsSection);
            const solutionsEl = $('solutions-container');
            if (solutionsEl) {
                solutionsEl.innerHTML = '';
                const { rho, p, Pr, Pt } = results.matter || {};
                if (rho)  solutionsEl.appendChild(makeCard('\\(\\rho\\)  — energy density', rho));
                if (Pr)   solutionsEl.appendChild(makeCard('\\(P_r\\)  — radial pressure',    Pr));
                if (Pt)   solutionsEl.appendChild(makeCard('\\(P_t\\)  — tangential pressure', Pt));
                if (p && !Pr) solutionsEl.appendChild(makeCard('\\(p\\)  — pressure', p));
            }
        }
    }

    // -- Exported Equations (early-exit mode only) --
    const exportedEqSection = $('exported-eq-section');
    const exportedEqContainer = $('exported-eq-container');
    if (exportedEqSection && exportedEqContainer) {
        if (isEarlyExit && results.exported_equations) {
            show(exportedEqSection);
            exportedEqContainer.innerHTML = '';
            
            // Store equations for copy-all functionality
            exportedEqContainer._equations = results.exported_equations;
            
            results.exported_equations.forEach((eq, idx) => {
                const card = makeCard(
                    `Field equation ${eq.index} — fully reduced`,
                    eq.equation_latex,
                    { latex: eq.lhs_latex + ' = ' + eq.rhs_latex }
                );
                card.style.marginBottom = '16px';
                exportedEqContainer.appendChild(card);
            });
            
            // Setup copy-all button
            const copyAllBtn = $('copy-all-equations-btn');
            if (copyAllBtn) {
                copyAllBtn.onclick = () => {
                    const eqs = exportedEqContainer._equations || [];
                    const alignBlock = '\\begin{align}\n' + 
                        eqs.map(eq => `    ${eq.lhs_latex} &= ${eq.rhs_latex} % ${eq.index}`).join(' \\\\\n') +
                        '\n\\end{align}';
                    navigator.clipboard.writeText(alignBlock).then(() => {
                        copyAllBtn.textContent = 'Copied!';
                        setTimeout(() => copyAllBtn.textContent = 'Copy All Equations', 2000);
                    });
                };
            }
        } else {
            hide(exportedEqSection);
        }
    }
    setupNumericSolvePanel(results);
    setupSymbolicPlotPanel(results);

    // ── Energy conditions (hidden or note in early-exit) ──
    const ecEl = $('ec-container');
    const tabEnergy = $('tab-energy');
    if (ecEl && tabEnergy) {
        if (isEarlyExit) {
            ecEl.innerHTML = '<p class="note">Not available — model requires numerical or perturbative methods</p>';
        } else if (requestedDiagnostics.energy_conditions === false) {
            ecEl.innerHTML = '<p class="note">Energy conditions were not selected for this run.</p>';
        } else {
            ecEl.innerHTML = '';
            const ec = results.energy_conditions || {};
            const isAnisotropic = isSelectedAnisotropic;
            const pLabel = isAnisotropic ? 'P_r' : 'p';
            const ptLabel = 'P_t';
            const ecDefs = [
                ['NEC', 'Null Energy Condition',
                    isAnisotropic ? '\\rho + P_{r,t} \\geq 0' : '\\rho + p \\geq 0',
                    isAnisotropic
                        ? [['NEC_r', `\\rho + ${pLabel}`], ['NEC_t', `\\rho + ${ptLabel}`]]
                        : [['NEC_r', `\\rho + p`]]
                ],
                ['WEC', 'Weak Energy Condition',     '\\rho \\geq 0',     [['WEC', '\\rho']]],
                ['SEC', 'Strong Energy Condition',
                    isAnisotropic ? '\\rho + P_r + 2P_t \\geq 0' : '\\rho + 3p \\geq 0',
                    [['SEC', isAnisotropic ? '\\rho + P_r + 2P_t' : '\\rho + 3p']]
                ],
                ['DEC', 'Dominant Energy Condition',
                    isAnisotropic ? '\\rho \\geq |P_{r,t}|' : '\\rho \\geq |p|',
                    isAnisotropic
                        ? [['DEC_r', '\\rho - |P_r|'], ['DEC_t', '\\rho - |P_t|']]
                        : [['DEC_r', '\\rho - |p|']]
                ],
            ];
            ecDefs.forEach(([, title, cond, terms]) => {
                const hdr = document.createElement('div');
                hdr.style.cssText = 'margin-bottom:6px;margin-top:16px;';
                hdr.innerHTML = `<strong style="color:var(--gold);font-size:0.82rem">${title}</strong>`;
                const condDisp = document.createElement('div');
                condDisp.className = 'latex-display';
                condDisp.style.fontSize = '0.85rem';
                renderKaTeX(condDisp, cond, true);
                ecEl.appendChild(hdr);
                ecEl.appendChild(condDisp);
                terms.forEach(([key, label]) => {
                    if (ec[key]) ecEl.appendChild(makeCard(`\\(${label}\\)`, ec[key]));
                });
            });
        }
    }

    // ── EOS (hidden or note in early-exit) ──
    const eosEl = $('eos-container');
    if (eosEl) {
        if (isEarlyExit) {
            eosEl.innerHTML = '<p class="note">Not available — model requires numerical or perturbative methods</p>';
        } else if (requestedDiagnostics.eos === false) {
            eosEl.innerHTML = '<p class="note">Equation of state was not selected for this run.</p>';
        } else {
            eosEl.innerHTML = '';
            const eos = results.eos || {};
            const isAnisotropicEoS = isSelectedAnisotropic;
            const eosDefs = isAnisotropicEoS ? [
                ['omega_r',   '\\omega_r = P_r/\\rho\ \\text{(radial)}'],
                ['omega_t',   '\\omega_t = P_t/\\rho\ \\text{(tangential)}'],
                ['omega_eff', '\\omega_{\\rm eff} = P_{\\rm eff}/\\rho'],
            ] : [
                ['omega_r',   '\\omega = p/\\rho\ \\text{(EoS parameter)}'],
                ['omega_eff', '\\omega_{\\rm eff} = P_{\\rm eff}/\\rho'],
            ];
            eosDefs.forEach(([key, label]) => {
                if (eos[key]) eosEl.appendChild(makeCard(`\\(${label}\\)`, eos[key]));
            });
        }
    }

    // ── Stability (hidden or note in early-exit) ──
    const stabEl = $('stability-container');
    if (stabEl) {
        if (isEarlyExit) {
            stabEl.innerHTML = '<p class="note">Not available — model requires numerical or perturbative methods</p>';
        } else if (requestedDiagnostics.stability === false) {
            stabEl.innerHTML = '<p class="note">Stability / sound speed was not selected for this run.</p>';
        } else {
            stabEl.innerHTML = '';
            const cs = results.speed_of_sound || {};
            if (cs.cs2_r) stabEl.appendChild(makeCard('\\(c^2_s{}_{,r} = dP_r/d\\rho\\)', cs.cs2_r));
            // Skip cs2_t for perfect fluid (check for None value)
            if (cs.cs2_t) stabEl.appendChild(makeCard('\\(c^2_s{}_{,t} = dP_t/d\\rho\\)', cs.cs2_t));

            // Herrera cracking (anisotropic only)
            if (cs.cs2_r && cs.cs2_t) {
                const note = document.createElement('p');
                note.className = 'note';
                note.style.marginTop = '12px';
                note.textContent = 'Herrera cracking condition: −1 ≤ c²ₜ − c²ᵣ ≤ 0 for stability against cracking.';
                stabEl.appendChild(note);
            }
        }
    }

    const tovEl = $('tov-container');
    if (tovEl) {
        const bgId = $('background')?.value || '';
        const canShowTov = TOV_BACKGROUNDS.has(bgId) &&
            (currentStressTensor === 'perfect_fluid' || currentStressTensor === 'anisotropic');
        if (isEarlyExit) {
            tovEl.innerHTML = '<p class="note">Not available â€” model requires numerical or perturbative methods</p>';
        } else if (requestedDiagnostics.tov === false) {
            tovEl.innerHTML = '<p class="note">TOV analysis was not selected for this run.</p>';
        } else if (!canShowTov) {
            tovEl.innerHTML = '<p class="note">TOV analysis is available for static spherical perfect-fluid and anisotropic runs.</p>';
        } else {
            tovEl.innerHTML = '';
            const tov = results.tov || {};
            const tovDefs = [
                ['mass', 'm(r)'],
                ['compactness', '2m(r)/r'],
                ['redshift_gradient', "\\Phi'(r)"],
                ['pressure_gradient', "dP_r/dr"],
                ['hydrostatic_force', 'F_h = -dP_r/dr'],
                ['gravitational_force', "F_g = -(\\rho + P_r)\\Phi'"],
                ['anisotropic_force', 'F_a = 2(P_t - P_r)/r'],
                ['residual', 'F_h + F_g + F_a'],
                ['mass_continuity_residual', "m'(r) - 4\\pi r^2\\rho"],
            ];
            tovDefs.forEach(([key, label]) => {
                if (tov[key]) tovEl.appendChild(makeCard(`\\(${label}\\)`, tov[key]));
            });
            if (!tovEl.children.length) {
                tovEl.innerHTML = '<p class="note">No TOV terms were returned for this run.</p>';
            }
        }
    }

    show($('results-panel'));
    hide($('idle-state'));
    hide($('error-banner'));

    // Switch to Tab 1
    document.querySelector('.tab-btn[data-tab="tab-matter"]')?.click();
}

// ── Tabs ──────────────────────────────────────────────────────────────────────
document.querySelectorAll('.tab-btn').forEach(btn => {
    btn.addEventListener('click', () => {
        document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
        document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
        btn.classList.add('active');
        const tab = document.getElementById(btn.getAttribute('data-tab'));
        if (tab) tab.classList.add('active');
    });
});

// ── Theory radio buttons ──────────────────────────────────────────────────────
document.querySelectorAll('input[name="theory"]').forEach(inp => {
    inp.addEventListener('change', onTheoryChange);
});
document.querySelectorAll('.radio-pill').forEach(pill => {
    pill.addEventListener('click', () => {
        const inp = pill.querySelector('input[type="radio"]');
        if (inp) { inp.checked = true; onTheoryChange(); }
    });
});

// ── Other event listeners ─────────────────────────────────────────────────────
$('background').addEventListener('change', onBackgroundChange);
$('stress-energy').addEventListener('change', onSetChange);
$('model-preset').addEventListener('change', onModelChange);
const customModelInput = $('custom-model');
if (customModelInput) {
    customModelInput.addEventListener('input', onModelChange);
}

// ── Initialise ────────────────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
    loadTheoryRegistry();
    onTheoryChange();
    onBackgroundChange();
    onSetChange();
});


const _customModelInput = $('custom-model');
if (_customModelInput) _customModelInput.addEventListener('input', onModelChange);
