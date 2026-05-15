"""
Model Preset Registry — all four theories.
"""

MODEL_PRESETS = {
    'fR': [
        {'id': 'GR',           'name': 'General Relativity',       'expr': 'R'},
        {'id': 'starobinsky',  'name': 'Starobinsky',              'expr': 'R + alpha*R**2'},
        {'id': 'power_corr',   'name': 'R + αRⁿ',                 'expr': 'R + alpha*R**n'},
        {'id': 'exponential',  'name': 'Exponential',              'expr': 'R + alpha*(1 - exp(-R/beta))'},
        {'id': 'logarithmic',  'name': 'Logarithmic',              'expr': 'R + alpha*log(R)'},
    ],
    'fT': [
        {'id': 'TEGR',         'name': 'TEGR (GR limit)',          'expr': 'T'},
        {'id': 'power_law',    'name': 'Power-law',                'expr': 'T + alpha*T**n'},
        {'id': 'exponential',  'name': 'Exponential',              'expr': 'T*(1 + alpha*exp(beta*T))'},
        {'id': 'logarithmic',  'name': 'Logarithmic',              'expr': 'T + alpha*log(T/T0)'},
    ],
    'fTB': [
        {'id': 'GR_recovery',  'name': 'GR Recovery (f=T−B)',      'expr': 'T - B'},
        {'id': 'fR_limit',     'name': 'f(R) Limit',               'expr': '-(T - B) + alpha*(T - B)**2'},
        {'id': 'linear_comb',  'name': 'Linear Combination',       'expr': 'alpha*T + beta*B'},
        {'id': 'power_mixed',  'name': 'Power-law Mixed',          'expr': 'T + alpha*B**n'},
        {'id': 'exp_T',        'name': 'Exponential-T',            'expr': 'T*exp(alpha*T) - B'},
    ],
    'fRTLm': [
        {'id': 'GR_matter',    'name': 'GR + matter coupling',     'expr': 'R + lam*L'},
        {'id': 'star_T',       'name': 'Starobinsky + T-coupling', 'expr': 'R + alpha*R**2 + beta*T_mat'},
        {'id': 'full_linear',  'name': 'Full Linear',              'expr': 'R + beta*T_mat + lam*L'},
        {'id': 'mixed_form',   'name': 'Mixed Form',               'expr': 'R + alpha*R**2 + beta*T_mat + gamma*L + delta*T_mat*L'},
        {'id': 'nonlinear_LT',  'name': 'Non-linear L×T',          'expr': 'R + alpha*L*T_mat'},
        {'id': 'non_min',      'name': 'Non-minimal Matter',       'expr': 'R*(1 + gamma*L)'},
    ],
    'fQ': [
        {'id': 'STEGR',        'name': 'STEGR (GR limit)',         'expr': 'Q'},
        {'id': 'power_law',    'name': 'Power-law',                'expr': 'Q + alpha*Q**n'},
        {'id': 'exponential',  'name': 'Exponential',              'expr': 'Q + alpha*(1 - exp(-Q/beta))'},
        {'id': 'logarithmic',  'name': 'Logarithmic',              'expr': 'Q + alpha*log(Q/Q0)'},
        {'id': 'sqrt',         'name': 'Square-root',              'expr': 'Q + alpha*sqrt(Q)'},
        {'id': 'quad',         'name': 'Quadratic',                'expr': 'Q + alpha*Q**2'},
    ],
    'fQC': [
        {'id': 'STEGR',        'name': 'STEGR (f = Q)',            'expr': 'Q'},
        {'id': 'boundary_lin', 'name': 'Linear boundary',          'expr': 'Q + alpha*C'},
        {'id': 'curvature_lim','name': 'Curvature equivalent',     'expr': 'Q + C'},
        {'id': 'qc_quad',      'name': 'Quadratic Q + C',          'expr': 'Q + alpha*Q**2 + beta*C'},
        {'id': 'boundary_log', 'name': 'Boundary logarithmic',     'expr': 'Q + alpha*C*log(C/C0)'},
        {'id': 'mixed_qc',     'name': 'Mixed Q-C coupling',       'expr': 'Q + alpha*Q*C'},
    ],
}
