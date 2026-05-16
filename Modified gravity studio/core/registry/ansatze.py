"""
Ansatz Preset Registry - cosmological and static spherically symmetric.
"""

ANSATZ_PRESETS = {
    'cosmological': [
        {'id': 'power_law',       'name': 'Power-law',           'expr': 'a0*t**h',               'params': ['a0', 'h']},
        {'id': 'de_sitter',       'name': 'de Sitter',           'expr': 'a0*exp(H0*t)',         'params': ['a0', 'H0']},
        {'id': 'sinh',            'name': 'Hyperbolic sine',     'expr': 'a0*sinh(H0*t)**h',     'params': ['a0', 'H0', 'h']},
        {'id': 'power_exp',       'name': 'Power-exponential',   'expr': 'a0*t**h*exp(beta*t)',  'params': ['a0', 'h', 'beta']},
    ],
    'wormhole_b': [
        {'id': 'exponential',     'name': 'Exponential',         'expr': 'r/exp(r - r0)',        'params': ['r0']},
        {'id': 'inverse_power',   'name': 'Inverse Power',       'expr': 'r0**2/r',              'params': ['r0']},
        {'id': 'power_law',       'name': 'Power-law',           'expr': 'r0*(r0/r)**(h-1)',     'params': ['r0', 'h']},
    ],
    'wormhole_Phi': [
        {'id': 'zero_tidal',      'name': 'Zero Tidal Force',    'expr': '0',                    'params': []},
        {'id': 'constant',        'name': 'Constant',            'expr': 'Phi0',                 'params': ['Phi0']},
    ],

    'blackhole_nu': [
        {'id': 'schwarzschild',      'name': 'Schwarzschild',      'expr': '1 - 2*M/r',                     'params': ['M']},
        {'id': 'reissner_nordstrom', 'name': 'Reissner-Nordstrom', 'expr': '1 - 2*M/r + Q**2/r**2',        'params': ['M', 'Q']},
    ],
    'blackhole_lam': [
        {'id': 'schwarzschild',      'name': 'Schwarzschild',      'expr': '1/(1 - 2*M/r)',                'params': ['M']},
        {'id': 'reissner_nordstrom', 'name': 'Reissner-Nordstrom', 'expr': '1/(1 - 2*M/r + Q**2/r**2)',   'params': ['M', 'Q']},
    ],
}

BACKGROUND_ANSATZ_MAP = {
    'FRW':              [('a', 'cosmological')],
    'Bianchi_I':        [('A', 'cosmological'), ('B', 'cosmological')],
    'Bianchi_III':      [('A', 'cosmological'), ('B', 'cosmological')],
    'Kantowski_Sachs':  [('A', 'cosmological'), ('B', 'cosmological')],
    'SS_wormhole':      [('b', 'wormhole_b'), ('Phi', 'wormhole_Phi')],
    'SS_blackhole':     [('nu_bh', 'blackhole_nu'), ('lam_bh', 'blackhole_lam')],
}

ANSATZ_PRESET_GROUPS = {
    'Bianchi_I': [
        {'id': 'power_law', 'name': 'Directional power-law', 'functions': {'A': 'A0*t**pA', 'B': 'B0*t**pB'}, 'params': ['A0', 'B0', 'pA', 'pB']},
        {'id': 'de_sitter', 'name': 'Directional de Sitter', 'functions': {'A': 'A0*exp(Hx*t)', 'B': 'B0*exp(Hy*t)'}, 'params': ['A0', 'B0', 'Hx', 'Hy']},
        {'id': 'sinh', 'name': 'Directional hyperbolic sine', 'functions': {'A': 'A0*sinh(H0*t)**pA', 'B': 'B0*sinh(H0*t)**pB'}, 'params': ['A0', 'B0', 'H0', 'pA', 'pB']},
        {'id': 'power_exp', 'name': 'Directional power-exponential', 'functions': {'A': 'A0*t**pA*exp(Hx*t)', 'B': 'B0*t**pB*exp(Hy*t)'}, 'params': ['A0', 'B0', 'pA', 'pB', 'Hx', 'Hy']},
    ],
    'Bianchi_III': [
        {'id': 'power_law', 'name': 'Directional power-law', 'functions': {'A': 'A0*t**pA', 'B': 'B0*t**pB'}, 'params': ['A0', 'B0', 'pA', 'pB']},
        {'id': 'de_sitter', 'name': 'Directional de Sitter', 'functions': {'A': 'A0*exp(Hx*t)', 'B': 'B0*exp(Hy*t)'}, 'params': ['A0', 'B0', 'Hx', 'Hy']},
        {'id': 'sinh', 'name': 'Directional hyperbolic sine', 'functions': {'A': 'A0*sinh(H0*t)**pA', 'B': 'B0*sinh(H0*t)**pB'}, 'params': ['A0', 'B0', 'H0', 'pA', 'pB']},
        {'id': 'power_exp', 'name': 'Directional power-exponential', 'functions': {'A': 'A0*t**pA*exp(Hx*t)', 'B': 'B0*t**pB*exp(Hy*t)'}, 'params': ['A0', 'B0', 'pA', 'pB', 'Hx', 'Hy']},
    ],
    'Kantowski_Sachs': [
        {'id': 'power_law', 'name': 'Directional power-law', 'functions': {'A': 'A0*t**pA', 'B': 'B0*t**pB'}, 'params': ['A0', 'B0', 'pA', 'pB']},
        {'id': 'de_sitter', 'name': 'Directional de Sitter', 'functions': {'A': 'A0*exp(Hx*t)', 'B': 'B0*exp(Hy*t)'}, 'params': ['A0', 'B0', 'Hx', 'Hy']},
        {'id': 'sinh', 'name': 'Directional hyperbolic sine', 'functions': {'A': 'A0*sinh(H0*t)**pA', 'B': 'B0*sinh(H0*t)**pB'}, 'params': ['A0', 'B0', 'H0', 'pA', 'pB']},
        {'id': 'power_exp', 'name': 'Directional power-exponential', 'functions': {'A': 'A0*t**pA*exp(Hx*t)', 'B': 'B0*t**pB*exp(Hy*t)'}, 'params': ['A0', 'B0', 'pA', 'pB', 'Hx', 'Hy']},
    ],
    'SS_blackhole': [
        {'id': 'schwarzschild', 'name': 'Schwarzschild', 'display': 'F(r) = 1 - 2*M/r', 'functions': {'nu_bh': '1 - 2*M/r', 'lam_bh': '1/(1 - 2*M/r)'}, 'params': ['M']},
        {'id': 'reissner_nordstrom', 'name': 'Reissner-Nordstrom', 'display': 'F(r) = 1 - 2*M/r + Q**2/r**2', 'functions': {'nu_bh': '1 - 2*M/r + Q**2/r**2', 'lam_bh': '1/(1 - 2*M/r + Q**2/r**2)'}, 'params': ['M', 'Q']},
    ],
}


def get_grouped_preset(background_id: str, preset_id: str):
    for preset in ANSATZ_PRESET_GROUPS.get(background_id, []):
        if preset['id'] == preset_id:
            return preset
    return None


def validate_grouped_ansatz(background_id: str, ansatz: dict):
    grouped = ANSATZ_PRESET_GROUPS.get(background_id, [])
    if not grouped:
        return True, ''

    fields = list(grouped[0]['functions'].keys())
    expr_to_preset = {}
    for preset in grouped:
        pid = preset['id']
        for fn, expr in preset['functions'].items():
            expr_to_preset.setdefault(fn, {})[expr] = pid

    selected_ids = []
    for fn in fields:
        expr = ansatz.get(fn)
        if expr is None:
            return False, f"Missing ansatz function '{fn}' for background '{background_id}'"
        pid = expr_to_preset.get(fn, {}).get(expr)
        if pid is None:
            return True, ''
        selected_ids.append(pid)

    if selected_ids and len(set(selected_ids)) != 1:
        return False, (
            f"Mixed preset families are not allowed for background '{background_id}'. "
            f"Choose one coupled metric preset or switch all coupled functions to custom."
        )
    return True, ''
