"""
Ansatz Preset Registry - cosmological and static spherically symmetric.
"""

ANSATZ_PRESETS = {
    'cosmological': [
        {'id': 'power_law',       'name': 'Power-law',           'expr': 'a0*t**h',               'params': ['a0', 'h']},
        {'id': 'de_sitter',       'name': 'de Sitter',           'expr': 'a0*exp(H0*t)',         'params': ['a0', 'H0']},
        {'id': 'sinh',            'name': 'Hyperbolic sine',     'expr': 'a0*sinh(H0*t)**n',     'params': ['a0', 'H0', 'n']},
        {'id': 'power_exp',       'name': 'Power-exponential',   'expr': 'a0*t**h*exp(beta*t)',  'params': ['a0', 'h', 'beta']},
    ],
    'wormhole_b': [
        {'id': 'exponential',     'name': 'Exponential',         'expr': 'r/exp(r - r0)',        'params': ['r0']},
        {'id': 'inverse_power',   'name': 'Inverse Power',       'expr': 'r0**2/r',              'params': ['r0']},
        {'id': 'power_law',       'name': 'Power-law',           'expr': 'r0*(r0/r)**(n-1)',     'params': ['r0', 'n']},
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
    'FRW_flat':         [('a', 'cosmological')],
    'FRW_curved':       [('a', 'cosmological')],
    'Bianchi_I':        [('A', 'cosmological'), ('B', 'cosmological')],
    'Bianchi_III':      [('A', 'cosmological'), ('B', 'cosmological')],
    'Kantowski_Sachs':  [('A', 'cosmological'), ('B', 'cosmological')],
    'SS_wormhole':      [('b', 'wormhole_b'), ('Phi', 'wormhole_Phi')],
    'SS_blackhole':     [('nu_bh', 'blackhole_nu'), ('lam_bh', 'blackhole_lam')],
}

ANSATZ_PRESET_GROUPS = {
    'Bianchi_I': [
        {'id': 'power_law', 'name': 'Power-law', 'functions': {'A': 'a0*t**h', 'B': 'a0*t**h'}, 'params': ['a0', 'h']},
        {'id': 'de_sitter', 'name': 'de Sitter', 'functions': {'A': 'a0*exp(H0*t)', 'B': 'a0*exp(H0*t)'}, 'params': ['a0', 'H0']},
        {'id': 'sinh', 'name': 'Hyperbolic sine', 'functions': {'A': 'a0*sinh(H0*t)**n', 'B': 'a0*sinh(H0*t)**n'}, 'params': ['a0', 'H0', 'n']},
        {'id': 'power_exp', 'name': 'Power-exponential', 'functions': {'A': 'a0*t**h*exp(beta*t)', 'B': 'a0*t**h*exp(beta*t)'}, 'params': ['a0', 'h', 'beta']},
    ],
    'Bianchi_III': [
        {'id': 'power_law', 'name': 'Power-law', 'functions': {'A': 'a0*t**h', 'B': 'a0*t**h'}, 'params': ['a0', 'h']},
        {'id': 'de_sitter', 'name': 'de Sitter', 'functions': {'A': 'a0*exp(H0*t)', 'B': 'a0*exp(H0*t)'}, 'params': ['a0', 'H0']},
        {'id': 'sinh', 'name': 'Hyperbolic sine', 'functions': {'A': 'a0*sinh(H0*t)**n', 'B': 'a0*sinh(H0*t)**n'}, 'params': ['a0', 'H0', 'n']},
        {'id': 'power_exp', 'name': 'Power-exponential', 'functions': {'A': 'a0*t**h*exp(beta*t)', 'B': 'a0*t**h*exp(beta*t)'}, 'params': ['a0', 'h', 'beta']},
    ],
    'Kantowski_Sachs': [
        {'id': 'power_law', 'name': 'Power-law', 'functions': {'A': 'a0*t**h', 'B': 'a0*t**h'}, 'params': ['a0', 'h']},
        {'id': 'de_sitter', 'name': 'de Sitter', 'functions': {'A': 'a0*exp(H0*t)', 'B': 'a0*exp(H0*t)'}, 'params': ['a0', 'H0']},
        {'id': 'sinh', 'name': 'Hyperbolic sine', 'functions': {'A': 'a0*sinh(H0*t)**n', 'B': 'a0*sinh(H0*t)**n'}, 'params': ['a0', 'H0', 'n']},
        {'id': 'power_exp', 'name': 'Power-exponential', 'functions': {'A': 'a0*t**h*exp(beta*t)', 'B': 'a0*t**h*exp(beta*t)'}, 'params': ['a0', 'h', 'beta']},
    ],
    'SS_blackhole': [
        {'id': 'schwarzschild', 'name': 'Schwarzschild', 'functions': {'nu_bh': '1 - 2*M/r', 'lam_bh': '1/(1 - 2*M/r)'}, 'params': ['M']},
        {'id': 'reissner_nordstrom', 'name': 'Reissner-Nordstrom', 'functions': {'nu_bh': '1 - 2*M/r + Q**2/r**2', 'lam_bh': '1/(1 - 2*M/r + Q**2/r**2)'}, 'params': ['M', 'Q']},
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
