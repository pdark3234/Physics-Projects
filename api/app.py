"""
Modified Gravity Studio — Flask API

Full registry endpoints and task orchestration.
"""

import builtins
import sys
if not hasattr(builtins, 'display'):
    builtins.display = lambda *args, **kwargs: None

from flask import Flask, request, jsonify, Response, render_template
from flask_cors import CORS
import json
import uuid
import threading
import time
from queue import Empty, Queue
import os
import contextlib
import io
import gc
from collections import OrderedDict

app = Flask(__name__, template_folder='../templates', static_folder='../static')
CORS(app)

from api.routes.numeric import numeric_bp
from api.routes.plotting import plotting_bp

app.register_blueprint(numeric_bp)
app.register_blueprint(plotting_bp)

# Task registry
TASK_REGISTRY: dict = {}
TASK_QUEUES: dict   = {}
TASK_META: "OrderedDict[str, dict]" = OrderedDict()

MAX_COMPLETED_TASKS = max(8, int(os.getenv("MGS_MAX_COMPLETED_TASKS", "24")))
MAX_TASK_AGE_SECONDS = max(60, int(os.getenv("MGS_MAX_TASK_AGE_SECONDS", "900")))


def _mark_task(task_id: str, **fields):
    meta = TASK_META.get(task_id, {})
    meta.update(fields)
    meta.setdefault("created_at", time.time())
    TASK_META[task_id] = meta
    TASK_META.move_to_end(task_id, last=True)


def _prune_task_state(force: bool = False):
    now = time.time()
    removable = []
    for task_id, meta in list(TASK_META.items()):
        is_done = meta.get("status") in {"complete", "error", "cancelled"}
        age = now - float(meta.get("completed_at") or meta.get("created_at") or now)
        if is_done and (force or age > MAX_TASK_AGE_SECONDS):
            removable.append(task_id)

    completed_ids = [
        task_id for task_id, meta in TASK_META.items()
        if meta.get("status") in {"complete", "error", "cancelled"}
    ]
    overflow = max(0, len(completed_ids) - MAX_COMPLETED_TASKS)
    if overflow:
        removable.extend(completed_ids[:overflow])

    for task_id in list(dict.fromkeys(removable)):
        TASK_REGISTRY.pop(task_id, None)
        TASK_QUEUES.pop(task_id, None)
        TASK_META.pop(task_id, None)


def log_info(message: str):
    """Print important server events even when noisy symbolic logs are muted."""
    print(message, file=sys.__stdout__, flush=True)


def _symbolic_logs_enabled() -> bool:
    """Return true only when raw symbolic library/debug logs are requested."""
    return os.environ.get("MGS_SYMBOLIC_LOGS", "false").lower() == "true"


@contextlib.contextmanager
def _quiet_symbolic_output():
    """Mute solver/library stdout while preserving explicit server progress logs."""
    if _symbolic_logs_enabled():
        yield
        return

    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        yield


def get_task_queue(task_id: str) -> Queue:
    if task_id not in TASK_QUEUES:
        TASK_QUEUES[task_id] = Queue()
        _mark_task(task_id, status="queued", created_at=time.time())
        _prune_task_state()
    return TASK_QUEUES[task_id]


def emit_event(task_id: str, event: dict):
    get_task_queue(task_id).put(event)


def event_stream(task_id: str):
    queue = get_task_queue(task_id)
    try:
        while True:
            try:
                event = queue.get(timeout=15)
                if event is None:
                    break
                yield f"data: {json.dumps(event)}\n\n"
                if event.get('type') in ('complete', 'error', 'cancelled'):
                    break
            except Empty:
                yield f"data: {json.dumps({'type': 'heartbeat'})}\n\n"
            except Exception as e:
                # Handle queue errors or timeouts
                yield f"data: {json.dumps({'type': 'error', 'message': f'Stream error: {str(e)}'})}\n\n"
                break
    except GeneratorExit:
        # Client disconnected
        pass
    except Exception as e:
        # Catastrophic error - try to send one last event
        try:
            yield f"data: {json.dumps({'type': 'error', 'message': f'Server error: {str(e)}'})}\n\n"
        except:
            pass  # If we can't even send an error, just give up
    finally:
        # Always cleanup resources
        TASK_QUEUES.pop(task_id, None)
        TASK_REGISTRY.pop(task_id, None)


# ─── Pages ────────────────────────────────────────────────────────────────────

@app.route('/')
def index():
    return render_template('index.html')


# ─── Registry endpoints ───────────────────────────────────────────────────────

@app.route('/api/backgrounds', methods=['GET'])
def get_backgrounds():
    from core.registry.metrics import METRIC_REGISTRY, BACKGROUND_NAMES, BACKGROUND_METRIC_LATEX
    backgrounds = [
        {
            'id':    bg_id,
            'name':  BACKGROUND_NAMES.get(bg_id, bg_id),
            'latex': BACKGROUND_METRIC_LATEX.get(bg_id, ''),
        }
        for bg_id in METRIC_REGISTRY
    ]
    return jsonify({'backgrounds': backgrounds})


@app.route('/api/background_info/<background_id>', methods=['GET'])
def get_background_info(background_id: str):
    from core.registry.metrics import BACKGROUND_NAMES, BACKGROUND_METRIC_LATEX, METRIC_REGISTRY
    from core.registry.ansatze import BACKGROUND_ANSATZ_MAP, ANSATZ_PRESETS, ANSATZ_PRESET_GROUPS
    if background_id not in METRIC_REGISTRY:
        return jsonify({'error': 'Unknown background'}), 404

    ansatz_fields = []
    for fn_name, category in BACKGROUND_ANSATZ_MAP.get(background_id, []):
        ansatz_fields.append({
            'fn_name':  fn_name,
            'presets':  ANSATZ_PRESETS.get(category, []),
        })

    ctx = METRIC_REGISTRY[background_id]()
    info = {
        'id':           background_id,
        'name':         BACKGROUND_NAMES.get(background_id, background_id),
        'latex':        BACKGROUND_METRIC_LATEX.get(background_id, ''),
        'coord_names':  list(ctx.coord_index.keys()),
        'metric_fns':   ctx.metric_fn_names,
        'ansatz_fields': ansatz_fields,
        'ansatz_groups': ANSATZ_PRESET_GROUPS.get(background_id, []),
        'set_restrictions': _set_restrictions(background_id),
    }
    return jsonify(info)


def _set_restrictions(background_id: str) -> dict:
    from core.registry.metrics import is_set_allowed
    all_sets = ['perfect_fluid', 'anisotropic', 'dust', 'radiation', 'vacuum']
    return {s: is_set_allowed(background_id, s) for s in all_sets}


@app.route('/api/theories', methods=['GET'])
def get_theories():
    from core.registry.theories import theory_payload
    return jsonify({'theories': theory_payload()})


@app.route('/api/theory_capabilities/<theory>', methods=['GET'])
def get_theory_capabilities(theory: str):
    from core.registry.theories import THEORY_REGISTRY
    spec = THEORY_REGISTRY.get(theory)
    if spec is None:
        return jsonify({'error': 'Unknown theory'}), 404
    return jsonify(spec.to_dict())


@app.route('/api/symmetry_group/<background_id>', methods=['GET'])
def get_symmetry_group(background_id: str):
    """Return Lie / isometry group info for a background."""
    from core.registry.metrics import (
        BACKGROUND_SYMMETRY_GROUP, BACKGROUND_GEOMETRY_INFO, METRIC_REGISTRY
    )
    if background_id not in METRIC_REGISTRY:
        return jsonify({'error': 'Unknown background'}), 404
    group_info = BACKGROUND_SYMMETRY_GROUP.get(background_id, {})
    geo_info   = BACKGROUND_GEOMETRY_INFO.get(background_id, {})
    return jsonify({
        'background_id':  background_id,
        'symmetry_group': group_info,
        'geometry_info':  geo_info,
    })


@app.route('/api/models/<theory>', methods=['GET'])
def get_models(theory: str):
    from core.registry.models import MODEL_PRESETS
    presets = MODEL_PRESETS.get(theory, [])
    return jsonify({'models': presets})


@app.route('/api/stress_energy', methods=['GET'])
def get_stress_energy_types():
    types = [
        {'id': 'perfect_fluid', 'name': 'Perfect Fluid',  'latex': r'T_{\mu\nu} = (\rho+p)u_\mu u_\nu + p g_{\mu\nu}', 'unknowns': ['rho', 'p']},
        {'id': 'anisotropic',   'name': 'Anisotropic Fluid', 'latex': r'T^{\mu\nu} = (\rho+P_t)u^\mu u^\nu + P_t g^{\mu\nu} + (P_r-P_t)x^\mu x^\nu', 'unknowns': ['rho', 'P_r', 'P_t']},
        {'id': 'dust',          'name': 'Dust',            'latex': r'T_{\mu\nu} = \rho u_\mu u_\nu',                    'unknowns': ['rho']},
        {'id': 'radiation',     'name': 'Radiation',       'latex': r'T_{\mu\nu} = \frac{\rho}{3}g_{\mu\nu}',            'unknowns': ['rho']},
        {'id': 'vacuum',        'name': 'Vacuum (Λ)',      'latex': r'T_{\mu\nu} = -\Lambda g_{\mu\nu}',                  'unknowns': ['Lambda']},
    ]
    return jsonify({'types': types})


@app.route('/api/ansatz/<background_id>', methods=['GET'])
def get_ansatz_presets(background_id: str):
    from core.registry.ansatze import BACKGROUND_ANSATZ_MAP, ANSATZ_PRESETS, ANSATZ_PRESET_GROUPS
    fields = []
    for fn_name, category in BACKGROUND_ANSATZ_MAP.get(background_id, []):
        fields.append({
            'fn_name': fn_name,
            'presets': ANSATZ_PRESETS.get(category, []),
        })
    return jsonify({'ansatz_fields': fields})


@app.route('/api/lm_choices', methods=['GET'])
def get_lm_choices():
    """Matter Lagrangian choices for f(R,T,Lm)."""
    choices = [
        {'id': 'rho',      'name': 'Lm = ρ',       'latex': r'\mathcal{L}_m = \rho'},
        {'id': 'neg_rho',  'name': 'Lm = −ρ',      'latex': r'\mathcal{L}_m = -\rho'},
        {'id': 'p',        'name': 'Lm = p',        'latex': r'\mathcal{L}_m = p'},
        {'id': 'T_scalar', 'name': 'Lm = T (trace)', 'latex': r'\mathcal{L}_m = T'},
    ]
    return jsonify({'choices': choices})


@app.route('/api/lm_compatibility/<set_type>', methods=['GET'])
def get_lm_compatibility(set_type: str):
    """Return Lm compatibility for given SET type."""
    from core.registry.metrics import LM_SET_COMPATIBILITY
    
    # Get all matter_lag choices from lm_choices endpoint
    all_choices = ['rho', 'neg_rho', 'p', 'T_scalar']
    
    compatibility = {}
    for matter_lag in all_choices:
        key = (set_type, matter_lag)
        allowed, reason = LM_SET_COMPATIBILITY.get(key, (True, ''))
        compatibility[matter_lag] = {
            'allowed': allowed,
            'reason': reason
        }
    
    return jsonify(compatibility)


@app.route('/api/scalar_map/<theory>', methods=['GET'])
def get_scalar_map(theory: str):
    """Return scalar name map for given theory."""
    if theory == 'fRTLm':
        from core.theories.fRTLm import SCALAR_NAME_MAP
        return jsonify(SCALAR_NAME_MAP)
    else:
        return jsonify({})


# ─── Validation ───────────────────────────────────────────────────────────────

@app.route('/api/validate_model', methods=['POST'])
def validate_model():
    data     = request.get_json()
    expr_str = data.get('expr', '')
    theory   = data.get('theory', 'fR')

    import sympy as sp

    # Reserved symbols per theory
    from core.registry.theories import THEORY_REGISTRY
    spec = THEORY_REGISTRY.get(theory)
    reserved = set(spec.model_symbols) if spec else {'R'}

    # Build local map with dummy symbols for reserved names
    dummy = {name: sp.Symbol(f'_dummy_{name}') for name in reserved}

    try:
        expr = sp.sympify(expr_str, locals=dummy)
        free_symbols = [str(s) for s in expr.free_symbols
                        if str(s) not in {f'_dummy_{n}' for n in reserved}]
        # Replace dummy names back for latex
        for name, d in dummy.items():
            expr = expr.subs(d, sp.Symbol(name))
        return jsonify({
            'valid':  True,
            'params': sorted(set(free_symbols)),
            'latex':  sp.latex(expr),
        })
    except Exception as e:
        return jsonify({'valid': False, 'error': str(e)}), 400


# ─── Computation ─────────────────────────────────────────────────────────────

@app.route('/api/compute', methods=['POST'])
def start_computation():
    data    = request.get_json()
    task_id = str(uuid.uuid4())
    get_task_queue(task_id)  # initialise queue before thread starts

    log_info(
        f"[MGS] Queued run {task_id[:8]} | "
        f"{data.get('theory', 'fR')} | "
        f"{data.get('background_id', 'FRW')} | "
        f"{data.get('stress_tensor', 'perfect_fluid')}"
    )

    thread = threading.Thread(
        target=_run_pipeline_task,
        args=(task_id, data),
        daemon=True,
    )
    TASK_REGISTRY[task_id] = thread
    _mark_task(task_id, status="queued", created_at=time.time())
    _prune_task_state()
    thread.start()
    return jsonify({'task_id': task_id, 'status': 'queued'})


def _run_pipeline_task(task_id: str, data: dict):
    from core.pipeline import Pipeline, PipelineInput

    start_time = time.perf_counter()

    last_progress_key = {"value": None}

    def callback(event):
        emit_event(task_id, event)
        event_type = event.get('type')
        if event_type == 'progress':
            _mark_task(task_id, status='running', last_progress_at=time.time())
            pct = int(event.get('pct') or 0)
            label = str(event.get('label') or 'Working')
            progress_key = (pct, label)
            if progress_key != last_progress_key["value"]:
                last_progress_key["value"] = progress_key
                log_info(f"  [{task_id[:8]}] {pct:3d}%  {label}")
        elif event_type == 'complete':
            elapsed = time.perf_counter() - start_time
            _mark_task(task_id, status='complete', completed_at=time.time(), elapsed=elapsed)
            log_info(f"[MGS] Complete run {task_id[:8]} | {elapsed:.2f}s")
        elif event_type == 'error':
            elapsed = time.perf_counter() - start_time
            _mark_task(task_id, status='error', completed_at=time.time(), elapsed=elapsed)
            log_info(f"[MGS] Error in run {task_id[:8]} | {elapsed:.2f}s | {event.get('message')}")
        elif event_type == 'cancelled':
            elapsed = time.perf_counter() - start_time
            _mark_task(task_id, status='cancelled', completed_at=time.time(), elapsed=elapsed)
            log_info(f"[MGS] Cancelled run {task_id[:8]} | {elapsed:.2f}s")

    pipeline = Pipeline(callback)

    # Parse ansatz — may be multi-field dict or legacy single-key dict
    raw_ansatz = data.get('ansatz', {})
    if isinstance(raw_ansatz, str):
        raw_ansatz = {'a': raw_ansatz}
    diagnostics = data.get('diagnostics', {})

    inp = PipelineInput(
        background_id  = data.get('background_id', 'FRW'),
        theory         = data.get('theory', 'fR'),
        model_expr     = data.get('model_expr', 'R'),
        model_params   = data.get('model_params', {}),
        stress_tensor  = data.get('stress_tensor', 'perfect_fluid'),
        ansatz         = raw_ansatz,
        ansatz_params  = data.get('ansatz_params', {}),
        curvature_k    = int(data.get('curvature_k', 0)),
        matter_lag     = data.get('matter_lag', 'rho'),
        compute_energy_conditions = bool(diagnostics.get(
            'energy_conditions', data.get('compute_energy_conditions', True)
        )),
        compute_eos = bool(diagnostics.get('eos', data.get('compute_eos', True))),
        compute_stability = bool(diagnostics.get(
            'stability', data.get('compute_stability', True)
        )),
        compute_tov = bool(diagnostics.get('tov', data.get('compute_tov', True))),
        simplify_mode = data.get('simplify_mode', 'fast'),
    )

    log_info(
        f"[MGS] Started run {task_id[:8]}\n"
        f"  Theory: {inp.theory}\n"
        f"  Background: {inp.background_id}\n"
        f"  Matter: {inp.stress_tensor}\n"
        f"  Model: {inp.model_expr}\n"
        f"  Diagnostics: EC={inp.compute_energy_conditions}, "
        f"EoS={inp.compute_eos}, Stability={inp.compute_stability}, TOV={inp.compute_tov}"
    )

    try:
        with _quiet_symbolic_output():
            pipeline.run(inp)
    except Exception as e:
        elapsed = time.perf_counter() - start_time
        log_info(f"[MGS] Task exception {task_id[:8]} | {elapsed:.2f}s | {e}")
        emit_event(task_id, {'type': 'error', 'message': str(e)})
    finally:
        with _quiet_symbolic_output():
            try:
                from core.results import clear_all_render_caches
                clear_all_render_caches()
            except Exception:
                pass
            try:
                from core.solver import flush_simplify_cache
                flush_simplify_cache()
            except Exception:
                pass
            try:
                from core.pipeline_cache import flush_sympy_caches, prune_all_caches
                prune_all_caches(lambda _msg: None)
                flush_sympy_caches(lambda _msg: None)
            except Exception:
                pass

        pipeline = None
        inp = None
        data = None
        gc.collect()

        meta = TASK_META.get(task_id, {})
        if meta.get('status') not in {'complete', 'error', 'cancelled'}:
            _mark_task(task_id, status='complete', completed_at=time.time(), elapsed=time.perf_counter() - start_time)
        get_task_queue(task_id).put(None)
        _prune_task_state()


@app.route('/api/stream/<task_id>')
def stream_events(task_id: str):
    if task_id not in TASK_QUEUES:
        return jsonify({'error': 'Task not found'}), 404
    return Response(
        event_stream(task_id),
        mimetype='text/event-stream',
        headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'},
    )


@app.route('/api/cancel/<task_id>', methods=['POST'])
def cancel_task(task_id: str):
    thread = TASK_REGISTRY.get(task_id)
    if thread:
        _mark_task(task_id, status='cancelled', completed_at=time.time())
        emit_event(task_id, {'type': 'cancelled'})
        _prune_task_state()
        return jsonify({'status': 'cancelled'})
    return jsonify({'error': 'Task not found'}), 404


@app.route('/api/quit', methods=['POST'])
def quit_server():
    try:
        from core.results import clear_all_temporary_caches
        clear_all_temporary_caches()
    except Exception as exc:
        log_info(f"[MGS] quit cleanup warning | {exc}")

    def _shutdown():
        import time
        time.sleep(0.5)
        os._exit(0)

    threading.Thread(target=_shutdown, daemon=True).start()
    return jsonify({'status': 'shutting_down', 'caches_cleared': True})


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000, threaded=True)
