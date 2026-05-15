"""
Pipeline configuration — timeouts (seconds) per stage.
"""
import os

STAGE_TIMEOUTS = {
    'geometry_curvature':   60,
    'geometry_teleparallel': 120,
    'boundary_term':        60,
    'model_derivatives':    30,
    'lhs_fR':               90,
    'lhs_fT':               90,
    'lhs_fTB':              300,
    'lhs_fRTLm':            120,
    'solve':                180,
    'derived':              60,
}

ALLOW_SHUTDOWN = os.environ.get('MGS_LOCAL', 'true').lower() == 'true'
MAX_WORKERS    = int(os.environ.get('MGS_WORKERS', '2'))
VERBOSE_LOGS   = os.environ.get('MGS_VERBOSE', 'false').lower() == 'true'
