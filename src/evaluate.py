"""
evaluate.py — DEPRECATED.

This entry point has been replaced by src.evaluate_forecast, which evaluates
clustering quality and STGNN forecast accuracy without UPF assignment logic.

UPF assignment, energy modelling, and SLA analysis belong in the downstream
orchestration / digital-twin repository, not here.

Run instead:
    python -m src.evaluate_forecast [--config config.yaml] [--with-coherence]
"""

raise RuntimeError(
    "\n\n"
    "  src.evaluate is deprecated.\n\n"
    "  Use:  python -m src.evaluate_forecast\n\n"
    "  The new script evaluates clustering quality and forecast accuracy.\n"
    "  UPF assignment analysis belongs in the orchestration repository.\n"
)
