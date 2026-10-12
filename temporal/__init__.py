"""A model of task initiation: recover reward sensitivity (f) and effort aversion (k)
from when people start work, and tune the task from the estimates.

    model/       the person: accumulator, TD value, competence, giving up, the simulator
    inference    fit (f, k) to decisions
    recovery     data-generating pipelines and recovery sweeps
    controller   choose task settings from the estimates
    app/         the playable task (desktop and web)
"""
