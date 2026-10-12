"""The person, as the environment models them.

    params       UserParams (f, k, ...) and ModelState
    accumulator  the drift-diffusion process behind start and redo decisions
    conditions   the (value, cost) each decision node feeds it
    value        TD value of a set, and reward timing
    competence   chance of a right answer, and how it grows
    persistence  giving up on a question
    questions    question types and Elo ratings
    simulation   the simulated learner
"""
