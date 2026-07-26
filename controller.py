from config import UserParams
from main import run_experiment

params = UserParams(f=.5,k=.5,b=.5)

for _ in range(3):
    run_experiment(params)
