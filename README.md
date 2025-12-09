# CS3346A
CS3346A Project code
Unity Version: 6000.2.14f1 
Python version: 3.10
Get it here: https://www.python.org/downloads/release/python-31011/

To install the package we're using do this in the root of the folders:
pip install mlagents

Before hitting play do mlagents-learn trainer_config.yaml --run-id=platformer2d_001 
Change the run-ID each time as the run will save as a file and keeping the name will overwrite the file

## Live training dashboard
- Start training as usual with `mlagents-learn trainer_config.yaml --run-id=<your_run>`.
- In another terminal from the repo root run `python monitor/dashboard_server.py --results-dir results --port 8000`.
- Open http://127.0.0.1:8000 to see live reward/episode stats; choose the run and behavior in the dropdowns.
- If you see an error about tensorboard not being installed, add it with `pip install tensorboard`.
