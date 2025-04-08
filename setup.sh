python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python3 setup.py develop
pip install -e .