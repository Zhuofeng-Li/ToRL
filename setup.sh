conda create -n sandbox-runtime python==3.12
conda activate sandbox-runtime

pip install -r SandboxFusion/runtime/python/requirements.txt
pip install -r requirements.txt
pip install poetry  
poetry install  
mkdir -p docs/build  
make run-online

