## Summary
+ It supports math (deepmath) or science (general-reasoner) tasks Python-use LLM RL Training.
+ It realizes LLL-as-Judge for GPQA-D evaluation during training.

## Env Set Up
```python 
uv venv --python 3.10
source .venv/bin/activate
uv pip install -r requirements.txt
uv pip install flash-attn==2.7.4.post1  --no-build-isolation 
cd octotools
uv pip install -e .
```

```python
git submodule update --init --recursive
cd octotools
uv pip install -e .
# set .env 
```

