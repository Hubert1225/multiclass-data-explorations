from yaml import safe_load

with open("params.yaml") as f:
    params_yaml = f.read()
params = safe_load(params_yaml)
