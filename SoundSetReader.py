from ruamel.yaml import YAML

yaml=YAML(typ='safe')   # default, if not specfied, is 'rt' (round-trip)

with open('./Sound.yaml','r') as f:
    print(yaml.load(f))