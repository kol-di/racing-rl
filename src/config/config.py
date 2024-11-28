import ruamel.yaml

def read_conf(config_path='./src/config/config.yaml'):
    yaml = ruamel.yaml.YAML(typ="unsafe", pure=True)
    with open(config_path) as f:
        config = yaml.load(f)
    conf_keys = list(config.keys())

    expected_keys = [
        'device', 'gamma', 'reward_steps', 'entropy_beta', 'batch_size', 
        'num_stack'
    ]
    for k in expected_keys:
        assert k in conf_keys

    return config
    
    