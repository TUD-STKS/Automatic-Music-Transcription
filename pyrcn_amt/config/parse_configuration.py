import configparser


def parse_configuration(config_file):
    config = configparser.ConfigParser()
    config.read(config_file)

    experiment_settings = dict(config['experiment'])

    input_to_node_settings = dict(config['input_to_node'])
    node_to_node_settings = dict(config['node_to_node'])
    regression_settings = dict(config['regression'])

    return experiment_settings, input_to_node_settings, node_to_node_settings, regression_settings