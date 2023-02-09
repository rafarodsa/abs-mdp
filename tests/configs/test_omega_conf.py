from omegaconf import OmegaConf as oc
from src.absmdp.configs import TrainerConfig
from src.models.configs import DistributionConfig, ConfigFactory


def test_omega_conf():
    # Load the config file
    path = 'configs/pinball_simple.yaml'
    dict_config = oc.load(path)
    config = oc.merge(oc.structured(TrainerConfig), dict_config)
    print(oc.to_yaml(config.model.decoder))
    print(oc.to_yaml(config.model.encoder))
    print(config.model.encoder.features.input_dim)
    
if __name__=="__main__":
    test_omega_conf()
