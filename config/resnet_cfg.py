from config.cfg import BaseConfig
class ResnetConfig(BaseConfig):

    def __init__(self):
        super().__init__()

        self.parser.add_argument('--resnet_layers', default=50, type=int, choices=[18, 34, 50, 101], help='Number of Resnet layers')
