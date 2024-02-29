from vinewschatbot.config import ConfigurationManager
from vinewschatbot.conponents import *

class ValidateProgramPipeline:
    def __init__(
        self,
        config: ConfigurationManager
    ):
        self.config = config
        self.validator = Validator(config=self.config)

    def main(self):
        self.validator.wandb_config()
        self.validator.word_segmentor()
        self.validator.search_model()
        self.validator.dataset()
        self.validator.base_model()