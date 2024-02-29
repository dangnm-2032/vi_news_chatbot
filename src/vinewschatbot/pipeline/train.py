from vinewschatbot.config import ConfigurationManager
from vinewschatbot.conponents import *

class TrainingPipeline:
    def __init__(
        self,
        config: ConfigurationManager
    ) -> None:
        self.config = config
        self.trainer = Trainer(config=self.config)

    def main(self):
        self.trainer.train()