from vinewschatbot.config import ConfigurationManager
from vinewschatbot.conponents import *

class CreateDatasetPipeline:
    def __init__(
        self,
        config: ConfigurationManager
    ) -> None:
        self.config = config
        self.create_dataset = CreateDataset(config=self.config)

    def main(self):
        self.create_dataset.load()
        self.create_dataset.merge()
        self.create_dataset.format()
        self.create_dataset.concatenate()
        self.create_dataset.save()