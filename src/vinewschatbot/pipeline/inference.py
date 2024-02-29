from vinewschatbot.config import ConfigurationManager
from vinewschatbot.conponents import *

class InferencePipeline:
    def __init__(
        self,
        config: ConfigurationManager
    ) -> None:
        self.config = config
        self.inference = Inference(config=self.config)

    def main(self):
        self.inference.init_word_segmentor()
        self.inference.init_search_model()
        self.inference.init_summary_model()
        self.inference.init_interface()

        server_config = self.config.get_inference_server_config()
        self.inference.interface.launch(
            server_name=server_config.server_name,
            share=server_config.share
        )