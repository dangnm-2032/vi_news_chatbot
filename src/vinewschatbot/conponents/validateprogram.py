from vinewschatbot.constants import *
from vinewschatbot.logging import logger
from vinewschatbot.config import *
from git import Repo
from sentence_transformers import SentenceTransformer
from datasets import load_dataset
from transformers import AutoConfig, T5ForConditionalGeneration, AutoTokenizer, GenerationConfig

class Validator:
    def __init__(
        self,
        config: ConfigurationManager
    ) -> None:
        self.config = config

    def wandb_config(self):
        logger.info("Checking wandb config...")
        wand_config = self.config.get_wandb_config()
        if wand_config.key == "your_wandb_api_key":
            msg = "Enter your wandb api key in config/config.yaml\nPlease visit https://wandb.ai/settings to get your api."
            logger.exception(msg)
            raise msg
        logger.info("Done")

    def word_segmentor(self):
        logger.info("Checking word segmentor...")
        config = self.config.get_word_segmentor_config()

        if os.path.exists(config.model_name_or_path):
           logger.info("Word segmentor is installed!")
           return 
        
        try:
            logger.info("Word segmentor is not installed! Cloning from github...")
            Repo.clone_from(
                config.source,
                config.model_name_or_path
            )
            logger.info("Done")
        except Exception as e:
            logger.exception(e)
            raise e
        return

    def search_model(self):
        logger.info("Checking search model...")
        config = self.config.get_search_model_config()
        try:
            model = SentenceTransformer(
                model_name_or_path=config.model_name_or_path,
                device='cpu'
            )
            del model
            logger.info("Done")
        except Exception as e:
            logger.exception(e)
            raise e
        return

    def dataset(self):
        logger.info("Checking datasets...")
        try:
            bltlab_ds = load_dataset(
                'bltlab/lr-sum', 
                'vie'
            )
            toanduc_ds = load_dataset(
                'toanduc/t5-sumary-dataset'
            )
            csebuetnlp_ds = load_dataset(
                'csebuetnlp/xlsum', 
                'vietnamese'
            )
            nlplabtdtu_ds = load_dataset(
                'nlplabtdtu/summarization_sft_prompted'
            )
            vietgpt_ds = load_dataset(
                'vietgpt/news_summarization_vi'
            )
            del bltlab_ds, toanduc_ds, csebuetnlp_ds, nlplabtdtu_ds, vietgpt_ds
            logger.info("Done")
        except Exception as e:
            logger.exception(e)
            raise e
        return

    def base_model(self):
        logger.info("Checking base model...")
        config = self.config.get_summary_model_config()
        try:
            tokenizer = AutoTokenizer.from_pretrained(config.base_model)  
            model_config = AutoConfig.from_pretrained(config.base_model)
            model = T5ForConditionalGeneration.from_pretrained(
                config.base_model, 
                config=model_config
            )
            del tokenizer, model_config, model
            logger.info("Done")
        except Exception as e:
            logger.exception(e)
            raise e
        return