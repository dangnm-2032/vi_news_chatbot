from vinewschatbot.constants import *
from vinewschatbot.logging import logger
from vinewschatbot.config import *
from datasets import Dataset, disable_caching, load_from_disk, load_dataset, concatenate_datasets

class CreateDataset:
    def __init__(
        self,
        config: ConfigurationManager
    ) -> None:
        self.config = config
    
    def load(self):
        logger.info('Loading datasets...')
        self.bltlab_ds = load_dataset('bltlab/lr-sum', 'vie')
        self.toanduc_ds = load_dataset('toanduc/t5-sumary-dataset')
        self.csebuetnlp_ds = load_dataset('csebuetnlp/xlsum', 'vietnamese')
        self.nlplabtdtu_ds = load_dataset('nlplabtdtu/summarization_sft_prompted')
        self.vietgpt_ds = load_dataset('vietgpt/news_summarization_vi')
        logger.info("Done")

    def merge(self):
        logger.info("Unspliting train test datasets...")
        self.bltlab_full_ds = concatenate_datasets([self.bltlab_ds['train'], self.bltlab_ds['test'], self.bltlab_ds['validation']])
        self.toanduc_full_ds = concatenate_datasets([self.toanduc_ds['train']])
        self.csebuetnlp_full_ds = concatenate_datasets([self.csebuetnlp_ds['train'], self.csebuetnlp_ds['test'], self.csebuetnlp_ds['validation']])
        self.nlplabtdtu_full_ds = concatenate_datasets([self.nlplabtdtu_ds['train'], self.nlplabtdtu_ds['test']])
        self.vietgpt_full_ds = concatenate_datasets([self.vietgpt_ds['train'], self.vietgpt_ds['test']])
        logger.info("Done")

    def format(self):
        logger.info("Formatting datasets...")
        self.bltlab_full_ds = self.bltlab_full_ds.remove_columns(['id', 'url', 'title'])
        self.toanduc_full_ds = self.toanduc_full_ds.rename_column('Original Text', 'text').rename_column('Summary', 'summary')
        self.csebuetnlp_full_ds = self.csebuetnlp_full_ds.remove_columns(['id', 'url', 'title'])
        self.nlplabtdtu_full_ds = self.nlplabtdtu_full_ds
        self.vietgpt_full_ds = self.vietgpt_full_ds.rename_column('content', 'text')
        logger.info("Done")

    def concatenate(self):
        logger.info("Concatenating datasets...")
        self.full_ds = concatenate_datasets([self.bltlab_full_ds, self.toanduc_full_ds, self.csebuetnlp_full_ds, self.nlplabtdtu_full_ds, self.vietgpt_full_ds])
        self.full_ds = self.full_ds.shuffle(seed=42)
        logger.info("Done")

    def save(self):
        logger.info("Saving...")
        config = self.config.get_summary_dataset_config()
        self.full_ds.save_to_disk(config.path)
        logger.info("Done")
    