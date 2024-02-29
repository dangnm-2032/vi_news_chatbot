from vinewschatbot.constants import *
from vinewschatbot.logging import logger
from vinewschatbot.config import *
from vinewschatbot.utils import *
import py_vncorenlp
import os
from sentence_transformers import SentenceTransformer
from transformers import AutoConfig, T5ForConditionalGeneration, AutoTokenizer
import gradio as gr

class Inference:
    def __init__(
        self,
        config: ConfigurationManager
    ) -> None:
        self.config = config

    def init_word_segmentor(self):
        logger.info("Initialize word segmentor...")
        config = self.config.get_word_segmentor_config()
        try:
            save_dir = f"{CURRENT_WORKING_DIRECTORY}/{config.model_name_or_path}"
            self.rdrsegmenter = py_vncorenlp.VnCoreNLP(annotators=["wseg"], save_dir=save_dir)
            os.chdir(CURRENT_WORKING_DIRECTORY)
            logger.info("Done")
        except Exception as e:
            logger.exception(e)
            raise e
        return

    def init_search_model(self):
        logger.info("Initialize search model...")
        config = self.config.get_search_model_config()
        try:
            self.search_model = SentenceTransformer(
                config.model_name_or_path,
                device="cuda" if config.is_gpu else "cpu"
            )
            logger.info("Done")
        except Exception as e:
            logger.exception(e)
            raise e
        return

    def init_summary_model(self):
        logger.info("Initialize search model...")
        config = self.config.get_summary_model_config()
        try:
            checkpoint = config.model_name_or_path
            self.tokenizer = AutoTokenizer.from_pretrained(checkpoint)  
            c = AutoConfig.from_pretrained(checkpoint)
            self.model = T5ForConditionalGeneration.from_pretrained(
                checkpoint, 
                config=c
            ).to("cuda" if config.is_gpu else "cpu")
            logger.info("Done")
        except Exception as e:
            logger.exception(e)
            raise e
        return

    def init_interface(self):
        def process(text):
            logger.info("Searching news on VNE...")
            titles, links = search_post_link(text)
            logger.info("Done")
            logger.info("Filtering related news...")
            idx = choose_most_related_news(
                titles,
                self.rdrsegmenter,
                self.search_model
            )
            links = np.array(links)
            related_links = links[idx]
            logger.info(f"Done - {len(related_links)}")
            news_data = []
            logger.info("Crawling news content and summarizing...")
            for i, link in enumerate(related_links):
                print(i, end='\r')
                ret = crawl_content(link)
                ret['summary'] = summarize(
                    ret['content'],
                    self.tokenizer,
                    self.model
                )
                news_data.append(ret)
            big_news = "\n".join([data['summary'] for data in news_data])
            # big_news_summary = summarize(big_news)
            ref_links = '\n\t'.join([data['link'] for data in news_data])
            ret_format = f"""{big_news}

Nguồn:
\t{ref_links}
"""         
            logger.info(f'User: {text}\n\nResponse: \n{big_news}\n\nLinks: {ref_links}')
            return ret_format
        self.interface = gr.Interface(
            fn=process, 
            inputs="textbox", 
            outputs="textbox"
        )
        self.interface.queue()
    