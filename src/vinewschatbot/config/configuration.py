from pathlib import Path
from vinewschatbot.constants import *
from vinewschatbot.utils import *
from vinewschatbot.entity import *

class ConfigurationManager:
    def __init__(
        self,
        config_filepath: Path = CONFIG_FILE_PATH,
        params_filepath: Path = PARAMS_FILE_PATH
    ) -> None:
        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)

    def get_inference_server_config(self) -> Server:
        config = self.config.server
        return Server(
            share=config.share,
            server_name=config.server_name
        )

    def get_training_args_params(self) -> TrainingArgs:
        config = self.params.TrainingArgs
        return TrainingArgs(
            output_dir=config.output_dir,
            evaluation_strategy=config.evaluation_strategy,
            per_device_train_batch_size=config.per_device_train_batch_size,
            per_device_eval_batch_size=config.per_device_eval_batch_size,
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            warmup_ratio=config.warmup_ratio,
            eval_steps=config.eval_steps,
            report_to=config.report_to,
            logging_steps=config.logging_steps,
            learning_rate=config.learning_rate,
            lr_scheduler_type=config.lr_scheduler_type,
            max_steps=config.max_steps,
        )

    def get_generation_config(self) -> GenerationConfig:
        config = self.params.GenerationConfig
        return GenerationConfig(
            num_beams=config.num_beams,
            top_k=config.top_k,
            top_p=config.top_p,
            temperature=config.temperature,
            max_length=config.max_length,
            min_length=config.min_length,
            num_return_sequences=config.num_return_sequences,
            repetition_penalty=config.repetition_penalty,
            do_sample=config.do_sample,
            penalty_alpha=config.penalty_alpha
        )

    def get_word_segmentor_config(self) -> Word_Segmentor:
        config = self.config.word_segmentor
        return Word_Segmentor(
            source=config.source,
            model_name_or_path=config.model_name_or_path
        )

    def get_search_model_config(self) -> Search_Model:
        config = self.config.search_model
        return Search_Model(
            model_name_or_path=config.model_name_or_path,
            is_gpu=config.is_gpu
        )
    
    def get_summary_model_config(self) -> Summary_Model:
        config = self.config.summary_model
        return Summary_Model(
            base_model=config.base_model,
            model_name_or_path=config.model_name_or_path,
            is_gpu=config.is_gpu
        )
    
    def get_summary_dataset_config(self) -> Summary_Dataset:
        config = self.config.summary_dataset
        return Summary_Dataset(
            path=config.path
        )