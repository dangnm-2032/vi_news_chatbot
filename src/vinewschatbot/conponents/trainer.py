from vinewschatbot.constants import *
from vinewschatbot.logging import logger
from vinewschatbot.config import *
from vinewschatbot.utils import *
from datasets import load_from_disk
from transformers import AutoConfig, T5ForConditionalGeneration, AutoTokenizer, GenerationConfig
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments, DataCollatorWithPadding
import wandb

class Trainer:
    def __init__(
        self,
        config: ConfigurationManager
    ) -> None:
        self.config = config

    def train(self):
        wandb_config = self.config.get_wandb_config()
        wandb.login(
            key=wandb_config.key
        )
        run = wandb.init(
            # Set the project where this run will be logged
            project="Vi_news_chatbot",
            # Track hyperparameters and run metadata
            # config={
            #     "learning_rate": lr,
            #     "epochs": epochs,
            # },
        )
        model_config = self.config.get_summary_model_config()
        ds_config = self.config.get_summary_dataset_config()
        base_model = model_config.base_model
        tokenizer = AutoTokenizer.from_pretrained(base_model)  
        ds = load_from_disk(ds_config.path)
        stat_ds = ds.map(get_len, num_proc=20, fn_kwargs={'tokenizer':tokenizer})
        filter_ds = stat_ds.filter(
            lambda x: (
                x['summary_len'] >= 41 and
                x['summary_len'] <= 129 and
                x['text_len'] >= 370 and
                x['text_len'] <= 896
            )
        )
        clean_ds = filter_ds.map(clean_data_text)
        config = AutoConfig.from_pretrained(base_model)
        tokenizer = AutoTokenizer.from_pretrained(base_model)  
        model = T5ForConditionalGeneration.from_pretrained(base_model, config=config)
        model.cuda()
        
        transform_ds = clean_ds.map(transform, num_proc=20, fn_kwargs={'tokenizer':tokenizer}).with_format('torch')
        ds = transform_ds.train_test_split(test_size=0.1).with_format('torch')
        data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

        tagrs = self.config.get_training_args_params()
        genconfig = self.config.get_generation_config()
        training_args = Seq2SeqTrainingArguments(
            output_dir=tagrs.output_dir,
            evaluation_strategy=tagrs.evaluation_strategy,
            per_device_train_batch_size=tagrs.per_device_train_batch_size,
            per_device_eval_batch_size=tagrs.per_device_eval_batch_size,
            gradient_accumulation_steps=tagrs.gradient_accumulation_steps,
            warmup_ratio=tagrs.warmup_ratio,
            eval_steps=tagrs.eval_steps,
            report_to=tagrs.report_to,
            logging_steps=tagrs.logging_steps,
            learning_rate=tagrs.learning_rate,
            lr_scheduler_type=tagrs.lr_scheduler_type,
            max_steps=tagrs.max_steps,
            generation_config=GenerationConfig(
                num_beams=genconfig.num_beams,
                top_k=genconfig.top_k,
                top_p=genconfig.top_p,
                temperature=genconfig.temperature,
                max_length=genconfig.max_length,
                min_length=genconfig.min_length,
                num_return_sequences=genconfig.num_return_sequences,
                repetition_penalty=genconfig.repetition_penalty,
                do_sample=genconfig.do_sample,
                penalty_alpha=genconfig.penalty_alpha
            )
        )

        trainer = Seq2SeqTrainer(
            model=model,
            args=training_args,
            train_dataset=ds['train'],
            eval_dataset=ds['test'],
            data_collator=data_collator,
        )
        trainer.train()
        save_name = model_config.model_name_or_path
        trainer.save_model(save_name)
        tokenizer.save_pretrained(save_name)
        config.save_pretrained(save_name)