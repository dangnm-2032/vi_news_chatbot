from dataclasses import dataclass
from pathlib import Path

@dataclass
class Word_Segmentor:
    source: Path
    model_name_or_path: Path

@dataclass
class Search_Model:
    model_name_or_path: Path
    is_gpu: bool

@dataclass
class Summary_Model:
    base_model: Path
    model_name_or_path: Path
    is_gpu: bool

@dataclass
class Summary_Dataset:
    path: Path

@dataclass
class TrainingArgs:
    output_dir: Path
    evaluation_strategy: str
    per_device_train_batch_size: int
    per_device_eval_batch_size: int
    gradient_accumulation_steps: int
    warmup_ratio: float
    eval_steps: int
    report_to: str
    logging_steps: int
    learning_rate: float
    lr_scheduler_type: str
    max_steps: int

@dataclass
class GenerationConfig:
    num_beams: int
    top_k: int
    top_p: float
    temperature: float
    max_length: int
    min_length: int
    num_return_sequences: int
    repetition_penalty: float
    do_sample: bool
    penalty_alpha: float

@dataclass
class Server:
    share: bool
    server_name: str