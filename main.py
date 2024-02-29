from vinewschatbot.logging import logger
from vinewschatbot.pipeline import *
from vinewschatbot.utils import *
from vinewschatbot.config import ConfigurationManager
from vinewschatbot.constants import *
from multiprocessing import Process
disable_caching()

def main():
    config_manager = ConfigurationManager(
        config_filepath=CONFIG_FILE_PATH
    )

    try:
        STAGE_NAME = stage_name("STAGE 1: VALIDATE PROGRAM")
        logger.info(STAGE_NAME)
        validate_program = ValidateProgramPipeline(config=config_manager)
        validate_program.main()
    except Exception as e:
        logger.exception(e)
        raise e
    
    try:
        STAGE_NAME = stage_name("STAGE 2: CREATE TRAINING DATASET")
        logger.info(STAGE_NAME)
        create_dataset = CreateDatasetPipeline(config=config_manager)
        create_dataset.main()
    except Exception as e:
        logger.exception(e)
        raise e
    
    try:
        STAGE_NAME = stage_name("STAGE 3: TRAIN SUMMARY MODEL")
        logger.info(STAGE_NAME)
        training = TrainingPipeline(config=config_manager)
        training.main()
    except Exception as e:
        logger.exception(e)
        raise e

    try:
        STAGE_NAME = stage_name("STAGE 4: RUNNING INFERENCE")
        logger.info(STAGE_NAME)
        inference = InferencePipeline(config=config_manager)
        proc = Process(target=inference.main)
        proc.start()
        proc.join()
    except KeyboardInterrupt as k:
        logger.exception(k)
        logger.info("Program shutdown")
    except Exception as e:
        logger.exception(e)
        raise e

if __name__ == "__main__":
    torch.multiprocessing.set_start_method('spawn')
    main()