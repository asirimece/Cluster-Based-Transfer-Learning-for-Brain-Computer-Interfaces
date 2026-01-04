import pickle
from omegaconf import DictConfig, OmegaConf
from lib.mtl.evaluate import MTLEvaluator
from lib.mtl.train import MTLTrainer
from lib.tl.evaluate import TargetTLEvaluator, TLEvaluator
from lib.tl.train import TLTrainer
from lib.pipeline.preprocess.preprocessor import Preprocessor
from lib.dataset.utils import save_preprocessed_data
from lib.pipeline.features.extract import FeatureExtractor, save_features
from lib.logging import logger

logger = logger.get()


def run(config: DictConfig) -> None:
     logger.info("==== Starting transfer learning pipeline ====")

     if not config.experiment.experiment.transfer.in_session.enabled:
          preprocessor = Preprocessor(config)
          preprocessed_data = preprocessor.run()
          save_preprocessed_data(preprocessed_data, config.dataset.preprocessing.output_file)
          
          features = FeatureExtractor.run(config, preprocessed_data, mode="cross-subject")
          save_features(features, config.transform.output_file)
          
          trainer = MTLTrainer(config, config.model)
          mtl_wrapper = trainer.run()

          evaluator = MTLEvaluator(mtl_wrapper, config)
          evaluator.evaluate()

          tl_trainer = TLTrainer(config)
          tl_results = tl_trainer.run()
          
          TLEvaluator(tl_results, config).evaluate()
     
     if config.experiment.experiment.transfer.in_session.enabled:
          preprocessor = Preprocessor(config)
          preprocessed_data = preprocessor.run()
          save_preprocessed_data(preprocessed_data, config.dataset.preprocessing.output_file)
          
          features = FeatureExtractor.run(config, preprocessed_data, mode="in-session")
          save_features(features, config.transform.output_file)
          
          logger.info("[In-Session] Skipping MTL pretraining.")
          
          tl_trainer = TLTrainer(config)
          tl_results = tl_trainer.run()
          
          TargetTLEvaluator(tl_trainer, config.experiment.experiment).evaluate()
          