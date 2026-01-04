
import os
from typing import Dict, List
from omegaconf import DictConfig
from moabb.datasets import BNCI2014_001
from collections import defaultdict
from .methods import (
    process_fif_subject,
    process_fif_subject_two_days,
    process_moabb_subject,
    split_fif_files_into_days,
)
from lib.logging import logger

logger = logger.get()


class Preprocessor:
    def __init__(self, config: DictConfig):
        self.config = config.dataset
        self.preproc_config = self.config.preprocessing
        self.data_source = self.config.source
        self.data_dir = self.config.get("home", None)

    def run(self) -> dict:
        if self.data_source == "fif":
            return self._run_fif()
        elif self.data_source == "moabb":
            return self._run_moabb()
        else:
            raise ValueError(f"Unsupported data source: {self.data_source}")

    def _run_fif(self) -> dict:
        """
        Run .fif recordings, including acquired subject recordings.
        Possibility to split files into days: `preprocessing.split_days` 
        and concatenate recordings: `preprocessing.concatenate_recordings`
        """
        results: Dict[str, dict] = {}

        if not os.path.isdir(self.data_dir):
            raise FileNotFoundError(f"FIF data directory not found: {self.data_dir}")

        fif_files = [os.path.join(self.data_dir, f) for f in os.listdir(self.data_dir) if f.endswith(".fif")]
        if len(fif_files) == 0:
            raise RuntimeError(f"No .fif files found in {self.data_dir}")

        fif_files = sorted(fif_files)
        split_days = bool(getattr(self.preproc_config, "split_days", True))

        # A simple subject label for the directory (assumes one acquired subject per folder)
        subj_label = getattr(self.preproc_config, "subject_label", "subject_001")

        if split_days:
            day1_files, day2_files = split_fif_files_into_days(fif_files)
            logger.info(f"\n--- Subject: {subj_label} | Day-1 files: {len(day1_files)} | Day-2 files: {len(day2_files)} ---")
            results[subj_label] = process_fif_subject_two_days(day1_files, day2_files, subj=subj_label, config=self.config)
            return results

        # No day split
        concat_enabled = bool(getattr(self.preproc_config, "concatenate_recordings", False))
        if concat_enabled:
            logger.info(f"\n--- Subject: {subj_label} ({len(fif_files)} recordings; concatenated) ---")
            results[subj_label] = process_fif_subject(fif_files, subj_label, self.config)
        else:
            for fname in fif_files:
                subj = os.path.basename(fname).split("_")[2] if "_" in os.path.basename(fname) else subj_label
                logger.info(f"\n--- Subject: {subj} ---")
                results[subj] = process_fif_subject(fname, subj, self.config)

        return results


    def _run_moabb(self) -> dict:
        dataset = BNCI2014_001()
        all_data = dataset.get_data()
        results = {}
        for subj in sorted(all_data.keys()):
            logger.info(f"\n--- Subject: {subj} ---")
            results[subj] = process_moabb_subject(all_data[subj], subj, self.config)
        return results
