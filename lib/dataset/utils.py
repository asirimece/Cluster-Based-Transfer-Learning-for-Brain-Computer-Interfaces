import os
import pickle
from lib.logging import logger

logger = logger.get()


def load_preprocessed_data(path: str):
    with open(path, 'rb') as f: 
        data = pickle.load(f)
    return data

def save_preprocessed_data(results: dict, output_file: str) -> None:
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    with open(output_file, "wb") as f:
        pickle.dump(results, f)
    logger.info(f"Preprocessed data saved to {output_file}")
    
def save_features(features, config, filename="./features.pkl"):
    methods_config = config.feature_extraction.get('methods', [])
    method_names = [str(method_config.get('name', 'unknown')) for method_config in methods_config]

    dr = config.feature_extraction.get('dimensionality_reduction', {}).get('name', '')
    fs = config.feature_extraction.get('feature_selection', {}).get('name', '')
    methods_str = "_".join(method_names)
    base, ext = os.path.splitext(filename)
    new_filename = f"{base}_{methods_str}_{dr}_{fs}{ext}"
    
    with open(new_filename, "wb") as f:
        pickle.dump(features, f)