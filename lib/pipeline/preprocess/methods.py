
import os
import re
import mne
import numpy as np
from .epoch import create_macro_epochs, crop_subepochs
from typing import List, Tuple, Dict
from lib.logging import logger


logger = logger.get()


_DAY_PATTERNS = [
    re.compile(r"(day[-_]?1|session[-_]?1|d1|s1)", re.IGNORECASE),
    re.compile(r"(day[-_]?2|session[-_]?2|d2|s2)", re.IGNORECASE),
]


def bandpass_filter(raw, low, high, method):
    return raw.copy().filter(l_freq=low, h_freq=high, method=method, verbose=False)


def apply_notch_filter(raw, freq):
    return raw.copy().notch_filter(freqs=[freq], verbose=False)

def resample_raw(raw, target_sfreq, npad="auto"):
    return raw.copy().resample(sfreq=target_sfreq, npad=npad, verbose=False)

def load_and_concatenate_fif_files(filepaths):
    raws = [mne.io.read_raw_fif(fp, preload=True, verbose=False) for fp in filepaths]
    return mne.concatenate_raws(raws)

def remove_eog_artifacts_ica(raw, eog_ch, n_components, method, random_state,
                             show_ica_plots, save_ica_plots, plots_output_dir,
                             subj_id=None, sess_label=None):
    raw.load_data()
    if save_ica_plots:
        abs_dir = os.path.abspath(plots_output_dir)
        os.makedirs(abs_dir, exist_ok=True)
    else:
        abs_dir = plots_output_dir

    ica = mne.preprocessing.ICA(n_components=n_components, method=method, random_state=random_state, verbose=False)
    ica.fit(raw)

    eog_inds_total = []
    for ch in eog_ch:
        if ch in raw.ch_names:
            eog_inds, _ = ica.find_bads_eog(raw, ch_name=ch)
            eog_inds_total.extend(eog_inds)
            logger.info(f"[ICA] Detected EOG components {eog_inds} correlating with channel {ch}.")

    eog_inds_total = list(set(eog_inds_total))
    logger.info(f"[ICA] Total EOG-related components to exclude: {eog_inds_total}")
    ica.exclude.extend(eog_inds_total)

    fig_label = f"subj{subj_id}_{sess_label}" if subj_id else ""

    figs_components = ica.plot_components(show=show_ica_plots)
    if save_ica_plots and figs_components:
        if isinstance(figs_components, list):
            for i, fig in enumerate(figs_components):
                fig.savefig(os.path.join(abs_dir, f"ica_components_{fig_label}_page{i}.png"))
        else:
            figs_components.savefig(os.path.join(abs_dir, f"ica_components_{fig_label}.png"))

    fig_sources = ica.plot_sources(raw, show=show_ica_plots)
    if save_ica_plots and fig_sources:
        fig_sources.savefig(os.path.join(abs_dir, f"ica_sources_{fig_label}.png"))

    ica.apply(raw)
    logger.info(f"[ICA] Applied ICA. Excluded {len(ica.exclude)} components.")
    return raw

def data_split_concatenate(subj_data, train_session, test_session):
    def get_session_data(data):
        if isinstance(data, dict):
            runs = list(data.values())
            return mne.concatenate_raws(runs) if len(runs) > 1 else runs[0]
        return data
    return get_session_data(subj_data.get(train_session)), get_session_data(subj_data.get(test_session))

def exponential_moving_standardization(epochs, smoothing_factor, eps, esm_params=None, return_params=False):
    data = epochs.get_data()
    n_epochs, n_channels, n_times = data.shape
    standardized_data = np.zeros_like(data)

    if esm_params is None:
        initial_means = np.mean(data[:, :, 0], axis=0)
        initial_vars = np.var(data, axis=(0, 2))
        esm_params = {"initial_means": initial_means, "initial_vars": initial_vars}

    initial_means = esm_params["initial_means"]
    initial_vars = esm_params["initial_vars"]

    for ep in range(n_epochs):
        for ch in range(n_channels):
            x = data[ep, ch, :]
            running_mean = initial_means[ch]
            running_var = initial_vars[ch]
            standardized_signal = np.zeros_like(x)
            for t in range(len(x)):
                running_mean = smoothing_factor * x[t] + (1 - smoothing_factor) * running_mean if t != 0 else running_mean
                running_var = smoothing_factor * ((x[t] - running_mean) ** 2) + (1 - smoothing_factor) * running_var if t != 0 else running_var
                standardized_signal[t] = (x[t] - running_mean) / np.sqrt(running_var + eps)
            standardized_data[ep, ch, :] = standardized_signal

    standardized_epochs = epochs.copy().load_data()
    standardized_epochs._data = standardized_data
    return (standardized_epochs, esm_params) if return_params else standardized_epochs

def train_test_event_split(epochs, train_frac=0.7, random_seed=42, shuffle=True):
    n_trials = len(epochs)
    indices = np.arange(n_trials)
    if shuffle:
        rng = np.random.default_rng(random_seed)
        rng.shuffle(indices)
    n_train = int(np.floor(n_trials * train_frac))
    return epochs[indices[:n_train]], epochs[indices[n_train:]]

def process_fif_subject(fpaths, subj, config):
    """
    Manages preprocessing of .fif recordings.
    """
    preproc_config = config.preprocessing 
    paths = fpaths if isinstance(fpaths, list) else [fpaths]

    all_subepochs = []
    for sess_i, fp in enumerate(paths):
        raw = mne.io.read_raw_fif(fp, preload=True, verbose=False)
        raw.pick_channels(list(config.eeg_channels))

        if (
            hasattr(preproc_config.raw_preprocessors, "resample") and
            hasattr(preproc_config.raw_preprocessors.resample, "kwargs")
        ):
            raw = resample_raw(
                raw,
                target_sfreq=preproc_config.raw_preprocessors.resample.kwargs.target_sfreq,
                npad=preproc_config.raw_preprocessors.resample.kwargs.get("npad", "auto")
            )

        if preproc_config.raw_preprocessors.get("notch_filter"):
            raw = apply_notch_filter(
                raw,
                freq=preproc_config.raw_preprocessors.notch_filter.kwargs.freq
            )

        if preproc_config.raw_preprocessors.get("bandpass_filter"):
            raw = bandpass_filter(
                raw,
                low=preproc_config.raw_preprocessors.bandpass_filter.kwargs.low,
                high=preproc_config.raw_preprocessors.bandpass_filter.kwargs.high,
                method=preproc_config.raw_preprocessors.bandpass_filter.kwargs.method
            )

        if preproc_config.remove_eog_artifacts:
            raw = remove_eog_artifacts_ica(
                raw,
                eog_ch=preproc_config.eog_channels,
                n_components=preproc_config.ica.kwargs.n_components,
                method=preproc_config.ica.kwargs.method,
                random_state=preproc_config.ica.kwargs.random_state,
                show_ica_plots=preproc_config.show_ica_plots,
                save_ica_plots=preproc_config.save_ica_plots,
                plots_output_dir=preproc_config.ica_plots_dir,
                subj_id=subj,
                sess_label=f"run{sess_i}"
            )

        raw.pick("eeg")

        macro = create_macro_epochs(raw, preproc_config)
        sub   = crop_subepochs(
                    macro,
                    preproc_config.epoching.kwargs.crop_window_length,
                    preproc_config.epoching.kwargs.crop_step_size
                )
        all_subepochs.append(sub)

    sub_epochs = mne.concatenate_epochs(all_subepochs)

    train_frac  = preproc_config.data_split.kwargs.train_fraction
    random_seed = preproc_config.data_split.kwargs.random_seed

    train_epochs, test_epochs = train_test_event_split(
        sub_epochs,
        train_frac=train_frac,
        random_seed=random_seed,
        shuffle=True
    )

    standardized_train, esm_params = exponential_moving_standardization(
        train_epochs,
        smoothing_factor=preproc_config.exponential_moving_standardization.kwargs.smoothing_factor,
        eps=preproc_config.exponential_moving_standardization.kwargs.eps,
        return_params=True
    )

    standardized_test = exponential_moving_standardization(
        test_epochs,
        smoothing_factor=preproc_config.exponential_moving_standardization.kwargs.smoothing_factor,
        eps=preproc_config.exponential_moving_standardization.kwargs.eps,
        esm_params=esm_params
    )

    return {"train": standardized_train, "test": standardized_test}


def process_moabb_subject(subj_data, subj, config):
    """
    Manages preprocessing of MOABB datasets
    """
    preproc_config = config.preprocessing
    train_session = preproc_config.data_split.kwargs.train_session
    test_session = preproc_config.data_split.kwargs.test_session

    train_raw, test_raw = data_split_concatenate(subj_data, train_session, test_session)

    train_raw.pick_channels(list(config.eeg_channels))
    test_raw.pick_channels(list(config.eeg_channels))

    if (
        hasattr(preproc_config.raw_preprocessors, "resample") and
        hasattr(preproc_config.raw_preprocessors.resample, "kwargs")
    ):
        target_sfreq = preproc_config.raw_preprocessors.resample.kwargs.target_sfreq
        npad         = preproc_config.raw_preprocessors.resample.kwargs.get("npad", "auto")
        train_raw = resample_raw(train_raw, target_sfreq, npad)
        test_raw  = resample_raw(test_raw, target_sfreq, npad)

    if preproc_config.raw_preprocessors.get("notch_filter"):
        train_raw = apply_notch_filter(
            train_raw, freq=preproc_config.raw_preprocessors.notch_filter.kwargs.freq
        )
        test_raw = apply_notch_filter(
            test_raw, freq=preproc_config.raw_preprocessors.notch_filter.kwargs.freq
        )

    if preproc_config.raw_preprocessors.get("bandpass_filter"):
        train_raw = bandpass_filter(
            train_raw,
            low=preproc_config.raw_preprocessors.bandpass_filter.kwargs.low,
            high=preproc_config.raw_preprocessors.bandpass_filter.kwargs.high,
            method=preproc_config.raw_preprocessors.bandpass_filter.kwargs.method
        )
        test_raw = bandpass_filter(
            test_raw,
            low=preproc_config.raw_preprocessors.bandpass_filter.kwargs.low,
            high=preproc_config.raw_preprocessors.bandpass_filter.kwargs.high,
            method=preproc_config.raw_preprocessors.bandpass_filter.kwargs.method
        )

    if preproc_config.remove_eog_artifacts:
        train_raw = remove_eog_artifacts_ica(
            train_raw,
            eog_ch=config.eog_channels,
            n_components=preproc_config.ica.kwargs.n_components,
            method=preproc_config.ica.kwargs.method,
            random_state=preproc_config.ica.kwargs.random_state,
            show_ica_plots=preproc_config.show_ica_plots,
            save_ica_plots=preproc_config.save_ica_plots,
            plots_output_dir=preproc_config.ica_plots_dir,
            subj_id=subj,
            sess_label=train_session
        )
        test_raw = remove_eog_artifacts_ica(
            test_raw,
            eog_ch=config.eog_channels,
            n_components=preproc_config.ica.kwargs.n_components,
            method=preproc_config.ica.kwargs.method,
            random_state=preproc_config.ica.kwargs.random_state,
            show_ica_plots=preproc_config.show_ica_plots,
            save_ica_plots=preproc_config.save_ica_plots,
            plots_output_dir=preproc_config.ica_plots_dir,
            subj_id=subj,
            sess_label=test_session
        )

    train_raw.pick_types(eeg=True, stim=False, exclude=[])
    test_raw.pick_types(eeg=True, stim=False, exclude=[])

    train_epochs = crop_subepochs(
        create_macro_epochs(train_raw, preproc_config),
        preproc_config.epoching.kwargs.crop_window_length,
        preproc_config.epoching.kwargs.crop_step_size
    )

    test_epochs = crop_subepochs(
        create_macro_epochs(test_raw, preproc_config),
        preproc_config.epoching.kwargs.crop_window_length,
        preproc_config.epoching.kwargs.crop_step_size
    )

    standardized_train, esm_params = exponential_moving_standardization(
        train_epochs,
        smoothing_factor=preproc_config.exponential_moving_standardization.kwargs.smoothing_factor,
        eps=preproc_config.exponential_moving_standardization.kwargs.eps,
        return_params=True
    )

    standardized_test = exponential_moving_standardization(
        test_epochs,
        smoothing_factor=preproc_config.exponential_moving_standardization.kwargs.smoothing_factor,
        eps=preproc_config.exponential_moving_standardization.kwargs.eps,
        esm_params=esm_params
    )

    return {
        train_session: standardized_train,
        test_session: standardized_test
    }


def split_fif_files_into_days(files: List[str]) -> Tuple[List[str], List[str]]:
    """Split filenames into Day-1 and Day-2 buckets using filename hints; fallback: first half vs second half."""
    if not files:
        return [], []
    files = sorted(files)
    day1, day2 = [], []
    for fp in files:
        name = os.path.basename(fp)
        if _DAY_PATTERNS[0].search(name):
            day1.append(fp)
        elif _DAY_PATTERNS[1].search(name):
            day2.append(fp)
    if len(day1) == 0 and len(day2) == 0:
        # fallback: first half:day1, second half:day2
        mid = len(files) // 2
        day1, day2 = files[:mid], files[mid:]
    logger.info(f"[split_fif_files_into_days] day1={len(day1)} day2={len(day2)}")
    return day1, day2

def _concat_and_preprocess_runs(
    file_list: List[str],
    subj: str,
    config,
    day_label: str
) -> mne.io.BaseRaw:
    """Load, resample, notch, bandpass, ICA/EOG per run, then concatenate runs."""
    preproc = config.preprocessing
    raws = []
    for i, fp in enumerate(sorted(file_list)):
        raw = mne.io.read_raw_fif(fp, preload=True, verbose=False)
        raw.pick_channels(list(config.eeg_channels))

        if hasattr(preproc.raw_preprocessors, "resample") and hasattr(preproc.raw_preprocessors.resample, "kwargs"):
            raw = resample_raw(raw,
                target_sfreq=preproc.raw_preprocessors.resample.kwargs.target_sfreq,
                npad=preproc.raw_preprocessors.resample.kwargs.get("npad", "auto"))

        if preproc.raw_preprocessors.get("notch_filter"):
            raw = apply_notch_filter(raw, freq=preproc.raw_preprocessors.notch_filter.kwargs.freq)

        if preproc.raw_preprocessors.get("bandpass_filter"):
            raw = bandpass_filter(
                raw,
                low=preproc.raw_preprocessors.bandpass_filter.kwargs.low,
                high=preproc.raw_preprocessors.bandpass_filter.kwargs.high,
                method=preproc.raw_preprocessors.bandpass_filter.kwargs.method
            )

        if preproc.remove_eog_artifacts:
            raw = remove_eog_artifacts_ica(
                raw,
                eog_ch=config.eog_channels,
                n_components=preproc.ica.kwargs.n_components,
                method=preproc.ica.kwargs.method,
                random_state=preproc.ica.kwargs.random_state,
                show_ica_plots=preproc.show_ica_plots,
                save_ica_plots=preproc.save_ica_plots,
                plots_output_dir=preproc.ica_plots_dir,
                subj_id=subj,
                sess_label=f"{day_label}_run{i}"
            )

        raw.pick("eeg")
        raws.append(raw)

    if len(raws) == 0:
        raise RuntimeError(f"No runs for {day_label}")
    return mne.concatenate_raws(raws) if len(raws) > 1 else raws[0]

def process_fif_subject_two_days(
    day1_files: List[str],
    day2_files: List[str],
    subj: str,
    config
) -> Dict[str, mne.Epochs]:
    """
    Day-aware path:
      - Day-1 files → preprocess+concat → epoch → TRAIN
      - Day-2 files → preprocess+concat → epoch → TEST
      - ESM: fit on TRAIN, apply same params to TEST
    """
    preproc = config.preprocessing

    # Day-1 → TRAIN
    raw_day1 = _concat_and_preprocess_runs(day1_files, subj, config, "day1")
    train_macro = create_macro_epochs(raw_day1, preproc)
    train_epochs = crop_subepochs(
        train_macro,
        preproc.epoching.kwargs.crop_window_length,
        preproc.epoching.kwargs.crop_step_size
    )

    # Day-2 → TEST
    raw_day2 = _concat_and_preprocess_runs(day2_files, subj, config, "day2")
    test_macro = create_macro_epochs(raw_day2, preproc)
    test_epochs = crop_subepochs(
        test_macro,
        preproc.epoching.kwargs.crop_window_length,
        preproc.epoching.kwargs.crop_step_size
    )

    # ESM: fit on TRAIN, reuse on TEST
    standardized_train, esm_params = exponential_moving_standardization(
        train_epochs,
        smoothing_factor=preproc.exponential_moving_standardization.kwargs.smoothing_factor,
        eps=preproc.exponential_moving_standardization.kwargs.eps,
        return_params=True
    )
    standardized_test = exponential_moving_standardization(
        test_epochs,
        smoothing_factor=preproc.exponential_moving_standardization.kwargs.smoothing_factor,
        eps=preproc.exponential_moving_standardization.kwargs.eps,
        esm_params=esm_params
    )

    logger.info(f"[process_fif_subject_two_days] subj={subj} "
                f"train={len(standardized_train)} test={len(standardized_test)}")
    return {"train": standardized_train, "test": standardized_test}

