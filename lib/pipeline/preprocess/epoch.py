import numpy as np
import mne
from omegaconf import OmegaConf

def create_macro_epochs(raw: mne.io.Raw, dataset_config) -> mne.Epochs:
    """
    Create macro MNE Epochs from tmin to tmax around each event.
    """
    tmin = dataset_config.epoching.kwargs.tmin
    tmax = dataset_config.epoching.kwargs.tmax
    
    events, event_id = mne.events_from_annotations(raw, verbose=False)
    epochs = mne.Epochs(
        raw, events, event_id=event_id,
        tmin=tmin, tmax=tmax,
        baseline=None, preload=True, verbose=False
    )
    return epochs

def extract_time_locked_epochs(raw, tmin, tmax, keep_codes=(1,2)):
    """
    Extract epochs for left hand, right hand, and rest events.
    """
    tmin = float(OmegaConf.to_container(tmin, resolve=True)) if hasattr(tmin, "keys") else float(tmin)
    tmax = float(OmegaConf.to_container(tmax, resolve=True)) if hasattr(tmax, "keys") else float(tmax)

    events, event_id = mne.events_from_annotations(raw, verbose=False)
    event_id = {name: code for name, code in event_id.items() if code in keep_codes}
    epochs = mne.Epochs(
        raw, events, event_id=event_id,
        tmin=tmin, tmax=tmax,
        baseline=None, preload=True, verbose=False
    )
    return epochs

def crop_subepochs(epochs, window_length, step_size, keep_codes=(1,2)):
    """
    Crops epochs to subepochs.
    """
    sfreq = epochs.info['sfreq']
    data = epochs.get_data()
    events = epochs.events

    mask = np.isin(events[:, -1], keep_codes)
    data = data[mask]
    events = events[mask]

    window_samples = int(round(window_length * sfreq))
    step_samples   = int(round(step_size   * sfreq))

    all_crops = []
    all_events = []
    counter   = 0

    label_map = {code: idx for idx, code in enumerate(keep_codes)}

    for trial_idx, raw_label in enumerate(events[:, -1]):
        mapped_label = label_map[raw_label]
        trial_data   = data[trial_idx]
        n_sub        = 1 + (trial_data.shape[1] - window_samples) // step_samples

        for j in range(n_sub):
            start = j * step_samples
            end   = start + window_samples
            if end > trial_data.shape[1]:
                break
            all_crops.append(trial_data[:, start:end])
            all_events.append([counter, trial_idx, mapped_label])
            counter += 1

    all_crops  = np.stack(all_crops)
    all_events = np.array(all_events, int)

    new_info = epochs.info.copy()
    new_tmin = 0.0

    new_epochs = mne.EpochsArray(
        data=all_crops,
        info=new_info,
        events=all_events,
        tmin=new_tmin,
        baseline=None,
        verbose=False
    )
    return new_epochs

def time_lock_and_slide_epochs(raw, tmin, tmax, window_length, step_size, keep_codes=(1,2)):
    epochs = extract_time_locked_epochs(raw, tmin, tmax, keep_codes=keep_codes)
    new_epochs = crop_subepochs(epochs, window_length, step_size, keep_codes=keep_codes)
    return new_epochs
