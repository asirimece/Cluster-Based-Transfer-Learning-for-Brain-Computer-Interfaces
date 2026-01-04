import mne
from lib.logging import logger

logger = logger.get()


def unify_events(raw, unify_annotations, event_markers):
    """
    Unify raw event annotations.
    """
    events, _ = mne.events_from_annotations(raw, verbose=False)
    new_events = events.copy()

    for i, evt in enumerate(new_events):
        raw_code = evt[2]
        # Convert raw code to string
        code_str = str(raw_code)
        if code_str in unify_annotations:
            unified_label = unify_annotations[code_str]
            if unified_label in event_markers:
                mapped_value = event_markers[unified_label]
                new_events[i, 2] = mapped_value
                logger.info(f"Event {i}: raw code {raw_code} -> unified '{unified_label}' -> mapped {mapped_value}")
    
    new_event_id = {label: code for label, code in event_markers.items()}
    return new_events, new_event_id

