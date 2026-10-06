# Dataset registry

The registry is metadata-only. Access class is enforced at the adapter
boundary, and no raw clinical data are redistributed by this repository.

| ID | Access | Modalities / use | Source |
|---|---|---|---|
| `mitdb` | OPEN | ECG, arrhythmia | [PhysioNet](https://physionet.org/content/mitdb/) |
| `afdb` | OPEN | ECG, AF | [PhysioNet](https://physionet.org/content/afdb/) |
| `physionet_challenge` | OPEN | challenge-specific multimodal tasks | [PhysioNet](https://physionet.org/about/database/) |
| `mimic_waveform` | CREDENTIALED | ECG/PPG/BP/respiration, deterioration/hypotension | [PhysioNet](https://physionet.org/content/mimic4wdb/) |
| `mimic_iv_ecg` | CREDENTIALED | ECG and diagnoses | [PhysioNet](https://physionet.org/content/mimic-iv-ecg/) |
| `sleep_edf` | OPEN | EEG/EOG/EMG, sleep staging | [PhysioNet](https://physionet.org/content/sleep-edfx/) |
| `wesad` | OPEN | ECG/EDA/temperature/respiration/IMU, stress | [UCI](https://archive.ics.uci.edu/dataset/465/wesad) |
| `pamap2` | OPEN | IMU/ECG/temperature, activity | [UCI](https://archive.ics.uci.edu/dataset/231/pamap2+physical+activity+monitoring) |
| `gaitpdb` | OPEN | gait, mobility/rehabilitation research | [PhysioNet](https://physionet.org/content/gaitpdb/1.0.0/) |

Each entry in `src/physioworld/datasets/registry.py` records citation,
version, subject unit, preprocessing, access instructions, limitations, and
source URL. Version/checksum must be recorded from the provider at acquisition.
Availability, license terms and credential requirements must be checked against
the provider for the specific release being used. See the [dataset guide](dataset_guide.md).
