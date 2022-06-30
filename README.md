# dialogue-generation-alignment
Alignment in contextualised dialogue generation

(To be updated with more info ie requirements, visuals etc)

## Run Experiments
Results directory for experiments (Switchboard): `results/Switchboard-Corpus/expreriments/`
### Experiments Type 1: Alignement in Human - Human Dialogue
- `run_human_normal.py`
- `run_human_scramble.py`

Results already in `results/Switchboard-Corpus/expreriments/base_human`.
### Experiments Type 2: Alignement in Human-Model Dialogue
Different models and tokenizers can be selected inside the scripts. Current default `DialoGPT` (DialoGPT-small).
- `run_model_normal_testAll.py`: Currently without parallelization. Takes ~2.5 hours for DialoGPT-small.
- `run_model_scramble.py`

Results already in `results/Switchboard-Corpus/expreriments/base_DialoGPT`.
### Fine tuning
(Coming)

# Folder Structure
- `external`: External Data and Utilities. 
  - `Switchboatd-Corpus`: It contains the Switchboard Data and utilities to process them and has been added as a git submodule. It is tracking the `develop` branch of [KonstSkouras/Switchboard-Corpus](https://github.com/KonstSkouras/Switchboard-Corpus/tree/develop). Forked repo with changes from [NathanDuran/Switchboard-Corpus](https://github.com/NathanDuran/Switchboard-Corpus).
  - `Maptak-Corpus` (coming).
- `results`: Results of running the experiments scripts, the dialign software etc.
  - `Switchboard-Corpus`: Results of the above scripts and dialign for Switchboard.
    - `data_modified`: Modified data from external directory. ie train/val data in contexed format.
    - `dialign_output`: Output of running dialign from the .tsv produced in the results. 
    - `experiments`: Results of running the experiments scripts.
  - `Maptask-Corpus`: (Coming)



