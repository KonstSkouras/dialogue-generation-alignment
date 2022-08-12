# Evaluating alignment in contextualised dialogue generation
MSc AI Thesis Project code

Supervisor: Arabella J. Sinclair
## Data
The data for the below experiments are located in the external submodules in the `external` directory: More specifically it containes:
- `Switchboatd-Corpus`: It contains the Switchboard Data and utilities to process them and has been added as a git submodule. It is tracking the `develop` branch of [KonstSkouras/Switchboard-Corpus](https://github.com/KonstSkouras/Switchboard-Corpus/tree/develop). Forked repo with changes from [NathanDuran/Switchboard-Corpus](https://github.com/NathanDuran/Switchboard-Corpus). 
- `Maptask-Corpus`: It contains the Maptask Data and utilities to process them and has been added as a git submodule. It is tracking the `develop` branch of [KonstSkouras/Maptask-Corpus](https://github.com/KonstSkouras/Maptask-Corpus/tree/develop). Forked repo with changes from [NathanDuran/Maptask-Corpus](https://github.com/NathanDuran/Maptask-Corpus). 

In these forked repositories the data have been additionally processed  for the project's needs, i.e concatened utterances per speaker, enforce 80/10/10 split, add addional metadata and they are ready to be used for the below experiments.

In the `external` directory the `Concreteness_ratings_Brysbaert_et_al_BRM.xlsx` by [Brysbaert, M., Warriner, A.B., & Kuperman, V. (2014)](http://crr.ugent.be/archives/1330), is also downloaded in order to be used for the Statistics.
## Run Experiments
Results for the experiments: `results/{dataset}/expreriments/` ,

where `dataset` = `Switchboard-Corpus` or `Maptask-Corpus/`

Before runnning the below experiments you can the desired paremeters within the scripts:
- Dataset: `Switchboard-Corpus` or `Maptask-Corpus`
- Models: `DialoGPT` or `human`
- Model type: `base` or `finetuned`

The output direcotries will be created automatically with your chosen parameters if they don't exist.  
### Experiments Type 1: Alignement in Human - Human Dialogue
- `run_human_normal.py`
- `run_human_scramble.py`

Results already in `results/{dataset}/expreriments/base_human`.
### Experiments Type 2: Alignement in Human-Model Dialogue
Different models and tokenizers can be selected inside the scripts. Current default `DialoGPT` (DialoGPT-small).
- `run_model_normal_testAll.py`
- `run_model_scramble.py`

Results already in `results/{dataset}/expreriments/base_DialoGPT` for the base model normal and scrambled dialogues, as well as, in `results/{dataset}/expreriments/finetuned_DialoGPT` for the finetuned model with normal dialogues.
### Fine tuning
For finetuning the `train_dialogpt.ipynb` notebook from Nathan Cooper's [Tutorial](https://nathancooper.io/i-am-a-nerd/chatbot/deep-learning/gpt2/2020/05/12/chatbot-part-1.html) was used to finetune the model per dataset with slight modifications in Google Collab.
. Before running the notebook:
- Make sure the runtime enviroment is set to GPU.
- Upload the 
  - `train_contexed_5.csv` and 
  - `val_contexed_5.csv` 

from the `results/{dataset}/data_modified/` for the dataset you want to finetune.

Finetuned models in Hugging Face: [Switchboard](https://huggingface.co/skouras/DialoGPT-small-swda) and [Maptask](https://huggingface.co/skouras/DialoGPT-small-maptask).

## Run Dialign Software
- Download the dialign software from the [official repository](https://github.com/GuillaumeDD/dialign).
- Run the script `create_normalized_dialogues.py`, to create the appropriate normalized_tsvs and prepare the dialogues (lowercase)for Dialign.
- Run `create_directories.py` script to create the appropriate dialign output directories, if they do not exist already.
- For every dataset, for every experiment run `dialign.jar` by providing the appropriate input `normalized_tsv` directory and the desired output directory i.e. for Human - Human normal Maptask dialogues:

`java -jar dialign.jar -i ".\dialogue-generation-alignment\results\Maptask-Corpus\experiments\base_human\tsv_normalized\" -o ".\dialogue-generation-alignment\results\Maptask-Corpus\dialign_normalized\base_human\normal\"`

For every dataset, for all the conducted experiments there are 8 kind of dialogues to run the software for:
human normal, human scrambled, model normal A-M, model normal B-M, model scrambled A-M, model scrambled B-M, finetuned normal A-M, finetuned normal B-M   

Dialign outputs of all the conducted experiments are already in the directory `results\{dataset}\dialign_normalized\`. 

Finally, the created lexicons can be additionally filtered to only include expressions in a window span of 5, initiated by the Human and established by the model by running:
- `create_filtered_lexicons.py`

The filtered lexicons are in `results\{dataset}\dialign_filtered\`.

## Create statistics
In order to create them run the script:
- `create_output_statistics.py`

Before running the script you can choose the desired parameters for the statistics you want to create. These incluse:
- Parameters for the desired `dataset`, `model`, `model_type` as in the experiment scripts.
- type: `responses` or `expressions` 
- do_pos_tags = `True` or `False` for pos tags
- do_word_freq = `True` or `False` for word frequency 
- do_word_conc = `True` or `False` for word concretness

Output statistics per dataset, per experiment are already created in the directory: `statistics\{dataset}\`.
## Other scripts
- `create_directories.py`: Create results directories proactively. No need to run in advance for the experiments.
- `utilities.py`: Multiple utilities functions used by the experiments scripts.
- `train_dialogpt.py`: The same as `train_dialogpt.ipynb` in case the training needs to be done in a local machine rather than Colab.

## Folder Structure
- `external`: External Data and Utilities. 
  - `Switchboatd-Corpus`: It contains the Switchboard Data and utilities to process them and has been added as a git submodule. It is tracking the `develop` branch of [KonstSkouras/Switchboard-Corpus](https://github.com/KonstSkouras/Switchboard-Corpus/tree/develop). Forked repo with changes from [NathanDuran/Switchboard-Corpus](https://github.com/NathanDuran/Switchboard-Corpus).
  - `Maptak-Corpus`: similar to Switchboatd-Corpus.
- `results`: Results of running the experiments scripts, the dialign software etc.
  - `Switchboard-Corpus`
    - `data_modified`: Modified data from external directory. ie train/val data in contexed format ready for training/evaluation.
    - `dialign_output`: Output of running dialign from the .tsv produced in the results. (Deprecated, older experiments).
    - `dialign_normalized`: Output of running dialign from the .normalized_tsvs produced in the results. (Current active).
    - `dialign_filtered`: Filteted lexicons created from the dialign_normalized lexicons. (Current active).
    - `experiments`: Results of running the experiments scripts.
  - `Maptask-Corpus`: Similar to Switchboatd-Corpus.
- `statistics`: Statistics for analysis after conducting the experiments 
  - `Switchboard-Corpus`:
    - `{experiment_type}`: ie. base_human, base_DialoGPT, finetuned_DialoGPT
  - `Maptask-Corpus`: Similar with Switchboard-Corpus.



