from utilities import create_result_directories, get_dialogue_uttr_dict_from_csv, create_scrambled_dialogues_txt, transform_txt_to_tsv

models_list = ["DialoGPT"]
model_res_dict = create_result_directories(models_list, to_local_directory=True)

"""Speaker A - M"""

source_dir_csv_A = model_res_dict["DialoGPT"]["directory_csv_A_M"] # "./Switchboard-Corpus/results/base_DialoGPT/csv/speaker_A_M/"

all_dialogue_uttr_dict_A_M = get_dialogue_uttr_dict_from_csv(source_dir_csv_A)

len(all_dialogue_uttr_dict_A_M)

output_dir = model_res_dict["DialoGPT"]["directory_scrambled_txt_A_M"]

create_scrambled_dialogues_txt(all_dialogue_uttr_dict_A_M, output_dir, speakers=["A", "M"])

transform_txt_to_tsv(model_res_dict["DialoGPT"]["directory_scrambled_txt_A_M"], 
                     "DialoGPT", 
                     model_res_dict["DialoGPT"]["directory_scrambled_tsv_A_M"],
                     speakers=["A", "M"])

"""Speaker B - M"""

source_dir_csv_B = model_res_dict["DialoGPT"]["directory_csv_B_M"] # "./Switchboard-Corpus/results/base_DialoGPT/csv/speaker_B_M/"

all_dialogue_uttr_dict_B_M = get_dialogue_uttr_dict_from_csv(source_dir_csv_B)
len(all_dialogue_uttr_dict_B_M)

output_dir = model_res_dict["DialoGPT"]["directory_scrambled_txt_B_M"]
create_scrambled_dialogues_txt(all_dialogue_uttr_dict_B_M, output_dir, speakers=["B", "M"])

transform_txt_to_tsv(model_res_dict["DialoGPT"]["directory_scrambled_txt_B_M"], 
                     "DialoGPT", 
                     model_res_dict["DialoGPT"]["directory_scrambled_tsv_B_M"],
                     speakers=["B", "M"])