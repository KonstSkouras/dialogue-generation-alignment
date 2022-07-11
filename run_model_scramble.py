import os
from utilities import create_result_directories, get_dialogue_uttr_dict_from_csv, create_scrambled_dialogues_txt, transform_txt_to_tsv, get_dataset_settings

# dataset = "Switchboard-Corpus"
dataset = "Maptask-Corpus"
model_type = "base"

models_list = ["DialoGPT"]

# data_dir = './external/Switchboard-Corpus/swda_data/test/'
test_dir, speakers = get_dataset_settings(dataset)
model_res_dict = create_result_directories(models_list, to_local_directory=True, dataset=dataset, model_type="base", speakers=speakers)

# print(model_res_dict)

for model_name in models_list:
  full_model_name = model_type + "_" + model_name
  for speaker in speakers:
    dir_name = "directory_csv_" + speaker + "_M" # ie directory_csv_A_M 
    source_dir_csv = model_res_dict[full_model_name][dir_name] # "./Switchboard-Corpus/results/base_DialoGPT/csv/speaker_A_M/"

    all_dialogue_uttr_dict_speaker_M = get_dialogue_uttr_dict_from_csv(source_dir_csv)
    # print(len(all_dialogue_uttr_dict_speaker_M))

    out_dir_name_txt = "directory_scrambled_txt_" + speaker + "_M" #i.e directory_scrambled_txt_A_M
    output_dir = model_res_dict[full_model_name][out_dir_name_txt]
    if(len(os.listdir(output_dir)) > 1):
      print("Output directory:", output_dir)
      print("Output directory is not empty. Clear the directory before running the experiment.")
    else:
      create_scrambled_dialogues_txt(all_dialogue_uttr_dict_speaker_M, output_dir, speakers=[speaker,"M"])

      out_dir_name_tsv = "directory_scrambled_tsv_" + speaker + "_M" #i.e directory_scrambled_txt_A_M
      transform_txt_to_tsv(model_res_dict[full_model_name][out_dir_name_txt], 
                          model_name, 
                          model_res_dict[full_model_name][out_dir_name_tsv],
                          speakers=[speaker, "M"])


# """Speaker B - M"""
# speaker_2 = speakers[0]

# source_dir_csv_B = model_res_dict["DialoGPT"]["directory_csv_B_M"] # "./Switchboard-Corpus/results/base_DialoGPT/csv/speaker_B_M/"

# all_dialogue_uttr_dict_B_M = get_dialogue_uttr_dict_from_csv(source_dir_csv_B)
# len(all_dialogue_uttr_dict_B_M)

# output_dir = model_res_dict["DialoGPT"]["directory_scrambled_txt_B_M"]
# create_scrambled_dialogues_txt(all_dialogue_uttr_dict_B_M, output_dir, speakers=["B", "M"])

# transform_txt_to_tsv(model_res_dict["DialoGPT"]["directory_scrambled_txt_B_M"], 
#                      "DialoGPT", 
#                      model_res_dict["DialoGPT"]["directory_scrambled_tsv_B_M"],
#                      speakers=["B", "M"])