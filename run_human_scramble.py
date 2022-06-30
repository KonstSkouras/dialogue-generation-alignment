from utilities import create_result_directories, get_dialogue_uttr_dict, create_scrambled_dialogues_txt, transform_txt_to_tsv

models_list = ["human"]
human_res_dict = create_result_directories(models_list, to_local_directory=True)

test_dir = './external/Switchboard-Corpus/swda_data/test/'
output_dir = human_res_dict["human"]["directory_tsv"]

all_dialogue_uttr_dict = get_dialogue_uttr_dict(test_dir)

output_dir = human_res_dict["human"]["directory_scrambled_txt"]

create_scrambled_dialogues_txt(all_dialogue_uttr_dict, output_dir)

transform_txt_to_tsv(human_res_dict["human"]["directory_scrambled_txt"], 
                     "human", 
                     human_res_dict["human"]["directory_scrambled_tsv"])

