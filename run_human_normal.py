from utilities import create_result_directories, transform_txt_to_tsv

# print(model_res_dict)
models_list = ["human"]
human_res_dict = create_result_directories(models_list, to_local_directory=True)

test_dir = './external/Switchboard-Corpus/swda_data/test/'
output_dir = human_res_dict["human"]["directory_tsv"]

transform_txt_to_tsv(test_dir, "human", output_dir)