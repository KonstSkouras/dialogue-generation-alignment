from utilities import create_result_directories

models_list = ["DialoGPT"]
create_result_directories(models_list, to_local_directory=True, dataset="Switchboard-Corpus", model_type="base")
create_result_directories(models_list, to_local_directory=True, dataset="Switchboard-Corpus", model_type="finetuned")
create_result_directories(models_list, to_local_directory=True, dataset="Maptask-Corpus", model_type="base", speakers=["f", "g"])
# print(model_res_dict)

models_list = ["human"]
create_result_directories(models_list, to_local_directory=True, dataset="Switchboard-Corpus")
create_result_directories(models_list, to_local_directory=True, dataset="Maptask-Corpus", model_type="base", speakers=["f", "g"])
# print(human_res_dict)