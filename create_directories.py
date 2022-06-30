from utilities import create_result_directories

models_list = ["DialoGPT"]
model_res_dict = create_result_directories(models_list, to_local_directory=True)

# print(model_res_dict)
models_list = ["human"]
human_res_dict = create_result_directories(models_list, to_local_directory=True)

# print(human_res_dict)