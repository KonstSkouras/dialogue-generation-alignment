# import matplotlib
import os
from io import StringIO
import csv
import pandas as pd
from collections import namedtuple, Counter
import random
import itertools
from datetime import datetime
import spacy


HISTORY_CONTEXT_LENGTH = 5

### Create Resutls Repositories
def create_directory_safe(directory):
  if not os.path.exists(directory):
    os.makedirs(directory)
    print("Creating directory %s", directory)

def create_result_directories(models_list, to_local_directory=False, dataset="Switchboard-Corpus", model_type="base", speakers=["A", "B"]):  
  if to_local_directory:
    res_directory = "./results/" + dataset + "/experiments/"
    data_directory = "./results/" + dataset + "/data_modified/"
    stats_directory = "./statistics/" + dataset + "/"
  else:
    res_directory = "/content/results/" + dataset + "/experiments/" 
    data_directory = "/content/results/" + dataset + "/data_modified/"
    stats_directory = "/content/statistics/" + dataset + "/"


  create_directory_safe(res_directory)
  create_directory_safe(data_directory)
  create_directory_safe(stats_directory)

  model_res_dict = {}
  for model_name in models_list:
    res_dict = {}
    full_model_name = model_type + "_" + model_name
    # Don't create finetuned directory for Human
    if (model_name == "human" and model_type == "finetuned"):
      continue
    model_directory = res_directory + full_model_name + "/"
    model_stats_directory = stats_directory + full_model_name + "/"

    res_dict["model_directory"] = model_directory
    res_dict["model_stats_directory"] = model_stats_directory
    # directory = "/content/Switchboard-Corpus/results/base_DialoGPT/"
    create_directory_safe(model_directory)
    create_directory_safe(model_stats_directory)

    # Create contexed_csv directorie(s).
    directory_c_csv = model_directory + "contexed_csv/"
    res_dict["directory_c_csv"] = directory_c_csv
    # directory_c_csv = "/content/Switchboard-Corpus/results/base_DialoGPT/contexed_csv/"
    create_directory_safe(directory_c_csv)

    # Create csv directorie(s).
    directory_csv = model_directory + "csv/"
    res_dict["directory_csv"] = directory_csv
    # directory_csv = "/content/Switchboard-Corpus/results/base_DialoGPT/csv/"
    create_directory_safe(directory_csv)

    if(model_name != "human"):
      dir_init_name = "directory_csv_"
      for speaker in speakers:
        dir_end_name = speaker + "_M"
        directory_csv_speaker = directory_csv + "speaker_" + dir_end_name + "/"
        
        res_dict[dir_init_name+dir_end_name] = directory_csv_speaker
        create_directory_safe(directory_csv_speaker)
    
    directory_tsv = model_directory + "tsv/"
    res_dict["directory_tsv"] = directory_tsv
    # directory_tsv = "/content/Switchboard-Corpus/results/base_DialoGPT/tsv/"
    create_directory_safe(directory_tsv)

    directory_tsv_normalized = model_directory + "tsv_normalized/"
    res_dict["directory_tsv_normalized"] = directory_tsv_normalized
    # directory_tsv = "/content/Switchboard-Corpus/results/base_DialoGPT/tsv_normalized/"
    create_directory_safe(directory_tsv_normalized)

    if(model_name != "human"):
      for speaker in speakers:
        dir_init_name = "directory_tsv_"
        dir_end_name =  speaker + "_M"
        directory_tsv_speaker = directory_tsv + "speaker_" + dir_end_name + "/"        
        res_dict[dir_init_name + dir_end_name] = directory_tsv_speaker
        create_directory_safe(directory_tsv_speaker)

        dir_init_name = "directory_tsv_normalized_"
        directory_tsv_normalized_speaker = directory_tsv_normalized + "speaker_" + dir_end_name + "/"
        res_dict[dir_init_name + dir_end_name] = directory_tsv_normalized_speaker
        create_directory_safe(directory_tsv_normalized_speaker)
    # else:
    #   speaker_1 = speakers[0]
    #   speaker_2 = speakers[1]
    #   dir_end_name = "speaker_" + speaker_1 + "_" + speaker_2
    #   directory_tsv_speaker = directory_tsv + dir_end_name + "/"
        
    #   res_dict[dir_init_name + dir_end_name] = directory_tsv_speaker
    #   create_directory_safe(directory_tsv_speaker)


    # Create Scrambled directorie(s).
    directory_scrambled = model_directory + "scrambled/"
    res_dict["directory_scrambled"] = directory_scrambled
    create_directory_safe(directory_scrambled)

    directory_scrambled_tsv = directory_scrambled + "tsv/"
    res_dict["directory_scrambled_tsv"] = directory_scrambled_tsv
    create_directory_safe(directory_scrambled_tsv)

    directory_scrambled_tsv_normalized = directory_scrambled + "tsv_normalized/"
    res_dict["directory_scrambled_tsv_normalized"] = directory_scrambled_tsv_normalized
    create_directory_safe(directory_scrambled_tsv_normalized)

    if(model_name != "human"):
      for speaker in speakers:
        dir_init_name = "directory_scrambled_tsv_"
        dir_end_name = speaker + "_M"
        directory_scrambled_tsv_speaker = directory_scrambled_tsv + "speaker_" + dir_end_name + "/"
        
        res_dict[dir_init_name + dir_end_name] = directory_scrambled_tsv_speaker
        create_directory_safe(directory_scrambled_tsv_speaker)

        dir_init_name = "directory_scrambled_tsv_normalized_"
        directory_scrambled_tsv_normalized_speaker = directory_scrambled_tsv_normalized + "speaker_" + dir_end_name + "/"
        
        res_dict[dir_init_name + dir_end_name] = directory_scrambled_tsv_normalized_speaker
        create_directory_safe(directory_scrambled_tsv_normalized_speaker)

    directory_scrambled_txt = directory_scrambled + "txt/"
    res_dict["directory_scrambled_txt"] = directory_scrambled_txt
    create_directory_safe(directory_scrambled_txt)

    if(model_name != "human"):
      dir_init_name = "directory_scrambled_txt_"
      for speaker in speakers:
        dir_end_name = speaker + "_M"
        directory_scrambled_txt_speaker = directory_scrambled_txt + "speaker_" + dir_end_name + "/"
        
        res_dict[dir_init_name + dir_end_name] = directory_scrambled_txt_speaker
        create_directory_safe(directory_scrambled_txt_speaker)

    model_res_dict[full_model_name] = res_dict
  return model_res_dict

def get_dataset_settings(dataset="Switchboard-Corpus"):
  DatasetSettings = namedtuple('Settings', ['test_dir', 'speakers'])
  if dataset == "Switchboard-Corpus":
    test_dir = './external/Switchboard-Corpus/swda_data/test/'
    speakers=["A", "B"]
  else:
    test_dir = './external/Maptask-Corpus/maptask_data/test/'
    speakers=["f", "g"]
  s = DatasetSettings(test_dir, speakers)
  return s

def create_dialign_directories(models_list, output_dir_name="dialign_output", to_local_directory=False, dataset="Switchboard-Corpus", model_type="base", speakers=["A", "B"]):
  if to_local_directory:
    res_directory = "./results/" + dataset + "/"+ output_dir_name+ "/"
  else:
    res_directory = "/content/results/" + dataset + "/"+ output_dir_name+ "/"

  create_directory_safe(res_directory)

  model_res_dict = {}
  for model_name in models_list:
    res_dict = {}
    full_model_name = model_type + "_" + model_name
    # Create Model directory
    if (model_name == "human" and model_type == "finetuned"):
      continue
    model_directory = res_directory + full_model_name + "/"

    res_dict["model_directory"] = model_directory
    
    res_dict["directory_normal"] = model_directory + "normal/"
    create_directory_safe(res_dict["directory_normal"])
    if(model_name != "human"):
      dir_init_name = "directory_normal"
      for speaker in speakers:
        dir_end_name = speaker + "_M"
        directory_normal_speaker = res_dict["directory_normal"] + "speaker_" + dir_end_name + "/"
        
        res_dict[dir_init_name + dir_end_name] = directory_normal_speaker
        create_directory_safe(directory_normal_speaker)


    res_dict["directory_scrambled"] = model_directory + "scrambled/"
    create_directory_safe(res_dict["directory_scrambled"])
    if(model_name != "human"):
      dir_init_name = "directory_scrambled"
      for speaker in speakers:
        dir_end_name = speaker + "_M"
        directory_scrambled_speaker = res_dict["directory_scrambled"] + "speaker_" + dir_end_name + "/"
        
        res_dict[dir_init_name + dir_end_name] = directory_scrambled_speaker
        create_directory_safe(directory_scrambled_speaker)
    model_res_dict[full_model_name] = res_dict
  return model_res_dict

"""## Experiments Type 1: Alignement in Human - Human Dialogue

### Human - Normal Dialogue
"""

def transform_txt_to_tsv(test_dir, model_name, output_dir, speakers=["A", "B"]):
  for txt_file in os.listdir(test_dir):
    txt_file_name = txt_file.replace(".", "_")
    speaker_1 = speakers[0]
    speaker_2 = speakers[1]
    
    filename = txt_file_name + '-base_' + model_name + '-speaker_' + speaker_1 + '_' + speaker_2 +  '.tsv'
    output_file = output_dir + filename

    with open(test_dir + txt_file, 'r') as f_input, open(output_file, 'w', newline='') as f_output:
      csv_output = csv.writer(f_output, delimiter='\t')
      lines = f_input.read().split('\n')

      for line in lines:
        attr_arr = line.split('|')
        # at least the speker and the uttr exists.
        if(len(attr_arr)>=2):
          speaker = attr_arr[0]+':'
          uttr = attr_arr[1]
          # print(speaker, uttr)
          csv_output.writerow([speaker, uttr])

"""### Human - Scrambled Dialogue"""

def create_dataframe_from_txt(txt_file):
  Item = namedtuple('Item', 'speaker utterance unified')
  items = []

  with open(txt_file, 'r') as f_input:
    lines = f_input.read().split('\n')
    
    for line in lines:
      attr_arr = line.split('|')
      # at least the speker and the uttr exists.
      if(len(attr_arr)>=2):
        speaker = attr_arr[0]
        uttr = attr_arr[1]
        unified = speaker + '|' + uttr
        items.append(Item(speaker, uttr, unified))
        # print(speaker, uttr)

  df = pd.DataFrame.from_records(items, columns=['Speaker', 'Utterance', 'Unified'])
  return df

def create_scrambled_dialogues_txt(dialogue_uttr_dict, 
                                   output_dir, 
                                   speakers=["A", "B"], seed=42):
  
  directory_scrambled_txt = output_dir

  random.seed(seed)
  all_utt_arr = list(itertools.chain.from_iterable(dialogue_uttr_dict.values()))
  # Remove empty string
  all_utt_arr = list(filter(None, all_utt_arr))
  
  speaker_1 = speakers[0]
  speaker_2 = speakers[1]

  # Separate utterances per Speaker.
  all_utt_arr_speaker_1 = [utt for utt in all_utt_arr if utt.startswith(speaker_1)]
  all_utt_arr_speaker_2 = [utt for utt in all_utt_arr if utt.startswith(speaker_2)]

  # Shuffle all utterances.
  random.shuffle(all_utt_arr_speaker_1)
  random.shuffle(all_utt_arr_speaker_2)

  for txt_file, dialogue in dialogue_uttr_dict.items():
    len_1 = len([utt for utt in dialogue if utt.startswith(speaker_1)])
    len_2 = len([utt for utt in dialogue if utt.startswith(speaker_2)])

    # Choose Random per speaker
    d_random_idxs_1 = random.sample(range(len(all_utt_arr_speaker_1)), len_1)
    d_uttr_list_1 = [all_utt_arr_speaker_1[i] for i in range(len(all_utt_arr_speaker_1)) if i in d_random_idxs_1]
    
    d_random_idxs_2 = random.sample(range(len(all_utt_arr_speaker_2)), len_2)
    d_uttr_list_2 = [all_utt_arr_speaker_2[i] for i in range(len(all_utt_arr_speaker_2)) if i in d_random_idxs_2]

    # Remove the selected from all uttr list so we don't select again.
    for index in sorted(d_random_idxs_1, reverse=True):
        del all_utt_arr_speaker_1[index]
    
    for index in sorted(d_random_idxs_2, reverse=True):
        del all_utt_arr_speaker_2[index]

    # Merge lists by alternating partner utterances.
    final_uttr_list = [x for x in itertools.chain.from_iterable(
        itertools.zip_longest(d_uttr_list_1,
                               d_uttr_list_2)) if x]
    # final_uttr_list.append('')
    # all_utt_arr_scrambled.append(final_uttr_list)
    
    txt_file = txt_file.replace(".txt", "")
    txt_file_arr = txt_file.split("_")

    filename = txt_file_arr[0] + '-scrambled'
    # Save to txt.
    filaname_txt = filename + '.txt'
    with open(directory_scrambled_txt+filaname_txt, mode='wt', encoding='utf-8') as myfile:
      myfile.write('\n'.join(final_uttr_list))
  
  # return all_utt_arr_scrambled

def get_uttr_list_from_txt(test_dir, txt_file):
  with open(test_dir + txt_file, 'r') as f_input:
    lines = f_input.read().split('\n')
    str_list = list(filter(None, lines)) # Remove empty string
  return str_list

def get_uttr_list_from_df(df, column='Unified'):
  str_list = df[column].tolist()
  return str_list

def get_dialogue_uttr_dict(test_dir):
  d_dict = {}
  for txt_file in os.listdir(test_dir):
    df_test_file = create_dataframe_from_txt(test_dir+txt_file)
    d_uttr_list = get_uttr_list_from_df(df_test_file)
    
    d_dict[txt_file] = d_uttr_list
  return d_dict

"""## Experiments Type 2: Alignement in Human-Model Dialogue"""
"""### Data Preparation

Keep n previous lines as context. Here n=5. The number was chosen after a few tries with the contexed datasets. More specifically, if context was increased ie in 6, 7 etc. lines, some dataset entries would exceed the maximum of 1024 tokens that can be processed by GPT2 type models.
"""
# Tranformation code from https://towardsdatascience.com/make-your-own-rick-sanchez-bot-with-transformers-and-dialogpt-fine-tuning-f85e6d1f4e30
def create_contexted_df(input_df, n=HISTORY_CONTEXT_LENGTH):
  contexted = []
  for i in range(n, len(input_df['Utterance'])):
    row = []
    prev = i - 1 - n # we additionally subtract 1, so row will contain current response and 4 previous responses  
    for j in range(i, prev, -1):
      row.append(input_df['Utterance'][j])
    contexted.append(row)
  columns = ['response', 'context'] 
  columns = columns + ['context/'+str(i) for i in range(n-1)]
  df = pd.DataFrame.from_records(contexted, columns=columns)
  return df

def create_contexted_csv_all(data_dir, 
                            output_dirs, 
                            model_name="human", 
                            model_type="base", 
                            n=HISTORY_CONTEXT_LENGTH):
 
  full_model_name = model_type + "_" + model_name
  directory_c_csv = output_dirs[full_model_name]["directory_c_csv"]

  for txt_file in os.listdir(data_dir):
    df_test = create_dataframe_from_txt(data_dir+txt_file)
    df_test_contexted = create_contexted_df(df_test, n)
    
    txt_file = txt_file.replace(".", "_")
    filename = txt_file + '-' + full_model_name + '-contexted_' + str(n) + '.csv'
    
    df_test_contexted.to_csv(directory_c_csv + filename)

"""Test model (DialogGPT, GPT2 etc) for 5 User inputs. Uncomment to try."""

def dialogue_with_model(model, tokenizer, turns=5, print_all_conversation=False, use_eos_token=True, use_history_context=True):
  for step in range(turns):
    # encode the new user input, add the eos_token and return a tensor in Pytorch
    if use_eos_token:
      new_user_input_ids = tokenizer.encode(input(">> User:") + tokenizer.eos_token, return_tensors='pt')
    else:
      new_user_input_ids = tokenizer.encode(input(), return_tensors='pt')
   
    # append the new user input tokens to the chat history
    if use_history_context:
      bot_input_ids = torch.cat([chat_history_ids, new_user_input_ids], dim=-1) if step > 0 else new_user_input_ids
    else:
      bot_input_ids = new_user_input_ids
    # print(type(bot_input_ids))
    print("bot_input_ids:", bot_input_ids)
    
    # decoder_input_ids  = tokenizer(">> Model:", return_tensors="pt").input_ids  # Batch size 1
    # generated a response while limiting the total chat history to 100 tokens    
    chat_history_ids = model.generate(
        input_ids = bot_input_ids, 
        max_length=100,
        pad_token_id=tokenizer.eos_token_id
    )
    
    # pretty print last ouput tokens from bot
    print("chat_history_ids:", chat_history_ids)
    print("chat_history_ids len:", len(chat_history_ids[0]))

    if print_all_conversation:
      print("Model: {}".format(tokenizer.decode(chat_history_ids[0], skip_special_tokens=True)))
    else:
      print("Model: {}".format(tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)))


def construct_conv_test(row, tokenizer, eos = True):
  # from: https://stackoverflow.com/questions/952914/how-to-make-a-flat-list-out-of-list-of-lists
  flatten = lambda l: [item for sublist in l for item in sublist]

  # /Modification: Changed row to row[1:] in order to ignore the response during testing.
  # Ignore the response column
  conv = list(reversed([tokenizer.encode(x) + [tokenizer.eos_token_id] for x in row[1:]]))
  conv = flatten(conv)
  return conv

# Tokenize df_test
def generate_model_responses(model, tokenizer, df_contexted, verbose=False):
  model_chats_list = []
  total_rows = df_contexted.shape[0]
  for idx, row in df_contexted.iterrows():
    if(verbose):
      print("Processing ", idx, "/", total_rows)
      print(row)
    # Construct context without the response.
    conv = construct_conv_test(row, tokenizer)
    # bot_input_ids = torch.tensor(conv, dtype=torch.long)
    # bot_ids_list = bot_input_ids.tolist()
    new_bot_input_ids = torch.tensor([conv], dtype=torch.long)
    
    # generated a response while limiting the total chat history to 1024 tokens, 
    chat_history_ids = model.generate(
        new_bot_input_ids, max_length=1024,
        pad_token_id=tokenizer.eos_token_id,  
        no_repeat_ngram_size=3,       
        do_sample=True, 
        top_k=100, 
        top_p=0.7,
        temperature = 0.8
    )

    history_chat = tokenizer.decode(new_bot_input_ids[0], skip_special_tokens=False).split('<|endoftext|>')
    model_response = tokenizer.decode(chat_history_ids[:, new_bot_input_ids.shape[-1]:][0], skip_special_tokens=True)
    
    if (verbose):
      print("History: {}".format(history_chat))
      print("Model Response: {}".format(model_response))
      print("------------------------ . ------------------------")
    # Remove empty str
    history_chat = list(filter(None, history_chat))
    history_chat.append(model_response)
    full_chat_rev = list(reversed(history_chat))
    
    model_chats_list.append(full_chat_rev)
    # print("model_chats_list", model_chats_list)
    # break
  return model_chats_list

# Source df could be the initial df of the test/train/val.
def create_model_df_from_contexed(source_df, contexted_df):
  # the response column is model's response. 
  # Basically n = HISTORY_CONTEXT_LENGTH
  n = len(contexted_df.columns) - 1

  Item = namedtuple('Item', 'speaker utterance unified')
  items = []

  # do the first row
  for idx, row in contexted_df.iterrows():
    # first row
    if idx == 0:
      # copy from the original dataset
      for i in range(0,n):
        speaker = source_df.loc[i, "Speaker"]
        uttr = source_df.loc[i, "Utterance"]
        unified = source_df.loc[i, "Unified"]
        items.append(Item(speaker, uttr, unified))
    # the response column is model's response
    speaker_m = 'M'
    uttr_m = row['response']
    unified_m = speaker_m + '|' + str(uttr_m)
    items.append(Item(speaker_m, uttr_m, unified_m))
  df = pd.DataFrame.from_records(items, columns=['Speaker', 'Utterance', 'Unified'])
  return df

def create_model_speaker_df(source_df, model_df, speaker="A", n=HISTORY_CONTEXT_LENGTH, remove_first_context=True):
  df = source_df.copy(deep=True)
  
  # Remove the first HISTORY_CONTEXT_LENGTH rows that the model 
  # didn't create a response.
  if remove_first_context:
    df = df.iloc[n: , :]
  
  for i, row in source_df.iterrows():
    # Keep the utterances of the requested speaker
    # and replace the rest with the model responses. 
    if row['Speaker'] != speaker:
      speaker_m = model_df.loc[i, "Speaker"] 
      uttr_m = model_df.loc[i, "Utterance"]
      unified_m = model_df.loc[i, "Unified"]
      
      df["Speaker"][i] = speaker_m
      df["Utterance"][i] = uttr_m
      df["Unified"][i] = unified_m

  return df

def test_all(model, 
             model_name,
             model_type,
             tokenizer,
             output_dirs,
             speakers=["A","B"],
             verbose = False, 
             data_dir = './external/Switchboard-Corpus/swda_data/test/', 
             text_file = None, 
             n = HISTORY_CONTEXT_LENGTH):
  
  full_model_name = model_type + "_" + model_name
  speaker_1 = speakers[0]
  speaker_2 = speakers[1]

  directory_c_csv = output_dirs[full_model_name]["directory_c_csv"]
  
  # directory_csv = output_dirs[full_model_name]["directory_csv"]
  dir_name_1 = "directory_csv_" + speaker_1 + "_M" # ie directory_csv_A_M 
  dir_name_2 = "directory_csv_" + speaker_2 + "_M"
  directory_csv_1_M = output_dirs[full_model_name][dir_name_1]
  directory_csv_2_M = output_dirs[full_model_name][dir_name_2]

  # directory_tsv = output_dirs[full_model_name]["directory_tsv"]
  dir_name_1 = "directory_tsv_" + speaker_1 + "_M" # ie directory_tsv_A_M 
  dir_name_2 = "directory_tsv_" + speaker_2 + "_M"
  directory_tsv_1_M = output_dirs[full_model_name][dir_name_1]
  directory_tsv_2_M = output_dirs[full_model_name][dir_name_2]
  
  force_file = False
  if (text_file):
    force_file = True

  total_files = len(os.listdir(data_dir))
  current_cnt = 0

  start_total = datetime.now()
  for txt_file in os.listdir(data_dir):
    start_file_time = datetime.now()
    if(force_file):
      txt_file = text_file
      total_files = 1
    current_cnt +=1
    
    print("Processing ", current_cnt, "/", total_files, ":", txt_file)
    # 1. Create test dataframe
    print("1. Create test dataframe")
    df_test = create_dataframe_from_txt(data_dir+txt_file)
    # display(df_test)
    
    # 2. Create test context dataframe
    print("2. Create test context dataframe")
    df_test_contexted = create_contexted_df(df_test, n)
    # display(df_test_contexted)
    
    # 3. Generate model responses
    print("3. Generate model responses")
    model_chats_list = generate_model_responses(model, tokenizer, 
                                                df_test_contexted, 
                                                verbose)
    # 3.1 Generate contexted dataframe with model responses
    print("3.1 Generate contexted dataframe with model responses")
    df_model_contexted = pd.DataFrame(model_chats_list, columns = df_test_contexted.columns)
    
    # 3.2. Save model contexted df to csv
    txt_file = txt_file.replace(".", "_")
    filename = txt_file + '-' + full_model_name + '-contexted_' + str(n) + '.csv'
    print("3.2. Save model contexted df to csv:", filename)
    df_model_contexted.to_csv(directory_c_csv + filename)
    
    # 4. Create dataframe with model responses only.
    df_model = create_model_df_from_contexed(source_df=df_test, 
                                             contexted_df=df_model_contexted)
    print("4. Create dataframe with model responses only, size:", df_model.size)
    # display(df_model)
    # 5. Create speaker A - model df and save as csv and tsv (for dialign).
    # -> remove the first n utterances that there is no model generated text 
    #    for replacement.
    print("5. Create speaker 1 - model df")
    df_model_speaker_1 = create_model_speaker_df(df_test, df_model, speaker=speaker_1,
                                                 n=n, remove_first_context=True)
    filename = txt_file + '-' + full_model_name + '-speaker_' + speaker_1
    # 5.1 Save to csv.
    filaname_csv = filename + '.csv'
    print("5.1 Save to csv:", filaname_csv)
    df_model_speaker_1.to_csv(directory_csv_1_M+filaname_csv)

    # 5.2 Save to tsv for dialign.
    filename_tsv = filename + '.tsv'
    print("5.2 Save to tsv for dialign:", filename_tsv)
    df_dialign = df_model_speaker_1.drop('Unified', axis=1)
    df_dialign['Speaker'] = df_dialign['Speaker'].apply(lambda x: x + ":")
    df_dialign.to_csv(directory_tsv_1_M+filename_tsv, sep="\t", header=False, index=False)
    
    # 6. Create speaker B - model df and save as csv and tsv (for dialign).
    # -> remove the first n utterances that there is no model generated text 
    #    for replacement.
    print("6. Create speaker 2 - model df")
    df_model_speaker_2 = create_model_speaker_df(df_test, df_model, speaker=speaker_2,
                                                 n=n, remove_first_context=True)
    # 6.1 Save to csv.
    filename = txt_file + '-' + full_model_name + '-speaker_' + speaker_2
    filaname_csv = filename + '.csv'
    print("6.1 Save to csv:", filaname_csv)
    df_model_speaker_2.to_csv(directory_csv_2_M+filaname_csv)

    # 6.2 Save to tsv for dialign.
    filename_tsv = filename + '.tsv'
    print("6.2 Save to tsv for dialign:", filename_tsv)
    df_dialign = df_model_speaker_2.drop('Unified', axis=1)
    df_dialign['Speaker'] = df_dialign['Speaker'].apply(lambda x: x + ":")
    df_dialign.to_csv(directory_tsv_2_M+filename_tsv, sep="\t", header=False, index=False)
    
    stop_file_time = datetime.now()
    print("Processing Time:", stop_file_time - start_file_time)
    print("######################.######################") 
    
    if(force_file):
      break
  stop_total = datetime.now()
  print("Total Time:", stop_total - start_total)


"""### Base Model - Scrambled Dialogue"""
def get_dialogue_uttr_dict_from_csv(source_csv_dir):
  d_dict = {}
  for csv_file in os.listdir(source_csv_dir):
    df_test_file = pd.read_csv(source_csv_dir+csv_file)
    d_uttr_list = get_uttr_list_from_df(df_test_file)
    
    d_dict[csv_file] = d_uttr_list
  return d_dict


""" ## Expression Evaluation - Output Statistics"""
def get_responses_list_from_contexed_dir(data_dir):
  total_pred_list = []

  for csv_file in os.listdir(data_dir):
    df = pd.read_csv(data_dir+csv_file)
    df.fillna('', inplace=True)
    pred_list = df.response.tolist()
    total_pred_list.append(pred_list)
    
  # from: https://stackoverflow.com/questions/952914/how-to-make-a-flat-list-out-of-list-of-lists
  flatten = lambda l: [item for sublist in total_pred_list for item in sublist]

  flatten_list = flatten(total_pred_list)
  return flatten_list

def create_postags_df(doc):
  Item = namedtuple('Item', 'id tag count')
  items = []

  num_pos = doc.count_by(spacy.attrs.POS)
  for k,v in sorted(num_pos.items()):
    print(f'{k}. {doc.vocab[k].text:{8}}: {v}')
    id = k
    tag = doc.vocab[k].text
    count = v
    items.append(Item(id, tag, count))

  df = pd.DataFrame.from_records(items, columns=['ID', 'Tag', 'Count'])
  return df

def create_word_freq_df(words_list):
  word_freq = Counter(words_list)
       
  word_freq_df = pd.DataFrame.from_dict(word_freq, orient='index').reset_index()
  word_freq_df.columns = ['Word', 'Count']
  word_freq_df.sort_values('Count', ascending=False, ignore_index=True, inplace=True)

  return word_freq_df

def create_word_concrete_df(word_df, concrete_df):  
  df_word_lower = word_df.copy()
  df_word_lower['Word'] = df_word_lower['Word'].str.lower()

  word_lower_list = df_word_lower['Word'].to_list()
  concrete_dict = concrete_df.set_index('Word')['Conc.M'].to_dict() 
  
  word_concrete_dict = {}
  for word in word_lower_list:
    # set default value to 0 if word doesn't exist
    word_concrete_dict[word] = concrete_dict.get(word,0)

  df = pd.DataFrame.from_dict(word_concrete_dict, orient='index')
  df.reset_index(inplace=True)
  df.columns = ['Word', 'Conc.M']
  return df

def top_common_content_word_counts(base_h_content_word_freq_df, base_d_content_word_freq_df, finetune_d_content_word_freq_df, top_n=10):
  base_h_content_list = base_h_content_word_freq_df['Word'].to_list()
  base_d_content_list = base_d_content_word_freq_df['Word'].str.lower().to_list()
  finetune_d_content_list = finetune_d_content_word_freq_df['Word'].str.lower().to_list()
  
  common_human_models_words = [x for x in base_h_content_list if x.lower() in base_d_content_list and x.lower() in finetune_d_content_list]

  common_top_n = common_human_models_words[:top_n]
  base_h_content_word_dict = base_h_content_word_freq_df.set_index('Word').to_dict(orient='index')
  base_d_content_word_dict = base_d_content_word_freq_df.set_index('Word').to_dict(orient='index')
  finetune_d_content_word_dict = finetune_d_content_word_freq_df.set_index('Word').to_dict(orient='index')

  merge_count_dict = {}
  for word in common_top_n:
    merge_count_dict[word] = {}
    merge_count_dict[word]["Count (Human)"] = base_h_content_word_dict[word]
    merge_count_dict[word]["Count (Base DialoGPT)"] = base_d_content_word_dict[word]
    merge_count_dict[word]["Count (Finetuned DialoGPT)"] = finetune_d_content_word_dict[word]
  return merge_count_dict
  


def create_normalized_tsv(directory_tsv, directory_tsv_normalized):
  for tsv in os.listdir(directory_tsv):
    tsv_df = pd.read_csv(directory_tsv+tsv, sep="\t", names=['Speaker', 'Utterance'])
    tsv_df["Utterance"] = tsv_df.apply(lambda x: x["Utterance"].replace(',','').lower() if isinstance(x["Utterance"], str) else "",axis=1)

    tsv_df.to_csv(directory_tsv_normalized+tsv, sep='\t', index=False, header=False)