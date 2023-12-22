import pickle

# 替换为你的Pickle文件路径
# pickle_file_path = 'e_piano_3/train/MIDI-Unprocessed_01_R1_2006_01-09_ORIG_MID--AUDIO_01_R1_2006_01_Track01_wav.pickle'

# # 以二进制读取模式打开Pickle文件
# with open(pickle_file_path, 'rb') as file:
#     # 加载（反序列化）Pickle文件
#     data = pickle.load(file)

#     # 打印加载的数据
#     print(data)

pickle_file_path2 = 'e_piano/train/MIDI-Unprocessed_01_R1_2006_01-09_ORIG_MID--AUDIO_01_R1_2006_01_Track01_wav.pkl'

# 以二进制读取模式打开Pickle文件
with open(pickle_file_path2, 'rb') as file:
    # 加载（反序列化）Pickle文件
    data = pickle.load(file)

    # 打印加载的数据
    print(data)