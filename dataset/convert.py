from pathlib import Path
from copy import deepcopy
import os
import pickle
from torch import Tensor, argmax
from torch.utils.data import DataLoader
from torch.cuda import is_available as cuda_available, is_bf16_supported
from torch.backends.mps import is_available as mps_available
from torchtoolkit.data import create_subsets
from miditok import REMI, TokenizerConfig, TokSequence
from miditok.pytorch_data import DatasetTok, DataCollator
from tqdm import tqdm
from miditoolkit import MidiFile
from transformers import GPT2LMHeadModel, GPT2Config, Trainer, TrainingArguments, GenerationConfig, TransfoXLConfig, TransfoXLLMHeadModel
from evaluate import load as load_metric

# Our tokenizer's configuration
PITCH_RANGE = (21, 109)
BEAT_RES = {(0, 1): 8, (1, 2): 4, (2, 4): 2, (4, 8): 1}
NB_VELOCITIES = 24
SPECIAL_TOKENS = ["PAD", "MASK", "BOS", "EOS"]
USE_CHORDS = False
USE_RESTS = False
USE_TEMPOS = True
USE_TIME_SIGNATURE = False
USE_PROGRAMS = False
NB_TEMPOS = 32
TEMPO_RANGE = (50, 200)  # (min_tempo, max_tempo)
TOKENIZER_PARAMS = {
    "pitch_range": PITCH_RANGE,
    "beat_res": BEAT_RES,
    "num_velocities": NB_VELOCITIES,
    "special_tokens": SPECIAL_TOKENS,
    "use_chords": USE_CHORDS,
    "use_rests": USE_RESTS,
    "use_tempos": USE_TEMPOS,
    "use_time_signatures": USE_TIME_SIGNATURE,
    "use_programs": USE_PROGRAMS,
    "num_tempos": NB_TEMPOS,
    "tempo_range": TEMPO_RANGE,
}
config = TokenizerConfig(**TOKENIZER_PARAMS)

# Creates the tokenizer convert MIDIs to tokens
tokens_path = Path('Maestro_tokens_no_bpe')
tokenizer = REMI(config)  # REMI
midi_paths = list(Path('Maestro').glob('**/*.mid')) + list(Path('Maestro').glob('**/*.midi'))
tokenizer.tokenize_midi_dataset(midi_paths, tokens_path)

tokenizer.save_params("tokenizer_no_bpe.conf")
tokens_paths = list(Path('Maestro_tokens_no_bpe').glob("**/*.json"))
# dataset = DatasetTok(
#     tokens_paths, max_seq_len=3072, min_seq_len=3072, one_token_stream=False,
# )
# subset_train, subset_valid_test = create_subsets(dataset, [0.2])
# subset_valid, subset_test = create_subsets(subset_valid_test, [0.5])

# print(len(subset_train))
# print(len(subset_valid))
# print(len(subset_test))
# print(len(dataset))
# os.makedirs('train', exist_ok=True)
# os.makedirs('val', exist_ok=True)
# os.makedirs('test', exist_ok=True)
# # 遍历subset_train中的每个元素
# for i, item in enumerate(subset_train):
#     filename = f'train/{i}.pickle'
#     with open(filename, 'wb') as f:
#         pickle.dump(item, f)
# print("Num Train:",len(subset_train))

# # 遍历subset_valid中的每个元素
# for i, item in enumerate(subset_valid):
#     filename = f'val/{i}.pickle'
#     with open(filename, 'wb') as f:
#         pickle.dump(item, f)
# print("Num valid:",len(subset_valid))

# # 遍历subset_test中的每个元素
# for i, item in enumerate(subset_test):
#     filename = f'test/{i}.pickle'
#     with open(filename, 'wb') as f:
#         pickle.dump(item, f)

# print("Num test:",len(subset_test))
os.makedirs('e_piano', exist_ok=True)
print("len_tokenizer:",len(tokenizer))
print("PAD_None", tokenizer["PAD_None"])
print("BOS_None", tokenizer["BOS_None"])
print("EOS_None", tokenizer["EOS_None"])
