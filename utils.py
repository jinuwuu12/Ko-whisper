from datetime import datetime
import pickle
from random import shuffle
import numpy as np
import pandas as pd
import librosa as lr
import soundfile as sf
import os
from tqdm import tqdm
from datasets import load_dataset, DatasetDict
from datasets import Audio

def get_unique_directory(dir_name : str, model_name : str) -> str:
    '''입력된 디렉토리 이름에 날짜/ 시간 정보를 추가해서 리턴'''
    model_name = model_name.split('/')[-1]
    now = datetime.now().strftime('%Y-%m-%d-%H%M')

    return os.path.join(dir_name, f'{model_name}-{now}')


class PrepareDataset:
    def __init__(self, audio_dir :str = './data/audio') -> None:
        self.VOICE_DIR = audio_dir

    # wav 파일로 변경
    def pcm2audio(
            self,
            audio_path: str,
            extention: str = 'wav',
            save_file: bool =True,
            remove: bool = False, #pcm 파일 삭제 여부
        ) -> object: # 객체반환
            buf = None
            with open(audio_path, 'rb') as tf:
                buf = tf.read()
                # zero (0) padding
                # 경우에 따라서 PCM 파일의 길이가 8bit(1byte)로
                # 나누어 떨어지지 않는 경우가 있어 0으로 패딩을 더해준다.
                # 패딩하지 않으면 numpy나 librosa 사용 시 오류가 날 수 있다.

                buf = buf+b'0' if len(buf)%2 else buf
            pcm_data = np.frombuffer(buf, dtype='int16')
            wav_data = lr.util.buf_to_float(x=pcm_data, n_bytes=2)

            # 음성 파일을 변환하여 저장: .pcm -> .wav
            if save_file:
                save_file_name = audio_path.replace('.pcm', f'.{extention}')
                sf.write(
                    file = save_file_name,
                    data = wav_data,
                    samplerate=16000,
                    format='wav',
                    endian='LITTLE',
                    subtype='PCM_16'
                    )
            
            # 파일 삭제 옵션
            if remove:
                 if os.path.isfile(audio_path):
                      os.remove(audio_path)
            
            return wav_data
    
    # 평가 오디오 데이터 전처리 
    def eval_audio_process(
            self,
            source_dir : str,
            remove_original_audio:bool = True,
    )-> None:
        print(f'source dir: {source_dir}')
        audio_files = os.listdir(source_dir)
        for audio in tqdm(audio_files, desc=f'Processing directory: {audio_files}'):
            file_name = audio
            if file_name.endswith('.pcm'):
                self.pcm2audio(
                    audio_path=os.path.join(source_dir, file_name),
                    extention='wav',
                    remove= remove_original_audio
                )
    
    # 전체 변경
    def process_audio(
              self,
              source_dir: str,
              remove_original_audio: bool = True,
    ) -> None:
         print(f'source_dir: {source_dir}')
         sub_directories = sorted(os.listdir(source_dir))
         print(f'Processing audios: {len(sub_directories)} directories')
         for directory in tqdm(sub_directories, desc=f'Processing directory: {source_dir}'):
                # 디렉토리 안에 있을 경우에 변환환
                if os.path.isdir(directory):
                    files = os.listdir(os.path.join(source_dir, directory))
                    for file_name in files:
                        if file_name.endswith('.pcm'):
                            self.pcm2audio(
                                audio_path=os.path.join(source_dir, directory, file_name),
                                extention='wav',
                                remove= remove_original_audio
                            )
                # 디렉토리가 아닐경우에도 변환이 가능함
                elif os.path.isfile(directory):
                     file_name = directory
                     if file_name.endswith('.pcm'):
                            self.pcm2audio(
                                audio_path=os.path.join(source_dir, file_name),
                                extention='wav',
                                remove= remove_original_audio
                            )

    # 인코딩 변경
    def convert_encoding(self,file_path: str)-> None:
        '''convert file encoding UTF-8 to CP949 '''
        try:
            with open(file_path, 'rt', encoding='cp949') as f:
                lines = f.readlines()
        except:
            with open(file_path, 'rt', encoding='utf-8') as f:
                lines = f.readlines()
        with open(file_path, 'wt', encoding='utf-8') as f:
            for line in lines:
                f.write(line)
    
    # 전체 인코딩 변경경
    def convert_all_encoding(self, target_dir: str)-> None:
        '''디렉토리 내부의 모든 텍스트 파일을 인코딩 변경'''
        print(f'Target directory: {target_dir}')
        sub_directories = sorted(os.listdir(target_dir))
        num_files =0
        for directory in tqdm(sub_directories, desc='converting cp949 -> utf-8'):
            files = os.listdir(os.path.join(target_dir, directory))
            for file_name in files:
                if file_name.endswith('.txt'):
                     self.convert_encoding(
                          os.path.join(target_dir, directory, file_name)
                     )
                     num_files +=1
        print(f'{num_files} txt files are converted')

    # trn 파일 그룹별로 나누기 
    def split_whole_data(self, target_file: str)->None:
        '''
        전체 데이터 파일(전사파일)을 그룹별로 나눔
        '''
        with open(target_file, 'rt', encoding='utf-8') as f:
            # 모든 라인 다 읽어옴
            lines = f.readlines()
            # 데이터 그룹을 만듬 ex)KsponSpeech_01
            data_group = set()

            for line in lines:
                data_group.add(line.split('/')[0])
            # 데이터 그룹을 리스트로 만들고 정렬렬
            data_group = sorted(list(data_group))

            data_dic = {group: [] for group in data_group} # dict comprehension         
            for line in lines:
                data_dic[line.split('/')[0]].append(line)

            # Save file seperately
            # target_file: data/info/train.trn
            # 파일 저장을 현재 디렉토리 경로로 지정정
            save_dir = target_file.split('/')[:-1]
            save_dir = '/'.join(save_dir)

            for group, line_list in data_dic.items():
                file_path = os.path.join(save_dir, f'train_{group}.trn')
                with open(file_path, 'wt', encoding='utf-8') as f:
                    for text in line_list:
                        f.write(text)
                    print(f'File created -> {file_path}')
                print('Done!')

    # from datasets import load_dataset('csv', /data/info/train.csv)
    def get_dataset_dict(self, file_name:str, extention:str ='wav')-> dict:
        ''' path_dir에 있는 파일을 dict형으로 가공하여 리턴턴
            retrun data_dict = {
                    'audio': ['file_path1', 'file_path2', ...], 
                    'text' : ['text1', 'text2', ...],
            }   
        '''
        data_dic = {'path': [], 'sentence': []}
        print(f'file name : {file_name}')
        with open(file_name, 'rt', encoding='utf-8') as f:
            lines = f.readlines()
            for line in lines:
                audio, text = line.split('::')
                # 공백 문자가 있을 경우를 생각해서 공백 제거
                audio = audio.strip()
                # join 함수가 자동으로 백슬래시를 넣어줌
                audio = os.path.join(
                    os.getcwd(), #현재 디렉토리 경로를 붙여줌 절대 경로로 만들어주기 위해 
                    self.VOICE_DIR.replace('./', ''),
                    audio
                )
                if audio.endswith('.pcm'):
                    audio = audio.replace('pcm', f'{extention}')
                text = text.strip()
                # 전체를 확인하는 지 체크크
                # print(f'{audio} : {text}')

                data_dic['path'].append(audio)
                data_dic['sentence'].append(text)

        return data_dic
    
    # pkl 형태의 파일로 저장 
    def save_trn_to_pkl(self, file_name:str)->None:
        '''.trn 파일을 Dict로 만든 후 바이너리로 저장'''
        data_dict = self.get_dataset_dict(file_name=file_name)
        # pickle file dump
        file_name_pickle = file_name + '.dic.pkl'
        with open(file_name_pickle, 'wb') as f:
            pickle.dump(data_dict,f)
        print(f'Dataset is saved as pickle file as dictionary')
        print(f'Dataset path : {file_name_pickle}')
    
    # csv 형태의 파일로 저장장
    def save_trn_to_csv(self, file_name:str)->None:
        ''' .trn 파일을 .csv로 저장'''
        print(f'경로 확인: {file_name}')
        data_dict = self.get_dataset_dict(file_name=file_name)
        # csv file dump
        file_name_csv = file_name.split('.')[:-1]
        file_name_csv = ''.join(file_name_csv) +'.csv'
        print(f'경로 확인: {file_name_csv}')
        if file_name_csv.startswith('./'):
            file_name_csv.replace('./','')
        data_frm = pd.DataFrame(data_dict)
        # header = True -> Whisper input에서 헤더 정보를 사용하기 때문에 True로 설정해줘야함
        data_frm.to_csv(file_name_csv, index=False, header=True)
        print(f'Dataset is saved as csv file as dictionary')
        print(f'Dataset path : {file_name_csv}')

    # Train/Test set 분리 
    def split_train_test(self, target_file: str, train_size: float = 0.8) -> None:
        '''입력 파일(.trn)을 train/test 분류하여 저장
            if train_size is 0.8,
                train:test = 80%:20%
            eng) Whisper model require header columns for example file_path/sentence
        '''
        with open(target_file, 'rt', encoding='utf-8') as f:
            data = f.readlines()
            train_num = int(len(data) * train_size)
            header =None
            if target_file.endswith('.csv'):
                header = data[0]
                data = data[1:]
                train_num = int(len(data)*train_size)
        shuffle(data)
        data_train = sorted(data[0:train_num])
        data_test = sorted(data[train_num:])

        # train_set 파일 저장
        train_file = target_file.split('.')[:-1]
        train_file = ''.join(train_file) + '_train.csv'
        if target_file.startswith('.'):
            train_file = '.' + train_file
        with open(train_file, 'wt', encoding='utf-8')as f:
            if header:
                f.write(header)
            for line in data_train:
                f.write(line)
        print(f'Train_dataset saved -> {train_file} ({train_size*100}%)')

        # test_set 파일 저장
        test_file = target_file.split('.')[:-1]
        test_file = ''.join(test_file) + '_test.csv'
        if target_file.startswith('.'):
            test_file = '.' +test_file
        with open(test_file, 'wt', encoding='utf-8')as f:
            if header:
                f.write(header)
            for line in data_test:
                f.write(line)
        print(f'Test_dataset saved -> {test_file} ({(1.0-train_size)*100:.1f}%)')
    
    # 디렉토리 내부의 모든 텍스트 파일을 삭제 
    def remove_all_test_files(self, target_dir: str, extention: str = '.txt') -> None:
        print(f'Target_directory : {target_dir}')
        sub_directories = sorted(os.listdir(target_dir))
        num_files = 0
        for directory in tqdm(sub_directories, desc=f'Delete all {extention} files'):
            files = os.listdir(os.path.join(target_dir, directory))
            for file_name in files:
                if file_name.endswith(f'{extention}'):
                    os.remove(
                        os.path.join(target_dir, directory, file_name)
                    )
                    num_files += 1
        print(f'Removed {num_files} {extention} files is removed')
    
class LoadHuggingFaceDataset:
    def __init__(self, ) -> None:
        print('Loading dataset from Hugging-face')
    
    def load_dataset(self, )-> DatasetDict:
        '''Build dataset from hugging-face dataset'''
        dataset = DatasetDict()
        dataset = load_dataset("google/fleurs", "ko_kr")
        dataset = dataset.remove_columns(['id','num_samples','raw_transcription', 'gender','lang_id','language','lang_group_id','audio'])
        # dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))
        
        print(dataset['train']['transcription'][0])
        return dataset
    
    def save2csv(self,target_dir)-> None:
        '''save dataset to csv'''
        dataset = load_dataset("google/fleurs", "ko_kr")
        dataset = dataset.remove_columns(['id','num_samples','raw_transcription', 'gender','lang_id','language','lang_group_id','audio'])
        transcription_lst = dataset['train']['transcription']
        fleursWav_lst = os.listdir(target_dir)
        fleursWav_paths = [os.path.join(os.path.abspath(target_dir), fname) for fname in fleursWav_lst]
        dataFrm = pd.DataFrame({
            'path' : fleursWav_paths,
            'sentence' : transcription_lst,
        })
        dataFrm.to_csv("data/info/fleurs_transcription.csv", index=False, encoding="utf-8")
        
    def save_dataset_audio(self, dataset_name, config_name, split="train", save_dir="./saved_wav", max_samples=None):
        # 데이터셋 로드
        dataset = load_dataset(dataset_name, config_name, split=split)
        # 저장 디렉토리 생성
        os.makedirs(save_dir, exist_ok=True)

        print(f"[INFO] Saving {split} split of {dataset_name} ({config_name}) to '{save_dir}'")

        for i, sample in enumerate(dataset):
            if max_samples is not None and i >= max_samples:
                break

            audio_array = sample["audio"]["array"]
            sampling_rate = sample["audio"]["sampling_rate"]

            save_path = os.path.join(save_dir, f"{split}_{i:05d}.wav")

            # soundfile을 사용해 저장
            sf.write(file=save_path, data=audio_array, samplerate=sampling_rate)

            if i % 100 == 0:
                print(f"  - 저장 경로: {save_path}")

        print("오디오 파일 저장완료")

# 테스트를 위한 자기 호출 
if __name__ == '__main__':
    # audio = r'D:\Whisper\data\audio\KsponSpeech_01\KsponSpeech_01\KsponSpeech_0001\KsponSpeech_000001.pcm'
    # prepareds = PrepareDataset()
    # prepareds.pcm2audio(audio_path=audio, remove=True)
    # source_dir = 'data/audio/KsponSpeech_01'
    # prepareds = PrepareDataset()
    # prepareds.process'_audio(source_dir=source_dir)
    # text_file = r'D:\Whisper\data\audio\KsponSpeech_01\KsponSpeech_0001\KsponSpeech_000001.txt'
    # target_file = r'data\audio\KsponSpeech_01'
    # prepareds = PrepareDataset()
    # prepareds.remove_all_test_files(target_file)
    Dataset = LoadHuggingFaceDataset()
    # Dataset.save_dataset_audio(
    #     dataset_name="google/fleurs",
    #     config_name="ko_kr",
    #     split="train",
    #     save_dir="./data/audio/fleurs",
    #     max_samples=None  # 저장할 개수 제한 (None이면 전체 저장)
    # )


    Dataset.save2csv(target_dir=r'data\audio\fleurs')