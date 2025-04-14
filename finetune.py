'''
Fine-tuning whisper (possible: tiny, small, medium, etc.)
References:
    - Master reference -> https://huggingface.co/blog/fine-tune-whisper
    - Korean blog -> https://velog.io/@mino0121/NLP-OpenAI-Whisper-Fine-tuning-for-Korean-ASR-with-HuggingFace-Transformers
Written by: Prof. Jin u HA
License: MIT license
'''
import argparse
import numpy as np
from datasets import load_dataset, DatasetDict
from utils import get_unique_directory
from scipy.io.wavfile import read


from transformers import WhisperFeatureExtractor, WhisperTokenizer, WhisperProcessor

def get_config() -> argparse.ArgumentParser:
    '''Whisper finetuning args parsing function'''
    parser =argparse.ArgumentParser()

    parser.add_argument('--base-model' , '-b',
                        required=True,
                        help='Base model for tokenizer, processor, feature extractor. \
                        Ex. openai/whisper-tiny, openai/whisper-small from huggingface'
                        )
    parser.add_argument('--pretrained-model', '-p',
                        default='',
                        help='pretrained model from hugging face or local \
                            if you trained several times you can use it'
                            )
    parser.add_argument('--out-dir','-od',
                        required=True,
                        help='If you finished your train your model will \
                        be saved this folder'
                        )
    parser.add_argument('--finetuned-model-dir', '-ftm',
                        required=True,
                        help='Directory for saving fine-tuned model \
                        (best model after train)'
                        )
    
    # train, test dataset
    parser.add_argument('--train-set', '-train', 
                        required=True, 
                        help='Train dataset name (file name or file path)'
                        )
    parser.add_argument('--valid-set', '-valid', 
                        required=True, 
                        help='Train dataset name (file name or file path)'
                        )
    parser.add_argument('--test-set', '-test', 
                        required=True, 
                        help='Train dataset name (file name or file path)'
                        )
    
    # language select
    parser.add_argument('--language',
                        default= 'Korean',
                        help='select tokenizer language that profit your data'
                        )
    parser.add_argument('--task',
                        default='transcribe',
                        help = 'you can choose transcribe(default) or translate'
                        )
    parser.add_argument('--sampling-rate', '-sample',
                        type=int,
                        default=16000,
                        help='wav files sampling rate'
                        )
    config = parser.parse_args()
    return config

class WhisperTrainer:
    '''Whisper finetune trainer'''
    def __init__(self, config) -> None:
        # 이렇게 해야 클래스 전체에서 config를 사용할 수 있음
        self.config = config

        # 사전 학습 모델 - 2개
        # Base model -> tokenizer, feature_exrtactor ,processor
        # pre-trained model -> model for fine-tuning

        if config.pretrained_model == True:
            self.pretrained_model = config.pretrained_model
        else:
            print('Pre-trained model is not given....')
            print(f'\nWe will set pre-trained model same to --base-model (-b): {config.base_model}')
            self.pretrained_model = config.base_model

        self.output_dir = get_unique_directory(
            dir_name = config.out_dir, 
            model_name = self.pretrained_model
            )
         
        self.finetuned_model_dir = get_unique_directory(
            dir_name = config.finetuned_model_dir,
            model_name = self.pretrained_model
            )
        print(f'\nTraining outputs will be saved -> {self.output_dir}')
        print(f'Fine-tuned model will be saved -> {self.finetuned_model_dir}\n')

        # Feature Extractor, Tokenizer 등록
        self.feature_extractor = WhisperFeatureExtractor.from_pretrained(
            pretrained_model_name_or_path = config.base_model
            )
        
        self.tokenizer = WhisperTokenizer.from_pretrained(
            pretrained_model_name_or_path = config.base_model, 
            language = config.language, 
            task = config.task,
            )

        #Processor 등록 
        self.processor = WhisperProcessor.from_pretrained(
            pretrained_model_name_or_path = config.base_model,
            language = config.language, 
            task = config.task,
            )


    def load_dataset(self, )-> DatasetDict:
        '''Build dataset containing train/valid.test sets'''
        dataset = DatasetDict()
        dataset['train'] = load_dataset(
            path='csv',
            name='aihub-ksponSpeech_dataset',
            split='train',
            data_files=self.config.train_set,
        )
        dataset['valid'] = load_dataset(
            path='csv',
            name='aihub-ksponSpeech_dataset',
            split='train',
            data_files=self.config.valid_set,
        )
        dataset['test'] = load_dataset(
            path='csv',
            name='aihub-ksponSpeech_dataset',
            split='train',
            data_files=self.config.test_set,
        )
        return dataset

    def compute_metrics(self, pred) -> dict:
        '''Prepare evaluation metric (WER, CER, MOS)'''
        pass

    def prepare_dataset(self, batch) -> object:
        '''Get input features with numpy array & sentence label'''
        # load and resample audio data from 48 to 16kHz
        audio = batch["path"]
        _ , data = read(audio)
        audio_array = np.array(data, dtype=np.float32)

        # compute log-Mel input features from input audio array
        batch["input_features"] = self.feature_extractor(
            audio_array, 
            sampling_rate = self.config.sampling_rate
            ).input_features[0] # 왜 0번째를 들고와?? 이거 ㄹㅇ 코드 까봐야함 

        # encode target text to label ids 
        batch["labels"] = self.tokenizer(batch["sentence"]).input_ids
        return batch

    def process_dataset(self, dataset: DatasetDict) -> tuple:
        '''Process loaded dataset applying prepare_dataset()'''
        # num_proc는 cpu 코어 갯수에 따라 달라짐짐
        train = dataset['train'].map(function = self.prepare_dataset, remove_columns=dataset.column_names['train'], num_proc=20)
        valid = dataset['valid'].map(function = self.prepare_dataset, remove_columns=dataset.column_names['valid'], num_proc=20)
        test = dataset['test'].map(function = self.prepare_dataset, remove_columns=dataset.column_names['test'], num_proc=20)
        
        return (train, valid, test)

    def enforce_fine_tune_lang(self) -> None:
        '''Enforce fine-tune language'''

    def create_trainer(self, train, valid) -> None:
        '''Create seq2seq trainer'''
        pass

    def run(self)-> None:
        '''Run trainer'''


if __name__ =='__main__':
    config = get_config()
    trainer = WhisperTrainer(config)
    result = trainer.load_dataset()
    print(result)
    print(result['train'][0]['sentence'])


    input_str = result['train'][0]['sentence']
    labels = trainer.tokenizer(input_str).input_ids
    print(f'labels : {labels}')


    decoded_with_special_tokens = trainer.tokenizer.decode(labels, skip_special_tokens=False)
    decoded_str_without_special_tokens = trainer.tokenizer.decode(labels, skip_special_tokens=True)
    print(f"Input:                 {input_str}")
    print(f"Decoded w/ special:    {decoded_with_special_tokens}")
    print(f"Decoded w/out special: {decoded_str_without_special_tokens}")
    print(f"Is equal:             {input_str == decoded_str_without_special_tokens}")

    train, valid, test = trainer.process_dataset(result)
    print(train, valid, test)