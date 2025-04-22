'''
Fine-tuning whisper (possible: tiny, small, medium, etc.)
References:
    - Master reference -> https://huggingface.co/blog/fine-tune-whisper
    - Korean blog -> https://velog.io/@mino0121/NLP-OpenAI-Whisper-Fine-tuning-for-Korean-ASR-with-HuggingFace-Transformers
Written by: Prof. Jin u HA
License: MIT license
'''
import argparse
import pprint
import numpy as np
import pandas as pd
import evaluate
import torch
from datasets import load_dataset, DatasetDict, Dataset
from trainer.collator import DataCollatorSpeechSeq2SeqWithPadding
from utils import get_unique_directory
from scipy.io.wavfile import read


from transformers import (
    WhisperFeatureExtractor, 
    WhisperTokenizer, 
    WhisperProcessor, 
    WhisperForConditionalGeneration,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    )

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
                        default='./model_output',
                        help='If you finished your train your model will \
                        be saved this folder'
                        )
    parser.add_argument('--finetuned-model-dir', '-ftm',
                        # required=True,
                        help='Directory for saving fine-tuned model \
                        (best model after train)'
                        )
    
    # train, test dataset
    parser.add_argument('--train-set', '-train', 
                        # required=True, 
                        help='Train dataset name (file name or file path)'
                        )
    parser.add_argument('--valid-set', '-valid', 
                        # required=True, 
                        help='Train dataset name (file name or file path)'
                        )
    parser.add_argument('--test-set', '-test', 
                        # required=True, 
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
    parser.add_argument('--metric',
                        type=str, 
                        default='cer',
                        help='select your evaluation-rate matric'
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
            pretrained_model_name_or_path = config.base_model,
            return_attention_mask=True,
            )
        
        self.tokenizer = WhisperTokenizer.from_pretrained(
            pretrained_model_name_or_path = config.base_model, 
            language = config.language, 
            task = config.task,
            )
        # 모델 등록
        self.model = WhisperForConditionalGeneration.from_pretrained(self.pretrained_model)

        # Processor 등록 
        self.processor = WhisperProcessor.from_pretrained(
            pretrained_model_name_or_path = config.base_model,
            language = config.language, 
            task = config.task,
            )
        
        # Label Collator 등록
        self.data_collator = DataCollatorSpeechSeq2SeqWithPadding(
            processor=self.processor,
            decoder_start_token_id=self.model.config.decoder_start_token_id,
        )

        # train args 등록록
        self.training_args = Seq2SeqTrainingArguments(
            output_dir = self.output_dir,                   # change to a repo name of your choice
            per_device_train_batch_size = 32,               # select your batch size 
            gradient_accumulation_steps = 1,                # increase by 2x for every 2x decrease in batch size
            learning_rate=1e-5,
            warmup_steps=500,                               # transformer는 adam이라는 옵티마이저를 사용 워밍업하는개념
            max_steps=5000,
            gradient_checkpointing=True,
            fp16 = True,                                    # 부동소수점 자릿수 defalut: fp32 -> fp16 학습이 빨라진다고 함(AMP)
            eval_strategy="steps",
            per_device_eval_batch_size=16,
            predict_with_generate=True,
            generation_max_length=225,
            save_steps=1000,
            eval_steps=1000,
            logging_steps=100,                               # 25 step 마다 output에 저장 (용량이 크니 주의해야함)
            # report_to=["tensorboard"],
            load_best_model_at_end=True,                     # 마지막에 베스트 모델 저장
            metric_for_best_model=config.metric, 
            greater_is_better=False, 
            push_to_hub=False,                              # push hugging-face hub
        )

    def load_dataset(self, )-> DatasetDict:
        '''Build dataset containing train/valid.test sets'''
        dataset = DatasetDict()
        if self.config.train_set:
            train_df = pd.read_csv(self.config.train_set, encoding='utf-8')
            dataset['train'] = Dataset.from_pandas(train_df)
        if self.config.valid_set:
            valid_df = pd.read_csv(self.config.valid_set, encoding='utf-8')
            dataset['valid'] = Dataset.from_pandas(valid_df)
        if self.config.test_set:
            test_df = pd.read_csv(self.config.test_set, encoding='utf-8')
            dataset['test'] = Dataset.from_pandas(test_df)
        return dataset
    
        # dataset['train'] = load_dataset(
        #     path='csv',
        #     name='aihub-ksponSpeech_dataset',
        #     split='train',
        #     data_files=self.config.train_set,
        # )
        # dataset['valid'] = load_dataset(
        #     path='csv',
        #     name='aihub-ksponSpeech_dataset',
        #     split='train',
        #     data_files=self.config.valid_set,
        # )
        # dataset['test'] = load_dataset(
        #     path='csv',
        #     name='aihub-ksponSpeech_dataset',
        #     split='train',
        #     data_files=self.config.test_set,
        # )
        # print(dataset['train']['path'][0])
        # return dataset

    # chracter morphs 별 metric 
    def compute_metrics(self, pred) -> dict:
        '''Prepare evaluation metric (WER, CER, MOS)'''
        cer_metric = evaluate.load("cer")
        wer_metric = evaluate.load("wer")
        pred_ids = pred.predictions
        label_ids = pred.label_ids
        label_ids[label_ids == -100] = self.tokenizer.pad_token_id
        pred_str = self.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = self.tokenizer.batch_decode(label_ids, skip_special_tokens=True)
        cer = 100 * cer_metric.compute(predictions=pred_str, references=label_str)
        wer = 100 * wer_metric.compute(predictions=pred_str, references=label_str)
        return {
            "cer": cer,
            'wer': wer,
            }
    

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
        
        # tokenize sentence (with attention mask)
        masked = self.tokenizer(
            batch['sentence'],
            padding='max_length',
            truncation = True,
            max_length=128,
            return_attention_mask=True,
            return_tensors="pt",
        )
        # encode target text to label ids 
        # batch["labels"] = self.tokenizer(batch["sentence"]).input_ids
        batch["labels"] = masked.input_ids[0]
        batch["label_attention_mask"] = masked.attention_mask[0]

        return batch

    # def process_dataset(self, dataset: DatasetDict) -> tuple:
        '''Process loaded dataset applying prepare_dataset()'''
        # num_proc는 cpu 코어 갯수에 따라 달라짐
        print(dataset['train'])
        train = dataset['train'].map(function = self.prepare_dataset, remove_columns=dataset.column_names['train'], num_proc=16)
        print(f'train cache file root :{train.cache_files}')
        print(dataset['valid'])
        valid = dataset['valid'].map(function = self.prepare_dataset, remove_columns=dataset.column_names['valid'], num_proc=16)
        print(f'valid cache file root :{valid.cache_files}')
        print(dataset['test'])
        test = dataset['test'].map(function = self.prepare_dataset, remove_columns=dataset.column_names['test'], num_proc=16)
        
        return (train, valid, test)
    
    # 조건부 process_dataset
    def process_dataset(self, dataset: DatasetDict) -> tuple:
        train = valid = test = None

        # 왜 cpu 코어 16개를 쓸때보다 4개를 쓸때가 더 빠른가.. 멍청한 컴퓨터녀석
        if 'train' in dataset:
            train = dataset['train'].map(
                function=self.prepare_dataset,
                remove_columns=dataset['train'].column_names,
                num_proc=16,
                load_from_cache_file=False
            )
        if 'valid' in dataset:
            valid = dataset['valid'].map(
                function=self.prepare_dataset,
                remove_columns=dataset['valid'].column_names,
                num_proc=16,
                load_from_cache_file=False
            )
        if 'test' in dataset:
            test = dataset['test'].map(
                function=self.prepare_dataset,
                remove_columns=dataset['test'].column_names,
                num_proc=16,
                load_from_cache_file=False
            )

        return train, valid, test
    

    # 한국어 강제 지정
    def enforce_fine_tune_lang(self) -> None:
        '''Enforce fine-tune language'''
        # 언어 강제 지정
        self.model.config.suppress_tokens = []
        self.model.generation_config.suppress_tokens = []
        # model_config_issue -> https://github.com/huggingface/transformers/issues/21994
        self.model.config.forced_decoder_ids = self.processor.tokenizer.get_decoder_prompt_ids(
            language=config.language,
            task=config.task,
        )
        # generations_config issue -> https://github.com/huggingface/transformers/issues/21937
        self.model.generation_config.forced_decoder_ids = self.processor.tokenizer.get_decoder_prompt_ids(
            language=config.language,
            task=config.task,
        )

    def create_trainer(self, train, valid) -> Seq2SeqTrainer:
        '''Create seq2seq trainer'''
        return Seq2SeqTrainer(
            args = self.training_args,
            model = self.model,
            train_dataset = train,
            eval_dataset = valid,
            data_collator = self.data_collator,
            compute_metrics = self.compute_metrics,
            tokenizer=self.processor.feature_extractor,
        )


    def eval(self,) -> None:
        self.enforce_fine_tune_lang()

        # Load and preprocess dataset
        dataset = self.load_dataset()
        _, _, test = self.process_dataset(dataset=dataset)

        # 이코드는 뭘까 지피티가 시켰다.......
        '''학습을 하지않을 땐 평가전략을 끄란다 /n
           뭔개소리일까?
        '''
        self.training_args.eval_strategy = "no"

        trainer = Seq2SeqTrainer(
            model=self.model,
            args=self.training_args,
            tokenizer=self.processor.feature_extractor,
            data_collator=self.data_collator,
            compute_metrics=self.compute_metrics,
        )
        # Evaluate
        print('\n start evaluation!!!!\n')
        result_dic = trainer.evaluate(eval_dataset=test)
        pprint.pprint(result_dic)

        print('\nClearing GPU cache')
        torch.cuda.empty_cache()
        print('\nTraining completed!!!')

    def run(self) -> None:
        '''Run trainer'''
        self.enforce_fine_tune_lang()
        dataset = self.load_dataset()
        train, valid, test = self.process_dataset(dataset=dataset)
        trainer = self.create_trainer(train, valid)
        print('\nStart training...\n')
        trainer.train()
        trainer.save_model(self.finetuned_model_dir)
        print('\nStart testing performance using test_dataset...\n')
        result_dic = trainer.evaluate(eval_dataset=test)
        pprint.pprint(result_dic)
        print('\nClearing GPU cache')
        torch.cuda.empty_cache()
        print('\nTraining completed!!!')

if __name__ =='__main__':
    config = get_config()
#     config = get_config()
#     trainer = WhisperTrainer(config)
#     result = trainer.load_dataset()
#     print(result)
#     print(result['train'][0]['sentence'])
#     input_str = result['train'][0]['sentence']
#     labels = trainer.tokenizer(input_str).input_ids
#     print(f'labels : {labels}')
#     decoded_with_special_tokens = trainer.tokenizer.decode(labels, skip_special_tokens=False)
#     decoded_str_without_special_tokens = trainer.tokenizer.decode(labels, skip_special_tokens=True)
#     print(f"Input:                 {input_str}")
#     print(f"Decoded w/ special:    {decoded_with_special_tokens}")
#     print(f"Decoded w/out special: {decoded_str_without_special_tokens}")
#     print(f"Is equal:             {input_str == decoded_str_without_special_tokens}")
#     train, valid, test = trainer.process_dataset(result)
#     print(train, valid, test)


    whisperClass = WhisperTrainer(config=config)
    whisperClass.run()


