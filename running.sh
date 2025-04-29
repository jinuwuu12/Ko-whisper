# pre-trained model에 fleurs 데이터 학습 및 테스트 결과 노션에 적어놓음 
python finetune.py -b openai/whisper-large-v3-turbo -p openai/whisper-large-v3-turbo -ftm .\model_finetuned -train .\data\info\fleurs_transcription_train.csv -valid .\data\info\fleurs_transcription_valid.csv -test .\data\info\fleurs_transcription_test.csv --metric cer
# 아무것도 없는 생 구조에 fleurs 데이터 학습 테스트 -> 결과 말안됨됨
python finetune_copy.py -b openai/whisper-small -p openai/whisper-small -ftm .\model_finetuned -train .\data\info\fleurs_transcription_train.csv -valid .\data\info\fleurs_transcription_valid.csv -test .\data\info\fleurs_transcription_test.csv --metric cer
# 아무것도 없는 모델에 KsponSpeech 데이터 학습 테스트 - 논문에 따르면 multilingual 은 모델의 큰 모델이 정확도가 높다고 나와있음
# ksponspeech 전처리를 whisper에 학습 된 데이터와 비슷하게 해서 학습예정.(v3-turbo-model)
python finetune_copy.py -b openai/whisper-large-v3-turbo -p openai/whisper-large-v3-turbo -ftm .\model_finetuned -train data/info/train_KsponSpeech_01_train.csv -valid data/info/train_KsponSpeech_01_test.csv -test data/info/eval_clean.csv --metric cer
# 전처리 한 ksponspeech를 large-v3-turbo에 학습
python finetune_copy.py -b openai/whisper-large-v3-turbo -p openai/whisper-large-v3-turbo -ftm .\model_finetuned -train data/info/train_KsponSpeech_01_train.csv -valid data/info/train_KsponSpeech_01_test.csv -test data/info/fleurs_transcription_test.csv --metric cer
# ksponspeech 전처리를 whisper에 학습 된 데이터와 비슷하게 해서 학습예정.(small)
python finetune_clear_model.py -b openai/whisper-small -p openai/whisper-small -ftm .\model_finetuned -train data/info/train_KsponSpeech_01_train.csv -valid data/info/train_KsponSpeech_01_test.csv -test data/info/eval_clean.csv --metric cer



# turbo 모델에 전처리 한 KsponSpeech 데이터 파인튜닝