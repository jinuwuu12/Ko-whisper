from transformers import Seq2SeqTrainingArguments

def get_training_arguments(output_dir: str, metric: str) -> Seq2SeqTrainingArguments:
    return Seq2SeqTrainingArguments(
            output_dir = output_dir,                        # change to a repo name of your choice
            per_device_train_batch_size = 32,               # select your batch size 
            gradient_accumulation_steps = 2,                # increase by 2x for every 2x decrease in batch size
            learning_rate=1e-5,
            warmup_steps=300,                               # transformer는 adam이라는 옵티마이저를 사용 워밍업하는개념
            max_steps=5000,                                 # 에폭 수 설정 해줬을 때는 step 을 없애야한다.  
            gradient_checkpointing=True,
            fp16 = True,                                    # 부동소수점 자릿수 defalut: fp32 -> fp16 학습이 빨라진다고 함(AMP)
            eval_strategy="steps",
            # num_train_epochs=30,                          # epochs 를 지정해주면 max_steps와 비교하여 더 작은 쪽에서 돌아감 혹시나 5000스텝 이상으로 돌아가는 경우에는 
            per_device_eval_batch_size=16,
            predict_with_generate=True,
            generation_max_length=225,
            save_steps=1000,
            eval_steps=1000,
            logging_steps=500,                              # 25 step 마다 output에 저장 (용량이 크니 주의해야함)
            # report_to=["tensorboard"],
            load_best_model_at_end=True,                    # 마지막에 베스트 모델 저장
            metric_for_best_model=metric, 
            greater_is_better=False, 
            push_to_hub=False,                              # push hugging-face hub
        )