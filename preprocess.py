import argparse
from utils import PrepareDataset

def audio_process(config) -> None:
    print('오디오 전처리 시작')
    preprocessor = PrepareDataset()
    preprocessor.process_audio(
        source_dir=config.target_dir,
        remove_original_audio=config.remove_original_audio,
    )

def file_process(config) -> None:
    print('파일 전처리 시작')
    preprocessor = PrepareDataset()
    if config.target_file:
        if not(
            config.csv or
            config.pkl
        ):
            print(f'If --target-file (-tf) is feed \
                one of --csv, --pkl, --split_whole_data (-w) must be set')
            return 
        
        if config.csv:
            preprocessor.save_trn_to_csv(config.target_file)
        if config.pkl:
            preprocessor.save_trn_to_pkl(config.target_file)
        if config.split_whole_data:
            preprocessor.split_whole_data(config.target_file)


    if config.convert_all_encoding:
        if not config.target_dir:
            print('If ')
        preprocessor.convert_all_encoding(config.target_dir)

def get_parser():
    parser = argparse.ArgumentParser(
        prog = 'Kspon dataset pre-processing',
        description = 'Process Korean speech dataset',
    )
    
    sub_parser = parser.add_subparsers(title='sub_command')

    # 오디오 전처리를 위한 파서 
    # Parser for sub-command 'audio'
    parser_audio = sub_parser.add_parser(
        'audio',
        help='sub-command for audio precessing'
    )
    parser_audio.add_argument(
        '--target-dir', '-t',
        required = True,
        help = 'directory of audio files'
    )
    parser_audio.add_argument(
        '--remove-original-audio', '-r',
        action = 'store_true',
    )
    parser_audio.set_defaults(func = audio_process)



    # 파일 전처리를 위한 파서 
    # Parser for sub-command 'file'
    parser_file = sub_parser.add_parser(
        'file',
        help = 'handling txt encoding, generate pkl/csv file,\
            or split file (train/test)'
    )
    parser_file.add_argument(
        '--target-file', '-tf',
        required= True,
        help='Target file name for processing'
    )
    parser_file.add_argument(
        '--convert-all-encoding', '-c',
        action='store_true',
        help='Convert all text files to utf-8 under target_dir'
    )
    parser_file.add_argument(
        '--target-dir', '-t',
        # required= True,
        help='directory of txt files'
    )
    parser_file.add_argument(
        '--split-whole-data', '-w',
        action='store_true',
        help='split whole trn data by sector'
    )
    parser_file.add_argument(
        '--pkl', 
        action='store_true',
        help='convert trn file to pkl file'
    )
    parser_file.add_argument(
        '--csv',
        action='store_true',
        help='convert trn file to csv file'
    )
    parser_file.set_defaults(func = file_process)




    config = parser.parse_args()
    return config


    



if __name__ == '__main__':
    config = get_parser()
    config.func(config)