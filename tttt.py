import tqdm
def eval_audio_process(
        self,
        source_dir : str,
        remove_original_audio:bool = True,
)-> None:
    print(f'source dir: {source_dir}')

    for audio in tqdm(source_dir, desc=f'Processing directory: {source_dir}'):
        file_name = audio
        if file_name.endswith('.pcm'):
            self.pcm2audio(
                audio_path=os.path.join(source_dir, file_name),
                extention='wav',
                remove= remove_original_audio
            )