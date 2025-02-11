import os 

orig_audio = "./demo/chunk-232_32-242_4_trimmed_norm_float32.wav"
orig_transcript = 'kéo dài thêm hai năm nữa. trong các trận đánh này, khu trục hạm đã nắm giữ vai trò tiền phong lần đầu tiên và có thể cũng là lần cuối cùng trong lịch sử.'
temp_folder = "./demo/temp"
os.makedirs(temp_folder, exist_ok=True)
os.system(f"cp {orig_audio} {temp_folder}")
filename = os.path.splitext(orig_audio.split("/")[-1])[0]
with open(f"{temp_folder}/{filename}.txt", "w") as f:
    f.write(orig_transcript)
# run MFA to get the alignment
align_temp = f"{temp_folder}/mfa_alignments"
beam_size = 50 
retry_beam_size = 200
alignments = f"{temp_folder}/mfa_alignments/{filename}.csv"

if not os.path.isfile(alignments):
    os.system(f"mfa align -v --clean -j 1 --output_format csv {temp_folder} \
            viIPA hp_dtn_acoustic {align_temp} --beam {beam_size} --retry_beam {retry_beam_size}")
