
# export method=visionzip
# export method=vispruner
# export method=atome-merge
# export method=atome-merge-attention
# export method=atome-merge-attention-dpp
# export method=cdpruner
# export method=cdpruner-attention
# export method=fastadasp
# export method=fastadasp-attention
# export method=fastadasp-attention-dpp
export method=SA-MAP

# remain_token_ratio is the ratio of tokens to retain
export remain_token_ratio=0.6
# threshold is used in SA-MAP
export threshold=0.85

bash run_audio.sh --model Qwen2-Audio-7B \
--model-path  /xxx/Qwen2-Audio-7B/ \
--data LibriSpeech,AISHELL-1,AISHELL-2,Fleurs-zh,Fleurs-en,WenetSpeech,mmau-test-mini,MELD,Nonspeech7k,TUT2017,VocalSound \
--work-dir eval_result/$method/ratio_$remain_token_ratio \
--eval-method 'vb-mcq'

# bash run_audio.sh --model Kimi-Audio  \
# --model-path /your/model/path/ \
# --data LibriSpeech,AISHELL-1,AISHELL-2,Fleurs-zh,Fleurs-en,WenetSpeech,mmau-test-mini,MELD,Nonspeech7k,TUT2017,VocalSound \
# --work-dir eval_result/$method/ratio_$remain_token_ratio \
# --eval-method 'vb-mcq'

# bash run_audio.sh --model GLM-ASR-Nano \
# --model-path /your/model/path/ \
# --data LibriSpeech,AISHELL-1,AISHELL-2,Fleurs-zh,Fleurs-en,WenetSpeech,mmau-test-mini,MELD,Nonspeech7k,TUT2017,VocalSound \
# --work-dir eval_result/$method/ratio_$remain_token_ratio \
# --eval-method 'vb-mcq'