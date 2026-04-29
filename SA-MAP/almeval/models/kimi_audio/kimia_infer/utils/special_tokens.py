from dataclasses import dataclass


@dataclass
class ExtraTokens:
    msg_end: int
    user_msg_start: int
    assistant_msg_start: int

    media_begin: int
    media_end: int

    kimia_text_blank: int
    kimia_text_eos: int

    kimia_user_msg_start: int
    kimia_assistant_msg_start: int

    kimia_speech_ct_id: int
    kimia_speech_ctd_id: int

    pad: int


def instantiate_extra_tokens(tokenizer):
    if hasattr(tokenizer, "special_tokens"):
        map_fn = lambda x: tokenizer.special_tokens[x]
    elif hasattr(tokenizer, "convert_tokens_to_ids"):
        map_fn = lambda x: tokenizer.convert_tokens_to_ids(x)
    else:
        raise ValueError(f"Invalid tokenizer type: {type(tokenizer)}")
    return ExtraTokens(
        msg_end=map_fn("<|im_msg_end|>"),  # 0
        user_msg_start=map_fn("<|im_user_msg_start|>"),  # 1
        assistant_msg_start=map_fn("<|im_assistant_msg_start|>"),  # 2
        media_begin=map_fn("<|im_media_begin|>"),  # 13
        media_end=map_fn("<|im_media_end|>"),  # 15
        kimia_text_blank=map_fn("<|im_kimia_text_blank|>"),  # 18
        kimia_text_eos=map_fn("<|im_kimia_text_eos|>"),  # 19
        kimia_user_msg_start=map_fn("<|im_kimia_user_msg_start|>"),  # 22
        kimia_assistant_msg_start=map_fn("<|im_kimia_assistant_msg_start|>"),  # 23
        kimia_speech_ct_id=map_fn("<|im_kimia_speech_ct_id|>"),  # 27
        kimia_speech_ctd_id=map_fn("<|im_kimia_speech_ctd_id|>"),  # 28
        pad=tokenizer.pad_id,
    )
