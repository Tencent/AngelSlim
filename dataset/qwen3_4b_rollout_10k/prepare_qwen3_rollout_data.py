import argparse
import json
import random
import re
from pathlib import Path

from datasets import load_dataset
from transformers import AutoTokenizer

DATASET_MIX = [
    {
        "name": "HuggingFaceH4/ultrachat_200k",
        "split": "train_sft",
        "category": "general_instruction",
        "quota": 3000,
    },
    {
        "name": "teknium/OpenHermes-2.5",
        "split": "train",
        "category": "general_instruction",
        "quota": 2000,
    },
    {
        "name": "TIGER-Lab/MathInstruct",
        "split": "train",
        "category": "reasoning_math",
        "quota": 2000,
    },
    {
        "name": "ise-uiuc/Magicoder-Evol-Instruct-110K",
        "split": "train",
        "category": "code",
        "quota": 1500,
    },
    {
        "name": "Yukang/LongAlpaca-12k",
        "split": "train",
        "category": "long_context",
        "quota": 1500,
    },
]


def normalize_text(text):
    text = str(text or "").replace("\x00", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def dedup_key(text):
    text = normalize_text(text).lower()
    return re.sub(r"[^a-z0-9\u4e00-\u9fff]+", "", text)[:4096]


def first_human_from_conversations(conversations):
    for turn in conversations or []:
        role = turn.get("from") or turn.get("role")
        if role in ("human", "user"):
            return turn.get("value") or turn.get("content")
    return None


def extract_prompt(row, dataset_name):
    if dataset_name == "HuggingFaceH4/ultrachat_200k":
        return row.get("prompt") or first_human_from_conversations(row.get("messages"))
    if dataset_name == "teknium/OpenHermes-2.5":
        return first_human_from_conversations(row.get("conversations"))
    if dataset_name == "TIGER-Lab/MathInstruct":
        return row.get("instruction")
    if dataset_name == "ise-uiuc/Magicoder-Evol-Instruct-110K":
        return row.get("instruction")
    if dataset_name == "Yukang/LongAlpaca-12k":
        instruction = normalize_text(row.get("instruction"))
        context = normalize_text(row.get("input"))
        if context:
            return f"{instruction}\n\nContext:\n{context}"
        return instruction
    return None


def should_skip_prompt(prompt, tokenizer, max_prompt_tokens):
    if not prompt:
        return True
    if len(prompt) < 80 or len(prompt) > 50000:
        return True
    lowered = prompt.lower()
    blocked = ("mmlu", "hellaswag", "winogrande", "arc challenge", "gsm8k")
    if any(item in lowered for item in blocked):
        return True
    token_count = len(tokenizer.encode(prompt, add_special_tokens=False))
    return token_count < 16 or token_count > max_prompt_tokens


def build_prompts(args):
    random.seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    prompts = []
    seen = set()

    for source in DATASET_MIX:
        dataset = load_dataset(source["name"], split=source["split"])
        indices = list(range(len(dataset)))
        random.shuffle(indices)
        kept = 0
        scanned = 0
        for idx in indices:
            if kept >= source["quota"]:
                break
            scanned += 1
            row = dataset[idx]
            prompt = normalize_text(extract_prompt(row, source["name"]))
            key = dedup_key(prompt)
            if key in seen or should_skip_prompt(prompt, tokenizer, args.max_prompt_tokens):
                continue
            seen.add(key)
            prompts.append(
                {
                    "id": f"{source['category']}_{kept:06d}",
                    "source": source["name"],
                    "category": source["category"],
                    "prompt": prompt,
                }
            )
            kept += 1
        print(
            f"source={source['name']} category={source['category']} "
            f"kept={kept} scanned={scanned}"
        )

    random.shuffle(prompts)
    prompts = prompts[: args.num_samples]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with args.prompts_path.open("w", encoding="utf-8") as f:
        for item in prompts:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"wrote {len(prompts)} prompts to {args.prompts_path}")


def load_jsonl(path):
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def apply_chat_template(tokenizer, prompt, thinking):
    directive = "/think" if thinking else "/no_think"
    content = f"{prompt}\n\n{directive}"
    messages = [{"role": "user", "content": content}]
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=thinking,
        )
    except TypeError:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )


def is_good_response(text, tokenizer, max_total_tokens, prompt_tokens):
    text = normalize_text(text)
    if len(text) < 40:
        return False
    lowered = text.lower()
    bad_phrases = (
        "i cannot answer",
        "i can't answer",
        "as an ai language model, i cannot",
    )
    if any(phrase in lowered for phrase in bad_phrases):
        return False
    response_tokens = len(tokenizer.encode(text, add_special_tokens=False))
    return response_tokens >= 16 and prompt_tokens + response_tokens <= max_total_tokens


def rollout(args):
    from vllm import LLM, SamplingParams

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    prompts = list(load_jsonl(args.prompts_path))
    if args.limit is not None:
        prompts = prompts[: args.limit]
    done_ids = set()
    if args.output_path.exists():
        for item in load_jsonl(args.output_path):
            done_ids.add(item.get("id"))
    prompts = [item for item in prompts if item["id"] not in done_ids]

    llm = LLM(
        model=args.model_path,
        tokenizer=args.model_path,
        trust_remote_code=True,
        tensor_parallel_size=args.tensor_parallel_size,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        dtype="auto",
    )
    vllm_tokenizer = llm.get_tokenizer()
    sampling = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        max_tokens=args.max_new_tokens,
        repetition_penalty=args.repetition_penalty,
    )

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    total_written = len(done_ids)
    with args.output_path.open("a", encoding="utf-8") as out:
        for start in range(0, len(prompts), args.batch_size):
            batch = prompts[start : start + args.batch_size]
            rendered = []
            metadata = []
            for item in batch:
                thinking = item["category"] in {"reasoning_math", "long_context"}
                prompt_text = apply_chat_template(vllm_tokenizer, item["prompt"], thinking)
                prompt_tokens = len(vllm_tokenizer.encode(prompt_text, add_special_tokens=False))
                rendered.append(prompt_text)
                metadata.append((item, thinking, prompt_tokens))

            outputs = llm.generate(rendered, sampling)
            for output, (item, thinking, prompt_tokens) in zip(outputs, metadata):
                response = output.outputs[0].text.strip()
                if not is_good_response(response, tokenizer, args.max_model_len, prompt_tokens):
                    continue
                response_tokens = len(tokenizer.encode(response, add_special_tokens=False))
                record = {
                    "id": item["id"],
                    "source": item["source"],
                    "category": item["category"],
                    "rollout_model": args.model_path,
                    "thinking": thinking,
                    "prompt_tokens": prompt_tokens,
                    "response_tokens": response_tokens,
                    "total_tokens": prompt_tokens + response_tokens,
                    "messages": [
                        {"role": "user", "content": item["prompt"]},
                        {"role": "assistant", "content": response},
                    ],
                }
                out.write(json.dumps(record, ensure_ascii=False) + "\n")
                total_written += 1
            out.flush()
            print(f"processed={start + len(batch)} written_total={total_written}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["build_prompts", "rollout"], required=True)
    parser.add_argument(
        "--model-path",
        default="Qwen/Qwen3-4B",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("qwen3_distill_rollout_staging"),
    )
    parser.add_argument("--num-samples", type=int, default=10000)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--seed", type=int, default=20260514)
    parser.add_argument("--max-prompt-tokens", type=int, default=6144)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--tensor-parallel-size", type=int, default=8)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--repetition-penalty", type=float, default=1.05)
    args = parser.parse_args()
    args.prompts_path = args.output_dir / "prompts_10k.jsonl"
    args.output_path = args.output_dir / "qwen3_4b_rollout_10k.jsonl"

    if args.mode == "build_prompts":
        build_prompts(args)
    else:
        rollout(args)


if __name__ == "__main__":
    main()
