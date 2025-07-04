# 如何准备数据

本章节介绍AngelSlim中如何准备压缩所需数据，准备完成后将数据路径填入config文件`data_path`字段即可。

## 支持文件格式
AngelSlim支持`jsonl`和`parquent`两种文件格式。

### jsonl文件

jsonl格式为一行一条data，每行data为json格式。支持如下格式

#### 格式1
此格式为类openai抓取的messages格式

```
{"system_prompt": "你是一个人工智能助手", "messages": [{"role": "user", "content": "回答脑筋急转弯：睡美人最怕的是什么？"}, {"role": "assistant", "content": "睡美人最怕的是闹钟。因为闹钟会把她从梦中唤醒。这个脑筋急转弯以幽默的方式表达了睡美人这一角色的特点。"}]}
```

- `"system_prompt"`：数据的system prompt
- `"messages"`：dict中的数据为prompt与answer信息。`"role"`为user表示prompt数据，assistant为模型answer数据

#### 格式2

简单dict格式
```
{"system_prompt": "你是一个人工智能助手", "input": "回答脑筋急转弯：睡美人最怕的是什么？", "output": "睡美人最怕的是闹钟。因为闹钟会把她从梦中唤醒。这个脑筋急转弯以幽默的方式表达了睡美人这一角色的特点。"}
```
- `"system_prompt"`：数据的system prompt
- `"input"`：prompt数据
- `"output"`：answer数据

#### 格式3

开源sharegpt数据集格式

```
{"conversations":[{"from":"human","value":"回答脑筋急转弯：睡美人最怕的是什么？"},{"from":"gpt","value":"睡美人最怕的是闹钟。因为闹钟会把她从梦中唤醒。这个脑筋急转弯以幽默的方式表达了睡美人这一角色的特点。"}]}
```

- `"conversations"`：dict中"from"字段为human为prompt数据，gpt为模型answer数据

### parquet文件

支持parquet文件喂入AngelSlim
需要parquet中的text列数据为下所示的开源sharegpt格式即可，AngelSlim会读取text列内容喂入tokenizer解码。
```
[ { "from": "human", "value": "这本著作极其出色。请用同义词替换加粗的词语，构建一个新的句子。" }, { "from": "gpt", "value": "这部作品极为杰出。" } ]
```

## DATA文件测试
```
from transformers import AutoModelForCausalLM, AutoTokenizer

from slim.data.dataloader import DataLoaderFactory

model_path = "CKPT/PATH"

tokenizer = AutoTokenizer.from_pretrained(
    model_path, trust_remote_code=True
)
dataloader = DataLoaderFactory.create_data_loader(
    data_type='TextDataset',
    processor=tokenizer,
    device='cpu',
    max_length=2048,
    batch_size=1,
    num_samples=1,
    data_source=model_path,
)

for data in dataloader:
    print(data)
```