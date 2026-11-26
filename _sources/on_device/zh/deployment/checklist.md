# 上线前检查清单

[← 部署总览](./index.md) ｜ [← 返回总览](../index.md)

- [ ] `config.json` 中存在 `quantization_config`，且 `ignore` / `ignored_layers` 字段与所用引擎一致；
- [ ] 使用 Transformers 或 `tools/infer.py` 能正常生成文本；
- [ ] 服务化路径（vLLM / SGLang）能通过 `openai.sh` 调用；
- [ ] 使用 STQ 低比特 GGUF 时，llama.cpp 已切换到对应支持分支（PR #22836）；Hy4-flash 需要 PR #32517；
- [ ] 使用 Strata 时，源码已切换到 PR #7120 对应分支，且所有 GGUF 分片位于同一目录；
- [ ] 端侧设备的内存、存储与线程数满足模型要求；
- [ ] 若使用 Hy4 preview 多机部署，两台机器的 manifest、receipt 与执行计划一致，并且端口互通；
- [ ] 按业务需要，在目标数据上重新验证精度，再决定是否进入训练流程。

如有问题，欢迎在 [GitHub Issues](https://github.com/Tencent/AngelSlim/issues) 反馈。

← [返回总览](./index.md)

[← 部署总览](./index.md) ｜ [← 返回总览](../index.md)
