# 移动端部署

[← 部署总览](./index.md) ｜ [← 返回总览](../index.md)

本页说明 Hy-MT1.5 在手机上的离线运行方式，包括官方 Android Demo 与支持 Arm SME2 指令集的设备。

## 1. Android 离线应用

AngelSlim 提供了 Hy-MT1.5 的 Android Demo，支持完全离线的翻译，并提供后台取词模式，可在其它应用中直接获取翻译。

下载地址：[Hy-MT-demo.apk](https://huggingface.co/AngelSlim/Hy-MT1.5-1.8B-1.25bit-GGUF/resolve/main/Hy-MT-demo.apk)

## 2. Arm SME2 设备

Hy-MT1.5 的 2-bit 版本面向支持 Arm SME2 指令集的移动设备，提供了 SME2 内核，并以 NEON 内核作为对照。内核的编译与切换方式请参考 [Hy-MT1.5 文档](../../../models/Hy-MT1.5/hy-mt1.5.md)。

---

[← 部署总览](./index.md) ｜ [← 返回总览](../index.md)
