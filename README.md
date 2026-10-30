# 灵流 API 节点（ComfyUI-Lingliu-API）

[English](#english)

灵流 API 节点是一个 ComfyUI 插件。装好后，在 ComfyUI 里直接调用灵流的生图、生视频、抠图、配音等模型，不用自己部署显卡和模型。

- 出品：温州灵流智能科技有限公司（灵流智能 / LINGLIU AI）
- 接口平台：灵流控制台 https://api.lingliuapi.com
- 价格：作图低至 0.1 元/张，视频 0.18 元/秒起，按实际用量从余额（流星 ✦）扣费。各模型的单价以控制台的模型价格页为准。

## 能做什么

| 类别 | 节点 |
|---|---|
| 设置与查询 | 灵流API设置、任务查询、模型列表 |
| 生图 / 改图 | GPT Image 2 / 2.5、Nano Banana / Banana 2、Seedream 5、Wan 2.7、Grok Imagine Image、Soul 2、JL 图像、MJ 全套（文生图、混图、局部重绘、换脸等） |
| 生视频 | Seedance 2.0 / 2.5（含低价、按次计费）、Wan 3.0（含低价）、Veo 3.1、MiniMax H3 首尾帧 / 参考、Grok Imagine Video、Omni 视频、JL 视频、MJ 图生视频 |
| 图像工具 | 图片抠图（透明底 PNG）、图像超分放大、批量图像输入 |
| 视频工具 | 动作模仿、视频转深度视频、视频超分放大、视频拼接 |
| 音频 | MiniMax 语音合成、语音克隆 |
| 文本与提示词 | 对话、多模态理解、视频提示词优化器、电商主图 / 详情提示词生成器、风格设定器、文本预设 |

所有节点都只走灵流线路，模型名和价格与灵流控制台一致。

## 安装

任选一种：

1. **ComfyUI Manager**：在 Manager 里搜索「Lingliu」，安装后重启 ComfyUI。
2. **git clone**：
   ```bash
   cd ComfyUI/custom_nodes
   git clone https://github.com/gutouy/ComfyUI-Lingliu-API.git
   pip install -r ComfyUI-Lingliu-API/requirements.txt
   ```
3. **zip**：在 GitHub 页面下载 zip，解压到 `ComfyUI/custom_nodes/`，再装依赖。

依赖：numpy、Pillow、requests、torch、av（ComfyUI 自带的环境一般已经有）。

## 第一次使用

1. 打开 https://api.lingliuapi.com 注册并登录，新建一个连接密钥，充值或兑换一些流星 ✦。
2. 在 ComfyUI 里添加「灵流API设置」节点，填入密钥，运行一次。插件不会把密钥写进本地配置文件；但节点输入框里的内容会随工作流一起保存，分享工作流前请先清空密钥。
3. 添加任意灵流节点，连好输入，运行。生成节点会真实调用平台并扣费，导入和查看工作流不会调用接口。

大多数生成节点也有自己的密钥输入框（API密钥 / api_key），填了就优先用节点上的密钥。

## 示例工作流

`examples/wan3_low/` 里有 Wan 3.0 低价视频的 3 个示例：单图参考、1 图 + 1 视频 + 1 音频、10 图 + 5 视频 + 5 音频。示例不含密钥，素材是本地合成的色块和正弦波，只用来检查连接。

## 常见问题

**提示「余额不足」或 403**：到控制台充值或兑换流星，或检查密钥是否已停用。

**提交后一直在生成**：把节点输出的任务 ID 填进「任务查询」节点查看。不要重复提交，重复提交会重复扣费。

**参考图上传失败**：如果开了代理或 VPN，请把 `r2.cloudflarestorage.com` 设为直连后重试。

**结果已生成但下载失败**：节点会输出结果 URL 和任务 ID，可以直接下载，或用任务 ID 重新取结果，不会重复扣费。

**旧工作流里的线路选项**：插件只有「灵流」一条线路，旧工作流里的其他线路选项会自动按灵流处理。

## 联系方式

- 公司：温州灵流智能科技有限公司
- 官网：https://ai.lingliuapi.com
- 控制台：https://api.lingliuapi.com
- 电话：138 6773 0625
- 地址：浙江省温州市瓯海区数安大厦 B 幢 2002 OPC 社区

问题和建议也可以在 GitHub 提 Issue。

## 许可

代码使用 MIT 许可，见 [LICENSE](LICENSE)。

`prompts/video_prompt_optimizer/minimax_h3/` 下的提示词写作文档来自 MiniMax-AI/MiniMax-H3，适用 MiniMax H3 Community License，见该目录的 LICENSE 和 NOTICE。

---

## English

Lingliu API Nodes is a ComfyUI custom node pack by LINGLIU AI (Wenzhou Lingliu Intelligent Technology Co., Ltd.). It calls the Lingliu platform for image generation and editing, video generation, background removal, motion imitation, upscaling, text-to-speech and prompt tools, so you do not need local GPUs or model weights.

### Install

- ComfyUI Manager: search for "Lingliu" and install, then restart ComfyUI.
- Manual: `git clone` this repository into `ComfyUI/custom_nodes/` and run `pip install -r requirements.txt`.

### Quick start

1. Sign up at https://api.lingliuapi.com, create an API key and top up your balance.
2. Add the 「灵流API设置」 (Lingliu API Settings) node, paste the key and run it once. The plugin does not write the key to its config file, but node inputs are saved with the workflow, so clear the key before sharing a workflow.
3. Add any Lingliu node and run it. Generation nodes call the platform and are billed per use; loading a workflow does not.

Pricing starts at ¥0.1 per image and ¥0.18 per second of video. See the model price page in the console for each model.

### License

The code is released under the MIT License. The MiniMax H3 prompt-writing documents in `prompts/video_prompt_optimizer/minimax_h3/` are covered by the MiniMax H3 Community License; see the LICENSE and NOTICE files in that folder.
