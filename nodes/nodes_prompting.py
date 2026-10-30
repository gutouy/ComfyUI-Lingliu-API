import asyncio
import base64
import json
import re

from ..utils import image_to_png_bytes, tensor2pil, video_to_data_url, video_to_remote_url
from .nodes_google import LingliuGoogleBase


DETAIL_SYSTEM_PROMPT = """[角色] 你是一名资深电商详情页设计师与 AI 文生图提示词工程师。
[任务] 根据产品信息生成电商详情页提示词。
[输出格式] JSON 字符串列表 (List[str])，每个元素包含该屏的完整提示词。
[语言规则]
- 总是遵循用户消息中的“输出语言要求”。
- 如果要求输出英文，则所有内容必须为英文（如果需要，翻译中文输入）。
- 如果要求输出中文，则所有内容必须为中文（如果需要，翻译英文输入）。
- 如果要求输出其他语种，则所有内容必须严格使用该语种（如有必要，自动翻译输入内容）。
- 永远不要混合使用语言。
[严格输出规则]
- 只允许输出 JSON 字符串列表 (List[str])，不要输出任何额外文字、解释、Markdown 或代码块。
- 列表长度必须严格等于 prompt_count。
- 每个元素对应一屏（一个字符串=一屏）。

[输出内容规则]
- 每个数组元素只输出该屏的“提示词正文”（纯文本字符串）。
- 不要在字符串里输出任何字段名或结构化包装（例如不要出现：prompt、consistency_id、花括号等）。

[参考图一致性规则 - 若用户提供产品参考图则必须遵守]
如果用户提供了 1 张或多张产品参考图（不同角度/局部），你必须：
- 把所有参考图视为“同一件产品”的不同角度/细节，不要生成不同款式的产品。
- 在所有屏幕中保持产品外观一致：结构比例、颜色与表面处理（哑光/高光/金属质感）、材质、LOGO 位置、按键/接口布局、纹理细节。
- 不要臆造参考图中不存在的部件、配件、文字标识。
- 参考图只用来约束“主体（产品/人物）外观”，不要把参考图的背景、场景、光影、道具当作必须复刻的内容。
- 必须倾向于“抠出主体/主体隔离”，然后为每一屏重建新的视觉场景（棚拍或氛围生活方式场景），以符合该屏的叙事目的。
- 即使参考图只有产品本体，也允许你生成“正在使用”的画面来提升吸引力，但必须保证产品外观严格一致、且不得新增参考图不存在的部件/配件/标识：
  - 吹风机：美女手持吹风机吹头发的画面
  - 衣服：模特穿上该衣服的上身展示
  - 沐浴露/洗面奶：近距离正在使用/挤压/涂抹的特写
- 如果参考图背景杂乱或有人手持：在画面描述中明确要求“抠出产品主体，移除手/杂物，重建为干净的棚拍场景”。
- 每一屏的提示词内容里必须加入一行“参考图说明”（使用输出语言要求的同一种语言，不能混用）：
  - 中文输出："参考图：以上传的产品参考图为严格外观依据，所有屏保持外观一致。"
  - 英文输出："Reference: Use the uploaded product image(s) as strict visual reference; keep appearance identical across screens."
  - 其他语种输出：将上述含义翻译成目标语种，且只能使用目标语种。
- 若存在多张参考图：按顺序指定用途并保持一致：
  - 图片1=主视觉标准（主角度/主色/主材质）；图片2/3=侧面/背面/局部细节补充。
  - 若存在多张参考图：每一屏还必须写清楚“本屏优先参考哪一张图”（使用输出语言要求的同一种语言，不能混用）：
    - 中文示例："优先参考：图片2（侧面角度）。"
    - 英文示例："Preferred Reference: Image 2 (side view)."
    - 其他语种输出：将上述含义翻译成目标语种，且只能使用目标语种。
  - 若只有 1 张参考图：禁止提及图片2/3。

[参考视频补充规则 - 若用户提供参考视频则必须遵守]
- 参考视频只作为动作、互动方式、镜头氛围、使用场景和主体动态细节的补充依据。
- 不要因为参考视频而改变产品主体的核心外观设定。
- 如果同时提供参考图与参考视频：参考图优先负责锁定主体外观，参考视频优先负责补充动作、情绪、镜头和场景感。

[防翻译指令]
- 中文：请确保画面中的全部文字只使用中文，不要翻译成英文，也不要混入任何英文字符。
- 英文：Keep all text in English only. Do not translate to Chinese. Do not mix any Chinese characters.
- 其他语种：请确保画面中的全部文字只使用目标语种，不要混入中文、英文或其他无关语种字符。

[重要指令]
请忽略上述 prompt 中的交互步骤，直接基于用户提供的产品信息，一次性生成指定数量的详情页提示词。
输出格式必须是 JSON 字符串列表 (List[str])，每个元素包含该屏的完整提示词（含主文案、副文案、画面描述等）。
不要输出 Markdown 表格，直接输出 JSON 列表。
每个屏的提示词必须包含以下 5 个模块：
（按输出语言要求二选一；如果是其他语种，则把对应模块标题翻译为目标语种并保持全程一致）

英文模板（仅当要求英文输出时使用）：
 1) Main Copy: "..."
 2) Sub Copy: "..."
 3) Layout & Typography: ...
 4) Visual & Composition: ...
 5) Quality & Details: ...

中文模板（仅当要求中文输出时使用）：
 1) 主文案: "..."
 2) 副文案: "..."
 3) 设计与排版: ...
 4) 画面主体与构图: ...
 5) 画质与细节: ...

其他语种模板：
 1) 将 “主文案 / Main Copy” 翻译为目标语种
 2) 将 “副文案 / Sub Copy” 翻译为目标语种
 3) 将 “设计与排版 / Layout & Typography” 翻译为目标语种
 4) 将 “画面主体与构图 / Visual & Composition” 翻译为目标语种
 5) 将 “画质与细节 / Quality & Details” 翻译为目标语种

[详情页分屏顺序 - 必须遵守]
你必须按以下“电商详情页叙事顺序”从上到下规划屏幕：
1) 首屏卖点总览（结果/痛点反转/技术背书 + 3-6 个 icon 卖点）
2) 痛点共鸣（3-5 条）
3) 解决方案一句话（你解决什么，怎么解决）
4) 核心机理（3步原理/结构剖面）
5) 功能卖点矩阵（6-10 个卡片）
6) 关键参数与性能证明（大数字KPI/图表）
7) 配件、多头、多模式适配
8) 使用步骤（3-5 步上手）
9) 应用场景或行业（4-8 个场景）
10) 对比模块（你 vs 传统/竞品）
11) 信任背书（认证、案例、口碑、服务）
12) 异议处理（FAQ 常见问题与风险消除）
13) 收口购买理由 + 售后承诺（CTA）

当 prompt_count < 13：合并相邻模块但不改变顺序；第1屏必须是首屏，第最后一屏必须是 CTA。
当 prompt_count > 13：优先把“卖点矩阵/参数证明/场景/背书”拆成更多屏，但顺序不变。

[每屏输出格式]
每屏（数组元素）必须是一段完整提示词正文字符串，其中必须包含：
- 该屏定位（例如："屏幕定位：首屏卖点总览"，使用用户要求的输出语言）。
- 严格按 5 模块顺序写内容。
- 文案短句化，保证生图可读；避免水印、避免乱码、避免多余随机字母。

[参考图在不同屏幕的用法建议]
当你写“画面主体与构图/Visual & Composition”时，优先按下列方式使用参考图角度：
- 首屏：优先使用图片1的主角度（常见为 45° 斜前方棚拍），突出产品主体。
- 机理/卖点/参数：使用能最好展示该卖点的角度做局部特写（按钮、接口、材质、结构），但不得改变产品设计。
- 使用步骤：如参考图本身有正确手持/使用姿势可复用；否则用干净的示意/图标步骤，不强行添加手。
- 对比/背书/FAQ/CTA：继续沿用首屏主角度或固定的一组角度，确保同款同色同质感。

[防翻译句放置位置 - 必须]
为保证 JSON 列表永远合法：防翻译句必须追加在每屏字符串内容末尾（在引号内）。

[模块标题语言要求]
- 如果要求输出英文：模块标题也必须使用英文标题（Main Copy/Sub Copy/...）。
- 如果要求输出中文：模块标题也必须使用中文标题（主文案/副文案/...）。
- 如果要求输出其他语种：模块标题也必须翻译为目标语种，且前后一致。

[备注]
- 防翻译句必须与输出语言一致：中文输出只允许中文；英文输出只允许英文；其他语种输出只允许目标语种。
"""


MAIN_IMAGE_SYSTEM_PROMPT = """[角色] 你是一名资深电商产品主图设计师与 AI 文生图提示词工程师。
[任务] 根据产品信息生成电商产品主图提示词。
[输出格式] JSON 字符串列表 (List[str])，每个元素包含一张产品主图的完整提示词。
[语言规则]
- 总是遵循用户消息中的“输出语言要求”。
- 如果要求输出英文，则所有内容必须为英文（如果需要，翻译中文输入）。
- 如果要求输出中文，则所有内容必须为中文（如果需要，翻译英文输入）。
- 如果要求输出其他语种，则所有内容必须严格使用该语种（如有必要，自动翻译输入内容）。
- 永远不要混合使用语言。

[严格输出规则]
- 只允许输出 JSON 字符串列表 (List[str])，不要输出任何额外文字、解释、Markdown 或代码块。
- 列表长度必须严格等于 prompt_count。
- 每个元素对应一张产品主图（一个字符串=一张主图）。

[输出内容规则]
- 每个数组元素只输出该张主图的“提示词正文”（纯文本字符串）。
- 不要在字符串里输出任何字段名或结构化包装（例如不要出现：prompt、consistency_id、花括号等）。
- 每张主图必须是可直接用于 AI 生图的完整提示词，包含产品主体、构图、背景、光影、材质、画质、文字控制等信息。

[产品主图核心规则]
- 主图必须突出单一产品主体，产品占画面主要视觉面积，构图清晰、商业化、适合电商平台。
- 产品必须清晰完整，不裁切关键部位，不遮挡核心卖点，不变形，不改变产品真实结构。
- 优先使用高质感棚拍、干净背景、真实光影、精细材质表现。
- 可根据产品类型生成白底主图、场景主图、氛围主图、卖点主图、细节特写主图，但不得做成长详情页、信息流海报或多屏排版。
- 避免复杂长文案，避免大量参数表、FAQ、步骤图、对比图、详情页式模块堆叠。
- 若需要画面文字，必须短句化、清晰可读；若无必要，明确要求“画面无文字”。
- 避免水印、乱码、随机字母、品牌误写、错误 LOGO、无关装饰文字。

[参考图一致性规则 - 若用户提供产品参考图则必须遵守]
如果用户提供了 1 张或多张产品参考图（不同角度/局部），你必须：
- 把所有参考图视为“同一件产品”的不同角度/细节，不要生成不同款式的产品。
- 在所有主图中保持产品外观一致：结构比例、颜色与表面处理（哑光/高光/金属质感）、材质、LOGO 位置、按键/接口布局、纹理细节。
- 不要臆造参考图中不存在的部件、配件、文字标识。
- 参考图只用来约束“主体产品外观”，不要把参考图的背景、场景、光影、道具当作必须复刻的内容。
- 必须倾向于“抠出主体/主体隔离”，然后为每张主图重建新的商业化视觉场景（白底棚拍、纯色背景、质感台面、生活方式场景等）。
- 即使参考图只有产品本体，也允许生成“正在使用”的主图来提升吸引力，但必须保证产品外观严格一致，且不得新增参考图不存在的部件、配件、标识：
  - 吹风机：美女手持吹风机吹头发的主图
  - 衣服：模特穿上该衣服的上身展示主图
  - 沐浴露/洗面奶：近距离正在使用/挤压/涂抹的特写主图
- 如果参考图背景杂乱或有人手持：在画面描述中明确要求“抠出产品主体，移除手/杂物，重建为干净的商业棚拍主图”。
- 每张主图的提示词内容里必须加入一行“参考图说明”（使用输出语言要求的同一种语言，不能混用）：
  - 中文输出："参考图：以上传的产品参考图为严格外观依据，所有主图保持外观一致。"
  - 英文输出："Reference: Use the uploaded product image(s) as strict visual reference; keep appearance identical across all main images."
  - 其他语种输出：将上述含义翻译成目标语种，且只能使用目标语种。
- 若存在多张参考图：按顺序指定用途并保持一致：
  - 图片1=主视觉标准（主角度/主色/主材质）；图片2/3=侧面/背面/局部细节补充。
  - 若存在多张参考图：每张主图还必须写清楚“本图优先参考哪一张图”（使用输出语言要求的同一种语言，不能混用）：
    - 中文示例："优先参考：图片1（主视觉角度）。"
    - 英文示例："Preferred Reference: Image 1 (main hero angle)."
    - 其他语种输出：将上述含义翻译成目标语种，且只能使用目标语种。
  - 若只有 1 张参考图：禁止提及图片2/3。

[参考视频补充规则 - 若用户提供参考视频则必须遵守]
- 参考视频只作为动作、互动方式、镜头氛围、使用场景和主体动态细节的补充依据。
- 不要因为参考视频而改变产品主体的核心外观设定。
- 如果同时提供参考图与参考视频：参考图优先负责锁定主体外观，参考视频优先负责补充动作、情绪、镜头和场景感。

[防翻译指令]
- 中文：请确保画面中的全部文字只使用中文，不要翻译成英文，也不要混入任何英文字符。
- 英文：Keep all text in English only. Do not translate to Chinese. Do not mix any Chinese characters.
- 其他语种：请确保画面中的全部文字只使用目标语种，不要混入中文、英文或其他无关语种字符。

[重要指令]
请忽略上述 prompt 中的交互步骤，直接基于用户提供的产品信息，一次性生成指定数量的产品主图提示词。
输出格式必须是 JSON 字符串列表 (List[str])，每个元素包含一张产品主图的完整提示词（含主图定位、画面文字、背景场景、产品构图、画质细节等）。
不要输出 Markdown 表格，直接输出 JSON 列表。
每张主图的提示词必须包含以下 5 个模块：
（按输出语言要求二选一；如果是其他语种，则把对应模块标题翻译为目标语种并保持全程一致）

英文模板（仅当要求英文输出时使用）：
 1) Main Image Goal: ...
 2) On-image Text: ...
 3) Background & Scene: ...
 4) Product & Composition: ...
 5) Quality & Details: ...

中文模板（仅当要求中文输出时使用）：
 1) 主图定位: ...
 2) 画面文字: ...
 3) 背景与场景: ...
 4) 产品主体与构图: ...
 5) 画质与细节: ...

其他语种模板：
 1) 将 “主图定位 / Main Image Goal” 翻译为目标语种
 2) 将 “画面文字 / On-image Text” 翻译为目标语种
 3) 将 “背景与场景 / Background & Scene” 翻译为目标语种
 4) 将 “产品主体与构图 / Product & Composition” 翻译为目标语种
 5) 将 “画质与细节 / Quality & Details” 翻译为目标语种

[主图生成顺序 - 必须遵守]
你必须根据 prompt_count 生成不同角度或不同风格的产品主图方案，优先按以下顺序规划：
1) 标准电商白底主图：产品完整清晰，主体突出，干净高级。
2) 高级棚拍主图：纯色或渐变背景，突出材质、质感和品牌调性。
3) 核心卖点主图：围绕最重要卖点做简洁视觉表达，可加入短文案或图形化元素。
4) 使用场景主图：展示产品真实使用状态或生活方式场景。
5) 细节特写主图：突出材质、结构、接口、纹理、工艺或关键功能部位。
6) 套装/配件主图：如产品包含配件、组合、套装，清晰展示完整内容。
7) 氛围视觉主图：强化情绪价值、礼品感、科技感、高端感或专业感。
8) CTA/促销感主图：如适用，可生成更有购买吸引力的商业主图，但避免详情页式堆字。

当 prompt_count < 8：按顺序选择最适合该产品的前若干类主图。
当 prompt_count > 8：优先围绕“白底主图、棚拍主图、卖点主图、使用场景、细节特写”生成更多角度或场景变体，但必须保证产品一致。

[每张主图输出格式]
每张主图（数组元素）必须是一段完整提示词正文字符串，其中必须包含：
- 该图定位（例如："主图定位：标准电商白底主图"，使用用户要求的输出语言）。
- 严格按 5 模块顺序写内容。
- 文案短句化，保证生图可读。
- 如果画面不需要文字，必须写明“画面文字：无文字，避免乱码、避免水印、避免随机字母。”
- 主图画面必须适合电商首图或商品橱窗展示，避免详情页长图、信息密集排版、多屏拼接。

[参考图在不同主图中的用法建议]
当你写“产品主体与构图/Product & Composition”时，优先按下列方式使用参考图角度：
- 白底主图：优先使用图片1的主角度，常见为正面或 45° 斜前方，完整展示产品。
- 棚拍主图：沿用图片1主角度，加入更精致的光影、台面、反射或背景层次。
- 卖点主图：使用最能展示该卖点的角度做局部强化，但不得改变产品设计。
- 使用场景主图：如参考图本身有正确手持/使用姿势可复用；否则根据产品类型生成自然使用场景。
- 细节特写主图：优先参考图片2/3中的侧面、背面、局部细节。
- 套装/配件主图：只展示用户信息或参考图中明确存在的配件，不得臆造。
- 氛围主图和 CTA 主图：继续沿用首图主角度或固定的一组角度，确保同款同色同质感。

[构图与平台适配]
- 默认主图比例为 1:1 方图，适合电商平台商品主图。
- 如用户明确指定比例，则严格遵守用户指定比例。
- 产品主体居中或略偏中心，视觉重心稳定。
- 产品占画面约 70%-85%，留出适当呼吸空间。
- 背景干净，不抢主体，不使用复杂杂乱道具。
- 避免过度夸张透视、畸变、错误阴影、悬浮不自然、比例失真。

[画质要求]
- 商业摄影级画质，高清锐利，真实材质，真实光影。
- 产品边缘干净，细节清晰，颜色准确。
- 不要低清、噪点、过曝、脏污、变形、重复产品、错位结构。
- 不要生成多余手指、畸形人物、错误反射、错误文字、错误 LOGO。
- 不要出现水印、二维码、平台角标、无关品牌标识。

[防翻译句放置位置 - 必须]
为保证 JSON 列表永远合法：防翻译句必须追加在每张主图字符串内容末尾（在引号内）。

[模块标题语言要求]
- 如果要求输出英文：模块标题也必须使用英文标题（Main Image Goal/On-image Text/...）。
- 如果要求输出中文：模块标题也必须使用中文标题（主图定位/画面文字/...）。
- 如果要求输出其他语种：模块标题也必须翻译为目标语种，且前后一致。

[备注]
- 防翻译句必须与输出语言一致：中文输出只允许中文；英文输出只允许英文；其他语种输出只允许目标语种。
"""


STYLE_SETTER_SYSTEM_PROMPT = """[角色] 你是一名资深电商视觉总监、品牌风格顾问与 AI 视觉提示词工程师。
[任务] 根据用户手动输入的信息，以及可选的产品参考图/参考视频，输出一组适合该产品的统一风格要求与统一主色系。

[输出目标]
你要输出可直接用于 AI 生图工作流的两段短文本：
1. 统一风格要求
2. 统一主色系

[输出格式]
只允许输出一个 JSON 对象，不要输出任何额外文字、解释、Markdown 或代码块。
JSON 结构必须是：
{
  "style_consistency": "...",
  "color_consistency": "..."
}

[语言规则]
- 总是遵循用户消息中的“输出语言要求”。
- 如果要求输出中文，则两个字段都只用中文。
- 如果要求输出英文，则两个字段都只用英文。
- 如果要求输出其他语种，则两个字段都只用目标语种。
- 永远不要混合使用语言。

[分析规则]
- 如果用户已经手动提供了“统一风格要求”，你必须尊重它，把它作为最高优先级约束。
- 如果用户已经手动提供了“统一主色系”，你必须尊重它，把它作为最高优先级约束。
- 如果其中一个字段为空，你需要结合产品图、产品视频、商品类型、核心卖点和已给出的另一个字段，补全缺失字段。
- 如果两个字段都为空，而用户提供了产品图或视频，你需要自动分析出最适合产品的统一风格要求与统一主色系。
- 如果图像/视频里主体已经有明显材质、颜色、定位和消费人群特征，必须优先从这些视觉信号出发，不要乱猜。
- 风格要求要偏“可执行”，适合直接作为后续主图、详情页提示词的统一风格锚点。
- 色系要求要偏“可执行”，清楚描述主色、辅助色、色温或整体色彩倾向。

[风格输出要求]
- “style_consistency” 应是 1~3 句可直接用于提示词的风格描述。
- 重点描述：视觉气质、商业调性、光影倾向、材质表达、版式气质、适合的背景处理方式。
- 不要写空泛形容词堆砌，要让后续模型知道该怎么统一整套图。

[色系输出要求]
- “color_consistency” 应是 1~2 句可直接用于提示词的色系描述。
- 重点描述：主色、辅助色、明暗关系、冷暖倾向、整体色温。
- 如果产品本身颜色非常明确，应以产品本身为核心扩展色系，而不是强行换色。

[电商适配要求]
- 输出内容必须有利于电商主图和详情页同时共用。
- 输出结果必须让“主图”和“详情页”在同一工作流里更容易保持同风格、同色系、同质感。
- 避免给出彼此冲突的风格或色彩建议。
"""


DESIGN_STYLES = [
    "简约 Ins 风",
    "高级奢华",
    "科技感",
    "清新自然",
    "国潮风",
    "活泼撞色",
    "极简工业风",
    "梦幻唯美",
    "亚马逊风格",
]

DETAIL_SCENE_PREFERENCES = [
    "混合（以使用场景为主）",
    "生活方式交互场景（人物/手部互动）",
    "棚拍干净背景（不复刻参考图背景）",
]

MAIN_IMAGE_SCENE_PREFERENCES = [
    "标准白底主图",
    "高级棚拍主图",
    "场景氛围主图",
    "混合（自动规划）",
]

MAIN_IMAGE_ASPECT_RATIOS = [
    "1:1",
    "4:5",
    "3:4",
    "16:9",
    "9:16",
    "自定义",
]


class LingliuPromptBase(LingliuGoogleBase):
    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("prompts_list", "debug_info")
    OUTPUT_IS_LIST = (True, False)
    FUNCTION = "generate_prompts"
    CATEGORY = "Lingliu/Prompting"

    def _count_image_batch(self, image):
        if image is None:
            return 0
        try:
            return len(tensor2pil(image))
        except Exception:
            return 0

    def _build_multimodal_content(self, text, image=None, video=None):
        content = [{"type": "text", "text": text}]

        if image is not None:
            for pil_image in tensor2pil(image):
                encoded = base64.b64encode(image_to_png_bytes(pil_image)).decode("utf-8")
                content.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{encoded}"},
                    }
                )

        if video is not None:
            remote_url = video_to_remote_url(video)
            if remote_url:
                content.append({"type": "image_url", "image_url": {"url": remote_url}})
            else:
                data_url = video_to_data_url(video)
                if data_url:
                    content.append({"type": "image_url", "image_url": {"url": data_url}})

        return content

    def _clean_code_fences(self, text):
        cleaned = str(text or "").strip()
        if cleaned.startswith("```json"):
            cleaned = cleaned[7:]
        elif cleaned.startswith("```"):
            cleaned = cleaned[3:]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
        return cleaned.strip()

    def _extract_prompt_text(self, item):
        if item is None:
            return None
        if isinstance(item, str):
            return item.strip()
        if isinstance(item, dict):
            for key in ("prompt", "text", "content"):
                value = item.get(key)
                if isinstance(value, str) and value.strip():
                    return value.strip()
            return json.dumps(item, ensure_ascii=False)
        return str(item).strip()

    def _split_response_to_screens(self, text, prompt_count):
        raw = str(text or "").replace("\r\n", "\n").replace("\r", "\n").strip()
        if not raw:
            return []
        if prompt_count <= 1:
            return [raw]

        numbered = re.split(r"(?m)^\s*(?:screen\s*\d+|panel\s*\d+|page\s*\d+|image\s*\d+|\d+[.)、．])\s*", raw, flags=re.IGNORECASE)
        numbered = [item.strip() for item in numbered if item and item.strip()]
        if len(numbered) >= 2:
            return numbered

        separated = [item.strip() for item in raw.split("\n---\n") if item.strip()]
        if len(separated) >= 2:
            return separated

        sections = [item.strip() for item in re.split(r"\n\s*\n\s*\n+", raw) if item.strip()]
        if len(sections) >= 2:
            return sections

        return [raw]

    def _enforce_prompt_count(self, prompts_list, prompt_count, fallback_text):
        prompts = [item for item in prompts_list if isinstance(item, str) and item.strip()]
        if len(prompts) == prompt_count:
            return prompts
        if len(prompts) > prompt_count:
            return prompts[:prompt_count]
        if len(prompts) == 1:
            parts = self._split_response_to_screens(prompts[0], prompt_count)
            if len(parts) >= prompt_count:
                return parts[:prompt_count]
        if len(prompts) < prompt_count:
            parts = self._split_response_to_screens(fallback_text, prompt_count)
            if len(parts) >= prompt_count:
                return parts[:prompt_count]
        while len(prompts) < prompt_count and prompts:
            prompts.append(prompts[-1])
        return prompts

    def _parse_response_to_prompts_list(self, response_text, prompt_count):
        method = "json"
        cleaned = self._clean_code_fences(response_text)
        try:
            parsed = json.loads(cleaned)
            if not isinstance(parsed, list):
                method = "split:not_list"
                parsed = self._split_response_to_screens(response_text, prompt_count)
        except Exception:
            method = "split:fallback"
            parsed = self._split_response_to_screens(response_text, prompt_count)

        normalized = []
        for item in parsed if isinstance(parsed, list) else [parsed]:
            text = self._extract_prompt_text(item)
            if text:
                normalized.append(text)

        normalized = self._enforce_prompt_count(normalized, prompt_count, response_text)
        if not normalized:
            normalized = [str(response_text or "").strip() or "[GENERATION_FAILED] Empty model response."]

        if len(normalized) < prompt_count:
            normalized.extend([normalized[-1]] * (prompt_count - len(normalized)))
        elif len(normalized) > prompt_count:
            normalized = normalized[:prompt_count]
        return normalized, method

    async def _stream_text_response(self, config, payload):
        return await asyncio.to_thread(self._stream_text_response_sync, config, payload)

    def _stream_text_response_sync(self, config, payload):
        import requests

        full_response = ""
        response = requests.post(
            self.endpoint(config, "chat_path"),
            headers=self.auth_headers(config, json_mode=True),
            json=payload,
            stream=True,
            timeout=self.timeout(config),
        )
        response.raise_for_status()
        for line in response.iter_lines():
            if not line:
                continue
            line_text = line.decode("utf-8", errors="ignore").strip()
            if not line_text.startswith("data: "):
                continue
            data = line_text[6:]
            if data == "[DONE]":
                break
            try:
                chunk = json.loads(data)
            except json.JSONDecodeError:
                continue
            choices = chunk.get("choices", [])
            if not choices:
                continue
            delta = choices[0].get("delta", {})
            if isinstance(delta.get("content"), str):
                full_response += delta["content"]
        return full_response

    def _finalize_system_prompt(self, custom_system_prompt, default_prompt, output_language, prompt_count):
        effective_system_prompt = custom_system_prompt.strip() or default_prompt
        effective_system_prompt += f"\n\n[节点运行参数]\n- 输出语言要求: {output_language}\n- prompt_count: {int(prompt_count)}"
        return effective_system_prompt

    async def _run_prompt_generation(
        self,
        *,
        model,
        image,
        video,
        output_language,
        prompt_count,
        custom_system_prompt,
        default_system_prompt,
        user_instruction,
        temperature,
        top_p,
        max_tokens,
        seed,
        stream,
        debug_payload,
    ):
        config = self.load_chat_config()
        if not (config.get("api_key") or "").strip():
            return (
                ["[GENERATION_FAILED] 请先使用 lingliu_api_set 设置全局 API key。"],
                "Global API key is missing.",
            )

        content = self._build_multimodal_content(user_instruction, image=image, video=video)
        effective_system_prompt = self._finalize_system_prompt(
            custom_system_prompt,
            default_system_prompt,
            output_language,
            prompt_count,
        )

        messages = [{"role": "system", "content": effective_system_prompt}, {"role": "user", "content": content}]

        payload = {
            "model": model,
            "messages": messages,
            "temperature": float(temperature),
            "top_p": float(top_p),
            "max_tokens": int(max_tokens),
            "stream": bool(stream),
        }
        if int(seed) > 0:
            payload["seed"] = int(seed)

        try:
            if stream:
                response_text = await self._stream_text_response(config, payload)
            else:
                response_payload = await self.async_request_json(
                    "POST",
                    self.endpoint(config, "chat_path"),
                    config,
                    headers=self.auth_headers(config, json_mode=True),
                    json=payload,
                )
                response_text = response_payload["choices"][0]["message"]["content"]
                debug_payload["response_payload"] = response_payload

            prompts_list, method = self._parse_response_to_prompts_list(response_text, int(prompt_count))
            debug_payload["parse_method"] = method
            debug_payload["raw_response"] = response_text
            debug_payload["returned_count"] = len(prompts_list)
            return prompts_list, json.dumps(debug_payload, ensure_ascii=False, indent=2)
        except Exception as exc:
            debug_payload["error"] = str(exc)
            return [f"[GENERATION_FAILED] {exc}"], json.dumps(debug_payload, ensure_ascii=False, indent=2)

    def _parse_json_object_response(self, response_text):
        cleaned = self._clean_code_fences(response_text)
        try:
            payload = json.loads(cleaned)
            if isinstance(payload, dict):
                return payload, "json"
        except Exception:
            pass

        style_match = re.search(r'"?style_consistency"?\s*[:：]\s*"?(.*?)"?(?:,|\n|$)', cleaned, flags=re.IGNORECASE | re.DOTALL)
        color_match = re.search(r'"?color_consistency"?\s*[:：]\s*"?(.*?)"?(?:,|\n|$)', cleaned, flags=re.IGNORECASE | re.DOTALL)

        parsed = {}
        if style_match:
            parsed["style_consistency"] = style_match.group(1).strip().strip('",}')
        if color_match:
            parsed["color_consistency"] = color_match.group(1).strip().strip('",}')
        return parsed, "regex"


class LingliuEcommercePromptGenerator(LingliuPromptBase):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "模型名": ("STRING", {"default": "gemini-2.5-pro"}),
                "商品类型": ("STRING", {"default": "商品"}),
                "核心卖点": ("STRING", {"multiline": True, "default": "请填写商品核心卖点"}),
                "设计风格": (DESIGN_STYLES, {"default": "简约 Ins 风"}),
                "场景偏好": (DETAIL_SCENE_PREFERENCES, {"default": "混合（以使用场景为主）"}),
                "输出语种": ("STRING", {"default": "中文"}),
                "输出屏数": ("INT", {"default": 6, "min": 1, "max": 20}),
            },
            "optional": {
                "参考图片": ("IMAGE",),
                "参考视频": ("VIDEO",),
                "统一风格要求": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "",
                        "placeholder": "可选：例如 奶油白+浅金色，高级轻奢，暖调柔光，整套图保持统一风格与色系",
                    },
                ),
                "统一主色系": ("STRING", {"default": "", "placeholder": "可选：例如 奶油白、莫兰迪粉、黑金、蓝银"}),
                "补充要求": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "",
                        "placeholder": "可选：补充你希望模型遵守的额外要求",
                    },
                ),
                "温度": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 2.0, "step": 0.01}),
                "采样范围": ("FLOAT", {"default": 0.95, "min": 0.0, "max": 1.0, "step": 0.01}),
                "最大输出长度": ("INT", {"default": 8192, "min": 1, "max": 32768}),
                "随机种子": ("INT", {"default": 0, "min": 0, "max": 2147483647, "control_after_generate": True}),
                "自定义系统提示词": ("STRING", {"multiline": True, "default": ""}),
                "流式输出": ("BOOLEAN", {"default": False}),
            },
        }

    def _scene_instruction(self, scene_preference):
        if scene_preference == "生活方式交互场景（人物/手部互动）":
            return "优先使用人物或手部互动的生活方式场景，不复刻参考图原背景。"
        if scene_preference == "棚拍干净背景（不复刻参考图背景）":
            return "优先使用干净棚拍、渐变背景、商业化展示构图，不复刻参考图原背景。"
        return "以使用场景为主，辅以少量干净棚拍场景，整体不复刻参考图原背景。"

    def _build_user_instruction(
        self,
        商品类型,
        核心卖点,
        设计风格,
        场景偏好,
        输出语种,
        输出屏数,
        参考图片数量,
        有参考视频,
        统一风格要求,
        统一主色系,
        补充要求,
    ):
        lines = [
            "[节点参数]",
            f"- 商品类型: {商品类型}",
            f"- 核心卖点: {核心卖点}",
            f"- 设计风格: {设计风格}",
            f"- 场景偏好: {场景偏好}",
            f"- 输出语言要求: {输出语种}",
            f"- prompt_count: {int(输出屏数)}",
            f"- 场景补充说明: {self._scene_instruction(场景偏好)}",
        ]
        if 参考图片数量 > 0:
            lines.extend(
                [
                    f"- 参考图片数量: {参考图片数量}",
                    "- 已提供参考图片，请严格遵守参考图一致性规则。",
                ]
            )
        if 有参考视频:
            lines.append("- 已提供参考视频，请严格遵守参考视频补充规则。")
        if str(统一风格要求 or "").strip():
            lines.extend(
                [
                    f"- 统一风格要求: {统一风格要求.strip()}",
                    "- 所有输出屏幕必须保持同一视觉语言、同一审美方向、同一材质表达与同一版式气质。",
                ]
            )
        if str(统一主色系 or "").strip():
            lines.extend(
                [
                    f"- 统一主色系: {统一主色系.strip()}",
                    "- 所有输出屏幕必须保持主色、辅助色、明暗倾向和整体色温一致，不要每一屏随意换色。",
                ]
            )
        if str(补充要求 or "").strip():
            lines.append(f"- 补充要求: {补充要求.strip()}")
        return "\n".join(lines)

    async def generate_prompts(
        self,
        模型名,
        商品类型,
        核心卖点,
        设计风格,
        场景偏好,
        输出语种,
        输出屏数,
        参考图片=None,
        参考视频=None,
        统一风格要求="",
        统一主色系="",
        补充要求="",
        温度=0.7,
        采样范围=0.95,
        最大输出长度=8192,
        随机种子=0,
        自定义系统提示词="",
        流式输出=False,
    ):
        参考图片数量 = self._count_image_batch(参考图片)
        用户指令 = self._build_user_instruction(
            商品类型=商品类型,
            核心卖点=核心卖点,
            设计风格=设计风格,
            场景偏好=场景偏好,
            输出语种=输出语种,
            输出屏数=输出屏数,
            参考图片数量=参考图片数量,
            有参考视频=bool(参考视频 is not None),
            统一风格要求=统一风格要求,
            统一主色系=统一主色系,
            补充要求=补充要求,
        )
        调试信息 = {
            "node_type": "detail_prompt",
            "model": 模型名,
            "product_type": 商品类型,
            "prompt_count": int(输出屏数),
            "reference_image_count": 参考图片数量,
            "has_video": bool(参考视频 is not None),
            "output_language": 输出语种,
            "style_consistency": 统一风格要求,
            "color_consistency": 统一主色系,
            "user_instruction": 用户指令,
        }
        return await self._run_prompt_generation(
            model=模型名,
            image=参考图片,
            video=参考视频,
            output_language=输出语种,
            prompt_count=输出屏数,
            custom_system_prompt=自定义系统提示词,
            default_system_prompt=DETAIL_SYSTEM_PROMPT,
            user_instruction=用户指令,
            temperature=温度,
            top_p=采样范围,
            max_tokens=最大输出长度,
            seed=随机种子,
            stream=流式输出,
            debug_payload=调试信息,
        )


class LingliuMainImagePromptGenerator(LingliuPromptBase):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "模型名": ("STRING", {"default": "gemini-2.5-pro"}),
                "商品类型": ("STRING", {"default": "商品"}),
                "核心卖点": ("STRING", {"multiline": True, "default": "请填写商品核心卖点"}),
                "主图风格": (DESIGN_STYLES, {"default": "简约 Ins 风"}),
                "主图场景": (MAIN_IMAGE_SCENE_PREFERENCES, {"default": "混合（自动规划）"}),
                "输出语种": ("STRING", {"default": "中文"}),
                "输出张数": ("INT", {"default": 4, "min": 1, "max": 20}),
            },
            "optional": {
                "参考图片": ("IMAGE",),
                "参考视频": ("VIDEO",),
                "画面比例": (MAIN_IMAGE_ASPECT_RATIOS, {"default": "1:1"}),
                "自定义比例": ("STRING", {"default": "", "placeholder": "例如 3:2，仅在画面比例选择自定义时使用"}),
                "统一风格要求": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "",
                        "placeholder": "可选：例如 与详情页保持同一高级暖调奶油白轻奢风",
                    },
                ),
                "统一主色系": ("STRING", {"default": "", "placeholder": "可选：例如 奶油白+浅金色"}),
                "补充要求": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "",
                        "placeholder": "可选：补充你希望模型遵守的额外要求",
                    },
                ),
                "温度": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 2.0, "step": 0.01}),
                "采样范围": ("FLOAT", {"default": 0.95, "min": 0.0, "max": 1.0, "step": 0.01}),
                "最大输出长度": ("INT", {"default": 8192, "min": 1, "max": 32768}),
                "随机种子": ("INT", {"default": 0, "min": 0, "max": 2147483647, "control_after_generate": True}),
                "自定义系统提示词": ("STRING", {"multiline": True, "default": ""}),
                "流式输出": ("BOOLEAN", {"default": False}),
            },
        }

    def _scene_instruction(self, 主图场景):
        if 主图场景 == "标准白底主图":
            return "优先生成标准白底电商主图，主体完整清晰，背景极简干净。"
        if 主图场景 == "高级棚拍主图":
            return "优先生成高级棚拍主图，强调材质、光影、反射和品牌质感。"
        if 主图场景 == "场景氛围主图":
            return "优先生成带情绪感或使用感的场景氛围主图，但主体仍然是唯一视觉焦点。"
        return "自动在白底主图、棚拍主图、卖点主图、场景主图之间规划组合。"

    def _normalized_ratio(self, 画面比例, 自定义比例):
        if 画面比例 != "自定义":
            return 画面比例
        custom_ratio = str(自定义比例 or "").strip()
        return custom_ratio or "1:1"

    def _build_user_instruction(
        self,
        商品类型,
        核心卖点,
        主图风格,
        主图场景,
        输出语种,
        输出张数,
        画面比例,
        参考图片数量,
        有参考视频,
        统一风格要求,
        统一主色系,
        补充要求,
    ):
        lines = [
            "[节点参数]",
            f"- 商品类型: {商品类型}",
            f"- 核心卖点: {核心卖点}",
            f"- 主图风格: {主图风格}",
            f"- 主图场景: {主图场景}",
            f"- 输出语言要求: {输出语种}",
            f"- prompt_count: {int(输出张数)}",
            f"- 画面比例: {画面比例}",
            f"- 场景补充说明: {self._scene_instruction(主图场景)}",
        ]
        if 参考图片数量 > 0:
            lines.extend(
                [
                    f"- 参考图片数量: {参考图片数量}",
                    "- 已提供参考图片，请严格遵守参考图一致性规则。",
                ]
            )
        if 有参考视频:
            lines.append("- 已提供参考视频，请严格遵守参考视频补充规则。")
        if str(统一风格要求 or "").strip():
            lines.extend(
                [
                    f"- 统一风格要求: {统一风格要求.strip()}",
                    "- 所有主图必须保持同一商业视觉语言、同一高级感来源、同一光影风格与同一材质表现逻辑。",
                ]
            )
        if str(统一主色系 or "").strip():
            lines.extend(
                [
                    f"- 统一主色系: {统一主色系.strip()}",
                    "- 所有主图必须保持统一主色、辅助色、背景色倾向、色温与对比关系，不要在不同主图之间大幅跳色。",
                ]
            )
        if str(补充要求 or "").strip():
            lines.append(f"- 补充要求: {补充要求.strip()}")
        return "\n".join(lines)

    async def generate_prompts(
        self,
        模型名,
        商品类型,
        核心卖点,
        主图风格,
        主图场景,
        输出语种,
        输出张数,
        参考图片=None,
        参考视频=None,
        画面比例="1:1",
        自定义比例="",
        统一风格要求="",
        统一主色系="",
        补充要求="",
        温度=0.7,
        采样范围=0.95,
        最大输出长度=8192,
        随机种子=0,
        自定义系统提示词="",
        流式输出=False,
    ):
        实际比例 = self._normalized_ratio(画面比例, 自定义比例)
        参考图片数量 = self._count_image_batch(参考图片)
        用户指令 = self._build_user_instruction(
            商品类型=商品类型,
            核心卖点=核心卖点,
            主图风格=主图风格,
            主图场景=主图场景,
            输出语种=输出语种,
            输出张数=输出张数,
            画面比例=实际比例,
            参考图片数量=参考图片数量,
            有参考视频=bool(参考视频 is not None),
            统一风格要求=统一风格要求,
            统一主色系=统一主色系,
            补充要求=补充要求,
        )
        调试信息 = {
            "node_type": "main_image_prompt",
            "model": 模型名,
            "product_type": 商品类型,
            "prompt_count": int(输出张数),
            "aspect_ratio": 实际比例,
            "reference_image_count": 参考图片数量,
            "has_video": bool(参考视频 is not None),
            "output_language": 输出语种,
            "style_consistency": 统一风格要求,
            "color_consistency": 统一主色系,
            "user_instruction": 用户指令,
        }
        return await self._run_prompt_generation(
            model=模型名,
            image=参考图片,
            video=参考视频,
            output_language=输出语种,
            prompt_count=输出张数,
            custom_system_prompt=自定义系统提示词,
            default_system_prompt=MAIN_IMAGE_SYSTEM_PROMPT,
            user_instruction=用户指令,
            temperature=温度,
            top_p=采样范围,
            max_tokens=最大输出长度,
            seed=随机种子,
            stream=流式输出,
            debug_payload=调试信息,
        )


class LingliuStyleSetter(LingliuPromptBase):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "模型名": ("STRING", {"default": "gemini-2.5-pro"}),
                "输出语种": ("STRING", {"default": "中文"}),
            },
            "optional": {
                "商品类型": ("STRING", {"default": "商品"}),
                "核心卖点": ("STRING", {"multiline": True, "default": ""}),
                "参考图片": ("IMAGE",),
                "参考视频": ("VIDEO",),
                "手动统一风格要求": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "",
                        "placeholder": "可选：如果你已经有明确风格要求，填在这里",
                    },
                ),
                "手动统一主色系": (
                    "STRING",
                    {
                        "default": "",
                        "placeholder": "可选：如果你已经有明确色系要求，填在这里",
                    },
                ),
                "补充要求": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "",
                        "placeholder": "可选：补充你希望模型分析时考虑的额外要求",
                    },
                ),
                "温度": ("FLOAT", {"default": 0.4, "min": 0.0, "max": 2.0, "step": 0.01}),
                "采样范围": ("FLOAT", {"default": 0.95, "min": 0.0, "max": 1.0, "step": 0.01}),
                "最大输出长度": ("INT", {"default": 2048, "min": 1, "max": 8192}),
                "随机种子": ("INT", {"default": 0, "min": 0, "max": 2147483647, "control_after_generate": True}),
                "自定义系统提示词": ("STRING", {"multiline": True, "default": ""}),
                "流式输出": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("统一风格要求", "统一主色系", "debug_info")
    OUTPUT_IS_LIST = (False, False, False)
    FUNCTION = "generate_style"
    CATEGORY = "Lingliu/Prompting"

    def _build_user_instruction(
        self,
        商品类型,
        核心卖点,
        输出语种,
        参考图片数量,
        有参考视频,
        手动统一风格要求,
        手动统一主色系,
        补充要求,
    ):
        lines = [
            "[节点参数]",
            f"- 商品类型: {商品类型}",
            f"- 核心卖点: {核心卖点}",
            f"- 输出语言要求: {输出语种}",
            f"- 参考图片数量: {参考图片数量}",
            f"- 是否提供参考视频: {'是' if 有参考视频 else '否'}",
            f"- 用户手动统一风格要求: {手动统一风格要求 or '未填写'}",
            f"- 用户手动统一主色系: {手动统一主色系 or '未填写'}",
        ]
        if str(补充要求 or "").strip():
            lines.append(f"- 补充要求: {补充要求.strip()}")
        lines.extend(
            [
                "",
                "[执行要求]",
                "- 若用户手动填写了某项，则该项优先保留。",
                "- 若某项未填写，则结合商品信息与参考图/参考视频自动补全。",
                "- 输出必须是 JSON 对象，且包含 style_consistency 和 color_consistency 两个字段。",
            ]
        )
        return "\n".join(lines)

    async def generate_style(
        self,
        模型名,
        输出语种,
        商品类型="商品",
        核心卖点="",
        参考图片=None,
        参考视频=None,
        手动统一风格要求="",
        手动统一主色系="",
        补充要求="",
        温度=0.4,
        采样范围=0.95,
        最大输出长度=2048,
        随机种子=0,
        自定义系统提示词="",
        流式输出=False,
    ):
        手动统一风格要求 = str(手动统一风格要求 or "").strip()
        手动统一主色系 = str(手动统一主色系 or "").strip()
        参考图片数量 = self._count_image_batch(参考图片)
        有参考视频 = bool(参考视频 is not None)

        if 手动统一风格要求 and 手动统一主色系:
            debug_payload = {
                "mode": "manual_both",
                "model": 模型名,
                "output_language": 输出语种,
                "reference_image_count": 参考图片数量,
                "has_video": 有参考视频,
            }
            return (
                手动统一风格要求,
                手动统一主色系,
                json.dumps(debug_payload, ensure_ascii=False, indent=2),
            )

        if not 手动统一风格要求 and not 手动统一主色系 and 参考图片 is None and 参考视频 is None:
            return (
                "",
                "",
                json.dumps(
                    {
                        "error": "未提供手动风格/色系，也未提供参考图片或参考视频，无法自动分析。",
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
            )

        用户指令 = self._build_user_instruction(
            商品类型=商品类型,
            核心卖点=核心卖点,
            输出语种=输出语种,
            参考图片数量=参考图片数量,
            有参考视频=有参考视频,
            手动统一风格要求=手动统一风格要求,
            手动统一主色系=手动统一主色系,
            补充要求=补充要求,
        )

        调试信息 = {
            "mode": "analyze_or_fill_missing",
            "model": 模型名,
            "product_type": 商品类型,
            "output_language": 输出语种,
            "reference_image_count": 参考图片数量,
            "has_video": 有参考视频,
            "manual_style": 手动统一风格要求,
            "manual_color": 手动统一主色系,
            "user_instruction": 用户指令,
        }

        config = self.load_chat_config()
        if not (config.get("api_key") or "").strip():
            return (
                "",
                "",
                json.dumps(
                    {"error": "请先使用 lingliu_api_set 设置全局 API key。"},
                    ensure_ascii=False,
                    indent=2,
                ),
            )

        content = self._build_multimodal_content(用户指令, image=参考图片, video=参考视频)
        effective_system_prompt = self._finalize_system_prompt(
            自定义系统提示词,
            STYLE_SETTER_SYSTEM_PROMPT,
            输出语种,
            1,
        )
        messages = [{"role": "system", "content": effective_system_prompt}, {"role": "user", "content": content}]
        payload = {
            "model": 模型名,
            "messages": messages,
            "temperature": float(温度),
            "top_p": float(采样范围),
            "max_tokens": int(最大输出长度),
            "stream": bool(流式输出),
        }
        if int(随机种子) > 0:
            payload["seed"] = int(随机种子)

        try:
            if 流式输出:
                response_text = await self._stream_text_response(config, payload)
            else:
                response_payload = await self.async_request_json(
                    "POST",
                    self.endpoint(config, "chat_path"),
                    config,
                    headers=self.auth_headers(config, json_mode=True),
                    json=payload,
                )
                response_text = response_payload["choices"][0]["message"]["content"]
                调试信息["response_payload"] = response_payload

            parsed, parse_method = self._parse_json_object_response(response_text)
            style_value = 手动统一风格要求 or str(parsed.get("style_consistency") or "").strip()
            color_value = 手动统一主色系 or str(parsed.get("color_consistency") or "").strip()

            调试信息["parse_method"] = parse_method
            调试信息["raw_response"] = response_text
            调试信息["style_consistency"] = style_value
            调试信息["color_consistency"] = color_value

            return (
                style_value,
                color_value,
                json.dumps(调试信息, ensure_ascii=False, indent=2),
            )
        except Exception as exc:
            调试信息["error"] = str(exc)
            return (
                手动统一风格要求,
                手动统一主色系,
                json.dumps(调试信息, ensure_ascii=False, indent=2),
            )
