from .nodes.nodes_openai_compatible import (
    LingliuApiSet,
    LingliuChat,
    LingliuImageEdit,
    LingliuImageGenerate,
    LingliuModels,
)
from .nodes.nodes_task_query import LingliuTaskQuery
from .nodes.nodes_google import (
    LingliuGeminiTextOnly,
    LingliuGoogleVeo3,
    LingliuNanoBanana,
    LingliuNanoBanana2Edit,
    LingliuNanoBananaEdit,
)
from .nodes.nodes_doubao import (
    LingliuDoubaoSeedance2,
    LingliuSeedance2AssetBundle,
    LingliuSeedance2AssetUpload,
)
from .nodes.nodes_seedance_official import LingliuR2StorageSet, LingliuSeedance2Official
from .nodes.nodes_seedance25 import LingliuSeedance25, LingliuSeedance25Official
from .nodes.nodes_video_prompt_optimizer import LingliuVideoPromptOptimizer
from .nodes.nodes_seedance_low import LingliuSeedanceLow
from .nodes.nodes_topaz_fal import LingliuTopazImageUpscale, LingliuTopazVideoUpscale
from .nodes.nodes_grok import LingliuGrokImage, LingliuGrokVideo
from .nodes.nodes_grok_imagine_image_2 import LingliuGrokImagineImage2
from .nodes.nodes_grok_imagine_video_1_5 import LingliuGrokImagineVideo15
from .nodes.nodes_grok_imagine_video_1_5_openai import LingliuGrokImagineVideo15OpenAI
from .nodes.nodes_omni import LingliuOmniVideo
from .nodes.nodes_agnes import LingliuAgnesChat, LingliuAgnesImage, LingliuAgnesImageReverse, LingliuAgnesVideo
from .nodes.nodes_openai import (
    LingliuChatGPTApi,
    LingliuGptImage,
    LingliuGptImage2,
    LingliuGptImage2Official,
    LingliuGptImageEdit,
)
from .nodes.nodes_gpt_image_25 import LingliuGptImage25
from .nodes.nodes_prompting import (
    LingliuEcommercePromptGenerator,
    LingliuMainImagePromptGenerator,
    LingliuStyleSetter,
)
from .nodes.nodes_text_presets import (
    LingliuTextPresetSelector,
    LingliuTextPresetSetter,
)
from .nodes.nodes_mj import (
    LingliuMJAction,
    LingliuMJAdvancedRead,
    LingliuMJBlend,
    LingliuMJChatCompat,
    LingliuMJDescribe,
    LingliuMJEdits,
    LingliuMJFetchTask,
    LingliuMJImagine,
    LingliuMJModalRegionEdit,
    LingliuMJShorten,
    LingliuMJSwapFace,
    LingliuMJUploadImages,
    LingliuMJVideo,
)
from .nodes.nodes_batch_image_input import LingliuBatchImageInput
from .nodes.nodes_video_tools import LingliuVideoConcat
from .nodes.nodes_seedream_responses import LingliuResponsesVision, LingliuSeedreamV5Image
from .nodes.nodes_minimax_h3 import LingliuMiniMaxH3Frames, LingliuMiniMaxH3Reference
from .nodes.nodes_minimax_tts import LingliuMiniMaxTTS, LingliuMiniMaxTTSOptions
from .nodes.nodes_minimax_voice_clone import LingliuMiniMaxVoiceClone
from .nodes.nodes_wan3 import LingliuWan3
from .nodes.nodes_wan3_low import LingliuWan3Low
from .nodes.nodes_jl_video import LingliuJLVideo
from .nodes.nodes_jl_image import LingliuJLImage
from .nodes.nodes_depth_video import LingliuDepthVideo
from .nodes.nodes_wan27_image import LingliuWan27Image
from .nodes.nodes_soul import LingliuSoul2Image, LingliuSoulIDManage, LingliuSoulIDTrain
from .nodes.nodes_matting import LingliuMatting
from .nodes.nodes_motion_imitation import LingliuMotionImitation

WEB_DIRECTORY = "./web"


NODE_CLASS_MAPPINGS = {
    "lingliu_api_set": LingliuApiSet,
    "lingliu_task_query": LingliuTaskQuery,
    "lingliu_models": LingliuModels,
    "lingliu_chat": LingliuChat,
    "lingliu_image_generate": LingliuImageGenerate,
    "lingliu_image_edit": LingliuImageEdit,
    "lingliu_google_veo3": LingliuGoogleVeo3,
    "lingliu_nano_banana": LingliuNanoBanana,
    "lingliu_nano_banana_edit": LingliuNanoBananaEdit,
    "lingliu_nano_banana2_edit": LingliuNanoBanana2Edit,
    "lingliu_gemini_text_only": LingliuGeminiTextOnly,
    "lingliu_seedance2": LingliuDoubaoSeedance2,
    "lingliu_seedance2_asset_bundle": LingliuSeedance2AssetBundle,
    "lingliu_seedance2_asset_upload": LingliuSeedance2AssetUpload,
    "lingliu_seedance2_official": LingliuSeedance2Official,
    "lingliu_seedance2.5": LingliuSeedance25,
    "lingliu_seedance2.5_official": LingliuSeedance25Official,
    "lingliu_video_prompt_optimizer": LingliuVideoPromptOptimizer,
    "lingliu_seedance_low": LingliuSeedanceLow,
    "lingliu_r2_storage_set": LingliuR2StorageSet,
    "lingliu_topaz_image_upscale": LingliuTopazImageUpscale,
    "lingliu_topaz_video_upscale": LingliuTopazVideoUpscale,
    "lingliu_grok_image": LingliuGrokImage,
    "lingliu_grok_imagine_image_2": LingliuGrokImagineImage2,
    "lingliu_grok_video": LingliuGrokVideo,
    "lingliu_grok_imagine_video_1_5": LingliuGrokImagineVideo15,
    "lingliu_grok_imagine_video_1_5_openai": LingliuGrokImagineVideo15OpenAI,
    "lingliu_omni_video": LingliuOmniVideo,
    "lingliu_agnes_chat": LingliuAgnesChat,
    "lingliu_agnes_image_reverse": LingliuAgnesImageReverse,
    "lingliu_agnes_image": LingliuAgnesImage,
    "lingliu_agnes_video": LingliuAgnesVideo,
    "lingliu_gpt_image": LingliuGptImage,
    "lingliu_gpt_image_edit": LingliuGptImageEdit,
    "lingliu_chatgpt_api": LingliuChatGPTApi,
    "lingliu_gpt_image_2": LingliuGptImage2,
    "lingliu_gpt_image_2_official": LingliuGptImage2Official,
    "lingliu_gpt_image_25": LingliuGptImage25,
    "lingliu_ecommerce_prompt_generator": LingliuEcommercePromptGenerator,
    "lingliu_main_image_prompt_generator": LingliuMainImagePromptGenerator,
    "lingliu_style_setter": LingliuStyleSetter,
    "lingliu_text_preset_setter": LingliuTextPresetSetter,
    "lingliu_text_preset_selector": LingliuTextPresetSelector,
    "lingliu_mj_imagine": LingliuMJImagine,
    "lingliu_mj_fetch_task": LingliuMJFetchTask,
    "lingliu_mj_action": LingliuMJAction,
    "lingliu_mj_modal_region_edit": LingliuMJModalRegionEdit,
    "lingliu_mj_blend": LingliuMJBlend,
    "lingliu_mj_describe": LingliuMJDescribe,
    "lingliu_mj_upload_images": LingliuMJUploadImages,
    "lingliu_mj_shorten": LingliuMJShorten,
    "lingliu_mj_edits": LingliuMJEdits,
    "lingliu_mj_video": LingliuMJVideo,
    "lingliu_mj_chat_compat": LingliuMJChatCompat,
    "lingliu_mj_swap_face": LingliuMJSwapFace,
    "lingliu_mj_advanced_read": LingliuMJAdvancedRead,
    "lingliu_batch_image_input": LingliuBatchImageInput,
    "lingliu_video_concat": LingliuVideoConcat,
    "lingliu_seedream_v5_image": LingliuSeedreamV5Image,
    "lingliu_responses_vision": LingliuResponsesVision,
    "lingliu_minimax_h3_frames": LingliuMiniMaxH3Frames,
    "lingliu_minimax_h3_reference": LingliuMiniMaxH3Reference,
    "lingliu_minimax_tts": LingliuMiniMaxTTS,
    "lingliu_minimax_tts_options": LingliuMiniMaxTTSOptions,
    "lingliu_minimax_voice_clone": LingliuMiniMaxVoiceClone,
    "lingliu_wan3": LingliuWan3,
    "lingliu_wan3_low": LingliuWan3Low,
    "lingliu_jl_video": LingliuJLVideo,
    "lingliu_jl_image": LingliuJLImage,
    "lingliu_depth_video": LingliuDepthVideo,
    "lingliu_wan27_image": LingliuWan27Image,
    "lingliu_soul2_image": LingliuSoul2Image,
    "lingliu_soul_id_train": LingliuSoulIDTrain,
    "lingliu_soul_id_manage": LingliuSoulIDManage,
    "lingliu_matting": LingliuMatting,
    "lingliu_motion_imitation": LingliuMotionImitation,
}


NODE_DISPLAY_NAME_MAPPINGS = {
    "lingliu_api_set": "灵流API设置",
    "lingliu_task_query": "lingliu_task_query",
    "lingliu_models": "lingliu_models",
    "lingliu_chat": "lingliu_chat",
    "lingliu_image_generate": "lingliu_image_generate",
    "lingliu_image_edit": "lingliu_image_edit",
    "lingliu_google_veo3": "lingliu_google_veo3",
    "lingliu_nano_banana": "lingliu_nano_banana",
    "lingliu_nano_banana_edit": "lingliu_nano_banana_edit",
    "lingliu_nano_banana2_edit": "lingliu_nano_banana2_edit",
    "lingliu_gemini_text_only": "lingliu_gemini_text_only",
    "lingliu_seedance2": "lingliu_seedance2",
    "lingliu_seedance2_asset_bundle": "lingliu_seedance2_asset_bundle",
    "lingliu_seedance2_asset_upload": "lingliu_seedance2_asset_upload",
    "lingliu_seedance2_official": "lingliu_Seedance2.0视频生成",
    "lingliu_seedance2.5": "lingliu_seedance2.5",
    "lingliu_seedance2.5_official": "lingliu_seedance2.5_official",
    "lingliu_video_prompt_optimizer": "lingliu_视频提示词优化器",
    "lingliu_seedance_low": "lingliu_seedance_low",
    "lingliu_r2_storage_set": "lingliu_R2_storage_set",
    "lingliu_topaz_image_upscale": "lingliu_图像超分放大",
    "lingliu_topaz_video_upscale": "lingliu_视频超分放大",
    "lingliu_grok_image": "lingliu_Grok_image",
    "lingliu_grok_imagine_image_2": "lingliu_Grok Imagine Image 2.0",
    "lingliu_grok_video": "lingliu_Grok_video",
    "lingliu_grok_imagine_video_1_5": "lingliu_Grok Imagine Video 1.5",
    "lingliu_grok_imagine_video_1_5_openai": "lingliu_Grok Imagine Video 1.5 OpenAI格式",
    "lingliu_omni_video": "灵流Omni视频",
    "lingliu_agnes_chat": "灵流Agnes文本",
    "lingliu_agnes_image_reverse": "灵流Agnes图像反推",
    "lingliu_agnes_image": "灵流Agnes图像",
    "lingliu_agnes_video": "灵流Agnes视频",
    "lingliu_gpt_image": "lingliu_gpt_image",
    "lingliu_gpt_image_edit": "lingliu_gpt_image_edit",
    "lingliu_chatgpt_api": "lingliu_chatgpt_api",
    "lingliu_gpt_image_2": "lingliu_gpt_image_2",
    "lingliu_gpt_image_2_official": "lingliu_gpt_image_2_official",
    "lingliu_gpt_image_25": "lingliu_GPT Image 2.5",
    "lingliu_ecommerce_prompt_generator": "lingliu_电商详情提示词生成器",
    "lingliu_main_image_prompt_generator": "lingliu_主图提示词生成器",
    "lingliu_style_setter": "lingliu_风格设定器",
    "lingliu_text_preset_setter": "灵流文本预设设置",
    "lingliu_text_preset_selector": "灵流文本预设选择器",
    "lingliu_mj_imagine": "灵流MJ文生图",
    "lingliu_mj_fetch_task": "灵流MJ查询任务",
    "lingliu_mj_action": "灵流MJ按钮操作",
    "lingliu_mj_modal_region_edit": "灵流MJ局部重绘",
    "lingliu_mj_blend": "灵流MJ混图",
    "lingliu_mj_describe": "灵流MJ图像描述",
    "lingliu_mj_upload_images": "灵流MJ上传图片",
    "lingliu_mj_shorten": "灵流MJ提示词精简",
    "lingliu_mj_edits": "灵流MJ图像编辑",
    "lingliu_mj_video": "灵流MJ图生视频",
    "lingliu_mj_chat_compat": "灵流MJ聊天兼容",
    "lingliu_mj_swap_face": "灵流MJ换脸",
    "lingliu_mj_advanced_read": "灵流MJ高级读取",
    "lingliu_batch_image_input": "灵流批量图像输入",
    "lingliu_video_concat": "灵流视频拼接",
    "lingliu_seedream_v5_image": "灵流Seedream5图像",
    "lingliu_responses_vision": "灵流GPT5.5多模态",
    "lingliu_minimax_h3_frames": "MiniMax-H3首尾帧生视频",
    "lingliu_minimax_h3_reference": "MiniMax-H3参考生视频",
    "lingliu_minimax_tts": "lingliu_minimax_tts",
    "lingliu_minimax_tts_options": "lingliu_minimax_tts_options",
    "lingliu_minimax_voice_clone": "lingliu_minimax_voice_clone",
    "lingliu_wan3": "lingliu_Wan3.0视频",
    "lingliu_wan3_low": "lingliu_Wan3.0低价视频",
    "lingliu_jl_video": "lingliu_JL视频",
    "lingliu_jl_image": "lingliu_JL图像",
    "lingliu_depth_video": "lingliu_视频转深度视频",
    "lingliu_wan27_image": "lingliu_Wan2.7图像",
    "lingliu_soul2_image": "lingliu_Soul 2 生图",
    "lingliu_soul_id_train": "lingliu_Soul ID 训练",
    "lingliu_soul_id_manage": "lingliu_Soul ID 管理",
    "lingliu_matting": "lingliu_图片抠图（透明底）",
    "lingliu_motion_imitation": "lingliu_动作模仿",
}


__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
