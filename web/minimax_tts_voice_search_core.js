import { MINIMAX_VOICE_BY_ID } from "./minimax_tts_voice_catalog.js";

export const MINIMAX_TTS_NODE_CLASS = "lingliu_minimax_tts";
export const RAW_VOICE_ID_WIDGET = "音色ID";
export const VOICE_SOURCE_WIDGET = "音色来源";
export const VOICE_SOURCE_PRESET = "预设音色";
export const VOICE_SOURCE_CUSTOM = "自定义ID";

export function isMiniMaxTtsVoiceSearchTarget(nodeName) {
	return nodeName === MINIMAX_TTS_NODE_CLASS;
}

export function modeForVoiceId(voiceId) {
	return MINIMAX_VOICE_BY_ID.has(String(voiceId ?? "").trim())
		? VOICE_SOURCE_PRESET
		: VOICE_SOURCE_CUSTOM;
}

export function syncRawVoiceId(rawWidget, voiceId) {
	if (!rawWidget) {
		return "";
	}
	const normalized = String(voiceId ?? "").trim();
	rawWidget.value = normalized;
	return normalized;
}
