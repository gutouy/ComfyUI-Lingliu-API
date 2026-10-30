import { app } from "../../scripts/app.js";
import {
	MINIMAX_SYSTEM_VOICES,
	MINIMAX_VOICE_BY_ID,
	filterMiniMaxVoices,
} from "./minimax_tts_voice_catalog.js";
import {
	RAW_VOICE_ID_WIDGET,
	VOICE_SOURCE_CUSTOM,
	VOICE_SOURCE_PRESET,
	VOICE_SOURCE_WIDGET,
	isMiniMaxTtsVoiceSearchTarget,
	modeForVoiceId,
	syncRawVoiceId,
} from "./minimax_tts_voice_search_core.js";

const INSTALL_FLAG = "__lingliuMiniMaxTtsVoiceSearchInstalled";
const CONTROL_HEIGHT = 38;
const DOM_MARGIN = 10;
const MENU_MIN_WIDTH = 320;
const MENU_MAX_WIDTH = 520;
const MENU_MAX_HEIGHT = 420;
const MAX_VISIBLE_RESULTS = MINIMAX_SYSTEM_VOICES.length;
const FILTER_ALL = "all";
const FILTER_MANDARIN = "mandarin";
const FILTER_CANTONESE = "cantonese";
const FILTER_ENGLISH = "english";
const FILTER_OTHER = "other";
const LANGUAGE_FILTERS = Object.freeze([
	{ id: FILTER_ALL, label: "全部" },
	{ id: FILTER_MANDARIN, label: "中文" },
	{ id: FILTER_CANTONESE, label: "粤语" },
	{ id: FILTER_ENGLISH, label: "英语" },
	{ id: FILTER_OTHER, label: "其他" },
]);
let menuSequence = 0;

function findWidget(node, name) {
	return node?.widgets?.find((widget) => widget?.name === name) || null;
}

function cssText(styles) {
	return Object.entries(styles)
		.map(([key, value]) => {
			const property = key.startsWith("--")
				? key
				: key.replace(/[A-Z]/gu, (letter) => `-${letter.toLowerCase()}`);
			return `${property}: ${value}`;
		})
		.join("; ");
}

function voiceFilterId(voice) {
	switch (voice?.language) {
		case "中文 (普通话)":
			return FILTER_MANDARIN;
		case "中文 (粤语)":
			return FILTER_CANTONESE;
		case "英文":
			return FILTER_ENGLISH;
		default:
			return FILTER_OTHER;
	}
}

function filterVoices(query, filterId = FILTER_ALL) {
	const matches = filterMiniMaxVoices(query);
	if (filterId === FILTER_ALL) {
		return matches;
	}
	return matches.filter((voice) => voiceFilterId(voice) === filterId);
}

function compactLanguageLabel(language) {
	switch (language) {
		case "中文 (普通话)":
			return "中文";
		case "中文 (粤语)":
			return "粤语";
		case "英文":
			return "英语";
		default:
			return String(language ?? "");
	}
}

const LANGUAGE_FILTER_COUNTS = new Map(
	LANGUAGE_FILTERS.map(({ id }) => [
		id,
		id === FILTER_ALL
			? MINIMAX_SYSTEM_VOICES.length
			: MINIMAX_SYSTEM_VOICES.filter((voice) => voiceFilterId(voice) === id).length,
	]),
);

function setNodeDirty(node) {
	node?.setDirtyCanvas?.(true, true);
	app?.graph?.setDirtyCanvas?.(true, false);
}

function resizeNode(node) {
	window.setTimeout(() => {
		if (!node || node.flags?.collapsed) {
			return;
		}
		const computed = node.computeSize?.();
		if (!Array.isArray(computed)) {
			return;
		}
		const currentWidth = Number(node.size?.[0]) || 0;
		node.setSize?.([Math.max(currentWidth, computed[0] || 0), computed[1] || 0]);
		setNodeDirty(node);
	}, 0);
}

function hideRawVoiceWidget(widget) {
	if (!widget || widget.__lingliuMiniMaxHidden) {
		return;
	}
	widget.__lingliuMiniMaxHidden = true;
	widget.hidden = true;
	widget.computeSize = () => [0, -4];
	widget.inputEl?.style?.setProperty("display", "none");
}

function setRawVoiceId(node, value, notify = true) {
	const widget = findWidget(node, RAW_VOICE_ID_WIDGET);
	if (!widget) {
		return "";
	}
	const previous = String(widget.value ?? "");
	const normalized = syncRawVoiceId(widget, value);
	if (notify && previous !== normalized) {
		try {
			widget.callback?.(normalized);
		} catch (error) {
			console.warn("[Lingliu MiniMax TTS] Voice ID callback failed.", error);
		}
	}
	setNodeDirty(node);
	return normalized;
}

function createControlContainer() {
	const container = document.createElement("div");
	container.className = "lingliu-minimax-tts-voice-control";
	container.style.cssText = cssText({
		width: "100%",
		maxWidth: "100%",
		minWidth: "0",
		height: `${CONTROL_HEIGHT}px`,
		boxSizing: "border-box",
		display: "flex",
		alignItems: "center",
	});
	return container;
}

function styleInlineControl(element) {
	element.style.cssText = cssText({
		width: "100%",
		minWidth: "0",
		height: "32px",
		boxSizing: "border-box",
		border: "1px solid var(--border-color, rgba(150, 150, 150, 0.45))",
		borderRadius: "6px",
		background: "var(--comfy-input-bg, rgba(20, 20, 20, 0.7))",
		color: "var(--input-text, currentColor)",
		font: "inherit",
		fontSize: "12px",
		lineHeight: "30px",
		padding: "0 10px",
	});
}

function closeVoiceMenu(node, { restoreFocus = false } = {}) {
	const state = node?.__lingliuMiniMaxVoiceMenu;
	if (!state) {
		return;
	}
	node.__lingliuMiniMaxVoiceMenu = null;
	if (state.anchorFrame != null) {
		window.cancelAnimationFrame?.(state.anchorFrame);
		state.anchorFrame = null;
	}
	state.controller.abort();
	state.panel.remove();
	state.trigger.setAttribute("aria-expanded", "false");
	if (restoreFocus && state.trigger.isConnected) {
		state.trigger.focus({ preventScroll: true });
	}
}

function anchorSignature(rect) {
	return [rect.left, rect.top, rect.width, rect.height]
		.map((value) => Math.round(Number(value || 0) * 4) / 4)
		.join(":");
}

function positionVoiceMenu(state, rect = state.trigger.getBoundingClientRect()) {
	if (!rect.width || !state.trigger.isConnected) {
		return false;
	}
	const viewportGap = 8;
	const viewportWidth = Math.max(0, window.innerWidth - viewportGap * 2);
	const width = Math.min(
		Math.max(MENU_MIN_WIDTH, rect.width),
		MENU_MAX_WIDTH,
		viewportWidth,
	);
	const left = Math.min(
		Math.max(viewportGap, rect.left),
		Math.max(viewportGap, window.innerWidth - width - viewportGap),
	);
	const below = window.innerHeight - rect.bottom - viewportGap;
	const above = rect.top - viewportGap;
	const useBelow = below >= 230 || below >= above;
	const viewportHeight = Math.max(120, window.innerHeight - viewportGap * 2);
	const sideHeight = Math.max(0, useBelow ? below : above);
	const availableHeight = Math.min(MENU_MAX_HEIGHT, viewportHeight, Math.max(120, sideHeight));
	const preferredTop = useBelow
		? rect.bottom + 4
		: rect.top - availableHeight - 4;
	const top = Math.min(
		Math.max(viewportGap, preferredTop),
		Math.max(viewportGap, window.innerHeight - availableHeight - viewportGap),
	);

	Object.assign(state.panel.style, {
		left: `${left}px`,
		top: `${top}px`,
		width: `${width}px`,
		maxHeight: `${availableHeight}px`,
	});
	state.anchorSignature = anchorSignature(rect);
	return true;
}

function startVoiceMenuTracking(node, state) {
	if (typeof window.requestAnimationFrame !== "function") {
		return;
	}
	const track = () => {
		if (node?.__lingliuMiniMaxVoiceMenu !== state || state.controller.signal.aborted) {
			return;
		}
		const rect = state.trigger.getBoundingClientRect();
		if (!rect.width || !state.trigger.isConnected) {
			closeVoiceMenu(node);
			return;
		}
		if (anchorSignature(rect) !== state.anchorSignature) {
			positionVoiceMenu(state, rect);
		}
		state.anchorFrame = window.requestAnimationFrame(track);
	};
	state.anchorFrame = window.requestAnimationFrame(track);
}

function createMenuOption(node, voice, state, index) {
	const option = document.createElement("button");
	option.type = "button";
	option.tabIndex = -1;
	option.id = `${state.optionIdPrefix}-${index}`;
	option.setAttribute("role", "option");
	option.setAttribute("aria-selected", String(voice.voiceId === state.selectedVoiceId));
	option.setAttribute("aria-label", `${voice.name}，${voice.language}`);
	option.dataset.voiceId = voice.voiceId;
	option.title = `${voice.name} · ${voice.language}`;
	option.style.cssText = cssText({
		width: "100%",
		minWidth: "0",
		minHeight: "36px",
		boxSizing: "border-box",
		display: "grid",
		gridTemplateColumns: "minmax(0, 1fr) auto auto",
		columnGap: "8px",
		alignItems: "center",
		flex: "0 0 auto",
		border: "0",
		borderRadius: "5px",
		background: "transparent",
		color: "var(--input-text, currentColor)",
		padding: "6px 9px",
		textAlign: "start",
		cursor: "pointer",
		font: "inherit",
	});

	const label = document.createElement("span");
	label.textContent = voice.name;
	label.title = label.textContent;
	label.style.cssText = cssText({
		gridColumn: "1",
		minWidth: "0",
		overflow: "hidden",
		textOverflow: "ellipsis",
		whiteSpace: "nowrap",
		fontSize: "12px",
		lineHeight: "18px",
	});

	const language = document.createElement("span");
	language.textContent = compactLanguageLabel(voice.language);
	language.title = voice.language;
	language.style.cssText = cssText({
		gridColumn: "2",
		maxWidth: "78px",
		overflow: "hidden",
		textOverflow: "ellipsis",
		whiteSpace: "nowrap",
		borderRadius: "999px",
		background: "rgba(128, 128, 128, 0.16)",
		padding: "2px 6px",
		fontSize: "10px",
		lineHeight: "14px",
		opacity: "0.84",
	});

	const selected = document.createElement("span");
	selected.textContent = voice.voiceId === state.selectedVoiceId ? "✓" : "";
	selected.setAttribute("aria-hidden", "true");
	selected.style.cssText = cssText({
		gridColumn: "3",
		width: "14px",
		textAlign: "center",
		fontSize: "13px",
		fontWeight: "700",
	});

	option.append(label, language, selected);
	option.addEventListener("pointermove", () => state.setActiveIndex(index));
	option.addEventListener("focus", () => state.setActiveIndex(index));
	option.addEventListener("click", () => {
		node.__lingliuMiniMaxLastPresetId = voice.voiceId;
		setRawVoiceId(node, voice.voiceId);
		updatePresetTrigger(node);
		closeVoiceMenu(node, { restoreFocus: true });
	});
	return option;
}

function renderVoiceResults(node, state, query, { revealSelected = false } = {}) {
	state.query = String(query ?? "");
	const results = filterVoices(state.query, state.activeFilter).slice(0, MAX_VISIBLE_RESULTS);
	state.results = results;
	state.options = [];
	state.list.replaceChildren();
	state.list.scrollTop = 0;
	const filterLabel = LANGUAGE_FILTERS.find(({ id }) => id === state.activeFilter)?.label || "全部";
	state.status.textContent = results.length
		? `${filterLabel} · 找到 ${results.length} 个音色`
		: `${filterLabel} · 没有匹配的音色`;

	if (results.length === 0) {
		const empty = document.createElement("div");
		empty.textContent = "未找到匹配项，请尝试音色名称、Voice ID 或语言。";
		empty.style.cssText = cssText({
			padding: "18px 12px",
			fontSize: "12px",
			lineHeight: "18px",
			textAlign: "center",
			opacity: "0.78",
		});
		state.list.append(empty);
		state.activeIndex = -1;
		state.search.removeAttribute("aria-activedescendant");
		return;
	}

	for (const [index, voice] of results.entries()) {
		const option = createMenuOption(node, voice, state, index);
		state.options.push(option);
		state.list.append(option);
	}

	const selectedIndex = results.findIndex((voice) => voice.voiceId === state.selectedVoiceId);
	const initialIndex = revealSelected && selectedIndex >= 0 ? selectedIndex : 0;
	state.setActiveIndex(initialIndex, false);
	const targetOption = state.options[initialIndex];
	if (targetOption && typeof window.requestAnimationFrame === "function") {
		window.requestAnimationFrame(() => {
			if (
				node?.__lingliuMiniMaxVoiceMenu === state
				&& state.options[initialIndex] === targetOption
			) {
				targetOption.scrollIntoView({ block: "nearest" });
			}
		});
	}
}

function openVoiceMenu(node, trigger) {
	closeVoiceMenu(node);
	const rawWidget = findWidget(node, RAW_VOICE_ID_WIDGET);
	const selectedVoiceId = String(rawWidget?.value ?? "").trim();
	const controller = new AbortController();
	const panel = document.createElement("div");
	const sequence = ++menuSequence;
	const listId = `lingliu-minimax-voice-list-${sequence}`;
	const optionIdPrefix = `${listId}-option`;
	panel.className = "lingliu-minimax-tts-voice-menu";
	panel.style.cssText = cssText({
		position: "fixed",
		zIndex: "10020",
		boxSizing: "border-box",
		display: "flex",
		flexDirection: "column",
		gap: "6px",
		padding: "8px",
		border: "1px solid var(--border-color, rgba(150, 150, 150, 0.55))",
		borderRadius: "8px",
		background: "var(--comfy-menu-bg, #202020)",
		color: "var(--input-text, #f2f2f2)",
		boxShadow: "0 10px 28px rgba(0, 0, 0, 0.34)",
		overflow: "hidden",
	});

	const search = document.createElement("input");
	search.type = "search";
	search.autocomplete = "off";
	search.spellcheck = false;
	search.placeholder = "搜索音色名称、Voice ID 或语言";
	search.setAttribute("role", "combobox");
	search.setAttribute("aria-autocomplete", "list");
	search.setAttribute("aria-controls", listId);
	search.setAttribute("aria-expanded", "true");
	search.setAttribute("aria-label", "搜索 MiniMax 官方音色");
	styleInlineControl(search);
	search.style.flex = "0 0 auto";

	let state = null;
	const filterBar = document.createElement("div");
	filterBar.setAttribute("role", "toolbar");
	filterBar.setAttribute("aria-label", "按语言筛选音色");
	filterBar.style.cssText = cssText({
		flex: "0 0 auto",
		minWidth: "0",
		display: "flex",
		gap: "4px",
		overflowX: "auto",
		overflowY: "hidden",
		overscrollBehavior: "contain",
		padding: "1px 0",
		scrollbarWidth: "thin",
	});
	const filterButtons = new Map();
	for (const filter of LANGUAGE_FILTERS) {
		const button = document.createElement("button");
		button.type = "button";
		button.textContent = `${filter.label} ${LANGUAGE_FILTER_COUNTS.get(filter.id) ?? 0}`;
		button.setAttribute("aria-pressed", String(filter.id === FILTER_ALL));
		button.style.cssText = cssText({
			flex: "0 0 auto",
			minHeight: "26px",
			boxSizing: "border-box",
			border: "1px solid var(--border-color, rgba(150, 150, 150, 0.38))",
			borderRadius: "999px",
			background: filter.id === FILTER_ALL
				? "var(--comfy-input-bg, rgba(128, 128, 128, 0.24))"
				: "transparent",
			color: "var(--input-text, currentColor)",
			padding: "0 8px",
			font: "inherit",
			fontSize: "11px",
			lineHeight: "24px",
			whiteSpace: "nowrap",
			cursor: "pointer",
		});
		button.addEventListener("click", () => {
			if (!state) {
				return;
			}
			state.activeFilter = filter.id;
			state.updateFilterButtons();
			renderVoiceResults(node, state, state.search.value, { revealSelected: true });
			state.search.focus({ preventScroll: true });
		}, { signal: controller.signal });
		filterButtons.set(filter.id, button);
		filterBar.append(button);
	}
	filterBar.addEventListener("keydown", (event) => {
		if (!["ArrowLeft", "ArrowRight", "Home", "End"].includes(event.key)) {
			return;
		}
		const buttons = Array.from(filterButtons.values());
		const current = buttons.indexOf(event.target);
		if (current < 0) {
			return;
		}
		event.preventDefault();
		let next = current;
		if (event.key === "ArrowLeft") next = Math.max(0, current - 1);
		if (event.key === "ArrowRight") next = Math.min(buttons.length - 1, current + 1);
		if (event.key === "Home") next = 0;
		if (event.key === "End") next = buttons.length - 1;
		const nextButton = buttons[next];
		nextButton?.focus({ preventScroll: true });
		nextButton?.scrollIntoView({ block: "nearest", inline: "nearest" });
	}, { signal: controller.signal });

	const status = document.createElement("div");
	status.setAttribute("role", "status");
	status.setAttribute("aria-live", "polite");
	status.style.cssText = cssText({
		flex: "0 0 auto",
		paddingInline: "2px",
		fontSize: "11px",
		lineHeight: "14px",
		opacity: "0.72",
	});

	const list = document.createElement("div");
	list.id = listId;
	list.setAttribute("role", "listbox");
	list.setAttribute("aria-label", "MiniMax 官方音色列表");
	list.style.cssText = cssText({
		flex: "1 1 auto",
		minHeight: "0",
		overflowY: "auto",
		overflowX: "hidden",
		overscrollBehavior: "contain",
		display: "flex",
		flexDirection: "column",
		gap: "1px",
	});

	panel.append(search, filterBar, status, list);
	document.body.append(panel);
	trigger.setAttribute("aria-expanded", "true");

	state = {
		activeFilter: FILTER_ALL,
		activeIndex: -1,
		anchorFrame: null,
		anchorSignature: "",
		controller,
		filterButtons,
		list,
		optionIdPrefix,
		options: [],
		panel,
		results: [],
		search,
		selectedVoiceId,
		status,
		trigger,
		updateFilterButtons() {
			for (const [filterId, button] of this.filterButtons) {
				const active = filterId === this.activeFilter;
				button.setAttribute("aria-pressed", String(active));
				button.style.background = active
					? "var(--comfy-input-bg, rgba(128, 128, 128, 0.24))"
					: "transparent";
				button.style.fontWeight = active ? "650" : "400";
			}
		},
		updateOptionStyles() {
			this.options.forEach((option, optionIndex) => {
				const active = optionIndex === this.activeIndex;
				const selected = option.dataset.voiceId === this.selectedVoiceId;
				option.style.background = active
					? "var(--comfy-input-bg, rgba(128, 128, 128, 0.24))"
					: selected
						? "rgba(128, 128, 128, 0.12)"
						: "transparent";
				option.style.boxShadow = selected
					? "inset 0 0 0 1px var(--border-color, rgba(150, 150, 150, 0.5))"
					: "none";
			});
		},
		setActiveIndex(index, scroll = true) {
			if (!this.options.length) {
				this.activeIndex = -1;
				this.search.removeAttribute("aria-activedescendant");
				return;
			}
			const next = Math.max(0, Math.min(index, this.options.length - 1));
			this.activeIndex = next;
			this.updateOptionStyles();
			const active = this.options[next];
			this.search.setAttribute("aria-activedescendant", active.id);
			if (scroll) {
				active.scrollIntoView({ block: "nearest" });
			}
		},
	};
	node.__lingliuMiniMaxVoiceMenu = state;

	state.updateFilterButtons();
	renderVoiceResults(node, state, "", { revealSelected: true });
	positionVoiceMenu(state);
	startVoiceMenuTracking(node, state);
	search.focus({ preventScroll: true });

	search.addEventListener("input", () => renderVoiceResults(node, state, search.value), {
		signal: controller.signal,
	});
	search.addEventListener("keydown", (event) => {
		if (event.isComposing || event.keyCode === 229) {
			return;
		}
		switch (event.key) {
			case "ArrowDown":
				event.preventDefault();
				state.setActiveIndex(state.activeIndex + 1);
				break;
			case "ArrowUp":
				event.preventDefault();
				state.setActiveIndex(state.activeIndex - 1);
				break;
			case "Home":
				event.preventDefault();
				state.setActiveIndex(0);
				break;
			case "End":
				event.preventDefault();
				state.setActiveIndex(state.options.length - 1);
				break;
			case "PageDown":
				event.preventDefault();
				state.setActiveIndex(state.activeIndex + 8);
				break;
			case "PageUp":
				event.preventDefault();
				state.setActiveIndex(state.activeIndex - 8);
				break;
			case "Enter":
				event.preventDefault();
				state.options[state.activeIndex]?.click();
				break;
			case "Escape":
				event.preventDefault();
				closeVoiceMenu(node, { restoreFocus: true });
				break;
			default:
				break;
		}
	}, { signal: controller.signal });
	panel.addEventListener("keydown", (event) => {
		if (event.target !== search && event.key === "Escape") {
			event.preventDefault();
			closeVoiceMenu(node, { restoreFocus: true });
		}
	}, { signal: controller.signal });
	panel.addEventListener("focusout", (event) => {
		if (!panel.contains(event.relatedTarget)) {
			closeVoiceMenu(node);
		}
	}, { signal: controller.signal });

	document.addEventListener("pointerdown", (event) => {
		if (!panel.contains(event.target) && event.target !== trigger) {
			closeVoiceMenu(node);
		}
	}, { capture: true, signal: controller.signal });
	window.addEventListener("resize", () => positionVoiceMenu(state), { signal: controller.signal });
	window.addEventListener("scroll", () => positionVoiceMenu(state), {
		capture: true,
		passive: true,
		signal: controller.signal,
	});
}

function updatePresetTrigger(node) {
	const trigger = node?.__lingliuMiniMaxVoiceTrigger;
	if (!trigger) {
		return;
	}
	const rawId = String(findWidget(node, RAW_VOICE_ID_WIDGET)?.value ?? "").trim();
	const voice = MINIMAX_VOICE_BY_ID.get(rawId);
	if (!voice) {
		trigger.textContent = "选择 MiniMax 官方音色";
		trigger.title = "当前音色不在官方系统音色快照中";
		return;
	}
	trigger.textContent = voice.name;
	trigger.title = `${voice.name} · ${voice.language}`;
}

function renderPresetControl(node, container) {
	const trigger = document.createElement("button");
	trigger.type = "button";
	trigger.setAttribute("aria-haspopup", "listbox");
	trigger.setAttribute("aria-expanded", "false");
	trigger.setAttribute("aria-label", "选择 MiniMax 官方音色，可按名称、Voice ID 或语言搜索");
	styleInlineControl(trigger);
	Object.assign(trigger.style, {
		cursor: "pointer",
		overflow: "hidden",
		textOverflow: "ellipsis",
		whiteSpace: "nowrap",
		textAlign: "start",
	});
	trigger.addEventListener("click", () => openVoiceMenu(node, trigger));
	trigger.addEventListener("keydown", (event) => {
		if (["ArrowDown", "Enter", " "].includes(event.key)) {
			event.preventDefault();
			openVoiceMenu(node, trigger);
		}
	});
	container.replaceChildren(trigger);
	node.__lingliuMiniMaxVoiceTrigger = trigger;
	updatePresetTrigger(node);
}

function renderCustomControl(node, container) {
	const input = document.createElement("input");
	input.type = "text";
	input.autocomplete = "off";
	input.spellcheck = false;
	input.placeholder = "输入克隆、设计或其他 Voice ID";
	input.setAttribute("aria-label", "自定义 MiniMax 音色 ID");
	input.value = String(findWidget(node, RAW_VOICE_ID_WIDGET)?.value ?? "");
	styleInlineControl(input);
	input.addEventListener("input", () => {
		node.__lingliuMiniMaxLastCustomId = input.value;
		setRawVoiceId(node, input.value, false);
	});
	input.addEventListener("change", () => {
		input.value = setRawVoiceId(node, input.value);
		node.__lingliuMiniMaxLastCustomId = input.value;
	});
	container.replaceChildren(input);
	node.__lingliuMiniMaxVoiceTrigger = null;
}

function renderVoiceControl(node) {
	const container = node?.__lingliuMiniMaxVoiceContainer;
	const sourceWidget = findWidget(node, VOICE_SOURCE_WIDGET);
	if (!container || !sourceWidget) {
		return;
	}
	closeVoiceMenu(node);
	if (sourceWidget.value === VOICE_SOURCE_PRESET) {
		renderPresetControl(node, container);
	} else {
		renderCustomControl(node, container);
	}
	setNodeDirty(node);
}

function setVoiceMode(node, mode, { preserveRaw = false } = {}) {
	const sourceWidget = findWidget(node, VOICE_SOURCE_WIDGET);
	const rawWidget = findWidget(node, RAW_VOICE_ID_WIDGET);
	if (!sourceWidget || !rawWidget) {
		return;
	}
	const normalizedMode = mode === VOICE_SOURCE_PRESET
		? VOICE_SOURCE_PRESET
		: VOICE_SOURCE_CUSTOM;
	sourceWidget.value = normalizedMode;

	if (!preserveRaw) {
		if (normalizedMode === VOICE_SOURCE_PRESET) {
			const current = String(rawWidget.value ?? "").trim();
			const voiceId = MINIMAX_VOICE_BY_ID.has(current)
				? current
				: node.__lingliuMiniMaxLastPresetId || MINIMAX_SYSTEM_VOICES[0].voiceId;
			node.__lingliuMiniMaxLastPresetId = voiceId;
			setRawVoiceId(node, voiceId);
		} else {
			const current = String(rawWidget.value ?? "").trim();
			const customId = MINIMAX_VOICE_BY_ID.has(current)
				? node.__lingliuMiniMaxLastCustomId || ""
				: current;
			node.__lingliuMiniMaxLastCustomId = customId;
			setRawVoiceId(node, customId);
		}
	}
	renderVoiceControl(node);
}

function syncVoiceControlsFromRaw(node) {
	const rawWidget = findWidget(node, RAW_VOICE_ID_WIDGET);
	const sourceWidget = findWidget(node, VOICE_SOURCE_WIDGET);
	if (!rawWidget || !sourceWidget) {
		return;
	}
	const voiceId = String(rawWidget.value ?? "").trim();
	const mode = modeForVoiceId(voiceId);
	if (mode === VOICE_SOURCE_PRESET) {
		node.__lingliuMiniMaxLastPresetId = voiceId;
	} else {
		node.__lingliuMiniMaxLastCustomId = voiceId;
	}
	setVoiceMode(node, mode, { preserveRaw: true });
}

function installVoiceControls(node) {
	if (!node) {
		return;
	}
	if (node[INSTALL_FLAG]) {
		syncVoiceControlsFromRaw(node);
		return;
	}

	const rawWidget = findWidget(node, RAW_VOICE_ID_WIDGET);
	if (!rawWidget) {
		console.warn(`[Lingliu MiniMax TTS] Missing widget: ${RAW_VOICE_ID_WIDGET}`);
		return;
	}
	if (typeof node.addDOMWidget !== "function") {
		console.warn("[Lingliu MiniMax TTS] addDOMWidget is unavailable; the raw Voice ID remains editable.");
		return;
	}

	const initialVoiceId = String(rawWidget.value ?? "").trim();
	const initialMode = modeForVoiceId(initialVoiceId);
	node.__lingliuMiniMaxLastPresetId = MINIMAX_VOICE_BY_ID.has(initialVoiceId)
		? initialVoiceId
		: MINIMAX_SYSTEM_VOICES[0].voiceId;
	node.__lingliuMiniMaxLastCustomId = MINIMAX_VOICE_BY_ID.has(initialVoiceId)
		? ""
		: initialVoiceId;

	const sourceWidget = node.addWidget(
		"combo",
		VOICE_SOURCE_WIDGET,
		initialMode,
		(value) => setVoiceMode(node, value),
		{ values: [VOICE_SOURCE_PRESET, VOICE_SOURCE_CUSTOM] },
	);
	sourceWidget.serialize = false;

	const container = createControlContainer();
	const domWidget = node.addDOMWidget("预设音色 / 自定义ID", "lingliu-minimax-tts-voice", container, {
		margin: DOM_MARGIN,
		hideOnZoom: false,
		getMinHeight: () => CONTROL_HEIGHT,
		getMaxHeight: () => CONTROL_HEIGHT,
		getHeight: () => CONTROL_HEIGHT,
		serialize: false,
	});
	domWidget.serialize = false;
	node.__lingliuMiniMaxVoiceContainer = container;
	hideRawVoiceWidget(rawWidget);
	node[INSTALL_FLAG] = true;

	setVoiceMode(node, initialMode, { preserveRaw: true });
	resizeNode(node);
}

app.registerExtension({
	name: "comfyui_lingliu_api.minimax_tts_voice_search",
	beforeRegisterNodeDef(nodeType, nodeData) {
		if (!isMiniMaxTtsVoiceSearchTarget(nodeData?.name)) {
			return;
		}

		const onNodeCreated = nodeType.prototype.onNodeCreated;
		nodeType.prototype.onNodeCreated = function (...args) {
			const result = onNodeCreated?.apply(this, args);
			try {
				installVoiceControls(this);
			} catch (error) {
				console.warn("[Lingliu MiniMax TTS] Voice search controls were not installed.", error);
			}
			return result;
		};

		const onConfigure = nodeType.prototype.onConfigure;
		nodeType.prototype.onConfigure = function (...args) {
			const result = onConfigure?.apply(this, args);
			try {
				installVoiceControls(this);
				syncVoiceControlsFromRaw(this);
			} catch (error) {
				console.warn("[Lingliu MiniMax TTS] Saved Voice ID was preserved without UI enhancement.", error);
			}
			return result;
		};

		const onRemoved = nodeType.prototype.onRemoved;
		nodeType.prototype.onRemoved = function (...args) {
			closeVoiceMenu(this);
			return onRemoved?.apply(this, args);
		};
	},
});
