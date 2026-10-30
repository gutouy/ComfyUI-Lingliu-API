import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";
import { ComfyWidgets } from "../../scripts/widgets.js";

const WIDGET_NAME = "task_id_live";
const TEXT_PRESET_SETTER_CLASS = "lingliu_text_preset_setter";
const TEXT_PRESET_SELECTOR_CLASS = "lingliu_text_preset_selector";
const TEXT_PRESET_CATEGORY_WIDGET_NAME = "preset_category";
const TEXT_PRESET_WIDGET_NAME = "preset";
const TEXT_PRESET_EMPTY = "暂无预设";
const PROVIDER_LABELS = {
	lingliu: "灵流",
};
const PROVIDER_ALIASES = {
	lingliu: "lingliu",
	"灵流": "lingliu",
	"灵流线路": "lingliu",
};
const BANANA_MODEL_GROUPS = {
	lingliu_gpt_image_2_official: {
		lingliu: ["gpt-image-2", "gpt-image-2-all", "gpt-image-2-1k", "gpt-image-2-hq2", "gpt-image-2-hq3", "gpt-image-2-official", "gpt-image-2-enterprise"],
	},
	lingliu_nano_banana_edit: {
		lingliu: ["nano-banana", "nano-banana-hd"],
	},
	lingliu_nano_banana2_edit: {
		lingliu: [
			"gemini-3.1-flash-image-preview",
			"gemini-3.1-flash-image",
			"nano-banana-2",
			"nano-banana-2-2k",
			"nano-banana-2-4k",
			"nano-banana-2.1",
			"nano-banana-pro",
			"nano-banana-pro-2k",
			"nano-banana-pro-4k",
			"gemini-3.1-flash-image-preview-2k",
			"gemini-3.1-flash-image-preview-4k",
			"gemini-3.1-flash-image-2k",
			"gemini-3.1-flash-image-4k",
		],
	},
};
const TASK_QUERY_TYPE_GROUPS = {
	lingliu_task_query: {
		lingliu: ["auto", "image_task", "video_task", "seedance_task", "seedance_official_task", "fal_task"],
	},
};

function normalizeProvider(value) {
	return PROVIDER_ALIASES[String(value || "").trim()] || "lingliu";
}

function displayProvider(value) {
	const provider = normalizeProvider(value);
	return PROVIDER_LABELS[provider] || String(value || "").trim();
}

function ensureTaskWidget(node) {
	if (!node) {
		return null;
	}

	if (!node.widgets) {
		node.widgets = [];
	}

	let widget = node.widgets.find((entry) => entry?.name === WIDGET_NAME);
	if (widget) {
		return widget;
	}

	widget = ComfyWidgets.STRING(node, WIDGET_NAME, ["STRING", { multiline: true }], app).widget;
	widget.inputEl.readOnly = true;
	widget.inputEl.style.opacity = 0.7;
	widget.value = "";
	return widget;
}

function refreshNode(node) {
	requestAnimationFrame(() => {
		const size = node.computeSize();
		if (size[0] < node.size[0]) {
			size[0] = node.size[0];
		}
		if (size[1] < node.size[1]) {
			size[1] = node.size[1];
		}
		node.onResize?.(size);
		app.graph.setDirtyCanvas(true, false);
	});
}

function findWidget(node, name) {
	return node?.widgets?.find((widget) => widget?.name === name) || null;
}

async function fetchTextPresetPayload() {
	const response = await api.fetchApi("/lingliu/text_presets", { cache: "no-store" });
	if (!response.ok) {
		throw new Error(`\u5237\u65b0\u6587\u672c\u9884\u8bbe\u5931\u8d25\uff1aHTTP ${response.status}`);
	}
	return await response.json();
}

function normalizeTextPresetPayload(payload) {
	const defaultCategory = String(payload?.default_category || "\u6682\u65e0\u9884\u8bbe").trim() || "\u6682\u65e0\u9884\u8bbe";
	const categories = Array.isArray(payload?.categories)
		? payload.categories.map((item) => String(item || "").trim()).filter(Boolean)
		: [];
	const presets = Array.isArray(payload?.presets)
		? payload.presets
				.map((item) => ({
					category: String(item?.category || defaultCategory).trim() || defaultCategory,
					title: String(item?.title || "").trim(),
				}))
				.filter((item) => item.title)
		: [];
	return {
		defaultCategory,
		placeholder: String(payload?.placeholder || TEXT_PRESET_EMPTY),
		categories: categories.length ? categories : [defaultCategory],
		presets,
	};
}

function textPresetTitlesForCategory(payload, category) {
	const selected = String(category || payload.defaultCategory || "").trim();
	const titles = payload.presets
		.filter((item) => item.category === selected)
		.map((item) => item.title);
	return titles.length ? titles : [payload.placeholder || TEXT_PRESET_EMPTY];
}

function updateTextPresetWidgets(node, payload, preferredCategory = null) {
	const categoryWidget = findWidget(node, TEXT_PRESET_CATEGORY_WIDGET_NAME);
	const presetWidget = findWidget(node, TEXT_PRESET_WIDGET_NAME);
	if (!categoryWidget || !presetWidget) {
		return;
	}

	categoryWidget.options ||= {};
	categoryWidget.options.values = payload.categories;
	const nextCategory = preferredCategory || categoryWidget.value;
	if (!payload.categories.includes(nextCategory)) {
		categoryWidget.value = payload.categories[0] || payload.defaultCategory || "\u6682\u65e0\u9884\u8bbe";
	} else {
		categoryWidget.value = nextCategory;
	}

	const presetValues = textPresetTitlesForCategory(payload, categoryWidget.value);
	presetWidget.options ||= {};
	presetWidget.options.values = presetValues;
	if (!presetValues.includes(presetWidget.value)) {
		presetWidget.value = presetValues[0] || TEXT_PRESET_EMPTY;
	}
	refreshNode(node);
}

function setupTextPresetCategoryFilter(node) {
	const categoryWidget = findWidget(node, TEXT_PRESET_CATEGORY_WIDGET_NAME);
	if (!categoryWidget || categoryWidget.__lingliuTextPresetCategoryHooked) {
		return;
	}
	const originalCallback = categoryWidget.callback;
	categoryWidget.callback = (...args) => {
		const result = typeof originalCallback === "function" ? originalCallback.apply(categoryWidget, args) : undefined;
		if (node.__lingliuTextPresetPayload) {
			updateTextPresetWidgets(node, node.__lingliuTextPresetPayload, categoryWidget.value);
		}
		return result;
	};
	categoryWidget.__lingliuTextPresetCategoryHooked = true;
}

function setupTextPresetRefresh(node) {
	if (node?.comfyClass !== TEXT_PRESET_SELECTOR_CLASS) {
		return;
	}
	setupTextPresetCategoryFilter(node);
	if (!node.__lingliuTextPresetRefreshHooked) {
		const widget = node.addWidget("button", "\u5237\u65b0\u9884\u8bbe", "\u5237\u65b0\u9884\u8bbe", async () => {
			try {
				const payload = normalizeTextPresetPayload(await fetchTextPresetPayload());
				node.__lingliuTextPresetPayload = payload;
				updateTextPresetWidgets(node, payload);
			} catch (error) {
				console.error("[lingliu] failed to refresh text presets", error);
				alert(error?.message || "\u5237\u65b0\u6587\u672c\u9884\u8bbe\u5931\u8d25\uff0c\u8bf7\u67e5\u770b\u6d4f\u89c8\u5668\u63a7\u5236\u53f0\u3002");
			}
		});
		widget.serialize = false;
		node.__lingliuTextPresetRefreshHooked = true;
	}
	refreshNode(node);
}

function updateTextPresetSetterCategoryWidget(node, payload) {
	const categoryWidget = findWidget(node, TEXT_PRESET_CATEGORY_WIDGET_NAME);
	if (!categoryWidget) {
		return;
	}
	categoryWidget.options ||= {};
	categoryWidget.options.values = payload.categories;
	if (!payload.categories.includes(categoryWidget.value)) {
		categoryWidget.value = payload.categories[0] || payload.defaultCategory || "默认分类";
	}
	refreshNode(node);
}

function setupTextPresetSetterCategoryRefresh(node) {
	if (node?.comfyClass !== TEXT_PRESET_SETTER_CLASS) {
		return;
	}
	if (!node.__lingliuTextPresetSetterRefreshHooked) {
		const widget = node.addWidget("button", "\u5237\u65b0\u5206\u7c7b", "\u5237\u65b0\u5206\u7c7b", async () => {
			try {
				const payload = normalizeTextPresetPayload(await fetchTextPresetPayload());
				updateTextPresetSetterCategoryWidget(node, payload);
			} catch (error) {
				console.error("[lingliu] failed to refresh text preset categories", error);
				alert(error?.message || "刷新文本预设分类失败，请查看浏览器控制台。");
			}
		});
		widget.serialize = false;
		node.__lingliuTextPresetSetterRefreshHooked = true;
	}
	refreshNode(node);
}

function updateBananaModelWidget(node) {
	if (!node?.comfyClass || !BANANA_MODEL_GROUPS[node.comfyClass]) {
		return;
	}
	const providerWidget = findWidget(node, "model_provider");
	const modelWidget = findWidget(node, "model");
	if (!providerWidget || !modelWidget) {
		return;
	}

	const provider = normalizeProvider(providerWidget.value);
	const displayValue = displayProvider(providerWidget.value);
	if (providerWidget.value !== displayValue) {
		providerWidget.value = displayValue;
	}
	const values = [...(BANANA_MODEL_GROUPS[node.comfyClass][provider] || [])];
	if (!values.length) {
		return;
	}

	modelWidget.options ||= {};
	modelWidget.options.values = values;
	if (!values.includes(modelWidget.value)) {
		modelWidget.value = values[0];
	}
	// V16.3: lingliu_image_channel_contract.js narrows the list per LL/JL channel.
	node.__lingliuImageChannelApply?.();
	refreshNode(node);
}

function setupBananaModelFilter(node) {
	if (!node?.comfyClass || !BANANA_MODEL_GROUPS[node.comfyClass]) {
		return;
	}
	const providerWidget = findWidget(node, "model_provider");
	if (!providerWidget || providerWidget.__lingliuBananaHooked) {
		updateBananaModelWidget(node);
		return;
	}

	const originalCallback = providerWidget.callback;
	providerWidget.callback = (...args) => {
		const result = typeof originalCallback === "function" ? originalCallback.apply(providerWidget, args) : undefined;
		updateBananaModelWidget(node);
		return result;
	};
	providerWidget.__lingliuBananaHooked = true;
	updateBananaModelWidget(node);
}

function updateTaskQueryTypeWidget(node) {
	if (!node?.comfyClass || !TASK_QUERY_TYPE_GROUPS[node.comfyClass]) {
		return;
	}
	const providerWidget = findWidget(node, "task_provider");
	const taskTypeWidget = findWidget(node, "task_type");
	if (!providerWidget || !taskTypeWidget) {
		return;
	}

	const provider = normalizeProvider(providerWidget.value);
	const displayValue = displayProvider(providerWidget.value);
	if (providerWidget.value !== displayValue) {
		providerWidget.value = displayValue;
	}
	const values = [...(TASK_QUERY_TYPE_GROUPS[node.comfyClass][provider] || [])];
	if (!values.length) {
		return;
	}

	taskTypeWidget.options ||= {};
	taskTypeWidget.options.values = values;
	if (!values.includes(taskTypeWidget.value)) {
		taskTypeWidget.value = values[0];
	}
	refreshNode(node);
}

function setupTaskQueryTypeFilter(node) {
	if (!node?.comfyClass || !TASK_QUERY_TYPE_GROUPS[node.comfyClass]) {
		return;
	}
	const providerWidget = findWidget(node, "task_provider");
	if (!providerWidget || providerWidget.__lingliuTaskQueryHooked) {
		updateTaskQueryTypeWidget(node);
		return;
	}

	const originalCallback = providerWidget.callback;
	providerWidget.callback = (...args) => {
		const result = typeof originalCallback === "function" ? originalCallback.apply(providerWidget, args) : undefined;
		updateTaskQueryTypeWidget(node);
		return result;
	};
	providerWidget.__lingliuTaskQueryHooked = true;
	updateTaskQueryTypeWidget(node);
}

function taskIdHandler(event) {
	const nodeId = event.detail?.node_id;
	if (!nodeId) {
		return;
	}

	const node = app.graph?._nodes_by_id?.[nodeId];
	if (!node) {
		return;
	}

	const widget = ensureTaskWidget(node);
	if (!widget) {
		return;
	}

	if (event.detail?.clear) {
		widget.value = "";
		refreshNode(node);
		return;
	}

	const message = String(event.detail?.message || event.detail?.task_id || "").trim();
	if (!message) {
		return;
	}

	if (event.detail?.append && widget.value) {
		const lines = widget.value.split("\n");
		if (!lines.includes(message)) {
			widget.value = `${widget.value}\n${message}`;
		}
	} else {
		widget.value = message;
	}

	refreshNode(node);
}

app.registerExtension({
	name: "lingliu.taskid",
	setup() {
		api.addEventListener("lingliu-task-id", taskIdHandler);
	},
	beforeRegisterNodeDef(nodeType, nodeData) {
		if (
			!BANANA_MODEL_GROUPS[nodeData?.name] &&
			!TASK_QUERY_TYPE_GROUPS[nodeData?.name] &&
			nodeData?.name !== TEXT_PRESET_SETTER_CLASS &&
			nodeData?.name !== TEXT_PRESET_SELECTOR_CLASS
		) {
			return;
		}

		const onNodeCreated = nodeType.prototype.onNodeCreated;
		nodeType.prototype.onNodeCreated = function () {
			const result = onNodeCreated?.apply(this, arguments);
			setupBananaModelFilter(this);
			setupTaskQueryTypeFilter(this);
			setupTextPresetSetterCategoryRefresh(this);
			setupTextPresetRefresh(this);
			return result;
		};

		const onConfigure = nodeType.prototype.onConfigure;
		nodeType.prototype.onConfigure = function () {
			const result = onConfigure?.apply(this, arguments);
			setupBananaModelFilter(this);
			setupTaskQueryTypeFilter(this);
			setupTextPresetSetterCategoryRefresh(this);
			setupTextPresetRefresh(this);
			return result;
		};
	},
});
