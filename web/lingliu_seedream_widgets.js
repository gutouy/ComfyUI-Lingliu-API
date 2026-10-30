import { app } from "../../scripts/app.js";

const NODE_CLASS = "lingliu_seedream_v5_image";
const MODEL_WIDGET = "模型";
const RESOLUTION_WIDGET = "清晰度";

const RESOLUTIONS_BY_MODEL = {
	"seedream-v5-pro": ["1K", "2K"],
	"doubao-seedream-5-0-260128": ["2K", "3K", "4K"],
};

function findWidget(node, name) {
	return node?.widgets?.find((widget) => widget.name === name);
}

function setComboValues(widget, values) {
	if (!widget || !Array.isArray(values) || values.length === 0) {
		return;
	}
	widget.options = widget.options || {};
	widget.options.values = values;
	if (!values.includes(widget.value)) {
		widget.value = values[0];
	}
}

function refreshResolutionChoices(node) {
	const modelWidget = findWidget(node, MODEL_WIDGET);
	const resolutionWidget = findWidget(node, RESOLUTION_WIDGET);
	if (!modelWidget || !resolutionWidget) {
		return;
	}

	const values = RESOLUTIONS_BY_MODEL[modelWidget.value] || RESOLUTIONS_BY_MODEL["seedream-v5-pro"];
	setComboValues(resolutionWidget, values);
	node?.setDirtyCanvas?.(true, true);
}

function installSeedreamWidgetFilter(node) {
	if (!node || node.__lingliuSeedreamWidgetFilterInstalled) {
		return;
	}
	node.__lingliuSeedreamWidgetFilterInstalled = true;

	const modelWidget = findWidget(node, MODEL_WIDGET);
	if (modelWidget) {
		const originalCallback = modelWidget.callback;
		modelWidget.callback = function (...args) {
			const result = originalCallback?.apply(this, args);
			refreshResolutionChoices(node);
			return result;
		};
	}

	refreshResolutionChoices(node);
	window.setTimeout(() => refreshResolutionChoices(node), 0);
}

app.registerExtension({
	name: "comfyui_lingliu_api.seedream_widgets",
	beforeRegisterNodeDef(nodeType, nodeData) {
		if (nodeData.name !== NODE_CLASS) {
			return;
		}

		const onNodeCreated = nodeType.prototype.onNodeCreated;
		nodeType.prototype.onNodeCreated = function (...args) {
			const result = onNodeCreated?.apply(this, args);
			installSeedreamWidgetFilter(this);
			return result;
		};

		const onConfigure = nodeType.prototype.onConfigure;
		nodeType.prototype.onConfigure = function (...args) {
			const result = onConfigure?.apply(this, args);
			installSeedreamWidgetFilter(this);
			refreshResolutionChoices(this);
			return result;
		};
	},
});
