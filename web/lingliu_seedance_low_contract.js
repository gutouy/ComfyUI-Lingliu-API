import { app } from "../../scripts/app.js";

const NODE_CLASS = "lingliu_seedance_low";
const MODEL_WIDGET = "模型";
const DURATION_WIDGET = "时长";
const RATIO_WIDGET = "比例";
const STATE_KEY = "__lingliuSeedanceLowDurationState";

const FIXED_RATIOS = ["21:9", "16:9", "4:3", "1:1", "3:4", "9:16"];
const WIDE_TALL = ["16:9", "9:16"];
const ALL_RATIOS = ["adaptive", ...FIXED_RATIOS];

// Mirrors LOW_MODEL_CONTRACTS in nodes/nodes_seedance_low.py; fixed models submit 15 s whatever the widget says.
const MODEL_RULES = {
	"seedance2.0-低价": { min: 4, max: 30, ratios: FIXED_RATIOS },
	"seedance2.0-低价-按次计费": { min: 4, max: 30, fixed: 15, ratios: FIXED_RATIOS },
	"seedance2.0-fast-低价": { min: 4, max: 30, ratios: ALL_RATIOS },
	"seedance2.0-fast-低价-按次计费": { min: 4, max: 30, fixed: 15, ratios: FIXED_RATIOS },
	"seedance2.0-mini-低价": { min: 4, max: 30, ratios: ALL_RATIOS },
	"seedance2.5-低价": { min: 4, max: 30, ratios: ALL_RATIOS },
	"seedance2.5-低价-按次计费": { min: 5, max: 30, ratios: WIDE_TALL },
	"seedance2.0-特价-按次计费": { min: 4, max: 30, fixed: 15, ratios: WIDE_TALL },
	"seedance2.5-低价-480p": { min: 5, max: 30, ratios: WIDE_TALL },
	"seedance2.5-低价-720p": { min: 5, max: 29, ratios: FIXED_RATIOS },
	"seedance2.0-图生-低价": { min: 4, max: 15, ratios: WIDE_TALL },
	"seedance2.0-低价-480p": { min: 5, max: 15, ratios: WIDE_TALL },
};
for (const model of [
	"seedance2.0-特惠-480p",
	"seedance2.0-特惠-720p",
	"seedance2.0-特惠-1080p",
	"seedance2.0-mini-特惠-480p",
	"seedance2.0-mini-特惠-720p",
	"seedance2.0-fast-特惠-480p",
	"seedance2.0-fast-特惠-720p",
]) {
	MODEL_RULES[model] = { min: 4, max: 15, ratios: FIXED_RATIOS };
}
for (const model of ["seedance2.5-特惠-480p", "seedance2.5-特惠-720p", "seedance2.5-特惠-1080p"]) {
	MODEL_RULES[model] = { min: 4, max: 30, ratios: FIXED_RATIOS };
}

function findWidget(node, name) {
	return node?.widgets?.find((widget) => widget?.name === name);
}

function setDisabled(widget, disabled) {
	widget.disabled = disabled;
	widget.options = widget.options || {};
	widget.options.disabled = disabled;
}

function updateRatio(node, rule, userChange) {
	const ratioWidget = findWidget(node, RATIO_WIDGET);
	if (!ratioWidget) {
		return;
	}
	ratioWidget.options = ratioWidget.options || {};
	ratioWidget.options.tooltip = `当前模型比例：${rule.ratios.join("、")}`;
	// Only a model switch moves the ratio; saved workflows keep theirs and the backend explains.
	if (userChange && !rule.ratios.includes(ratioWidget.value)) {
		ratioWidget.value = "16:9";
	}
}

function updateDurationLock(node, userChange = false) {
	const modelWidget = findWidget(node, MODEL_WIDGET);
	const durationWidget = findWidget(node, DURATION_WIDGET);
	if (!modelWidget || !durationWidget) {
		return;
	}

	const rule = MODEL_RULES[String(modelWidget.value || "").trim()];
	const state = node[STATE_KEY] || (node[STATE_KEY] = { locked: false, previous: 5 });
	durationWidget.options = durationWidget.options || {};
	if (rule?.fixed) {
		if (!state.locked) {
			const numeric = Number(durationWidget.value);
			if (Number.isInteger(numeric) && numeric >= 4 && numeric <= 30 && numeric !== rule.fixed) {
				state.previous = numeric;
			}
		}
		durationWidget.value = rule.fixed;
		durationWidget.options.min = 4;
		durationWidget.options.max = 30;
		durationWidget.options.tooltip = `当前模型固定 ${rule.fixed} 秒`;
		setDisabled(durationWidget, true);
		state.locked = true;
		updateRatio(node, rule, userChange);
		return;
	}

	if (state.locked) {
		durationWidget.value = state.previous;
	}
	setDisabled(durationWidget, false);
	state.locked = false;
	if (!rule) {
		return;
	}
	durationWidget.options.min = rule.min;
	durationWidget.options.max = rule.max;
	durationWidget.options.tooltip = `当前模型时长范围：${rule.min}–${rule.max} 秒`;
	if (userChange) {
		const numeric = Number(durationWidget.value);
		if (!Number.isFinite(numeric) || numeric < rule.min) {
			durationWidget.value = rule.min;
		} else if (numeric > rule.max) {
			durationWidget.value = rule.max;
		}
	}
	updateRatio(node, rule, userChange);
}

app.registerExtension({
	name: "comfyui_lingliu_api.seedance_low_contract",
	beforeRegisterNodeDef(nodeType, nodeData) {
		if (nodeData?.name !== NODE_CLASS) {
			return;
		}

		const originalCreated = nodeType.prototype.onNodeCreated;
		nodeType.prototype.onNodeCreated = function (...args) {
			const result = originalCreated?.apply(this, args);
			const modelWidget = findWidget(this, MODEL_WIDGET);
			if (modelWidget && !modelWidget.__lingliuSeedanceLowWrapped) {
				const callback = modelWidget.callback;
				modelWidget.callback = (...callbackArgs) => {
					const callbackResult = callback?.apply(modelWidget, callbackArgs);
					updateDurationLock(this, true);
					return callbackResult;
				};
				modelWidget.__lingliuSeedanceLowWrapped = true;
			}
			updateDurationLock(this);
			return result;
		};

		const originalConfigure = nodeType.prototype.onConfigure;
		nodeType.prototype.onConfigure = function (...args) {
			const result = originalConfigure?.apply(this, args);
			updateDurationLock(this);
			return result;
		};
	},
});
