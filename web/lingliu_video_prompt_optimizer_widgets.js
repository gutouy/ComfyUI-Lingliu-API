import { app } from "../../scripts/app.js";

const NODE_CLASS = "lingliu_video_prompt_optimizer";
const TARGET_WIDGET = "目标视频模型";
const MODE_WIDGET = "生成模式";
const DURATION_WIDGET = "目标时长";

const MODES_BY_TARGET = {
	"Seedance 2.5": ["文生视频", "首帧图生视频", "首尾帧生视频", "参考生视频"],
	"MiniMax H3": ["文生视频", "首帧图生视频", "首尾帧生视频", "尾帧图生视频", "全参考生视频"],
};

const MAX_DURATION_BY_TARGET = {
	"Seedance 2.5": 30,
	"MiniMax H3": 15,
};

function findWidget(node, name) {
	return node?.widgets?.find((widget) => widget?.name === name) || null;
}

function syncContract(node) {
	const targetWidget = findWidget(node, TARGET_WIDGET);
	const modeWidget = findWidget(node, MODE_WIDGET);
	const durationWidget = findWidget(node, DURATION_WIDGET);
	if (!targetWidget || !modeWidget || !durationWidget) {
		return;
	}

	const target = Object.hasOwn(MODES_BY_TARGET, targetWidget.value)
		? targetWidget.value
		: "Seedance 2.5";
	const modes = MODES_BY_TARGET[target];
	modeWidget.options = modeWidget.options || {};
	modeWidget.options.values = [...modes];
	if (!modes.includes(modeWidget.value)) {
		modeWidget.value = "文生视频";
	}

	durationWidget.options = durationWidget.options || {};
	durationWidget.options.min = 4;
	durationWidget.options.max = MAX_DURATION_BY_TARGET[target];
	const numericDuration = Number(durationWidget.value);
	if (Number.isFinite(numericDuration)) {
		durationWidget.value = Math.max(4, Math.min(MAX_DURATION_BY_TARGET[target], numericDuration));
	}
	node?.setDirtyCanvas?.(true, true);
}

function install(node) {
	if (!node || node.__lingliuVideoPromptOptimizerWidgetsInstalled) {
		return;
	}
	node.__lingliuVideoPromptOptimizerWidgetsInstalled = true;

	const targetWidget = findWidget(node, TARGET_WIDGET);
	if (targetWidget) {
		const originalCallback = targetWidget.callback;
		targetWidget.callback = function (...args) {
			const result = originalCallback?.apply(this, args);
			syncContract(node);
			return result;
		};
	}

	syncContract(node);
	window.setTimeout(() => syncContract(node), 0);
}

app.registerExtension({
	name: "comfyui_lingliu_api.video_prompt_optimizer_widgets",
	beforeRegisterNodeDef(nodeType, nodeData) {
		if (nodeData?.name !== NODE_CLASS) {
			return;
		}

		const onNodeCreated = nodeType.prototype.onNodeCreated;
		nodeType.prototype.onNodeCreated = function (...args) {
			const result = onNodeCreated?.apply(this, args);
			install(this);
			return result;
		};

		const onConfigure = nodeType.prototype.onConfigure;
		nodeType.prototype.onConfigure = function (...args) {
			const result = onConfigure?.apply(this, args);
			install(this);
			syncContract(this);
			window.setTimeout(() => syncContract(this), 0);
			return result;
		};
	},
});
