import { app } from "../../scripts/app.js";

const NODE_CLASSES = new Set([
	"lingliu_seedance2.5",
	"lingliu_seedance2.5_official",
	"lingliu_seedance_low",
]);

const UPDATE_BUTTON_NAME = "更新节点";
const INSTALL_FLAG = "__lingliuSeedance25PortsInstalled";

const FIXED_PORTS = [
	{ name: "首帧图", fallbackType: "IMAGE" },
	{ name: "尾帧图", fallbackType: "IMAGE" },
];

const DYNAMIC_PORTS = [
	{
		prefix: "参考图",
		countWidget: "参考图接口数量",
		defaultCount: 3,
		maxCount: 30,
		fallbackType: "IMAGE",
	},
	{
		prefix: "参考视频",
		countWidget: "参考视频接口数量",
		defaultCount: 1,
		maxCount: 10,
		fallbackType: "VIDEO",
	},
	{
		prefix: "参考音频",
		countWidget: "参考音频接口数量",
		defaultCount: 1,
		maxCount: 10,
		fallbackType: "AUDIO",
	},
];

function findWidget(node, name) {
	return node?.widgets?.find((widget) => widget?.name === name);
}

function findInputIndex(node, name) {
	return node?.inputs?.findIndex((input) => input?.name === name) ?? -1;
}

function inputIsConnected(input) {
	return input?.link != null
		|| (Array.isArray(input?.links) && input.links.length > 0)
		|| (Array.isArray(input?.linkIds) && input.linkIds.length > 0);
}

function declaredInputType(nodeData, name, fallbackType) {
	for (const groupName of ["required", "optional"]) {
		const definition = nodeData?.input?.[groupName]?.[name];
		const declared = Array.isArray(definition) ? definition[0] : definition;
		if (typeof declared === "string" && declared) {
			return declared;
		}
	}
	return fallbackType;
}

function parseDynamicIndex(name, prefix, maxCount) {
	if (typeof name !== "string" || !name.startsWith(prefix)) {
		return null;
	}
	const suffix = name.slice(prefix.length);
	if (!/^[1-9]\d*$/.test(suffix)) {
		return null;
	}
	const index = Number(suffix);
	return Number.isInteger(index) && index <= maxCount ? index : null;
}

function listDynamicInputs(node, spec) {
	const result = [];
	for (let slot = 0; slot < (node?.inputs?.length ?? 0); slot += 1) {
		const input = node.inputs[slot];
		const index = parseDynamicIndex(input?.name, spec.prefix, spec.maxCount);
		if (index != null) {
			result.push({ slot, index, input });
		}
	}
	result.sort((left, right) => left.index - right.index || left.slot - right.slot);
	return result;
}

function readCounts(node, specs) {
	const counts = new Map();
	for (const spec of specs) {
		const widget = findWidget(node, spec.countWidget);
		if (!widget) {
			console.warn(`[Lingliu Seedance 2.5] Missing widget: ${spec.countWidget}`);
			return null;
		}

		const numericValue = Number(widget.value);
		const integerValue = Number.isFinite(numericValue)
			? Math.trunc(numericValue)
			: spec.defaultCount;
		const count = Math.min(spec.maxCount, Math.max(0, integerValue));
		if (widget.value !== count) {
			widget.value = count;
		}
		counts.set(spec.prefix, count);
	}
	return counts;
}

function ensureInput(node, name, type) {
	if (findInputIndex(node, name) >= 0) {
		return false;
	}
	node.addInput(name, type);
	return true;
}

function materializeInputs(node, spec, count) {
	let changed = false;
	for (let index = 1; index <= count; index += 1) {
		changed = ensureInput(node, `${spec.prefix}${index}`, spec.type) || changed;
	}
	return changed;
}

function trimTrailingUnusedInputs(node, spec, count) {
	let changed = false;
	while (true) {
		const inputs = listDynamicInputs(node, spec);
		const tail = inputs.at(-1);
		if (!tail || tail.index <= count || inputIsConnected(tail.input)) {
			break;
		}
		node.removeInput(tail.slot);
		changed = true;
	}
	return changed;
}

function resizeNode(node) {
	if (!node || node.flags?.collapsed) {
		return;
	}
	const computed = node.computeSize?.();
	if (!Array.isArray(computed) || !Number.isFinite(computed[1])) {
		return;
	}
	const currentWidth = Number.isFinite(node.size?.[0]) ? node.size[0] : 0;
	const computedWidth = Number.isFinite(computed[0]) ? computed[0] : currentWidth;
	node.setSize?.([Math.max(currentWidth, computedWidth), computed[1]]);
	node.setDirtyCanvas?.(true, true);
}

function scheduleResize(node) {
	window.setTimeout(() => {
		try {
			resizeNode(node);
		} catch (error) {
			console.warn("[Lingliu Seedance 2.5] Unable to resize node.", error);
		}
	}, 0);
	window.setTimeout(() => {
		try {
			resizeNode(node);
		} catch (error) {
			console.warn("[Lingliu Seedance 2.5] Unable to resize node.", error);
		}
	}, 120);
}

function syncPorts(node, fixedSpecs, dynamicSpecs) {
	const counts = readCounts(node, dynamicSpecs);
	if (!counts) {
		return false;
	}

	let changed = false;
	for (const spec of fixedSpecs) {
		changed = ensureInput(node, spec.name, spec.type) || changed;
	}
	for (const spec of dynamicSpecs) {
		const count = counts.get(spec.prefix) ?? spec.defaultCount;
		changed = materializeInputs(node, spec, count) || changed;
		changed = trimTrailingUnusedInputs(node, spec, count) || changed;
	}

	node.setDirtyCanvas?.(true, true);
	scheduleResize(node);
	return changed;
}

function restoreFullInterface(node, fixedSpecs, dynamicSpecs) {
	const allSpecs = [
		...fixedSpecs.map((spec) => ({ name: spec.name, type: spec.type })),
		...dynamicSpecs.flatMap((spec) => (
			Array.from({ length: spec.maxCount }, (_, offset) => ({
				name: `${spec.prefix}${offset + 1}`,
				type: spec.type,
			}))
		)),
	];

	for (const spec of allSpecs) {
		if (findInputIndex(node, spec.name) >= 0) {
			continue;
		}
		try {
			node.addInput(spec.name, spec.type);
		} catch (error) {
			// Last-resort fail-open path for a partially broken LiteGraph node.
			if (!Array.isArray(node.inputs)) {
				node.inputs = [];
			}
			node.inputs.push({ name: spec.name, type: spec.type, link: null });
		}
	}
	node.setDirtyCanvas?.(true, true);
	scheduleResize(node);
}

function safeSyncPorts(node, fixedSpecs, dynamicSpecs) {
	try {
		return syncPorts(node, fixedSpecs, dynamicSpecs);
	} catch (error) {
		console.warn("[Lingliu Seedance 2.5] Port update was skipped.", error);
		try {
			restoreFullInterface(node, fixedSpecs, dynamicSpecs);
		} catch (restoreError) {
			console.warn("[Lingliu Seedance 2.5] Full interface restoration failed.", restoreError);
		}
		node?.setDirtyCanvas?.(true, true);
		return false;
	}
}

function installPortControls(node, fixedSpecs, dynamicSpecs) {
	if (!node || node[INSTALL_FLAG]) {
		return;
	}

	const button = node.addWidget(
		"button",
		UPDATE_BUTTON_NAME,
		UPDATE_BUTTON_NAME,
		() => {
			const graph = node.graph;
			try {
				graph?.beforeChange?.();
			} catch (error) {
				console.warn("[Lingliu Seedance 2.5] Unable to start graph update.", error);
			}

			safeSyncPorts(node, fixedSpecs, dynamicSpecs);

			try {
				graph?.afterChange?.();
			} catch (error) {
				console.warn("[Lingliu Seedance 2.5] Unable to finish graph update.", error);
			}
			node.setDirtyCanvas?.(true, true);
		},
	);
	button.serialize = false;
	node[INSTALL_FLAG] = true;

	safeSyncPorts(node, fixedSpecs, dynamicSpecs);
}

app.registerExtension({
	name: "comfyui_lingliu_api.seedance25_ports",
	beforeRegisterNodeDef(nodeType, nodeData) {
		if (!NODE_CLASSES.has(nodeData?.name)) {
			return;
		}

		const fixedSpecs = FIXED_PORTS.map((spec) => ({
			...spec,
			type: declaredInputType(nodeData, spec.name, spec.fallbackType),
		}));
		const dynamicSpecs = DYNAMIC_PORTS.map((spec) => ({
			...spec,
			type: declaredInputType(nodeData, `${spec.prefix}1`, spec.fallbackType),
		}));

		const onNodeCreated = nodeType.prototype.onNodeCreated;
		nodeType.prototype.onNodeCreated = function (...args) {
			const result = onNodeCreated?.apply(this, args);
			try {
				installPortControls(this, fixedSpecs, dynamicSpecs);
			} catch (error) {
				console.warn("[Lingliu Seedance 2.5] Port controls were not installed.", error);
			}
			return result;
		};

		const onConfigure = nodeType.prototype.onConfigure;
		nodeType.prototype.onConfigure = function (...args) {
			const result = onConfigure?.apply(this, args);
			try {
				installPortControls(this, fixedSpecs, dynamicSpecs);
				safeSyncPorts(this, fixedSpecs, dynamicSpecs);
			} catch (error) {
				console.warn("[Lingliu Seedance 2.5] Saved ports were left unchanged.", error);
			}
			return result;
		};
	},
});
