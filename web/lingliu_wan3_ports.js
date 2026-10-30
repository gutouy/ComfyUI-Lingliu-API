import { app } from "../../scripts/app.js";

const NODE_NAME = "lingliu_wan3";
const UPDATE_BUTTON_NAME = "更新节点";
const INSTALL_FLAG = "__lingliuWan3PortsInstalled";

const FIXED_PORTS = [
	{ name: "首帧图", type: "IMAGE" },
	{ name: "尾帧图", type: "IMAGE" },
];

const DYNAMIC_PORTS = [
	{ prefix: "参考图", countWidget: "参考图接口数量", defaultCount: 3, maxCount: 10, type: "IMAGE" },
	{ prefix: "参考视频", countWidget: "参考视频接口数量", defaultCount: 1, maxCount: 5, type: "VIDEO" },
	{ prefix: "参考音频", countWidget: "参考音频接口数量", defaultCount: 1, maxCount: 5, type: "AUDIO" },
];

function findWidget(node, name) {
	return node?.widgets?.find((widget) => widget?.name === name);
}

function findInputIndex(node, name) {
	return node?.inputs?.findIndex((input) => input?.name === name) ?? -1;
}

function connected(input) {
	return input?.link != null
		|| (Array.isArray(input?.links) && input.links.length > 0)
		|| (Array.isArray(input?.linkIds) && input.linkIds.length > 0);
}

function inputIndex(name, prefix, max) {
	if (typeof name !== "string" || !name.startsWith(prefix)) return null;
	const suffix = name.slice(prefix.length);
	if (!/^[1-9]\d*$/.test(suffix)) return null;
	const value = Number(suffix);
	return Number.isInteger(value) && value <= max ? value : null;
}

function dynamicInputs(node, spec) {
	return (node?.inputs ?? [])
		.map((input, slot) => ({ input, slot, index: inputIndex(input?.name, spec.prefix, spec.maxCount) }))
		.filter((item) => item.index != null)
		.sort((left, right) => left.index - right.index || left.slot - right.slot);
}

function countFor(node, spec) {
	const widget = findWidget(node, spec.countWidget);
	if (!widget) return spec.defaultCount;
	const raw = Number(widget.value);
	const value = Number.isFinite(raw) ? Math.trunc(raw) : spec.defaultCount;
	const count = Math.min(spec.maxCount, Math.max(0, value));
	widget.value = count;
	return count;
}

function ensureInput(node, name, type) {
	if (findInputIndex(node, name) >= 0) return false;
	node.addInput(name, type);
	return true;
}

function trimInputs(node, spec, count) {
	let changed = false;
	while (true) {
		const items = dynamicInputs(node, spec);
		const tail = items.at(-1);
		if (!tail || tail.index <= count || connected(tail.input)) break;
		node.removeInput(tail.slot);
		changed = true;
	}
	return changed;
}

function resize(node) {
	const size = node?.computeSize?.();
	if (!Array.isArray(size) || !Number.isFinite(size[1])) return;
	const width = Number.isFinite(node.size?.[0]) ? node.size[0] : size[0];
	node.setSize?.([Math.max(width, size[0]), size[1]]);
	node.setDirtyCanvas?.(true, true);
}

function scheduleResize(node) {
	window.setTimeout(() => resize(node), 0);
	window.setTimeout(() => resize(node), 120);
}

function sync(node) {
	let changed = false;
	for (const spec of FIXED_PORTS) changed = ensureInput(node, spec.name, spec.type) || changed;
	for (const spec of DYNAMIC_PORTS) {
		const count = countFor(node, spec);
		for (let index = 1; index <= count; index += 1) {
			changed = ensureInput(node, `${spec.prefix}${index}`, spec.type) || changed;
		}
		changed = trimInputs(node, spec, count) || changed;
	}
	node.setDirtyCanvas?.(true, true);
	scheduleResize(node);
	return changed;
}

function restoreAll(node) {
	for (const spec of FIXED_PORTS) ensureInput(node, spec.name, spec.type);
	for (const spec of DYNAMIC_PORTS) {
		for (let index = 1; index <= spec.maxCount; index += 1) {
			ensureInput(node, `${spec.prefix}${index}`, spec.type);
		}
	}
	resize(node);
}

function safeSync(node) {
	try {
		return sync(node);
	} catch (error) {
		console.warn("[Lingliu Wan3] Port update was skipped.", error);
		try { restoreAll(node); } catch (restoreError) {
			console.warn("[Lingliu Wan3] Full interface restoration failed.", restoreError);
		}
		return false;
	}
}

function install(node) {
	if (!node || node[INSTALL_FLAG]) return;
	const button = node.addWidget("button", UPDATE_BUTTON_NAME, UPDATE_BUTTON_NAME, () => {
		try { node.graph?.beforeChange?.(); } catch (error) { console.warn("[Lingliu Wan3] beforeChange failed", error); }
		safeSync(node);
		try { node.graph?.afterChange?.(); } catch (error) { console.warn("[Lingliu Wan3] afterChange failed", error); }
	});
	button.serialize = false;
	node[INSTALL_FLAG] = true;
	safeSync(node);
}

app.registerExtension({
	name: "comfyui_lingliu_api.wan3_ports",
	beforeRegisterNodeDef(nodeType, nodeData) {
		if (nodeData?.name !== NODE_NAME) return;
		const created = nodeType.prototype.onNodeCreated;
		nodeType.prototype.onNodeCreated = function (...args) {
			const result = created?.apply(this, args);
			try { install(this); } catch (error) { console.warn("[Lingliu Wan3] Install failed", error); }
			return result;
		};
		const configured = nodeType.prototype.onConfigure;
		nodeType.prototype.onConfigure = function (...args) {
			const result = configured?.apply(this, args);
			try { install(this); safeSync(this); } catch (error) { console.warn("[Lingliu Wan3] Configure failed", error); }
			return result;
		};
	},
});

