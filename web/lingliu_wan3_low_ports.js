import { app } from "../../scripts/app.js";

const NODE_NAME = "lingliu_wan3_low";
const INSTALL_FLAG = "__lingliuWan3LowPortsInstalled";
const SYNC_FLAG = "__lingliuWan3LowPortsSyncQueued";
const RUNNING_FLAG = "__lingliuWan3LowPortsSyncRunning";

const PORTS = [
    { prefix: "参考图", type: "IMAGE", max: 10 },
    { prefix: "参考视频", type: "VIDEO", max: 5 },
    { prefix: "参考音频", type: "AUDIO", max: 5 },
];

const DURATION_RANGES = {
    "wan3.0-低价-1080p": [2, 30],
    "wan3.0-按次计费": [5, 30],
};

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
        .map((input, slot) => ({ input, slot, index: inputIndex(input?.name, spec.prefix, spec.max) }))
        .filter((item) => item.index != null)
        .sort((left, right) => left.index - right.index || left.slot - right.slot);
}

function findInput(node, name) {
    return node?.inputs?.findIndex((input) => input?.name === name) ?? -1;
}

function ensureInput(node, name, type) {
    if (findInput(node, name) >= 0) return false;
    node.addInput(name, type);
    return true;
}

function syncSpec(node, spec) {
    const items = dynamicInputs(node, spec);
    const connectedItems = items.filter((item) => connected(item.input));
    const highestConnected = connectedItems.reduce((max, item) => Math.max(max, item.index), 0);
    const desired = Math.min(spec.max, Math.max(1, highestConnected + 1));
    let changed = false;

    for (let index = 1; index <= desired; index += 1) {
        changed = ensureInput(node, `${spec.prefix}${index}`, spec.type) || changed;
    }

    // removeInput updates the graph's target slot indices. Never splice or
    // rebuild node.inputs: the other media groups may already have links.
    while (true) {
        const current = dynamicInputs(node, spec);
        const tail = current.at(-1);
        if (!tail || tail.index <= desired || connected(tail.input)) break;
        node.removeInput(tail.slot);
        changed = true;
    }
    return changed;
}

function resize(node) {
    const computed = node?.computeSize?.();
    if (!Array.isArray(computed) || !Number.isFinite(computed[1])) return;
    const width = Number.isFinite(node.size?.[0]) ? node.size[0] : computed[0];
    node.setSize?.([Math.max(width, computed[0]), computed[1]]);
    node.setDirtyCanvas?.(true, true);
}

function syncAll(node) {
    if (node[RUNNING_FLAG]) return false;
    node[RUNNING_FLAG] = true;
    let changed = false;
    try {
    const legacy = findInput(node, "参考图像");
    const keepLegacy = node.properties?.lingliuWan3LegacyInput || (legacy >= 0 && connected(node.inputs[legacy]));
    if (keepLegacy) {
        changed = ensureInput(node, "参考图像", "IMAGE") || changed;
    } else if (legacy >= 0) {
        node.removeInput(legacy);
        changed = true;
    }
    for (const spec of PORTS) changed = syncSpec(node, spec) || changed;
    updateDurationWidget(node);
    const ratio = findWidget(node, "比例");
    if (ratio) {
        node.properties = node.properties || {};
        if (ratio.value && !["模型默认", "adaptive"].includes(ratio.value)) {
            node.properties.lingliuWan3LegacyRatio = ratio.value;
        }
        ratio.options = ratio.options || {};
        ratio.options.values = ["模型默认"];
        ratio.value = "模型默认";
        ratio.options.tooltip = node.properties.lingliuWan3LegacyRatio
            ? "旧工作流保存的比例没有应用，当前使用模型默认比例。"
            : "当前模型使用默认比例。";
    }
    node.properties = node.properties || {};
    node.properties.lingliuWan3LowSchemaVersion = 2;
    if (changed) {
        node.setDirtyCanvas?.(true, true);
        window.setTimeout(() => resize(node), 0);
        window.setTimeout(() => resize(node), 120);
    }
    return changed;
    } finally {
        node[RUNNING_FLAG] = false;
    }
}

function queueSync(node) {
    if (!node || node[SYNC_FLAG] || node[RUNNING_FLAG] || node.__lingliuWan3Removed) return;
    node[SYNC_FLAG] = true;
    window.setTimeout(() => {
        node[SYNC_FLAG] = false;
        if (node.__lingliuWan3Removed) return;
        if (app.configuringGraph || node.graph?._is_configuring) {
            window.setTimeout(() => queueSync(node), 30);
            return;
        }
        try {
            syncAll(node);
        } catch (error) {
            console.warn("[Lingliu WAN3 low] Port synchronization failed.", error);
        }
    }, 0);
}

function findWidget(node, name) {
    return node?.widgets?.find((widget) => widget?.name === name);
}

function updateDurationWidget(node) {
    const modelWidget = findWidget(node, "模型");
    const durationWidget = findWidget(node, "时长");
    if (!modelWidget || !durationWidget) return;
    const range = DURATION_RANGES[String(modelWidget.value)] || [2, 30];
    durationWidget.options = durationWidget.options || {};
    durationWidget.options.min = range[0];
    durationWidget.options.max = range[1];
    durationWidget.options.step = 1;
    durationWidget.options.tooltip = `当前模型时长范围：${range[0]}–${range[1]} 秒`;
    node.setDirtyCanvas?.(true, true);
}

function install(node) {
    if (!node || node[INSTALL_FLAG]) return;
    node[INSTALL_FLAG] = true;

    const modelWidget = findWidget(node, "模型");
    if (modelWidget) {
        const previous = modelWidget.callback;
        modelWidget.callback = function (...args) {
            const result = previous?.apply(this, args);
            updateDurationWidget(node);
            return result;
        };
    }

    const previousConnectionsChange = node.onConnectionsChange;
    node.onConnectionsChange = function (...args) {
        const result = previousConnectionsChange?.apply(this, args);
        queueSync(this);
        return result;
    };
    const previousMenu = node.getExtraMenuOptions;
    node.getExtraMenuOptions = function (canvas, options) {
        const result = previousMenu?.call(this, canvas, options);
        options.push({
            content: "显示旧版图片批次接口",
            callback: () => {
                this.properties = this.properties || {};
                this.properties.lingliuWan3LegacyInput = true;
                queueSync(this);
            },
        });
        return result;
    };
    const previousRemoved = node.onRemoved;
    node.onRemoved = function (...args) {
        this.__lingliuWan3Removed = true;
        return previousRemoved?.apply(this, args);
    };
    queueSync(node);
}

app.registerExtension({
    name: "comfyui_lingliu_api.wan3_low_ports",
    beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData?.name !== NODE_NAME) return;
        const created = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function (...args) {
            const result = created?.apply(this, args);
            try { install(this); } catch (error) { console.warn("[Lingliu WAN3 low] Install failed.", error); }
            return result;
        };
        const configured = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function (...args) {
            const result = configured?.apply(this, args);
            try { install(this); queueSync(this); } catch (error) { console.warn("[Lingliu WAN3 low] Configure failed.", error); }
            return result;
        };
    },
});
