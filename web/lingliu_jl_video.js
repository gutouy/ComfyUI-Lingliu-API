import { app } from "../../scripts/app.js";

const NODE_NAME = "lingliu_jl_video";
const INSTALL_FLAG = "__lingliuJLVideoInstalled";
const SYNC_FLAG = "__lingliuJLVideoSyncQueued";
const RUNNING_FLAG = "__lingliuJLVideoSyncRunning";
const REMOVED_FLAG = "__lingliuJLVideoRemoved";
const STATE_KEY = "__lingliuJLVideoDurationState";

const PORTS = [
    { prefix: "参考图", type: "IMAGE", key: "images", max: 30 },
    { prefix: "参考视频", type: "VIDEO", key: "videos", max: 3 },
    { prefix: "参考音频", type: "AUDIO", key: "audios", max: 10 },
];

const WIDE_TALL = ["16:9", "9:16"];
const VEO = { min: 8, max: 8, ratios: WIDE_TALL, images: 3, videos: 0, audios: 0, audio: false };

// Mirrors JL_VIDEO_CONTRACTS in nodes/nodes_jl_video.py; the backend re-checks everything.
const MODEL_RULES = {
    "seedance-2.5": { min: 4, max: 25, resolutions: ["480p", "720p"], ratios: ["21:9", "16:9", "4:3", "1:1", "3:4", "9:16"], images: 30, videos: 0, audios: 10, audio: false },
    "seedance-2.0": { min: 15, max: 15, resolutions: ["1080p"], ratios: WIDE_TALL, images: 9, videos: 3, audios: 3, audio: false },
    "seedance-dola-2.5": { min: 30, max: 30, resolutions: ["720p"], ratios: WIDE_TALL, images: 9, videos: 0, audios: 0, audio: false },
    "veo-3.1": { ...VEO, resolutions: ["720p"], audio: true },
    "veo-3.1-fast": { ...VEO, resolutions: ["720p"] },
    "veo-3.1-1080p": { ...VEO, resolutions: ["1080p"] },
    "veo-3.1-fast-1080p": { ...VEO, resolutions: ["1080p"] },
    "grok-imagine-video": { min: 1, max: 15, excluded: 7, resolutions: ["480p", "720p"], ratios: ["1:1", "16:9", "9:16", "3:2", "4:3", "3:4", "2:3"], images: 7, videos: 0, audios: 0, audio: false },
    "minimax-h3-optimized": { min: 4, max: 15, resolutions: ["768p"], ratios: ["16:9", "9:16", "1:1", "4:3", "3:4", "21:9"], images: 9, videos: 3, audios: 3, audio: false },
    "grok-video": { min: 6, max: 6, resolutions: ["720p"], ratios: ["2:3", "3:2", "1:1", "9:16", "16:9"], images: 0, videos: 0, audios: 0, audio: false },
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

function findWidget(node, name) {
    return node?.widgets?.find((widget) => widget?.name === name);
}

function currentRule(node) {
    return MODEL_RULES[String(findWidget(node, "模型")?.value ?? "").trim()];
}

function ensureInput(node, name, type) {
    if (findInput(node, name) >= 0) return false;
    node.addInput(name, type);
    return true;
}

// One empty port after the highest connected one, up to the model limit.
// Trimming stops at a connected port, so links above the limit stay and the backend explains.
function syncSpec(node, spec, limit) {
    const items = dynamicInputs(node, spec);
    const highestConnected = items.filter((item) => connected(item.input))
        .reduce((max, item) => Math.max(max, item.index), 0);
    const desired = Math.min(limit, highestConnected + 1);
    let changed = false;

    for (let index = 1; index <= desired; index += 1) {
        changed = ensureInput(node, `${spec.prefix}${index}`, spec.type) || changed;
    }

    // removeInput updates the graph's target slot indices. Never splice or
    // rebuild node.inputs: the other media groups may already have links.
    while (true) {
        const tail = dynamicInputs(node, spec).at(-1);
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
        const rule = currentRule(node);
        for (const spec of PORTS) changed = syncSpec(node, spec, rule ? rule[spec.key] : spec.max) || changed;
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
    if (!node || node[SYNC_FLAG] || node[RUNNING_FLAG] || node[REMOVED_FLAG]) return;
    node[SYNC_FLAG] = true;
    window.setTimeout(() => {
        node[SYNC_FLAG] = false;
        if (node[REMOVED_FLAG]) return;
        if (app.configuringGraph || node.graph?._is_configuring) {
            window.setTimeout(() => queueSync(node), 30);
            return;
        }
        try {
            syncAll(node);
        } catch (error) {
            console.warn("[Lingliu JL video] Port synchronization failed.", error);
        }
    }, 0);
}
function setDisabled(widget, disabled) {
    widget.disabled = disabled;
    widget.options = widget.options || {};
    widget.options.disabled = disabled;
}

function updateDuration(node, rule, userChange) {
    const widget = findWidget(node, "时长");
    if (!widget) return;
    const state = node[STATE_KEY] || (node[STATE_KEY] = { locked: false, previous: 5 });
    widget.options = widget.options || {};
    if (rule.min === rule.max) {
        // Fixed-length models submit their own length; keep the user's value for the way back.
        if (!state.locked) {
            const numeric = Number(widget.value);
            if (Number.isInteger(numeric) && numeric !== rule.min) state.previous = numeric;
        }
        widget.value = rule.min;
        widget.options.min = rule.min;
        widget.options.max = rule.max;
        widget.options.tooltip = `当前模型固定 ${rule.min} 秒`;
        setDisabled(widget, true);
        state.locked = true;
        return;
    }
    if (state.locked) widget.value = state.previous;
    state.locked = false;
    setDisabled(widget, false);
    widget.options.min = rule.min;
    widget.options.max = rule.max;
    widget.options.step = 1;
    widget.options.tooltip = rule.excluded
        ? `当前模型时长范围：${rule.min}–${rule.max} 秒（不支持 ${rule.excluded} 秒）`
        : `当前模型时长范围：${rule.min}–${rule.max} 秒`;
    if (!userChange) return;
    const numeric = Number(widget.value);
    if (!Number.isFinite(numeric) || numeric < rule.min) widget.value = rule.min;
    else if (numeric > rule.max) widget.value = rule.max;
    else if (numeric === rule.excluded) widget.value = rule.excluded - 1;
}

function updateChoice(node, name, allowed, userChange, preferred) {
    const widget = findWidget(node, name);
    if (!widget) return;
    widget.options = widget.options || {};
    if (allowed.length === 1) {
        widget.value = allowed[0];
        widget.options.tooltip = `当前模型固定 ${allowed[0]}`;
        setDisabled(widget, true);
        return;
    }
    setDisabled(widget, false);
    widget.options.tooltip = `当前模型支持：${allowed.join("、")}`;
    // Only a model switch moves the value; saved workflows keep theirs and the backend explains.
    if (userChange && !allowed.includes(widget.value)) {
        widget.value = allowed.includes(preferred) ? preferred : allowed[0];
    }
}

function applyContract(node, userChange = false) {
    const rule = currentRule(node);
    if (!rule) return;
    updateDuration(node, rule, userChange);
    updateChoice(node, "比例", rule.ratios, userChange, "16:9");
    updateChoice(node, "清晰度", rule.resolutions, userChange, "720p");
    const audio = findWidget(node, "生成音频");
    if (audio) setDisabled(audio, !rule.audio);
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
            applyContract(node, true);
            queueSync(node);
            return result;
        };
    }

    const previousConnectionsChange = node.onConnectionsChange;
    node.onConnectionsChange = function (...args) {
        const result = previousConnectionsChange?.apply(this, args);
        queueSync(this);
        return result;
    };
    const previousRemoved = node.onRemoved;
    node.onRemoved = function (...args) {
        this[REMOVED_FLAG] = true;
        return previousRemoved?.apply(this, args);
    };
}

app.registerExtension({
    name: "comfyui_lingliu_api.jl_video",
    beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData?.name !== NODE_NAME) return;
        const created = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function (...args) {
            const result = created?.apply(this, args);
            try { install(this); applyContract(this); queueSync(this); } catch (error) { console.warn("[Lingliu JL video] Install failed.", error); }
            return result;
        };
        const configured = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function (...args) {
            const result = configured?.apply(this, args);
            try { install(this); applyContract(this); queueSync(this); } catch (error) { console.warn("[Lingliu JL video] Configure failed.", error); }
            return result;
        };
    },
});
