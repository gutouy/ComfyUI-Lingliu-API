import { app } from "../../scripts/app.js";

// V16.3 LL/JL channel contract for the Lingliu image nodes.  Keep the model
// lists in sync with nodes/lingliu_image_channels.py and lingliu_taskid.js.
const CHANNEL_WIDGET = "渠道";
const CHANNELS = ["LL", "JL"];
const LEGACY_CHANNEL = "LL";
const NEW_NODE_CHANNEL = "JL";
const CHANNEL_PROVIDERS = new Set(["lingliu"]);
const PROVIDER_ALIASES = {
    lingliu: "lingliu",
    "灵流": "lingliu",
    "灵流线路": "lingliu",
};
const JL_QUALITIES = ["auto", "low", "medium", "high"];
const BANANA2_RATIOS = ["1:1", "2:3", "3:2", "3:4", "4:3", "4:5", "5:4", "9:16", "16:9", "21:9"];
const JL_CONTRACTS = {
    "nano-banana-2": { ratios: BANANA2_RATIOS, resolutions: ["1K", "2K", "4K"] },
    "nano-banana-2-2k": { ratios: BANANA2_RATIOS, resolutions: ["2K"] },
    "nano-banana-2-4k": { ratios: BANANA2_RATIOS, resolutions: ["4K"] },
    "nano-banana-pro": { ratios: BANANA2_RATIOS, resolutions: ["1K", "2K", "4K"] },
    "nano-banana-pro-2k": { ratios: BANANA2_RATIOS, resolutions: ["2K"] },
    "nano-banana-pro-4k": { ratios: BANANA2_RATIOS, resolutions: ["4K"] },
    "nano-banana-2.1": { ratios: ["1:1", "16:9", "9:16", "3:2", "2:3", "4:3", "3:4", "21:9"], resolutions: ["1K", "2K", "4K"] },
    "gpt-image-2": { ratios: ["1:1", "16:9", "9:16", "4:3", "3:4", "3:2", "2:3", "5:4", "4:5", "21:9"], resolutions: ["1K", "2K", "4K"] },
    "gpt-image-2-1k": { ratios: ["1:1", "16:9", "9:16", "4:3", "3:4"], resolutions: ["1K"] },
    "gpt-image-2.5-flare": { ratios: ["1:1", "16:9", "9:16", "4:3", "3:2", "3:4", "2:3"], resolutions: ["1K", "2K", "4K"] },
    "gpt-image-2.5-sunburst": { ratios: ["1:1", "16:9", "9:16", "5:4", "4:3", "3:2", "3:4"], resolutions: ["1K", "2K", "4K"] },
    "gpt-image-2.5-flare-1k": { ratios: ["1:1", "16:9", "9:16", "5:4", "4:3", "3:2", "4:5", "3:4"], resolutions: ["1K"] },
    "gpt-image-2.5-sunburst-1k": { ratios: ["1:1", "16:9", "9:16", "4:3", "21:9", "4:5", "3:4"], resolutions: ["1K"] },
    "gpt-image-2.5-1k": { ratios: ["1:1", "16:9", "9:16", "5:4", "4:3", "3:2", "21:9", "4:5", "3:4", "2:3"], resolutions: ["1K"] },
};
// V17.4: aliases that carry their own channel (优惠渠道2/高质渠道2/优惠渠道3/官方/官方-企业).
// They are listed on both channels, use the JL-style fixed controls and
// disable the 渠道 widget, which the backend ignores for them.
const FIXED_RATIOS = ["1:1", "16:9", "9:16", "3:2", "2:3", "4:3", "3:4", "5:4", "4:5", "21:9"];
const ALL_TIERS = ["1K", "2K", "4K"];
// 官方/官方-企业 price by quality and reject auto.
const OFFICIAL2_QUALITIES = ["low", "medium", "high"];
const OFFICIAL25_QUALITIES = ["low", "medium", "high", "xhigh", "max"];

const SPECS = {
    lingliu_nano_banana2_edit: {
        provider: "model_provider",
        model: "model",
        ratio: "aspect_ratio",
        resolution: "image_size",
        ratioAuto: true,
        models: {
            LL: [
                "gemini-3.1-flash-image-preview", "gemini-3.1-flash-image", "nano-banana-2", "nano-banana-2-2k", "nano-banana-2-4k", "nano-banana-2.1",
                "nano-banana-pro", "nano-banana-pro-2k", "nano-banana-pro-4k",
                "gemini-3.1-flash-image-preview-2k", "gemini-3.1-flash-image-preview-4k", "gemini-3.1-flash-image-2k", "gemini-3.1-flash-image-4k",
            ],
            // JL picks the tier with image_size, so the -2k/-4k aliases are LL only.
            JL: ["nano-banana-2", "nano-banana-2.1", "nano-banana-pro"],
        },
        llRatios: ["auto", "16:9", "4:3", "4:5", "3:2", "1:1", "2:3", "3:4", "5:4", "9:16", "21:9", "1:4", "4:1", "1:8", "8:1"],
        llResolutions: ["1K", "2K", "4K"],
        // -2k/-4k pick their tier from the model name; image_size is not sent.
        resolutionModels: ["nano-banana-2", "nano-banana-2.1", "nano-banana-pro"],
        // A JL node saved with a tier alias keeps its tier on the base model.
        jlAliases: {
            "nano-banana-2-2k": ["nano-banana-2", "2K"],
            "nano-banana-2-4k": ["nano-banana-2", "4K"],
            "nano-banana-pro-2k": ["nano-banana-pro", "2K"],
            "nano-banana-pro-4k": ["nano-banana-pro", "4K"],
        },
    },
    lingliu_gpt_image_2_official: {
        provider: "model_provider",
        model: "model",
        ratio: "aspect_ratio",
        resolution: "resolution",
        quality: "quality",
        ratioAuto: true,
        models: {
            LL: ["gpt-image-2", "gpt-image-2-all"],
            JL: ["gpt-image-2", "gpt-image-2-1k"],
        },
        llRatios: ["auto", "1:1", "16:9", "9:16", "4:3", "3:4", "3:2", "2:3", "5:4", "4:5", "21:9"],
        llResolutions: ["1K", "2K", "4K"],
        llQualities: ["auto", "high", "medium", "low"],
        jlFixed: { background: "auto", output_format: "png", moderation: "auto" },
        fixedModels: {
            "gpt-image-2-hq2": { ratios: FIXED_RATIOS, resolutions: ALL_TIERS },
            "gpt-image-2-hq3": { ratios: FIXED_RATIOS, resolutions: ALL_TIERS },
            "gpt-image-2-official": { ratios: FIXED_RATIOS, resolutions: ALL_TIERS, qualities: OFFICIAL2_QUALITIES },
            "gpt-image-2-enterprise": { ratios: FIXED_RATIOS, resolutions: ALL_TIERS, qualities: OFFICIAL2_QUALITIES },
        },
    },
    lingliu_gpt_image_25: {
        provider: null,
        model: "模型",
        ratio: "比例",
        resolution: "清晰度",
        quality: "质量",
        ratioAuto: false,
        models: {
            LL: ["gpt-image-2.5-flare", "gpt-image-2.5-sunburst"],
            JL: ["gpt-image-2.5-flare", "gpt-image-2.5-sunburst", "gpt-image-2.5-1k", "gpt-image-2.5-flare-1k", "gpt-image-2.5-sunburst-1k"],
        },
        llRatios: ["1:1", "16:9", "9:16", "3:2", "2:3", "4:3", "3:4"],
        llResolutions: ["1K", "2K", "4K"],
        llQualities: ["auto", "low", "medium", "high", "xhigh", "max"],
        jlFixed: { 输出格式: "png", 背景: "auto", 内容审核: "auto" },
        fixedModels: {
            "gpt-image-2.5-flare-all": { ratios: ["1:1", "16:9", "9:16", "4:3", "3:2", "21:9", "3:4", "2:3"], resolutions: ALL_TIERS },
            "gpt-image-2.5-sunburst-all": { ratios: ["1:1", "16:9", "9:16", "3:2", "4:3", "3:4", "2:3"], resolutions: ALL_TIERS },
            "gpt-image-2.5-flare-hq2": { ratios: FIXED_RATIOS, resolutions: ALL_TIERS },
            "gpt-image-2.5-sunburst-hq2": { ratios: FIXED_RATIOS, resolutions: ALL_TIERS },
            "gpt-image-2.5-flare-hq3": { ratios: FIXED_RATIOS, resolutions: ALL_TIERS },
            "gpt-image-2.5-sunburst-hq3": { ratios: FIXED_RATIOS, resolutions: ALL_TIERS },
            "gpt-image-2.5-flare-official": { ratios: FIXED_RATIOS, resolutions: ALL_TIERS, qualities: OFFICIAL25_QUALITIES },
            "gpt-image-2.5-sunburst-official": { ratios: FIXED_RATIOS, resolutions: ALL_TIERS, qualities: OFFICIAL25_QUALITIES },
            "gpt-image-2.5-flare-enterprise": { ratios: FIXED_RATIOS, resolutions: ALL_TIERS, qualities: OFFICIAL25_QUALITIES },
            "gpt-image-2.5-sunburst-enterprise": { ratios: FIXED_RATIOS, resolutions: ALL_TIERS, qualities: OFFICIAL25_QUALITIES },
        },
    },
};

function widget(node, name) {
    return name ? node?.widgets?.find((item) => item?.name === name) || null : null;
}

function setValues(target, values, fallback) {
    if (!target || !values?.length) return;
    target.options = target.options || {};
    target.options.values = [...values];
    if (!values.includes(target.value)) {
        target.value = fallback !== undefined && values.includes(fallback) ? fallback : values[0];
    }
}

function channelApplies(node, spec) {
    if (!spec.provider) return true;
    const value = String(widget(node, spec.provider)?.value || "").trim();
    return CHANNEL_PROVIDERS.has(PROVIDER_ALIASES[value] || "");
}

function applyContract(node, spec) {
    const channelWidget = widget(node, CHANNEL_WIDGET);
    if (!channelWidget) return;
    const applies = channelApplies(node, spec);
    let channel = CHANNELS.includes(channelWidget.value) ? channelWidget.value : LEGACY_CHANNEL;
    if (!applies) channel = LEGACY_CHANNEL;
    if (channelWidget.value !== channel) channelWidget.value = channel;
    // Other lines have no channel choice: lock LL.
    channelWidget.disabled = !applies;
    if (!applies) {
        node.setDirtyCanvas?.(true, true);
        return;
    }

    const modelWidget = widget(node, spec.model);
    const alias = channel === "JL" ? spec.jlAliases?.[modelWidget?.value] : null;
    if (alias) {
        modelWidget.value = alias[0];
        const resolutionWidget = widget(node, spec.resolution);
        if (resolutionWidget) resolutionWidget.value = alias[1];
    }
    const fixedModels = spec.fixedModels || {};
    setValues(modelWidget, [...spec.models[channel], ...Object.keys(fixedModels)]);
    const model = modelWidget?.value;
    const fixed = fixedModels[model] || null;
    channelWidget.disabled = Boolean(fixed);
    const contract = fixed || (channel === "JL" ? JL_CONTRACTS[model] : null);

    if (contract) {
        // Fixed aliases always send an explicit size, so they never offer auto.
        setValues(widget(node, spec.ratio), spec.ratioAuto && !fixed ? ["auto", ...contract.ratios] : contract.ratios);
        const usesResolution = !spec.resolutionModels || spec.resolutionModels.includes(model);
        setValues(widget(node, spec.resolution), usesResolution ? contract.resolutions : spec.llResolutions);
        if (spec.quality) setValues(widget(node, spec.quality), contract.qualities || JL_QUALITIES);
        for (const [name, value] of Object.entries(spec.jlFixed || {})) setValues(widget(node, name), [value]);
    } else {
        setValues(widget(node, spec.ratio), spec.llRatios);
        setValues(widget(node, spec.resolution), spec.llResolutions);
        if (spec.quality && spec.llQualities) setValues(widget(node, spec.quality), spec.llQualities);
        for (const name of Object.keys(spec.jlFixed || {})) {
            const target = widget(node, name);
            if (target?.__lingliuOriginalValues) {
                target.options.values = [...target.__lingliuOriginalValues];
            }
        }
    }
    node.setDirtyCanvas?.(true, true);
}

function rememberOriginalValues(node, spec) {
    for (const name of Object.keys(spec.jlFixed || {})) {
        const target = widget(node, name);
        if (target && !target.__lingliuOriginalValues && Array.isArray(target.options?.values)) {
            target.__lingliuOriginalValues = [...target.options.values];
        }
    }
}

function hook(node, name, spec) {
    const target = widget(node, name);
    if (!target || target.__lingliuImageChannelHooked) return;
    const previous = target.callback;
    target.callback = function (...args) {
        const result = previous?.apply(this, args);
        applyContract(node, spec);
        return result;
    };
    target.__lingliuImageChannelHooked = true;
}

function install(node, spec) {
    if (!node) return;
    // lingliu_taskid.js resets the model list per line, then calls this.
    node.__lingliuImageChannelApply = () => applyContract(node, spec);
    rememberOriginalValues(node, spec);
    hook(node, CHANNEL_WIDGET, spec);
    hook(node, spec.model, spec);
    if (spec.provider) hook(node, spec.provider, spec);
    applyContract(node, spec);
}

// Workflows saved before V16.3 have no channel value: the slot holds the next
// widget's value (for example the voice widget's [false,true]) or nothing.
// Insert LL there and hand the shifted values back to the later widgets.
function migrateLegacyValues(node, info) {
    const values = info?.widgets_values;
    if (!Array.isArray(values)) return;
    const serializable = (node.widgets || []).filter((item) => item?.serialize !== false);
    const index = serializable.findIndex((item) => item?.name === CHANNEL_WIDGET);
    if (index < 0 || CHANNELS.includes(values[index])) return;
    if (index > values.length) {
        // Older layouts saved fewer values; the channel slot was never reached.
        serializable[index].value = LEGACY_CHANNEL;
        return;
    }
    // Shift in place: other extensions (for example the voice widget) read
    // info.widgets_values by position in their own onConfigure.
    values.splice(index, 0, LEGACY_CHANNEL);
    for (let position = index; position < serializable.length; position += 1) {
        if (position >= values.length) break;
        serializable[position].value = values[position];
    }
}

app.registerExtension({
    name: "comfyui_lingliu_api.image_channel_contract",
    beforeRegisterNodeDef(nodeType, nodeData) {
        const spec = SPECS[nodeData?.name];
        if (!spec) return;
        const created = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function (...args) {
            const result = created?.apply(this, args);
            const channelWidget = widget(this, CHANNEL_WIDGET);
            if (channelWidget && !CHANNELS.includes(channelWidget.value)) channelWidget.value = NEW_NODE_CHANNEL;
            install(this, spec);
            return result;
        };
        const configured = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function (info, ...rest) {
            migrateLegacyValues(this, info);
            const result = configured?.call(this, info, ...rest);
            install(this, spec);
            return result;
        };
    },
});
