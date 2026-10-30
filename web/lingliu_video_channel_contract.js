import { app } from "../../scripts/app.js";

const SPECS = {
    lingliu_wan3: {
        channels: ["LL", "XJ"],
        defaultChannel: "LL",
        ratios: { LL: ["adaptive", "16:9", "4:3", "1:1", "3:4", "9:16"], XJ: ["16:9", "4:3", "1:1", "3:4", "9:16"] },
        duration: { LL: [2, 30], XJ: [1, 30] },
    },
    lingliu_minimax_h3_frames: {
        channels: ["LL", "MT"],
        defaultChannel: "LL",
        ratios: { LL: ["adaptive", "21:9", "16:9", "4:3", "1:1", "3:4", "9:16"], MT: ["16:9", "9:16", "1:1", "4:3", "3:4", "21:9"] },
        duration: { LL: [4, 15], MT: [4, 15] },
        resolutions: { LL: ["2K", "768P"], MT: ["2K", "1080p", "768P", "480p"] },
        // MT 480p/768P run up to 30 s; 1080p/2K stay at 15 s.
        longDuration: { MT: { resolutions: ["480p", "768P"], max: 30 } },
        channelOnly: { 提示词优化: ["MT"] },
    },
    lingliu_minimax_h3_reference: {
        channels: ["LL", "MT"],
        defaultChannel: "LL",
        ratios: { LL: ["adaptive", "21:9", "16:9", "4:3", "1:1", "3:4", "9:16"], MT: ["16:9", "9:16", "1:1", "4:3", "3:4", "21:9"] },
        duration: { LL: [4, 15], MT: [4, 15] },
        resolutions: { LL: ["2K", "768P"], MT: ["2K", "1080p", "768P", "480p"] },
        // MT 480p/768P run up to 30 s; 1080p/2K stay at 15 s.
        longDuration: { MT: { resolutions: ["480p", "768P"], max: 30 } },
        channelOnly: { 提示词优化: ["MT"] },
    },
};

function widget(node, name) {
    return node?.widgets?.find((item) => item?.name === name);
}

function applyContract(node, spec) {
    const channelWidget = widget(node, "渠道");
    const channel = spec.channels.includes(channelWidget?.value) ? channelWidget.value : spec.defaultChannel;
    if (channelWidget && channelWidget.value !== channel) channelWidget.value = channel;
    const resolutionWidget = widget(node, "清晰度");
    if (resolutionWidget && spec.resolutions?.[channel]) {
        const values = spec.resolutions[channel];
        resolutionWidget.options = resolutionWidget.options || {};
        resolutionWidget.options.values = values;
        if (!values.includes(resolutionWidget.value)) resolutionWidget.value = values[0];
    }
    const durationWidget = widget(node, "时长");
    let range = spec.duration[channel];
    const long = spec.longDuration?.[channel];
    if (range && long && long.resolutions.includes(resolutionWidget?.value)) range = [range[0], long.max];
    if (durationWidget && range) {
        durationWidget.options = durationWidget.options || {};
        durationWidget.options.min = range[0];
        durationWidget.options.max = range[1];
        durationWidget.options.step = 1;
        if (spec.longDuration && Number(durationWidget.value) > range[1]) durationWidget.value = range[1];
    }
    for (const [name, channels] of Object.entries(spec.channelOnly || {})) {
        const item = widget(node, name);
        if (item) item.disabled = !channels.includes(channel);
    }
    const ratioWidget = widget(node, "比例");
    if (ratioWidget && spec.ratios[channel]) {
        const values = spec.ratios[channel];
        ratioWidget.options = ratioWidget.options || {};
        ratioWidget.options.values = values;
        if (!values.includes(ratioWidget.value)) ratioWidget.value = values[0];
    }
    node.setDirtyCanvas?.(true, true);
}

function install(node, spec) {
    if (!node || node.__lingliuVideoChannelContractInstalled) return;
    node.__lingliuVideoChannelContractInstalled = true;
    const watched = spec.resolutions ? ["渠道", "清晰度"] : ["渠道"];
    for (const name of watched) {
        const item = widget(node, name);
        if (!item) continue;
        const previous = item.callback;
        item.callback = function (...args) {
            const result = previous?.apply(this, args);
            applyContract(node, spec);
            return result;
        };
    }
    applyContract(node, spec);
}

app.registerExtension({
    name: "comfyui_lingliu_api.video_channel_contract",
    beforeRegisterNodeDef(nodeType, nodeData) {
        const spec = SPECS[nodeData?.name];
        if (!spec) return;
        const created = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function (...args) {
            const result = created?.apply(this, args);
            install(this, spec);
            return result;
        };
        const configured = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function (...args) {
            const result = configured?.apply(this, args);
            install(this, spec);
            applyContract(this, spec);
            return result;
        };
    },
});
