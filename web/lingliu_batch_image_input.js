import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const NODE_CLASS = "lingliu_batch_image_input";
const IMAGE_LIST_WIDGET = "图片列表";
const SUMMARY_WIDGET = "已选择图片";
const PREVIEW_WIDGET = "图片预览";
const PREVIEW_HEIGHT = 174;
const PREVIEW_TILE_SIZE = 64;
const PREVIEW_TILE_GAP = 6;
const PREVIEW_MIN_WIDTH = 96;
const PREVIEW_DOM_MARGIN = 10;
const PREVIEW_CONTAINER_CHROME = 14;

function findWidget(node, name) {
	return node?.widgets?.find((widget) => widget?.name === name) || null;
}

function refreshNode(node) {
	requestAnimationFrame(() => {
		syncPreviewLayout(node);
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

function cssText(styles) {
	return Object.entries(styles)
		.map(([key, value]) => `${key}: ${value}`)
		.join("; ");
}

function createPreviewElement() {
	const container = document.createElement("div");
	container.className = "lingliu-batch-image-preview";
	container.style.cssText = cssText({
		width: "100%",
		maxWidth: "100%",
		minWidth: "0",
		height: `${PREVIEW_HEIGHT}px`,
		boxSizing: "border-box",
		display: "block",
		padding: "6px",
		border: "1px solid rgba(128, 128, 128, 0.35)",
		borderRadius: "6px",
		background: "rgba(0, 0, 0, 0.12)",
		overflowY: "auto",
		overflowX: "hidden",
	});
	return container;
}

function parseImageList(widget) {
	try {
		const value = String(widget?.value || "[]").trim() || "[]";
		const parsed = JSON.parse(value);
		return Array.isArray(parsed) ? parsed : [];
	} catch (_error) {
		return [];
	}
}

function setImageList(node, images) {
	const widget = findWidget(node, IMAGE_LIST_WIDGET);
	if (!widget) {
		return;
	}
	widget.value = JSON.stringify(images, null, 2);
	updateSummary(node, images);
	renderPreviewGrid(node, images);
	refreshNode(node);
}

function updateSummary(node, images = null) {
	const listWidget = findWidget(node, IMAGE_LIST_WIDGET);
	const summaryWidget = findWidget(node, SUMMARY_WIDGET);
	if (!summaryWidget) {
		return;
	}
	const items = images || parseImageList(listWidget);
	const names = items.map((item) => item?.original_name || item?.name).filter(Boolean);
	summaryWidget.value = names.length ? `已选择 ${names.length} 张` : "未选择图片";
}

function buildViewUrl(imageRef) {
	const params = new URLSearchParams();
	params.set("filename", imageRef?.name || imageRef?.filename || "");
	params.set("type", imageRef?.type || imageRef?.file_type || "input");
	params.set("subfolder", imageRef?.subfolder || "");
	params.set("preview", "webp");
	params.set("_", Date.now().toString());
	return api.apiURL(`/view?${params.toString()}`);
}

function renderEmptyPreview(container) {
	const empty = document.createElement("div");
	empty.textContent = "未选择图片";
	empty.style.cssText = cssText({
		width: "100%",
		height: "100%",
		display: "flex",
		alignItems: "center",
		justifyContent: "center",
		color: "rgba(180, 180, 180, 0.9)",
		fontSize: "12px",
	});
	container.appendChild(empty);
}

function renderPreviewGrid(node, images = null) {
	const container = node.__lingliuPreviewContainer;
	if (!container) {
		return;
	}
	constrainPreviewContainer(node, container);
	const listWidget = findWidget(node, IMAGE_LIST_WIDGET);
	const items = images || parseImageList(listWidget);
	container.replaceChildren();

	if (!items.length) {
		renderEmptyPreview(container);
		return;
	}

	const grid = document.createElement("div");
	grid.className = "lingliu-batch-image-preview-grid";
	grid.style.cssText = cssText({
		display: "grid",
		gridTemplateColumns: getPreviewColumnsCss(node),
		gridAutoRows: `${PREVIEW_TILE_SIZE}px`,
		gap: `${PREVIEW_TILE_GAP}px`,
		justifyContent: "flex-start",
		alignContent: "flex-start",
		alignItems: "flex-start",
		width: "100%",
		maxWidth: "100%",
		minWidth: "0",
		overflow: "visible",
	});

	for (const item of items) {
		const name = item?.original_name || item?.name || item?.filename || "图片";
		const tile = document.createElement("div");
		tile.title = name;
		tile.style.cssText = cssText({
			position: "relative",
			width: `${PREVIEW_TILE_SIZE}px`,
			height: `${PREVIEW_TILE_SIZE}px`,
			maxWidth: "100%",
			borderRadius: "5px",
			overflow: "hidden",
			background: "rgba(80, 80, 80, 0.35)",
			border: "1px solid rgba(128, 128, 128, 0.25)",
		});

		const img = document.createElement("img");
		img.alt = name;
		img.loading = "lazy";
		img.draggable = false;
		img.src = buildViewUrl(item);
		img.style.cssText = cssText({
			width: "100%",
			height: "100%",
			objectFit: "cover",
			display: "block",
		});

		const fallback = document.createElement("div");
		fallback.textContent = name;
		fallback.style.cssText = cssText({
			display: "none",
			width: "100%",
			height: "100%",
			boxSizing: "border-box",
			padding: "5px",
			alignItems: "center",
			justifyContent: "center",
			textAlign: "center",
			wordBreak: "break-all",
			fontSize: "10px",
			color: "rgba(235, 235, 235, 0.95)",
			background: "rgba(80, 40, 40, 0.45)",
		});

		img.addEventListener("error", () => {
			img.style.display = "none";
			fallback.style.display = "flex";
		});

		tile.append(img, fallback);
		grid.appendChild(tile);
	}

	container.appendChild(grid);
	syncPreviewLayout(node);
}

function getPreviewWidth(node) {
	const nodeWidth = Number(node?.size?.[0]) || PREVIEW_MIN_WIDTH;
	return Math.max(PREVIEW_MIN_WIDTH, Math.floor(nodeWidth - PREVIEW_DOM_MARGIN * 2 - PREVIEW_CONTAINER_CHROME));
}

function getPreviewColumnCount(node) {
	const width = getPreviewWidth(node);
	const stride = PREVIEW_TILE_SIZE + PREVIEW_TILE_GAP;
	return Math.max(1, Math.floor((width + PREVIEW_TILE_GAP) / stride));
}

function getPreviewColumnsCss(node) {
	return `repeat(${getPreviewColumnCount(node)}, ${PREVIEW_TILE_SIZE}px)`;
}

function syncPreviewLayout(node) {
	const container = node?.__lingliuPreviewContainer;
	if (!container) {
		return;
	}
	container.style.width = "100%";
	container.style.maxWidth = "100%";
	container.style.minWidth = "0";
	container.style.overflowX = "hidden";

	const parent = container.parentElement;
	if (parent) {
		parent.style.width = "100%";
		parent.style.maxWidth = "100%";
		parent.style.minWidth = "0";
		parent.style.overflow = "hidden";
	}

	const grid = container.querySelector(".lingliu-batch-image-preview-grid");
	if (grid) {
		grid.style.gridTemplateColumns = getPreviewColumnsCss(node);
	}
}

function applyPreviewWidth(node) {
	syncPreviewLayout(node);
}

function constrainPreviewContainer(node, container) {
	syncPreviewLayout(node);
	container.style.minWidth = "0";
	container.style.overflowX = "hidden";

	const parent = container.parentElement;
	if (parent) {
		parent.style.minWidth = "0";
		parent.style.overflow = "hidden";
	}
}

function hideImageListWidget(node) {
	const widget = findWidget(node, IMAGE_LIST_WIDGET);
	if (!widget) {
		return;
	}
	widget.hidden = true;
	widget.computeSize = () => [0, -4];
	widget.inputEl?.style?.setProperty("display", "none");
}

async function uploadImageFile(file) {
	const body = new FormData();
	body.append("image", file, file.name);
	body.append("type", "input");
	body.append("subfolder", "");
	body.append("overwrite", "false");

	const response = await api.fetchApi("/upload/image", {
		method: "POST",
		body,
	});
	if (!response.ok) {
		throw new Error(`上传失败：${file.name}，HTTP ${response.status}`);
	}
	const payload = await response.json();
	return {
		name: payload.name,
		subfolder: payload.subfolder || "",
		type: payload.type || "input",
		original_name: file.name,
	};
}

async function uploadFilesToNode(node, fileList) {
	const files = Array.from(fileList || []).filter((file) => file?.type?.startsWith("image/"));
	if (!files.length) {
		alert("请拖入或选择图片文件。");
		return;
	}

	const listWidget = findWidget(node, IMAGE_LIST_WIDGET);
	const images = parseImageList(listWidget);
	const failed = [];

	for (const file of files) {
		try {
			images.push(await uploadImageFile(file));
		} catch (error) {
			console.error("[lingliu] batch image upload failed", error);
			failed.push(error?.message || file.name);
		}
	}

	setImageList(node, images);
	if (failed.length) {
		alert(`部分图片上传失败：\n${failed.join("\n")}`);
	}
}

function openFilePicker(node) {
	const input = document.createElement("input");
	input.type = "file";
	input.accept = "image/*";
	input.multiple = true;
	input.style.display = "none";
	input.addEventListener("change", async () => {
		try {
			await uploadFilesToNode(node, input.files);
		} finally {
			input.remove();
		}
	});
	document.body.appendChild(input);
	input.click();
}

function setupBatchImageInputNode(node) {
	if (!node || node.__lingliuBatchImageInputReady) {
		return;
	}
	node.__lingliuBatchImageInputReady = true;

	const listWidget = findWidget(node, IMAGE_LIST_WIDGET);
	if (listWidget) {
		listWidget.inputEl?.setAttribute?.("spellcheck", "false");
		hideImageListWidget(node);
	}

	const summaryWidget = node.addWidget("text", SUMMARY_WIDGET, "0 张", () => {}, {});
	summaryWidget.serialize = false;
	summaryWidget.disabled = true;

	const button = node.addWidget("button", "加载图像", "加载图像", () => openFilePicker(node));
	button.serialize = false;

	if (typeof node.addDOMWidget === "function") {
		const previewContainer = createPreviewElement();
		node.__lingliuPreviewContainer = previewContainer;
		const previewWidget = node.addDOMWidget(PREVIEW_WIDGET, "lingliu-batch-image-preview", previewContainer, {
			margin: PREVIEW_DOM_MARGIN,
			hideOnZoom: false,
			getMinWidth: () => PREVIEW_MIN_WIDTH,
			getMinHeight: () => PREVIEW_HEIGHT,
			getMaxHeight: () => PREVIEW_HEIGHT,
			getHeight: () => PREVIEW_HEIGHT,
			afterResize: () => syncPreviewLayout(node),
			onDraw: () => syncPreviewLayout(node),
			serialize: false,
		});
		previewWidget.serialize = false;
		syncPreviewLayout(node);
	} else {
		console.warn("[lingliu] addDOMWidget is not available; batch image preview is disabled.");
	}

	const originalOnResize = node.onResize;
	node.onResize = function (size) {
		const result = originalOnResize?.apply(this, arguments);
		if (Array.isArray(size)) {
			this.size = size;
		}
		applyPreviewWidth(this);
		app.graph.setDirtyCanvas(true, false);
		return result;
	};

	const originalOnDragOver = node.onDragOver;
	node.onDragOver = function (event) {
		if (event?.dataTransfer?.types?.includes("Files")) {
			event.preventDefault();
			return true;
		}
		return originalOnDragOver?.apply(this, arguments);
	};

	const originalOnDrop = node.onDropFile;
	node.onDropFile = async function (file) {
		if (file?.type?.startsWith("image/")) {
			await uploadFilesToNode(node, [file]);
			return true;
		}
		return originalOnDrop?.apply(this, arguments);
	};

	const originalOnDropItem = node.onDropItem;
	node.onDropItem = async function (event) {
		const files = event?.dataTransfer?.files;
		if (files?.length) {
			event.preventDefault();
			await uploadFilesToNode(node, files);
			return true;
		}
		return originalOnDropItem?.apply(this, arguments);
	};

	updateSummary(node);
	renderPreviewGrid(node);
	refreshNode(node);
}

app.registerExtension({
	name: "lingliu.batch_image_input",
	beforeRegisterNodeDef(nodeType, nodeData) {
		if (nodeData?.name !== NODE_CLASS) {
			return;
		}

		const onNodeCreated = nodeType.prototype.onNodeCreated;
		nodeType.prototype.onNodeCreated = function () {
			const result = onNodeCreated?.apply(this, arguments);
			setupBatchImageInputNode(this);
			return result;
		};

		const onConfigure = nodeType.prototype.onConfigure;
		nodeType.prototype.onConfigure = function () {
			const result = onConfigure?.apply(this, arguments);
			setupBatchImageInputNode(this);
			hideImageListWidget(this);
			updateSummary(this);
			renderPreviewGrid(this);
			return result;
		};
	},
});
