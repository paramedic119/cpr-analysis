importScripts("https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.1/vision_bundle.js");
const { FilesetResolver, PoseLandmarker } = vision;

let poseLandmarker = null;
let currentModelType = null;
let vision = null;

const MODEL_URLS = {
  lite: "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task",
  full: "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task"
};

const DB_NAME = "cpr-ai-models";
const STORE_NAME = "models";

async function getDB() {
  return new Promise((resolve, reject) => {
    const req = indexedDB.open(DB_NAME, 1);
    req.onupgradeneeded = (e) => {
      e.target.result.createObjectStore(STORE_NAME);
    };
    req.onsuccess = () => resolve(req.result);
    req.onerror = () => reject(req.error);
  });
}

async function getCachedModel(type) {
  const db = await getDB();
  return new Promise((resolve) => {
    const tx = db.transaction(STORE_NAME, "readonly");
    const store = tx.objectStore(STORE_NAME);
    const req = store.get(type);
    req.onsuccess = () => resolve(req.result);
    req.onerror = () => resolve(null);
  });
}

async function cacheModel(type, buffer) {
  const db = await getDB();
  return new Promise((resolve) => {
    const tx = db.transaction(STORE_NAME, "readwrite");
    const store = tx.objectStore(STORE_NAME);
    const req = store.put(buffer, type);
    req.onsuccess = () => resolve();
    req.onerror = () => resolve();
  });
}

async function loadModel(type) {
  console.log(`loadModel(${type}) start`);
  let buffer = await getCachedModel(type);
  if (!buffer) {
    console.log(`Model cache miss for ${type}`);
    self.postMessage({ type: "progress", message: `AIモデル(${type})をダウンロード中...` });
    const res = await fetch(MODEL_URLS[type]);
    if (!res.ok) throw new Error("モデルのダウンロードに失敗しました");
    buffer = await res.arrayBuffer();
    await cacheModel(type, buffer);
    console.log(`Model ${type} downloaded and cached`);
  } else {
    console.log(`Model cache hit for ${type}, size: ${buffer.byteLength}`);
    self.postMessage({ type: "progress", message: `AIモデル(${type})をローカルから展開中...` });
  }
  console.log(`Converting buffer to Uint8Array for ${type}`);
  const uint8 = new Uint8Array(buffer);
  console.log(`Conversion done for ${type}`);
  return uint8;
}

async function initPoseLandmarker(runningMode, modelType) {
  const needNewModel = !poseLandmarker || currentModelType !== modelType;

  if (poseLandmarker && !needNewModel) {
    await poseLandmarker.setOptions({ runningMode });
    return;
  }

  if (poseLandmarker) {
    poseLandmarker.close();
    poseLandmarker = null;
  }

  if (!vision) {
    vision = await FilesetResolver.forVisionTasks(
      "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.1/wasm"
    );
  }

  console.log("Loading model buffer...");
  const buffer = await loadModel(modelType);
  currentModelType = modelType;

  console.log(`Initializing PoseLandmarker for ${modelType} (runningMode: ${runningMode})`);
  try {
    console.log("Attempting GPU delegate...");
    poseLandmarker = await PoseLandmarker.createFromOptions(vision, {
      baseOptions: {
        modelAssetBuffer: buffer,
        delegate: "GPU"
      },
      runningMode: runningMode,
      numPoses: 1
    });
    console.log("GPU initialization successful");
  } catch (gpuErr) {
    console.warn("GPU delegate failed, falling back to CPU", gpuErr);
    console.log("Attempting CPU delegate...");
    poseLandmarker = await PoseLandmarker.createFromOptions(vision, {
      baseOptions: {
        modelAssetBuffer: buffer,
        delegate: "CPU"
      },
      runningMode: runningMode,
      numPoses: 1
    });
    console.log("CPU initialization successful");
  }
}

self.onmessage = async (e) => {
  const { type, payload } = e.data;

  if (type === "init") {
    try {
      await initPoseLandmarker(payload.mode, payload.modelType);
      self.postMessage({ type: "init_done" });
    } catch (err) {
      self.postMessage({ type: "error", message: err.message });
    }
  } else if (type === "detect") {
    if (!poseLandmarker) {
      self.postMessage({ type: "error", message: "モデルが初期化されていません" });
      return;
    }
    const { imageBitmap, timestamp, frameId } = payload;
    try {
      const results = poseLandmarker.detectForVideo(imageBitmap, timestamp);

      let wy = null;
      if (results && results.landmarks && results.landmarks.length > 0) {
        wy = results.landmarks[0][12].y; // RIGHT_SHOULDER
      }

      self.postMessage({
        type: "detect_result",
        payload: { wy, timestamp, frameId }
      });
    } catch (err) {
      self.postMessage({ type: "error", message: err.message });
    } finally {
      if (imageBitmap) {
        imageBitmap.close();
      }
    }
  }
};
