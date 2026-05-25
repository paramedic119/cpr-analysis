importScripts("https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.1/vision_bundle.js");

console.log("Worker Globals:", Object.keys(self).filter(k => !k.startsWith("webkit") && !k.startsWith("moz")));

const FilesetResolver = self.FilesetResolver || (self.vision && self.vision.FilesetResolver);
const PoseLandmarker = self.PoseLandmarker || (self.vision && self.vision.PoseLandmarker);

console.log("Detected FilesetResolver:", !!FilesetResolver);
console.log("Detected PoseLandmarker:", !!PoseLandmarker);

let poseLandmarker = null;
let currentModelType = null;
let visionInstance = null;

const MODEL_URLS = {
  lite: "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task",
  full: "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task"
};

// MediaPipe pose landmark indices used for CPR detection
// 11: left shoulder, 12: right shoulder, 13: left elbow, 14: right elbow
const TRACK_INDICES = [11, 12, 13, 14];
const MIN_VISIBILITY = 0.3;

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
  let buffer = await getCachedModel(type);
  if (!buffer) {
    self.postMessage({ type: "progress", message: `AIモデル(${type})をダウンロード中...` });
    const res = await fetch(MODEL_URLS[type]);
    if (!res.ok) throw new Error("モデルのダウンロードに失敗しました");
    buffer = await res.arrayBuffer();
    await cacheModel(type, buffer);
  } else {
    self.postMessage({ type: "progress", message: `AIモデル(${type})をローカルから展開中...` });
  }
  return new Uint8Array(buffer);
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

  if (!visionInstance) {
    self.postMessage({ type: "progress", message: "MediaPipe WASMを準備中..." });
    visionInstance = await FilesetResolver.forVisionTasks(
      "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.1/wasm"
    );
  }

  const buffer = await loadModel(modelType);
  currentModelType = modelType;

  self.postMessage({ type: "progress", message: `AIモデル(${modelType})を初期化中(GPU試行)...` });
  try {
    poseLandmarker = await PoseLandmarker.createFromOptions(visionInstance, {
      baseOptions: { modelAssetBuffer: buffer, delegate: "GPU" },
      runningMode: runningMode,
      numPoses: 1
    });
  } catch (gpuErr) {
    console.warn("GPU delegate failed, falling back to CPU", gpuErr);
    self.postMessage({ type: "progress", message: `AIモデル(${modelType})を初期化中(CPU)...` });
    poseLandmarker = await PoseLandmarker.createFromOptions(visionInstance, {
      baseOptions: { modelAssetBuffer: buffer, delegate: "CPU" },
      runningMode: runningMode,
      numPoses: 1
    });
  }
}

// Extract tracked landmarks and compute mean Y from visible ones
function extractTrackedSignal(results) {
  if (!results || !results.landmarks || results.landmarks.length === 0) {
    return { wy: null, points: [], visibleCount: 0 };
  }
  const lms = results.landmarks[0];
  const points = [];
  let ySum = 0;
  let visibleCount = 0;
  for (const idx of TRACK_INDICES) {
    const lm = lms[idx];
    if (!lm) {
      points.push(null);
      continue;
    }
    const vis = lm.visibility ?? 1;
    const point = { idx, x: lm.x, y: lm.y, visibility: vis };
    points.push(point);
    if (vis >= MIN_VISIBILITY) {
      ySum += lm.y;
      visibleCount++;
    }
  }
  const wy = visibleCount >= 2 ? ySum / visibleCount : null;
  return { wy, points, visibleCount };
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
      const sig = extractTrackedSignal(results);
      self.postMessage({
        type: "detect_result",
        payload: {
          wy: sig.wy,
          points: sig.points,
          visibleCount: sig.visibleCount,
          timestamp,
          frameId
        }
      });
    } catch (err) {
      self.postMessage({ type: "error", message: err.message });
    } finally {
      if (imageBitmap) imageBitmap.close();
    }
  }
};
