// Classic worker for MediaPipe pose detection.
//
// iPhone Safari blocks cross-origin importScripts for both unpkg and
// jsdelivr CDNs. To work around this, the main thread fetches
// vision_bundle.js via fetch() (CORS works fine in the page context),
// wraps the source in a Blob URL, and posts that URL to this worker.
// importScripts on a blob URL is treated as same-origin, so it loads.
//
// Until the main thread sends "load_mp", FilesetResolver/PoseLandmarker
// are unavailable and only "load_mp" messages are honored.

let mpLoaded = false;
// NOTE: do NOT declare FilesetResolver/PoseLandmarker here. The MediaPipe
// bundle, once stripped of its trailing ESM export and importScripts'd,
// declares its own top-level `class FilesetResolver` / `const PoseLandmarker`.
// Any local `let`/`const` of the same name would throw
// "Identifier already declared" inside importScripts. We read them off
// `self.vision` / global scope below after loading.
let MpFilesetResolver = null;
let MpPoseLandmarker = null;

const MP_VERSION = "0.10.1";
const MP_WASM_URLS = [
  `https://unpkg.com/@mediapipe/tasks-vision@${MP_VERSION}/wasm`,
  `https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@${MP_VERSION}/wasm`
];

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
    let res;
    try {
      res = await fetch(MODEL_URLS[type], { mode: "cors", credentials: "omit", redirect: "follow" });
    } catch (e) {
      throw new Error(`モデル取得失敗(${type}): ${e.message || e}`);
    }
    if (!res.ok) throw new Error(`モデル取得失敗(${type}): HTTP ${res.status}`);
    buffer = await res.arrayBuffer();
    await cacheModel(type, buffer);
  } else {
    self.postMessage({ type: "progress", message: `AIモデル(${type})をローカルから展開中...` });
  }
  return new Uint8Array(buffer);
}

async function loadVisionInstance() {
  let lastErr;
  for (const url of MP_WASM_URLS) {
    try {
      return await MpFilesetResolver.forVisionTasks(url);
    } catch (e) {
      lastErr = e;
    }
  }
  throw new Error(`WASM準備失敗: ${lastErr && lastErr.message || lastErr}`);
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
    visionInstance = await loadVisionInstance();
  }

  const buffer = await loadModel(modelType);
  currentModelType = modelType;

  self.postMessage({ type: "progress", message: `AIモデル(${modelType})を初期化中(GPU試行)...` });
  try {
    poseLandmarker = await MpPoseLandmarker.createFromOptions(visionInstance, {
      baseOptions: { modelAssetBuffer: buffer, delegate: "GPU" },
      runningMode: runningMode,
      numPoses: 1
    });
  } catch (gpuErr) {
    console.warn("GPU delegate failed, falling back to CPU", gpuErr);
    self.postMessage({ type: "progress", message: `AIモデル(${modelType})を初期化中(CPU)...` });
    poseLandmarker = await MpPoseLandmarker.createFromOptions(visionInstance, {
      baseOptions: { modelAssetBuffer: buffer, delegate: "CPU" },
      runningMode: runningMode,
      numPoses: 1
    });
  }
}

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

  if (type === "load_mp") {
    try {
      importScripts(payload.blobUrl);
      // Bundle either sets globals directly or assigns onto `self.vision`.
      const visionGlobal = self.vision || {};
      MpFilesetResolver = self.FilesetResolver || visionGlobal.FilesetResolver;
      MpPoseLandmarker = self.PoseLandmarker || visionGlobal.PoseLandmarker;
      if (!MpFilesetResolver || !MpPoseLandmarker) {
        throw new Error("MediaPipeのグローバルが見つかりません");
      }
      mpLoaded = true;
      self.postMessage({ type: "mp_loaded" });
    } catch (err) {
      self.postMessage({ type: "error", message: "MP読み込み失敗: " + (err.message || err) });
    }
    return;
  }

  if (!mpLoaded) {
    self.postMessage({ type: "error", message: "MediaPipeが未読み込みです" });
    return;
  }

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
