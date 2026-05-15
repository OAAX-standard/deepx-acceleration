#pragma once

/*
 * YOLO post-processing for the DeepX inference_runner test binary.
 *
 * Supports three families, auto-detected from the model name:
 *
 *   YOLO8_11  — YOLOv8, YOLOv9, YOLOv10, YOLO11, YOLO12
 *               Output tensor: [4+C, N] or [1, 4+C, N]
 *               First 4 rows: cx, cy, w, h; remaining C rows: class scores.
 *               Requires greedy per-class NMS.
 *
 *   YOLO26    — YOLO26
 *               Output tensor: [N, 6] or [1, N, 6]
 *               Already decoded + NMS-filtered by the model graph.
 *               Each row: [x1, y1, x2, y2, confidence, class_id].
 *
 * Default thresholds match the reference nxai postprocessors:
 *   conf_threshold = 0.25, iou_threshold = 0.45, max_detections = 300
 */

#include "runtime_core.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace yolo {

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

enum class Family { YOLO8_11, YOLO26, UNKNOWN };

struct Detection {
    float x1, y1, x2, y2;
    float score;
    int   class_id;
};

// ---------------------------------------------------------------------------
// Infer YOLO family from model name (stem of the .dxnn filename)
// ---------------------------------------------------------------------------

inline Family detect_family(const std::string &model_name) {
    // Lowercase copy for case-insensitive matching
    std::string name = model_name;
    std::transform(name.begin(), name.end(), name.begin(), ::tolower);

    if (name.find("yolo26") != std::string::npos) return Family::YOLO26;
    if (name.find("yolo8")  != std::string::npos) return Family::YOLO8_11;
    if (name.find("yolo9")  != std::string::npos) return Family::YOLO8_11;
    if (name.find("yolo10") != std::string::npos) return Family::YOLO8_11;
    if (name.find("yolo11") != std::string::npos) return Family::YOLO8_11;
    if (name.find("yolo12") != std::string::npos) return Family::YOLO8_11;
    return Family::UNKNOWN;
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

namespace detail {

inline float iou(const Detection &a, const Detection &b) {
    float ix1 = std::max(a.x1, b.x1);
    float iy1 = std::max(a.y1, b.y1);
    float ix2 = std::min(a.x2, b.x2);
    float iy2 = std::min(a.y2, b.y2);

    float inter = std::max(0.0f, ix2 - ix1) * std::max(0.0f, iy2 - iy1);
    float area_a = (a.x2 - a.x1) * (a.y2 - a.y1);
    float area_b = (b.x2 - b.x1) * (b.y2 - b.y1);
    float uni = area_a + area_b - inter;
    return (uni <= 0.0f) ? 0.0f : inter / uni;
}

// Greedy per-class NMS (same algorithm as nxai reference postprocessors)
inline std::vector<Detection> nms(std::vector<Detection> boxes,
                                   float iou_threshold, int max_detections) {
    std::sort(boxes.begin(), boxes.end(),
              [](const Detection &a, const Detection &b) { return a.score > b.score; });

    std::vector<bool>      suppressed(boxes.size(), false);
    std::vector<Detection> result;
    result.reserve(std::min((int)boxes.size(), max_detections));

    for (size_t i = 0; i < boxes.size() && (int)result.size() < max_detections; ++i) {
        if (suppressed[i]) continue;
        result.push_back(boxes[i]);
        for (size_t j = i + 1; j < boxes.size(); ++j) {
            if (suppressed[j]) continue;
            if (boxes[j].class_id != boxes[i].class_id) continue;
            if (iou(boxes[i], boxes[j]) > iou_threshold)
                suppressed[j] = true;
        }
    }
    return result;
}

// Return float pointer for the first FLOAT32 tensor, or nullptr
inline const float *first_float_tensor(const Tensors *out, int &rank,
                                        const int *&shape) {
    if (!out || out->num_tensors == 0) return nullptr;
    const TensorDescriptor &td = out->tensors[0];
    if (td.data_type != DATA_TYPE_FLOAT) return nullptr;
    rank  = td.rank;
    shape = td.shape;
    return static_cast<const float *>(td.data);
}

} // namespace detail

// ---------------------------------------------------------------------------
// Decoders
// ---------------------------------------------------------------------------

// YOLOv8/11 family: tensor [4+C, N] or [1, 4+C, N]
inline std::vector<Detection> decode_yolo8_11(const Tensors *out,
                                               float conf_threshold,
                                               float iou_threshold,
                                               int   max_detections) {
    int         rank  = 0;
    const int  *shape = nullptr;
    const float *data = detail::first_float_tensor(out, rank, shape);
    if (!data) {
        fprintf(stderr, "[yolo8/11] output tensor is missing or not FLOAT32\n");
        return {};
    }

    size_t rows, n_anchors;
    if (rank == 3 && shape[0] == 1) {
        rows      = (size_t)shape[1];
        n_anchors = (size_t)shape[2];
    } else if (rank == 2) {
        rows      = (size_t)shape[0];
        n_anchors = (size_t)shape[1];
    } else {
        fprintf(stderr, "[yolo8/11] unexpected tensor rank %d\n", rank);
        return {};
    }

    if (rows < 5) {
        fprintf(stderr, "[yolo8/11] too few rows %zu (expected >= 5)\n", rows);
        return {};
    }

    size_t n_classes = rows - 4;
    std::vector<Detection> candidates;
    candidates.reserve(256);

    for (size_t i = 0; i < n_anchors; ++i) {
        float best_score = 0.0f;
        int   best_class = 0;
        for (size_t c = 0; c < n_classes; ++c) {
            float s = data[(4 + c) * n_anchors + i];
            if (s > best_score) { best_score = s; best_class = (int)c; }
        }
        if (best_score < conf_threshold) continue;

        float cx = data[0 * n_anchors + i];
        float cy = data[1 * n_anchors + i];
        float w  = data[2 * n_anchors + i];
        float h  = data[3 * n_anchors + i];
        candidates.push_back({cx - w / 2.0f, cy - h / 2.0f,
                               cx + w / 2.0f, cy + h / 2.0f,
                               best_score, best_class});
    }

    return detail::nms(candidates, iou_threshold, max_detections);
}

// YOLO26 family: tensor [N, 6] or [1, N, 6] — already decoded + NMS'd
inline std::vector<Detection> decode_yolo26(const Tensors *out,
                                             float conf_threshold,
                                             int   max_detections) {
    int         rank  = 0;
    const int  *shape = nullptr;
    const float *data = detail::first_float_tensor(out, rank, shape);
    if (!data) {
        fprintf(stderr, "[yolo26] output tensor is missing or not FLOAT32\n");
        return {};
    }

    size_t n_preds, row_size;
    if (rank == 3 && shape[0] == 1) {
        n_preds  = (size_t)shape[1];
        row_size = (size_t)shape[2];
    } else if (rank == 2) {
        n_preds  = (size_t)shape[0];
        row_size = (size_t)shape[1];
    } else {
        fprintf(stderr, "[yolo26] unexpected tensor rank %d\n", rank);
        return {};
    }

    if (row_size != 6) {
        fprintf(stderr, "[yolo26] expected row size 6, got %zu\n", row_size);
        return {};
    }

    std::vector<Detection> result;
    result.reserve(n_preds);

    for (size_t i = 0; i < n_preds && (int)result.size() < max_detections; ++i) {
        const float *row = data + i * 6;
        if (row[4] < conf_threshold) continue;
        result.push_back({row[0], row[1], row[2], row[3], row[4], (int)row[5]});
    }
    return result;
}

// ---------------------------------------------------------------------------
// Top-level entry point
// ---------------------------------------------------------------------------

inline std::vector<Detection> postprocess(const Tensors *out,
                                           Family family,
                                           float  conf_threshold  = 0.25f,
                                           float  iou_threshold   = 0.45f,
                                           int    max_detections  = 300) {
    switch (family) {
        case Family::YOLO8_11:
            return decode_yolo8_11(out, conf_threshold, iou_threshold, max_detections);
        case Family::YOLO26:
            return decode_yolo26(out, conf_threshold, max_detections);
        default:
            return {};
    }
}

// ---------------------------------------------------------------------------
// Logging helper — prints detections to stderr
// ---------------------------------------------------------------------------

inline void log_detections(const std::vector<Detection> &dets, const char *tag) {
    fprintf(stderr, "[%s] %zu detection(s) (conf>=0.25)\n", tag, dets.size());
    // Print at most 10 to keep logs readable
    size_t limit = std::min(dets.size(), (size_t)10);
    for (size_t i = 0; i < limit; ++i) {
        const Detection &d = dets[i];
        fprintf(stderr, "[%s]   [%zu] class=%d score=%.3f box=[%.1f,%.1f,%.1f,%.1f]\n",
                tag, i, d.class_id, d.score, d.x1, d.y1, d.x2, d.y2);
    }
    if (dets.size() > limit)
        fprintf(stderr, "[%s]   ... and %zu more\n", tag, dets.size() - limit);
}

} // namespace yolo
