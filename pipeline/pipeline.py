"""
DeepStream 7.1 vehicle detection + counting pipeline.
Usage: python3 pipeline/pipeline.py --input <video> --output <video> --engine <path>
"""

import argparse
import ctypes
import os
import sys

import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib
import numpy as np
import pyds

sys.path.insert(0, os.path.dirname(__file__))
from vehicle_counter import VehicleCounter

VEHICLE_CLASSES = {2, 3, 5, 7}
CONF_THRESH = 0.5
IOU_THRESH = 0.45
MUXER_W = 1920
MUXER_H = 1080


def _nms(boxes, scores, iou_thresh):
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
        iou = inter / (areas[i] + areas[order[1:]] - inter)
        order = order[1:][iou <= iou_thresh]
    return keep


def _decode_yolov8(raw, conf_thresh=CONF_THRESH, iou_thresh=IOU_THRESH):
    """
    raw: numpy array shape (84, 8400)
    YOLOv8 output: rows 0-3 = cx,cy,w,h; rows 4-83 = class scores.
    Returns list of dicts: {class_id, confidence, left, top, width, height}
    """
    pred = raw.T  # (8400, 84)
    scores = pred[:, 4:]
    class_ids = np.argmax(scores, axis=1)
    confidences = scores[np.arange(len(scores)), class_ids]

    mask = confidences > conf_thresh
    if not mask.any():
        return []

    pred = pred[mask]
    class_ids = class_ids[mask]
    confidences = confidences[mask]

    cx, cy, w, h = pred[:, 0], pred[:, 1], pred[:, 2], pred[:, 3]
    x1, y1 = cx - w / 2, cy - h / 2
    x2, y2 = cx + w / 2, cy + h / 2

    results = []
    for cls in np.unique(class_ids):
        m = class_ids == cls
        bboxes = np.stack([x1[m], y1[m], x2[m], y2[m]], axis=1)
        confs = confidences[m]
        for i in _nms(bboxes, confs, iou_thresh):
            results.append({
                'class_id': int(cls),
                'confidence': float(confs[i]),
                'left': float(bboxes[i, 0]),
                'top': float(bboxes[i, 1]),
                'width': float(bboxes[i, 2] - bboxes[i, 0]),
                'height': float(bboxes[i, 3] - bboxes[i, 1]),
            })
    return results


def _probe_callback(pad, info, counter):
    gst_buffer = info.get_buffer()
    if not gst_buffer:
        return Gst.PadProbeReturn.OK

    try:
        batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buffer))
        l_frame = batch_meta.frame_meta_list

        while l_frame:
            try:
                frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
            except StopIteration:
                break

            frame_num = frame_meta.frame_num
            detections = []

            l_user = frame_meta.frame_user_meta_list
            while l_user:
                try:
                    user_meta = pyds.NvDsUserMeta.cast(l_user.data)
                except StopIteration:
                    break

                if user_meta.base_meta.meta_type == pyds.NvDsMetaType.NVDSINFER_TENSOR_OUTPUT_META:
                    try:
                        tensor_meta = pyds.NvDsInferTensorMeta.cast(user_meta.user_meta_data)
                        layer = pyds.get_nvds_LayerInfo(tensor_meta, 0)
                        ptr = ctypes.cast(
                            pyds.get_ptr(layer.buffer),
                            ctypes.POINTER(ctypes.c_float)
                        )
                        raw = np.ctypeslib.as_array(ptr, shape=(84, 8400)).copy()
                        dets = _decode_yolov8(raw)

                        for det in dets:
                            if det['class_id'] not in VEHICLE_CLASSES:
                                continue
                            detections.append(det['class_id'])

                            obj_meta = pyds.nvds_acquire_obj_meta_from_pool(batch_meta)
                            rect = obj_meta.rect_params
                            rect.left = det['left']
                            rect.top = det['top']
                            rect.width = det['width']
                            rect.height = det['height']
                            rect.border_width = 3
                            rect.border_color.set(0.0, 1.0, 0.0, 1.0)
                            obj_meta.class_id = det['class_id']
                            obj_meta.confidence = det['confidence']
                            pyds.nvds_add_obj_meta_to_frame(frame_meta, obj_meta, None)
                    except Exception as e:
                        print(f"Tensor decode error frame {frame_num}: {e}", file=sys.stderr)

                try:
                    l_user = l_user.next
                except StopIteration:
                    break

            result = counter.update(frame_num, detections)
            f = result['frame']
            m = result['per_minute']

            display_meta = pyds.nvds_acquire_display_meta_from_pool(batch_meta)
            display_meta.num_labels = 1
            txt = display_meta.text_params[0]
            txt.display_text = (
                f"Frame {frame_num}  |  "
                f"Car:{f['car']} Truck:{f['truck']} Bus:{f['bus']} Moto:{f['motorcycle']}  |  "
                f"/min — Car:{m['car']} Truck:{m['truck']} Bus:{m['bus']} Moto:{m['motorcycle']}"
            )
            txt.x_offset = 10
            txt.y_offset = 30
            txt.font_params.font_name = "Serif"
            txt.font_params.font_size = 14
            txt.font_params.font_color.set(1.0, 1.0, 1.0, 1.0)
            txt.set_bg_clr = 1
            txt.text_bg_clr.set(0.0, 0.0, 0.0, 0.8)
            pyds.nvds_add_display_meta_to_frame(frame_meta, display_meta)

            try:
                l_frame = l_frame.next
            except StopIteration:
                break

    except Exception as e:
        print(f"Probe error: {e}", file=sys.stderr)

    return Gst.PadProbeReturn.OK


def _build_pipeline(input_path, output_path):
    pipeline = Gst.Pipeline()

    source = Gst.ElementFactory.make('filesrc', 'source')
    decoder = Gst.ElementFactory.make('decodebin', 'decoder')
    muxer = Gst.ElementFactory.make('nvstreammux', 'muxer')
    infer = Gst.ElementFactory.make('nvinfer', 'infer')
    osd = Gst.ElementFactory.make('nvdsosd', 'osd')
    nvconv = Gst.ElementFactory.make('nvvideoconvert', 'nvconv')
    conv = Gst.ElementFactory.make('videoconvert', 'conv')
    encoder = Gst.ElementFactory.make('theoraenc', 'encoder')
    muxout = Gst.ElementFactory.make('oggmux', 'muxout')
    sink = Gst.ElementFactory.make('filesink', 'sink')

    for name, el in [
        ('filesrc', source), ('decodebin', decoder), ('nvstreammux', muxer),
        ('nvinfer', infer), ('nvdsosd', osd),
        ('nvvideoconvert', nvconv), ('videoconvert', conv),
        ('theoraenc', encoder), ('oggmux', muxout), ('filesink', sink),
    ]:
        if not el:
            raise RuntimeError(f"Failed to create GStreamer element: {name}")
        pipeline.add(el)

    source.set_property('location', input_path)
    muxer.set_property('width', MUXER_W)
    muxer.set_property('height', MUXER_H)
    muxer.set_property('batch-size', 1)
    muxer.set_property('batched-push-timeout', 4000000)
    infer.set_property('config-file-path', '/app/pipeline/config/nvinfer_config.txt')
    sink.set_property('location', output_path)
    sink.set_property('sync', False)

    def on_pad_added(src, pad, mux):
        sink_pad = mux.get_request_pad('sink_0')
        if not sink_pad:
            print("ERROR: Could not get nvstreammux sink pad", file=sys.stderr)
            return
        if pad.link(sink_pad) != Gst.PadLinkReturn.OK:
            print("ERROR: Failed to link decoder → muxer", file=sys.stderr)

    decoder.connect('pad-added', on_pad_added, muxer)

    source.link(decoder)
    muxer.link(infer)
    infer.link(osd)
    osd.link(nvconv)
    nvconv.link(conv)
    conv.link(encoder)
    encoder.link(muxout)
    muxout.link(sink)

    return pipeline, infer


def main():
    ap = argparse.ArgumentParser(description='DeepStream vehicle detection pipeline')
    ap.add_argument('--input', required=True, help='Input video path')
    ap.add_argument('--output', default='/app/pipeline/output/out.ogv', help='Output video path')
    ap.add_argument('--engine', default='/models/yolov8n.engine', help='TensorRT engine path')
    ap.add_argument('--fps', type=float, default=30.0, help='Video FPS for per-minute window')
    args = ap.parse_args()

    if not os.path.exists(args.input):
        sys.exit(f"ERROR: Input not found: {args.input}")
    if not os.path.exists(args.engine):
        sys.exit(f"ERROR: Engine not found: {args.engine}")

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    Gst.init(None)
    counter = VehicleCounter(fps=args.fps)
    pipeline, infer_el = _build_pipeline(args.input, args.output)

    infer_src_pad = infer_el.get_static_pad('src')
    infer_src_pad.add_probe(Gst.PadProbeType.BUFFER, _probe_callback, counter)

    loop = GLib.MainLoop()
    bus = pipeline.get_bus()
    bus.add_signal_watch()

    def on_message(bus, msg, loop):
        if msg.type == Gst.MessageType.EOS:
            print(f"\nDone. Output: {args.output}")
            loop.quit()
        elif msg.type == Gst.MessageType.ERROR:
            err, debug = msg.parse_error()
            print(f"ERROR: {err}\n{debug}", file=sys.stderr)
            loop.quit()

    bus.connect('message', on_message, loop)
    pipeline.set_state(Gst.State.PLAYING)
    print(f"Pipeline running: {args.input} → {args.output}")

    try:
        loop.run()
    finally:
        pipeline.set_state(Gst.State.NULL)


if __name__ == '__main__':
    main()
