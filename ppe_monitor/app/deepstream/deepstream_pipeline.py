"""DeepStream pipeline runner (Phase 1)."""

from __future__ import annotations

import json
import queue
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional

import cv2
import numpy as np

from .config_builder import DeepStreamSettings
from .engine_utils import validate_engine_exists
from .metadata_adapter import AdaptedDeepStreamFrame, DsFrameMeta, DsObjectMeta, adapt_frame


def _log_event(event_type: str, **fields: object) -> None:
    print(json.dumps({"event_type": event_type, **fields}, default=str))


class DeepStreamUnavailableError(RuntimeError):
    """Raised when DeepStream Python dependencies are unavailable."""


def import_deepstream_modules():
    """Import DeepStream Python modules lazily."""
    import gi

    gi.require_version("Gst", "1.0")
    from gi.repository import GLib, Gst
    import pyds

    return Gst, GLib, pyds


@dataclass
class DeepStreamFrameBundle:
    frame_id: int
    source_id: int
    camera_id: str
    timestamp_s: float
    adapted: AdaptedDeepStreamFrame
    frame_bgr: Optional[np.ndarray] = None
    input_fps: float = 0.0
    camera_fps_map: Dict[str, float] | None = None
    primary_infer_latency_ms: float = 0.0
    tracker_latency_ms: float = 0.0
    end_to_end_latency_ms: float = 0.0


class DeepStreamPipelineRunner:
    """Owns a single-source DeepStream pipeline and emits frame metadata bundles."""

    def __init__(
        self,
        *,
        settings: DeepStreamSettings,
        label_map: Mapping[int, str],
        person_classes: Iterable[str],
        ppe_classes: Iterable[str],
        alias_to_canonical: Mapping[str, str],
    ) -> None:
        self.settings = settings
        self.label_map = dict(label_map)
        self.person_classes = set(str(v).strip().lower() for v in person_classes)
        self.ppe_classes = set(str(v).strip().lower() for v in ppe_classes)
        self.alias_to_canonical = dict(alias_to_canonical)
        self._q: "queue.Queue[DeepStreamFrameBundle]" = queue.Queue(
            maxsize=max(20, int(settings.appsink_max_buffers) * 8)
        )
        self._drop_count = 0
        self._running = False
        self._main_loop = None
        self._loop_thread: Optional[threading.Thread] = None
        self._pipeline = None
        self._bus = None
        self._Gst = None
        self._GLib = None
        self._pyds = None
        self._source_last_ts: Dict[int, float] = {}
        self._source_last_frame: Dict[int, int] = {}
        self._source_input_fps: Dict[int, float] = {}

    @property
    def dropped(self) -> int:
        return int(self._drop_count)

    def start(self) -> None:
        if self._running:
            return
        self._import_deepstream_dependencies()
        validate_engine_exists(self.settings.engine_path)
        self._ensure_config_exists(self.settings.gie_config, "nvinfer config")
        self._ensure_config_exists(self.settings.tracker_config, "tracker config")
        self._effective_gie_config = self._materialize_gie_config()

        self._build_pipeline()
        assert self._Gst is not None
        assert self._pipeline is not None
        assert self._GLib is not None

        self._main_loop = self._GLib.MainLoop()
        self._loop_thread = threading.Thread(
            target=self._main_loop.run,
            name="deepstream-mainloop",
            daemon=True,
        )
        self._loop_thread.start()

        ret = self._pipeline.set_state(self._Gst.State.PLAYING)
        if ret == self._Gst.StateChangeReturn.FAILURE:
            self.stop()
            raise RuntimeError("Failed to set DeepStream pipeline to PLAYING state")
        self._running = True
        _log_event(
            "deepstream_pipeline_started",
            source_uris=self.settings.source_uris,
            camera_ids=self.settings.camera_ids,
            gie_config=str(getattr(self, "_effective_gie_config", self.settings.gie_config).as_posix()),
            tracker_config=self.settings.tracker_config.as_posix(),
            engine_path=self.settings.engine_path.as_posix(),
        )

    def stop(self) -> None:
        if self._pipeline is not None and self._Gst is not None:
            try:
                self._pipeline.set_state(self._Gst.State.NULL)
            except Exception:
                pass
        if self._main_loop is not None:
            try:
                self._main_loop.quit()
            except Exception:
                pass
        if self._loop_thread is not None:
            self._loop_thread.join(timeout=2.0)
        self._running = False

    def read_bundle(self, timeout_seconds: float = 1.0) -> Optional[DeepStreamFrameBundle]:
        if not self._running:
            return None
        try:
            return self._q.get(timeout=max(0.01, float(timeout_seconds)))
        except queue.Empty:
            return None

    def _import_deepstream_dependencies(self) -> None:
        try:
            Gst, GLib, pyds = import_deepstream_modules()
        except Exception as exc:
            raise DeepStreamUnavailableError(
                "DeepStream backend selected but dependencies are missing.\n"
                "Please follow docs/deepstream_jetson_setup.md and ensure `pyds`, GStreamer, and DeepStream plugins are installed."
            ) from exc
        self._Gst = Gst
        self._GLib = GLib
        self._pyds = pyds
        self._Gst.init(None)

    def _ensure_config_exists(self, path: Path, label: str) -> None:
        if not path.exists():
            raise FileNotFoundError(f"DeepStream {label} not found: {path.as_posix()}")

    def _materialize_gie_config(self) -> Path:
        src = Path(self.settings.gie_config)
        lines = src.read_text(encoding="utf-8").splitlines()
        out_lines: List[str] = []
        found_engine = False
        for line in lines:
            stripped = line.strip()
            if stripped.startswith("model-engine-file="):
                out_lines.append(f"model-engine-file={self.settings.engine_path.as_posix()}")
                found_engine = True
                continue
            out_lines.append(line)
        if not found_engine:
            out_lines.append(f"model-engine-file={self.settings.engine_path.as_posix()}")
        runtime_cfg = src.with_name(f"{src.stem}.runtime.txt")
        runtime_cfg.write_text("\n".join(out_lines) + "\n", encoding="utf-8")
        return runtime_cfg

    def _build_pipeline(self) -> None:
        Gst = self._Gst
        assert Gst is not None
        self._pipeline = Gst.Pipeline.new("ppe-monitor-deepstream")
        if self._pipeline is None:
            raise RuntimeError("Unable to create DeepStream Gst.Pipeline")

        streammux = Gst.ElementFactory.make("nvstreammux", "stream-muxer")
        pgie = Gst.ElementFactory.make("nvinfer", "primary-inference")
        tracker = Gst.ElementFactory.make("nvtracker", "tracker")
        convert = Gst.ElementFactory.make("nvvideoconvert", "convert-rgba")
        capsfilter = Gst.ElementFactory.make("capsfilter", "caps-rgba")
        queue_after_tracker = Gst.ElementFactory.make("queue", "queue-after-tracker")
        appsink = Gst.ElementFactory.make("appsink", "app-sink")

        required = {
            "nvstreammux": streammux,
            "nvinfer": pgie,
            "nvtracker": tracker,
            "nvvideoconvert": convert,
            "capsfilter": capsfilter,
            "queue": queue_after_tracker,
            "appsink": appsink,
        }
        missing = [name for name, elem in required.items() if elem is None]
        if missing:
            raise RuntimeError(
                "Failed to create DeepStream elements: "
                + ", ".join(missing)
                + ". Verify DeepStream 7.1 plugins are installed and in GST_PLUGIN_PATH."
            )

        streammux.set_property("batch-size", int(self.settings.batch_size))
        streammux.set_property("width", int(self.settings.width))
        streammux.set_property("height", int(self.settings.height))
        streammux.set_property("batched-push-timeout", int(40000))
        streammux.set_property("live-source", 1 if self.settings.live_source else 0)

        gie_cfg = getattr(self, "_effective_gie_config", self.settings.gie_config)
        pgie.set_property("config-file-path", gie_cfg.as_posix())

        tracker.set_property("ll-lib-file", str(self.settings.tracker_ll_lib_file))
        tracker.set_property("ll-config-file", self.settings.tracker_config.as_posix())

        capsfilter.set_property("caps", Gst.Caps.from_string("video/x-raw(memory:NVMM), format=RGBA"))
        appsink.set_property("emit-signals", True)
        appsink.set_property("sync", False)
        appsink.set_property("max-buffers", int(self.settings.appsink_max_buffers))
        appsink.set_property("drop", True)
        appsink.connect("new-sample", self._on_new_sample)

        source_bins = []
        for idx, src in enumerate(self.settings.source_uris):
            source_bin = self._create_source_bin(idx, self._normalize_uri(src))
            source_bins.append(source_bin)
            self._pipeline.add(source_bin)
        self._pipeline.add(streammux)
        self._pipeline.add(pgie)
        self._pipeline.add(tracker)
        self._pipeline.add(queue_after_tracker)
        self._pipeline.add(convert)
        self._pipeline.add(capsfilter)
        self._pipeline.add(appsink)

        for idx, source_bin in enumerate(source_bins):
            sinkpad = streammux.get_request_pad(f"sink_{idx}")
            if sinkpad is None:
                raise RuntimeError(f"Unable to request sink_{idx} pad from nvstreammux")
            srcpad = source_bin.get_static_pad("src")
            if srcpad is None:
                raise RuntimeError(f"Unable to get source bin src pad for source {idx}")
            if srcpad.link(sinkpad) != Gst.PadLinkReturn.OK:
                raise RuntimeError(f"Failed to link source bin {idx} to nvstreammux")

        if not streammux.link(pgie):
            raise RuntimeError("Failed to link nvstreammux -> nvinfer")
        if not pgie.link(tracker):
            raise RuntimeError("Failed to link nvinfer -> nvtracker")
        if not tracker.link(queue_after_tracker):
            raise RuntimeError("Failed to link nvtracker -> queue")
        if not queue_after_tracker.link(convert):
            raise RuntimeError("Failed to link queue -> nvvideoconvert")
        if not convert.link(capsfilter):
            raise RuntimeError("Failed to link nvvideoconvert -> capsfilter")
        if not capsfilter.link(appsink):
            raise RuntimeError("Failed to link capsfilter -> appsink")

        tracker_src = tracker.get_static_pad("src")
        if tracker_src is None:
            raise RuntimeError("Unable to get nvtracker src pad for metadata probe")
        tracker_src.add_probe(Gst.PadProbeType.BUFFER, self._on_tracker_buffer_probe, None)

        self._bus = self._pipeline.get_bus()
        if self._bus is not None:
            self._bus.add_signal_watch()
            self._bus.connect("message", self._on_bus_message)

    def _normalize_uri(self, source_uri: str) -> str:
        raw = str(source_uri).strip()
        if "://" in raw:
            return raw
        path = Path(raw)
        if not path.is_absolute():
            path = path.resolve()
        return path.as_uri()

    def _camera_id_for_source(self, source_id: int) -> str:
        if 0 <= source_id < len(self.settings.camera_ids):
            return str(self.settings.camera_ids[source_id])
        return f"camera_{source_id}"

    def _create_source_bin(self, index: int, uri: str):
        Gst = self._Gst
        assert Gst is not None
        bin_name = f"source-bin-{index:02d}"
        source_bin = Gst.Bin.new(bin_name)
        if source_bin is None:
            raise RuntimeError(f"Unable to create source bin: {bin_name}")

        uri_decode_bin = Gst.ElementFactory.make("uridecodebin", f"uri-decode-bin-{index:02d}")
        if uri_decode_bin is None:
            raise RuntimeError("Unable to create uridecodebin element")
        uri_decode_bin.set_property("uri", uri)
        uri_decode_bin.connect("pad-added", self._on_decodebin_pad_added, source_bin)
        uri_decode_bin.connect("child-added", self._on_decodebin_child_added, source_bin)
        source_bin.add(uri_decode_bin)

        ghost_pad = Gst.GhostPad.new_no_target("src", Gst.PadDirection.SRC)
        if ghost_pad is None:
            raise RuntimeError("Unable to create ghost pad for source bin")
        source_bin.add_pad(ghost_pad)
        return source_bin

    def _on_decodebin_child_added(self, child_proxy, obj, name: str, user_data) -> None:
        # Keep this hook for parity with NVIDIA sample apps; no-op in Phase 1.
        _ = child_proxy
        _ = obj
        _ = name
        _ = user_data

    def _on_decodebin_pad_added(self, decodebin, decoder_src_pad, source_bin) -> None:
        Gst = self._Gst
        assert Gst is not None
        caps = decoder_src_pad.get_current_caps()
        if caps is None:
            return
        features = caps.get_features(0)
        if features is None or not features.contains("memory:NVMM"):
            return
        ghost_pad = source_bin.get_static_pad("src")
        if ghost_pad is None:
            return
        if not ghost_pad.set_target(decoder_src_pad):
            raise RuntimeError("Failed to link decoder src pad to source bin ghost pad")

    def _on_tracker_buffer_probe(self, pad, info, user_data):
        Gst = self._Gst
        pyds = self._pyds
        if Gst is None or pyds is None:
            return 0
        gst_buffer = info.get_buffer()
        if gst_buffer is None:
            return Gst.PadProbeReturn.OK

        batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buffer))
        if batch_meta is None:
            return Gst.PadProbeReturn.OK

        l_frame = batch_meta.frame_meta_list
        while l_frame is not None:
            try:
                frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
            except StopIteration:
                break

            objects: List[DsObjectMeta] = []
            l_obj = frame_meta.obj_meta_list
            while l_obj is not None:
                try:
                    obj_meta = pyds.NvDsObjectMeta.cast(l_obj.data)
                except StopIteration:
                    break
                rect = obj_meta.rect_params
                objects.append(
                    DsObjectMeta(
                        class_id=int(obj_meta.class_id),
                        class_label=str(obj_meta.obj_label),
                        confidence=float(obj_meta.confidence),
                        bbox=(
                            float(rect.left),
                            float(rect.top),
                            float(rect.width),
                            float(rect.height),
                        ),
                        object_id=int(obj_meta.object_id),
                        source_id=int(frame_meta.pad_index),
                    )
                )
                try:
                    l_obj = l_obj.next
                except StopIteration:
                    break

            ds_frame = DsFrameMeta(
                frame_num=int(frame_meta.frame_num),
                source_id=int(frame_meta.pad_index),
                pts_ns=int(frame_meta.buf_pts),
                objects=objects,
            )
            adapted = adapt_frame(
                ds_frame,
                camera_id=self._camera_id_for_source(int(frame_meta.pad_index)),
                label_map=self.label_map,
                person_classes=self.person_classes,
                ppe_classes=self.ppe_classes,
                alias_to_canonical=self.alias_to_canonical,
            )

            frame_bgr = None
            # Phase 1 Decision 1b keeps pose/compliance in Python, so frame access is required.
            needs_frame = True
            if needs_frame:
                try:
                    n_frame = pyds.get_nvds_buf_surface(hash(gst_buffer), frame_meta.batch_id)
                    frame_copy = np.array(n_frame, copy=True, order="C")
                    if frame_copy.ndim == 3 and frame_copy.shape[2] == 4:
                        frame_bgr = cv2.cvtColor(frame_copy, cv2.COLOR_RGBA2BGR)
                    elif frame_copy.ndim == 3 and frame_copy.shape[2] == 3:
                        frame_bgr = cv2.cvtColor(frame_copy, cv2.COLOR_RGB2BGR)
                except Exception:
                    frame_bgr = None

            now = time.time()
            source_id = int(adapted.source_id)
            prev_ts = self._source_last_ts.get(source_id)
            prev_frame = self._source_last_frame.get(source_id)
            input_fps = 0.0
            if prev_ts is not None and now > prev_ts:
                dt = now - prev_ts
                df = max(1, int(adapted.frame_id - prev_frame)) if prev_frame is not None else 1
                input_fps = float(df) / max(1e-6, dt)
            self._source_last_ts[source_id] = now
            self._source_last_frame[source_id] = int(adapted.frame_id)
            self._source_input_fps[source_id] = round(float(input_fps), 2)
            camera_fps_map = {
                self._camera_id_for_source(src_id): float(fps)
                for src_id, fps in self._source_input_fps.items()
            }

            bundle = DeepStreamFrameBundle(
                frame_id=int(adapted.frame_id),
                source_id=source_id,
                camera_id=adapted.camera_id,
                timestamp_s=float(adapted.timestamp_s),
                adapted=adapted,
                frame_bgr=frame_bgr,
                input_fps=round(float(input_fps), 2),
                camera_fps_map=camera_fps_map,
            )
            try:
                self._q.put_nowait(bundle)
            except queue.Full:
                self._drop_count += 1

            try:
                l_frame = l_frame.next
            except StopIteration:
                break

        return Gst.PadProbeReturn.OK

    def _on_new_sample(self, sink):
        sample = sink.emit("pull-sample")
        return self._Gst.FlowReturn.OK if sample is not None else self._Gst.FlowReturn.ERROR

    def _on_bus_message(self, bus, message) -> None:
        Gst = self._Gst
        if Gst is None:
            return
        mtype = message.type
        if mtype == Gst.MessageType.ERROR:
            err, debug = message.parse_error()
            _log_event("deepstream_bus_error", error=str(err), debug=str(debug or ""))
        elif mtype == Gst.MessageType.WARNING:
            err, debug = message.parse_warning()
            _log_event("deepstream_bus_warning", warning=str(err), debug=str(debug or ""))
        elif mtype == Gst.MessageType.EOS:
            _log_event("deepstream_bus_eos")
