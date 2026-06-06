from stt_providers.ondevice.runtime import pick_device, realtime_model_size

def test_pick_device_prefers_cuda_when_available():
    assert pick_device(cuda_available=lambda: True) == ("cuda", "float16")
    assert pick_device(cuda_available=lambda: False) == ("cpu", "int8")

def test_realtime_model_size_by_device():
    assert realtime_model_size("cuda") == "large-v3"
    assert realtime_model_size("cpu") == "small"   # CPU default (configurable)

def test_realtime_model_size_respects_override():
    assert realtime_model_size("cpu", override="medium") == "medium"
