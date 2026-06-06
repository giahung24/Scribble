def test_packages_importable():
    import stt_providers
    import stt_providers.ondevice
    assert stt_providers.ondevice is not None
