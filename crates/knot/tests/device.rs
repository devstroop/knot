//! SPEC §10: `KNOT_DEVICE` parsing and the cuda feature gate.

use knot::runtime::Device;

#[test]
fn parse_explicit_devices_only() {
    assert_eq!(Device::parse(None).unwrap(), Device::Cpu);
    assert_eq!(Device::parse(Some("cpu")).unwrap(), Device::Cpu);
    assert_eq!(Device::parse(Some("cuda")).unwrap(), Device::Cuda);
    // forgiving about case and stray whitespace, nothing else
    assert_eq!(Device::parse(Some(" CUDA ")).unwrap(), Device::Cuda);
    assert_eq!(Device::parse(Some("Cpu")).unwrap(), Device::Cpu);
    // no auto-selection: every device is a decision (SPEC §10)
    for raw in ["auto", "gpu", "cuda:0", "cpu0", ""] {
        let err = Device::parse(Some(raw)).unwrap_err().to_string();
        assert!(err.contains("KNOT_DEVICE"), "{raw}: {err}");
    }
}

#[test]
fn device_wire_values() {
    assert_eq!(Device::Cpu.as_str(), "cpu");
    assert_eq!(Device::Cuda.as_str(), "cuda");
}

#[cfg(all(feature = "onnx", not(feature = "cuda")))]
#[test]
fn cuda_without_the_feature_fails_before_touching_the_filesystem() {
    use knot::engine::Engine;
    use knot::router::Router;

    // The feature check runs before any path is read, so no checkpoint is
    // needed: a non-existent directory still produces the feature error.
    let err = match Engine::load_with_device(
        Router::new(),
        &[("english", std::path::Path::new("/nonexistent"))],
        Device::Cuda,
    ) {
        Ok(_) => panic!("cuda load without the cuda feature must fail"),
        Err(err) => err,
    };
    let msg = err.to_string();
    assert!(msg.contains("cuda"), "{msg}");
    assert!(msg.contains("features"), "{msg}");
}
