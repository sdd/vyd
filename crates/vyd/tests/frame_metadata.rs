use std::sync::Arc;
use vyd::frame_metadata::TypedFrameMetadata;
use vyd::{Frame, FrameContext};

#[derive(Debug, PartialEq, Eq)]
struct Label(&'static str);

#[test]
fn cloned_maps_share_values_but_replace_and_remove_independently() {
    let mut original = TypedFrameMetadata::new();
    original.insert(Label("original"));
    let retained = original.get_shared::<Label>().unwrap();
    let mut branch = original.clone();
    assert!(Arc::ptr_eq(
        &retained,
        &branch.get_shared::<Label>().unwrap()
    ));

    branch.insert(Label("branch"));
    assert_eq!(original.get::<Label>(), Some(&Label("original")));
    assert_eq!(branch.get::<Label>(), Some(&Label("branch")));
    assert_eq!(branch.remove::<Label>().unwrap().as_ref(), &Label("branch"));
    assert!(branch.get::<Label>().is_none());
    assert_eq!(original.get::<Label>(), Some(&Label("original")));
    assert_eq!(retained.as_ref(), &Label("original"));
}

#[test]
fn frame_context_exposes_typed_metadata_alongside_legacy_keys() {
    let mut frame = FrameContext::new(Frame::from_raw(1, 1, vec![7]).unwrap(), 3);
    frame.put("label", String::from("legacy"));
    frame.put_typed(Label("typed"));
    assert_eq!(frame.get_typed::<Label>(), Some(&Label("typed")));
    assert_eq!(frame.get_as::<String>("label"), "legacy");
}
