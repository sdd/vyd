use vyd::caps::{
    FrameFormatRelation, FrameMemory, FramePortSpec, FramePortSpecs, ProcessorPortSpecs,
};
use vyd::frame::PixelFormat;

#[test]
fn alternatives_and_processor_relations_are_nonempty_and_deduplicated() {
    let gray = FramePortSpec::luma8_cpu_packed();
    let nv12 = FramePortSpec::new(PixelFormat::Nv12, FrameMemory::CpuPacked);
    assert!(FramePortSpecs::try_from_vec(Vec::new()).is_err());
    assert!(ProcessorPortSpecs::transforming(Vec::new()).is_err());

    let alternatives = FramePortSpecs::one(nv12)
        .with_alternative(gray)
        .with_alternative(nv12);
    assert_eq!(alternatives.as_slice(), &[nv12, gray]);

    let passthrough = ProcessorPortSpecs::passthrough(alternatives);
    assert_eq!(
        passthrough.relations(),
        &[
            FrameFormatRelation {
                input: nv12,
                output: nv12
            },
            FrameFormatRelation {
                input: gray,
                output: gray
            },
        ]
    );

    let transform = FrameFormatRelation {
        input: gray,
        output: nv12,
    };
    let relations = ProcessorPortSpecs::transforming(vec![transform, transform]).unwrap();
    assert_eq!(relations.relations(), &[transform]);
}
