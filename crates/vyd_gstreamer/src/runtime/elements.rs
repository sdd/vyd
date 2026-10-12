use super::{invoke, Shared};
use gst::{glib, prelude::*, subclass::prelude::*};
use gst_base::{prelude::*, subclass::prelude::*};
use once_cell::sync::Lazy;
use std::sync::OnceLock;

fn templates(directions: &[gst::PadDirection]) -> Vec<gst::PadTemplate> {
    let caps = gst::Caps::builder("video/x-raw")
        .field("format", "GRAY8")
        .field("width", gst::IntRange::<i32>::new(1, i32::MAX))
        .field("height", gst::IntRange::<i32>::new(1, i32::MAX))
        .build();
    directions
        .iter()
        .map(|&direction| {
            gst::PadTemplate::new(
                if direction == gst::PadDirection::Src {
                    "src"
                } else {
                    "sink"
                },
                direction,
                gst::PadPresence::Always,
                &caps,
            )
            .unwrap()
        })
        .collect()
}

glib::wrapper! {
    pub struct Source(ObjectSubclass<source_imp::Source>) @extends gst_base::PushSrc, gst_base::BaseSrc, gst::Element, gst::Object;
}
glib::wrapper! {
    pub struct Processor(ObjectSubclass<processor_imp::Processor>) @extends gst::Element, gst::Object;
}
glib::wrapper! {
    pub struct Sink(ObjectSubclass<sink_imp::Sink>) @extends gst_base::BaseSink, gst::Element, gst::Object;
}

pub(super) fn source(shared: Shared) -> gst::Element {
    let element: Source = glib::Object::builder()
        .property("name", "vyd-source")
        .build();
    assert!(element.imp().shared.set(shared).is_ok());
    element.set_format(gst::Format::Time);
    element.upcast()
}
pub(super) fn processor(shared: Shared, start: usize, end: usize) -> gst::Element {
    let element: Processor = glib::Object::builder()
        .property("name", format!("vyd-processor-{start}"))
        .build();
    assert!(element.imp().config.set((shared, start, end)).is_ok());
    element.upcast()
}
pub(super) fn sink(shared: Shared) -> gst::Element {
    let element: Sink = glib::Object::builder().property("name", "vyd-sink").build();
    assert!(element.imp().shared.set(shared).is_ok());
    element.set_sync(false);
    element.set_async(false);
    element.upcast()
}

mod source_imp {
    use super::*;
    #[derive(Default)]
    pub struct Source {
        pub(super) shared: OnceLock<Shared>,
    }
    #[glib::object_subclass]
    impl ObjectSubclass for Source {
        const NAME: &'static str = "VydSerialSourceAdapter";
        type Type = super::Source;
        type ParentType = gst_base::PushSrc;
    }
    impl ObjectImpl for Source {}
    impl GstObjectImpl for Source {}
    impl ElementImpl for Source {
        fn metadata() -> Option<&'static gst::subclass::ElementMetadata> {
            static META: Lazy<gst::subclass::ElementMetadata> = Lazy::new(|| {
                gst::subclass::ElementMetadata::new(
                    "Vyd Rust source",
                    "Source/Video",
                    "Serial Rust source adapter",
                    "Vyd",
                )
            });
            Some(&META)
        }
        fn pad_templates() -> &'static [gst::PadTemplate] {
            static PADS: Lazy<Vec<gst::PadTemplate>> =
                Lazy::new(|| templates(&[gst::PadDirection::Src]));
            &PADS
        }
    }
    impl BaseSrcImpl for Source {
        fn negotiate(&self) -> Result<(), gst::LoggableError> {
            // Legacy sources reveal their concrete dimensions on the first
            // frame. create installs fixed caps before returning that buffer.
            Ok(())
        }
    }
    impl PushSrcImpl for Source {
        fn create(
            &self,
            _: Option<&mut gst::BufferRef>,
        ) -> Result<gst_base::subclass::base_src::CreateSuccess, gst::FlowError> {
            let packet = invoke(self.shared.get().unwrap(), |state| state.source())?;
            let Some((buffer, caps)) = packet else {
                return Err(gst::FlowError::Eos);
            };
            if self.obj().src_pad().current_caps().as_ref() != Some(&caps) {
                self.obj()
                    .set_caps(&caps)
                    .map_err(|_| gst::FlowError::NotNegotiated)?;
            }
            Ok(gst_base::subclass::base_src::CreateSuccess::NewBuffer(
                buffer,
            ))
        }
    }
}

mod processor_imp {
    use super::*;
    pub struct Processor {
        pub(super) config: OnceLock<(Shared, usize, usize)>,
        sinkpad: gst::Pad,
        srcpad: gst::Pad,
    }
    impl Processor {
        fn chain(&self, buffer: gst::Buffer) -> Result<gst::FlowSuccess, gst::FlowError> {
            let (shared, start, end) = self.config.get().unwrap();
            let output = invoke(shared, |state| {
                state.process(*start, *end, &self.sinkpad, &buffer)
            })?;
            // Never retain the session lock while pushing into another adapter.
            match output {
                Some(output) => self.srcpad.push(output),
                None => Ok(gst::FlowSuccess::Ok),
            }
        }
    }
    #[glib::object_subclass]
    impl ObjectSubclass for Processor {
        const NAME: &'static str = "VydSerialProcessorAdapter";
        type Type = super::Processor;
        type ParentType = gst::Element;
        fn with_class(klass: &Self::Class) -> Self {
            let sinkpad = gst::Pad::builder_from_template(&klass.pad_template("sink").unwrap())
                .chain_function(|_, parent, buffer| {
                    Self::catch_panic_pad_function(
                        parent,
                        || Err(gst::FlowError::Error),
                        |imp| imp.chain(buffer),
                    )
                })
                .event_function(|_, parent, event| {
                    Self::catch_panic_pad_function(
                        parent,
                        || false,
                        |imp| imp.srcpad.push_event(event),
                    )
                })
                .query_function(|_, parent, query| {
                    Self::catch_panic_pad_function(
                        parent,
                        || false,
                        |imp| imp.srcpad.peer_query(query),
                    )
                })
                .build();
            let srcpad = gst::Pad::builder_from_template(&klass.pad_template("src").unwrap())
                .event_function(|_, parent, event| {
                    Self::catch_panic_pad_function(
                        parent,
                        || false,
                        |imp| imp.sinkpad.push_event(event),
                    )
                })
                .query_function(|_, parent, query| {
                    Self::catch_panic_pad_function(
                        parent,
                        || false,
                        |imp| imp.sinkpad.peer_query(query),
                    )
                })
                .build();
            Self {
                config: OnceLock::new(),
                sinkpad,
                srcpad,
            }
        }
    }
    impl ObjectImpl for Processor {
        fn constructed(&self) {
            self.parent_constructed();
            self.obj().add_pad(&self.sinkpad).unwrap();
            self.obj().add_pad(&self.srcpad).unwrap();
        }
    }
    impl GstObjectImpl for Processor {}
    impl ElementImpl for Processor {
        fn metadata() -> Option<&'static gst::subclass::ElementMetadata> {
            static META: Lazy<gst::subclass::ElementMetadata> = Lazy::new(|| {
                gst::subclass::ElementMetadata::new(
                    "Vyd Rust processor",
                    "Filter/Video",
                    "Serial Rust processor adapter",
                    "Vyd",
                )
            });
            Some(&META)
        }
        fn pad_templates() -> &'static [gst::PadTemplate] {
            static PADS: Lazy<Vec<gst::PadTemplate>> =
                Lazy::new(|| templates(&[gst::PadDirection::Sink, gst::PadDirection::Src]));
            &PADS
        }
    }
}

mod sink_imp {
    use super::*;
    #[derive(Default)]
    pub struct Sink {
        pub(super) shared: OnceLock<Shared>,
    }
    #[glib::object_subclass]
    impl ObjectSubclass for Sink {
        const NAME: &'static str = "VydSerialSinkAdapter";
        type Type = super::Sink;
        type ParentType = gst_base::BaseSink;
    }
    impl ObjectImpl for Sink {}
    impl GstObjectImpl for Sink {}
    impl ElementImpl for Sink {
        fn metadata() -> Option<&'static gst::subclass::ElementMetadata> {
            static META: Lazy<gst::subclass::ElementMetadata> = Lazy::new(|| {
                gst::subclass::ElementMetadata::new(
                    "Vyd Rust sink",
                    "Sink/Video",
                    "Serial Rust sink adapter",
                    "Vyd",
                )
            });
            Some(&META)
        }
        fn pad_templates() -> &'static [gst::PadTemplate] {
            static PADS: Lazy<Vec<gst::PadTemplate>> =
                Lazy::new(|| templates(&[gst::PadDirection::Sink]));
            &PADS
        }
    }
    impl BaseSinkImpl for Sink {
        fn render(&self, buffer: &gst::Buffer) -> Result<gst::FlowSuccess, gst::FlowError> {
            let pad = self.obj().static_pad("sink").ok_or(gst::FlowError::Error)?;
            invoke(self.shared.get().unwrap(), |state| state.sink(&pad, buffer))?;
            Ok(gst::FlowSuccess::Ok)
        }
    }
}
