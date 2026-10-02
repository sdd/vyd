use serde::{Deserialize, Serialize};

/// Information about a video being annotated.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct VideoMetadata {
    pub file_path: String,
    pub width: u32,
    pub height: u32,
    pub frame_count: u32,
    pub fps: f32,
    pub duration_seconds: f32,
}

/// A labelled position at a frame in any video track.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Keyframe {
    pub frame_number: u32,
    pub x: f32,
    pub y: f32,
    pub confidence: f32,
    pub notes: Option<String>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct AnnotationSessionMetadata {
    pub created_at: chrono::DateTime<chrono::Utc>,
    pub annotator: String,
    pub tool_version: String,
    pub notes: Option<String>,
}

/// Interpolate between the closest keyframes on either side of a frame.
/// Returns `None` outside the annotated range.
pub fn position_at_frame(keyframes: &[Keyframe], frame_number: u32) -> Option<(f32, f32)> {
    if let Some(keyframe) = keyframes
        .iter()
        .find(|keyframe| keyframe.frame_number == frame_number)
    {
        return Some((keyframe.x, keyframe.y));
    }

    let before = keyframes
        .iter()
        .filter(|keyframe| keyframe.frame_number < frame_number)
        .max_by_key(|keyframe| keyframe.frame_number)?;
    let after = keyframes
        .iter()
        .filter(|keyframe| keyframe.frame_number > frame_number)
        .min_by_key(|keyframe| keyframe.frame_number)?;

    let t = (frame_number - before.frame_number) as f32
        / (after.frame_number - before.frame_number) as f32;
    Some((
        before.x + t * (after.x - before.x),
        before.y + t * (after.y - before.y),
    ))
}

pub fn frame_range(keyframes: &[Keyframe]) -> Option<(u32, u32)> {
    let min = keyframes
        .iter()
        .map(|keyframe| keyframe.frame_number)
        .min()?;
    let max = keyframes
        .iter()
        .map(|keyframe| keyframe.frame_number)
        .max()?;
    Some((min, max))
}

/// Insert a keyframe in frame order, replacing an existing position for that frame.
pub fn add_keyframe(keyframes: &mut Vec<Keyframe>, keyframe: Keyframe) {
    match keyframes.binary_search_by_key(&keyframe.frame_number, |existing| existing.frame_number) {
        Ok(index) => keyframes[index] = keyframe,
        Err(index) => keyframes.insert(index, keyframe),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn keyframe(frame_number: u32) -> Keyframe {
        Keyframe {
            frame_number,
            x: frame_number as f32,
            y: 0.0,
            confidence: 1.0,
            notes: None,
        }
    }

    #[test]
    fn keyframes_support_interpolation_and_replacement() {
        let mut keyframes = Vec::new();
        add_keyframe(&mut keyframes, keyframe(20));
        add_keyframe(&mut keyframes, keyframe(10));
        assert_eq!(frame_range(&keyframes), Some((10, 20)));
        assert_eq!(position_at_frame(&keyframes, 15), Some((15.0, 0.0)));
        assert_eq!(position_at_frame(&keyframes, 5), None);
        add_keyframe(
            &mut keyframes,
            Keyframe {
                x: 12.0,
                ..keyframe(10)
            },
        );
        assert_eq!(keyframes.len(), 2);
        assert_eq!(position_at_frame(&keyframes, 10), Some((12.0, 0.0)));
    }
}
