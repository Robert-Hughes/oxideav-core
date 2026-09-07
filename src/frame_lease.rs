//! Retainable decoded-frame storage for zero-copy pipelines.
//!
//! [`FrameLease`] is the ownership-carrying counterpart to [`crate::Frame`].
//! A decoder hands a lease downstream when the decoded storage must remain
//! alive beyond the decoder callback. Cloning a lease is cheap: it retains the
//! underlying CPU arena, heap frame, or hardware surface rather than copying
//! decoded media bytes.
//!
//! # Lifetime contract
//!
//! The storage referenced by a lease is immutable and remains valid until the
//! last clone of that lease is dropped. Producers may recycle pooled storage
//! only after all decoder-internal references and all consumer leases have been
//! released. Hardware consumers that submit asynchronous GPU work must retain
//! the lease until that work no longer reads the surface; dropping the Rust
//! handle is the consumer's declaration that the storage may be recycled.
//!
//! CPU arena frames and opaque hardware surfaces deliberately share this same
//! contract. Consumers that cannot operate on the native representation may
//! call [`FrameLease::materialize`] or [`FrameLease::into_frame`] to obtain the
//! legacy heap-backed [`crate::Frame`], paying a copy only at that explicit
//! compatibility boundary.

use std::any::Any;
use std::fmt;
use std::sync::Arc;

use crate::arena;
use crate::{Error, Frame, PixelFormat, Result, VideoFrame, VideoPlane};

/// Backend-owned GPU/video storage that can travel through a generic pipeline.
///
/// Implementations normally wrap a refcounted decoder surface lease. The
/// concrete backend handle remains available through [`Self::as_any`] for an
/// application that knows how to import/sample it directly. Generic consumers
/// can instead use [`Self::materialize`] as an explicit CPU fallback.
///
/// The implementation must uphold the module-level lease contract: the
/// underlying surface remains valid and immutable while this value (or any
/// clone of the enclosing [`HardwareVideoFrame`]) is alive.
pub trait HardwareVideoFrameStorage: Any + Send + Sync {
    /// Stable backend identifier, for example `"vdpau"` or `"vulkan"`.
    fn backend(&self) -> &'static str;

    /// Visible picture width in pixels.
    fn width(&self) -> u32;

    /// Visible picture height in pixels.
    fn height(&self) -> u32;

    /// Pixel format represented by the hardware surface.
    fn pixel_format(&self) -> PixelFormat;

    /// Presentation timestamp in the stream's time-base units.
    fn pts(&self) -> Option<i64>;

    /// Downcasting hook for backend-aware consumers.
    fn as_any(&self) -> &dyn Any;

    /// Copy/read the hardware surface into a legacy CPU [`VideoFrame`].
    ///
    /// This is a compatibility fallback and may be expensive. Consumers that
    /// understand [`Self::backend`] should prefer the opaque surface directly.
    fn materialize(&self) -> Result<VideoFrame>;
}

/// Cloneable handle to an opaque hardware-decoded video frame.
#[derive(Clone)]
pub struct HardwareVideoFrame {
    inner: Arc<dyn HardwareVideoFrameStorage>,
}

impl HardwareVideoFrame {
    /// Wrap backend-specific hardware storage in a retainable generic handle.
    pub fn new<T>(storage: T) -> Self
    where
        T: HardwareVideoFrameStorage + 'static,
    {
        Self {
            inner: Arc::new(storage),
        }
    }

    /// Backend identifier supplied by the hardware implementation.
    pub fn backend(&self) -> &'static str {
        self.inner.backend()
    }

    /// Visible picture width in pixels.
    pub fn width(&self) -> u32 {
        self.inner.width()
    }

    /// Visible picture height in pixels.
    pub fn height(&self) -> u32 {
        self.inner.height()
    }

    /// Pixel format represented by the hardware surface.
    pub fn pixel_format(&self) -> PixelFormat {
        self.inner.pixel_format()
    }

    /// Presentation timestamp in stream time-base units.
    pub fn pts(&self) -> Option<i64> {
        self.inner.pts()
    }

    /// Borrow the type-erased backend storage.
    pub fn storage(&self) -> &dyn HardwareVideoFrameStorage {
        self.inner.as_ref()
    }

    /// Downcast the backend storage to a concrete implementation type.
    pub fn downcast_ref<T: Any>(&self) -> Option<&T> {
        self.inner.as_any().downcast_ref::<T>()
    }

    /// Materialize this hardware frame into legacy CPU video planes.
    pub fn materialize(&self) -> Result<VideoFrame> {
        self.inner.materialize()
    }
}

impl fmt::Debug for HardwareVideoFrame {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("HardwareVideoFrame")
            .field("backend", &self.backend())
            .field("width", &self.width())
            .field("height", &self.height())
            .field("pixel_format", &self.pixel_format())
            .field("pts", &self.pts())
            .finish_non_exhaustive()
    }
}

/// Owned, cheaply retainable decoded-frame storage.
///
/// This is the preferred currency between a decoder and an asynchronous player
/// sink. The variants intentionally cover both CPU and GPU storage under one
/// lifetime rule:
///
/// - [`Self::Owned`] retains an ordinary heap-backed [`Frame`] behind an
///   [`Arc`], avoiding deep clones as it crosses queues.
/// - [`Self::ArenaVideo`] retains a reusable CPU arena allocation; the backing
///   buffer returns to its [`arena::sync::ArenaPool`] after the last clone is
///   dropped.
/// - [`Self::HardwareVideo`] retains an opaque decoder surface; its backend
///   decides how the resource is recycled after the final lease disappears.
#[derive(Clone)]
#[non_exhaustive]
pub enum FrameLease {
    /// Legacy heap-backed frame retained behind an atomic reference count.
    Owned(Arc<Frame>),
    /// Pooled CPU video frame retained directly from a sync arena.
    ArenaVideo(arena::sync::Frame),
    /// Opaque hardware-resident video frame.
    HardwareVideo(HardwareVideoFrame),
}

impl FrameLease {
    /// Wrap a legacy frame without copying its media buffers.
    pub fn from_frame(frame: Frame) -> Self {
        Self::Owned(Arc::new(frame))
    }

    /// Wrap an arena-backed CPU video frame.
    pub fn from_arena_video(frame: arena::sync::Frame) -> Self {
        Self::ArenaVideo(frame)
    }

    /// Wrap an opaque hardware video frame.
    pub fn from_hardware_video(frame: HardwareVideoFrame) -> Self {
        Self::HardwareVideo(frame)
    }

    /// Presentation timestamp in stream time-base units, when available.
    pub fn pts(&self) -> Option<i64> {
        match self {
            Self::Owned(frame) => frame.pts(),
            Self::ArenaVideo(frame) => frame.header().presentation_timestamp,
            Self::HardwareVideo(frame) => frame.pts(),
        }
    }

    /// Borrow the legacy frame when this lease already uses legacy storage.
    pub fn as_frame(&self) -> Option<&Frame> {
        match self {
            Self::Owned(frame) => Some(frame.as_ref()),
            _ => None,
        }
    }

    /// Borrow the arena-backed video frame when present.
    pub fn as_arena_video(&self) -> Option<&arena::sync::Frame> {
        match self {
            Self::ArenaVideo(frame) => Some(frame),
            _ => None,
        }
    }

    /// Borrow the opaque hardware video frame when present.
    pub fn as_hardware_video(&self) -> Option<&HardwareVideoFrame> {
        match self {
            Self::HardwareVideo(frame) => Some(frame),
            _ => None,
        }
    }

    /// Return `true` when the decoded video is still hardware-resident.
    pub fn is_hardware_video(&self) -> bool {
        matches!(self, Self::HardwareVideo(_))
    }

    /// Obtain a legacy heap-backed frame, copying only when the current storage
    /// representation cannot already be borrowed as one.
    pub fn materialize(&self) -> Result<Frame> {
        match self {
            Self::Owned(frame) => Ok(frame.as_ref().clone()),
            Self::ArenaVideo(frame) => Ok(Frame::Video(materialize_arena_video(frame)?)),
            Self::HardwareVideo(frame) => Ok(Frame::Video(frame.materialize()?)),
        }
    }

    /// Consume the lease and obtain a legacy heap-backed frame.
    ///
    /// A uniquely-owned [`Self::Owned`] frame is moved out of its [`Arc`] with
    /// no media-buffer copy. Shared owned frames, arena frames, and hardware
    /// frames materialize as needed.
    pub fn into_frame(self) -> Result<Frame> {
        match self {
            Self::Owned(frame) => match Arc::try_unwrap(frame) {
                Ok(frame) => Ok(frame),
                Err(frame) => Ok(frame.as_ref().clone()),
            },
            Self::ArenaVideo(frame) => Ok(Frame::Video(materialize_arena_video(&frame)?)),
            Self::HardwareVideo(frame) => Ok(Frame::Video(frame.materialize()?)),
        }
    }
}

impl From<Frame> for FrameLease {
    fn from(value: Frame) -> Self {
        Self::from_frame(value)
    }
}

impl From<arena::sync::Frame> for FrameLease {
    fn from(value: arena::sync::Frame) -> Self {
        Self::from_arena_video(value)
    }
}

impl From<HardwareVideoFrame> for FrameLease {
    fn from(value: HardwareVideoFrame) -> Self {
        Self::from_hardware_video(value)
    }
}

impl fmt::Debug for FrameLease {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Owned(frame) => f.debug_tuple("FrameLease::Owned").field(frame).finish(),
            Self::ArenaVideo(frame) => f
                .debug_struct("FrameLease::ArenaVideo")
                .field("header", frame.header())
                .field("planes", &frame.plane_count())
                .finish(),
            Self::HardwareVideo(frame) => f
                .debug_tuple("FrameLease::HardwareVideo")
                .field(frame)
                .finish(),
        }
    }
}

fn materialize_arena_video(frame: &arena::sync::Frame) -> Result<VideoFrame> {
    let header = frame.header();
    let mut planes = Vec::with_capacity(frame.plane_count());
    for index in 0..frame.plane_count() {
        let data = frame
            .plane(index)
            .ok_or_else(|| Error::invalid(format!("arena video frame is missing plane {index}")))?;
        let stride = frame
            .plane_stride(index)
            .or_else(|| header.pixel_format.plane_row_bytes(index, header.width));
        let stride = stride.ok_or_else(|| {
            Error::invalid(format!(
                "arena video frame has no stride for plane {index} ({:?} {}x{})",
                header.pixel_format, header.width, header.height
            ))
        })?;
        planes.push(VideoPlane {
            stride,
            data: data.to_vec(),
        });
    }
    let mut video = VideoFrame {
        pts: header.presentation_timestamp,
        planes,
    };
    if let Some(bits) = header.significant_bits() {
        video.set_significant_bits(bits.to_vec());
    }
    Ok(video)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::arena::sync::{ArenaPool, FrameHeader, FrameInner};
    use std::sync::atomic::{AtomicUsize, Ordering};

    #[test]
    fn owned_lease_moves_unique_frame_without_changing_payload() {
        let frame = Frame::Video(VideoFrame {
            pts: Some(7),
            planes: vec![VideoPlane {
                stride: 2,
                data: vec![1, 2, 3, 4],
            }],
        });
        let lease = FrameLease::from_frame(frame);
        let out = lease.into_frame().expect("materialize owned frame");
        let Frame::Video(out) = out else {
            panic!("expected video frame");
        };
        assert_eq!(out.pts, Some(7));
        assert_eq!(out.planes[0].data, [1, 2, 3, 4]);
    }

    #[test]
    fn arena_lease_retains_pool_buffer_until_last_clone_drops() {
        let pool = ArenaPool::new(1, 16);
        let arena = pool.lease().expect("lease arena");
        let pixels = arena.alloc::<u8>(4).expect("allocate pixels");
        pixels.copy_from_slice(&[1, 2, 3, 4]);
        let frame = FrameInner::new_with_strides(
            arena,
            &[(0, 4)],
            &[2],
            FrameHeader::new(2, 2, PixelFormat::Gray8, Some(11)),
        )
        .expect("arena frame");
        let lease = FrameLease::from_arena_video(frame);
        let retained = lease.clone();

        assert!(matches!(pool.lease(), Err(Error::ResourceExhausted(_))));
        drop(lease);
        assert!(matches!(pool.lease(), Err(Error::ResourceExhausted(_))));
        drop(retained);
        assert!(pool.lease().is_ok());
    }

    #[test]
    fn arena_materialization_preserves_stride_and_significant_bits() {
        let pool = ArenaPool::new(1, 8);
        let arena = pool.lease().unwrap();
        arena
            .alloc::<u8>(8)
            .unwrap()
            .copy_from_slice(&[1, 2, 9, 9, 3, 4, 9, 9]);
        let arena_frame = FrameInner::new_with_strides(
            arena,
            &[(0, 8)],
            &[4],
            FrameHeader::new(2, 2, PixelFormat::Gray8, Some(3))
                .with_significant_bits(&[7])
                .unwrap(),
        )
        .unwrap();
        let frame = FrameLease::from_arena_video(arena_frame)
            .into_frame()
            .unwrap();
        let Frame::Video(video) = frame else {
            panic!("expected video");
        };
        assert_eq!(video.planes[0].stride, 4);
        assert_eq!(video.planes[0].data, [1, 2, 9, 9, 3, 4, 9, 9]);
        assert_eq!(video.significant_bits(), Some(&[7][..]));
    }

    struct FakeHardwareFrame {
        drops: Arc<AtomicUsize>,
    }

    impl Drop for FakeHardwareFrame {
        fn drop(&mut self) {
            self.drops.fetch_add(1, Ordering::SeqCst);
        }
    }

    impl HardwareVideoFrameStorage for FakeHardwareFrame {
        fn backend(&self) -> &'static str {
            "fake"
        }

        fn width(&self) -> u32 {
            2
        }

        fn height(&self) -> u32 {
            2
        }

        fn pixel_format(&self) -> PixelFormat {
            PixelFormat::Gray8
        }

        fn pts(&self) -> Option<i64> {
            Some(5)
        }

        fn as_any(&self) -> &dyn Any {
            self
        }

        fn materialize(&self) -> Result<VideoFrame> {
            Ok(VideoFrame {
                pts: Some(5),
                planes: vec![VideoPlane {
                    stride: 2,
                    data: vec![9, 8, 7, 6],
                }],
            })
        }
    }

    #[test]
    fn hardware_lease_is_refcounted_and_materializes_only_on_request() {
        let drops = Arc::new(AtomicUsize::new(0));
        let hardware = HardwareVideoFrame::new(FakeHardwareFrame {
            drops: Arc::clone(&drops),
        });
        let lease = FrameLease::from_hardware_video(hardware.clone());
        let retained = lease.clone();
        assert_eq!(hardware.backend(), "fake");
        assert!(hardware.downcast_ref::<FakeHardwareFrame>().is_some());
        assert_eq!(drops.load(Ordering::SeqCst), 0);

        let materialized = retained.materialize().unwrap();
        let Frame::Video(video) = materialized else {
            panic!("expected video");
        };
        assert_eq!(video.planes[0].data, [9, 8, 7, 6]);
        assert_eq!(drops.load(Ordering::SeqCst), 0);

        drop(retained);
        drop(lease);
        drop(hardware);
        assert_eq!(drops.load(Ordering::SeqCst), 1);
    }
}
