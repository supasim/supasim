/* BEGIN LICENSE
  SupaSim, a GPGPU and simulation toolkit.
  Copyright (C) 2025 Magnus Larsson
  SPDX-License-Identifier: MIT OR Apache-2.0
END LICENSE */

/// The size must be >0 or equality comparison is undefined
macro_rules! api_type {
    ($name: ident, { $($field:tt)* }, $($attr: meta),*) => {
        paste::paste! {
            // Inner type
            pub(crate) struct [<$name Inner>] <B: hal::Backend> {
                pub(crate) _phantom: std::marker::PhantomData<B>, // Ensures B is always used
                $($field)*
            }

            #[derive(Clone)]
            pub(crate) struct [<$name Weak>] <B: hal::Backend>(std::sync::Weak<parking_lot::RwLock<[<$name Inner>]<B>>>);
            #[allow(dead_code)]
            impl<B: hal::Backend> [<$name Weak>] <B> {
                pub(crate) fn upgrade(&self) -> crate::SupaSimResult<B, $name<B>> {
                    Ok($name(self.0.upgrade().ok_or(crate::SupaSimError::AlreadyDestroyed(stringify!($name).to_owned()))?))
                }
            }
            // Mirrors the strong type's blanket Send + Sync. A `Weak` handle is a
            // dangling-safe pointer to the same `Arc<RwLock<Inner>>`; sharing it across
            // threads is exactly as sound as sharing the strong handle. The sync thread
            // deliberately holds a `Weak` instance handle (see `create_sync_thread`).
            unsafe impl<B: hal::Backend> Send for [<$name Weak>] <B> {}
            unsafe impl<B: hal::Backend> Sync for [<$name Weak>] <B> {}

            // Outer type, with some helper methods
            #[derive(Clone)]
            $(
                #[$attr]
            )*
            pub struct $name <B: hal::Backend> (std::sync::Arc<parking_lot::RwLock<[<$name Inner>]<B>>>);
            #[allow(dead_code)]
            impl<B: hal::Backend> $name <B> {
                pub(crate) fn from_inner(inner: [<$name Inner>]<B>) -> Self {
                    Self(std::sync::Arc::new(parking_lot::RwLock::new(inner)))
                }
                pub(crate) fn inner(&'_ self) -> crate::SupaSimResult<B, crate::InnerRef<'_, [<$name Inner>]<B>>> {
                    let r = self.0.read();
                    Ok(crate::InnerRef(r))
                }
                pub(crate) fn inner_mut(&'_ self) -> crate::SupaSimResult<B, crate::InnerRefMut<'_, [<$name Inner>]<B>>> {
                    let r = self.0.write();
                    Ok(crate::InnerRefMut(r))
                }
                pub(crate) fn downgrade(&self) -> [<$name Weak>]<B> {
                    [<$name Weak>](std::sync::Arc::downgrade(&self.0))
                }
            }
            impl<B: hal::Backend> PartialEq for $name <B> {
                fn eq(&self, other: &Self) -> bool {
                    std::ptr::eq(self.0.as_ref(), other.0.as_ref())
                }
            }
            impl<B: hal::Backend> Eq for $name <B> {}
            unsafe impl<B: hal::Backend> Send for $name <B> {}
            unsafe impl<B: hal::Backend> Sync for $name <B> {}
        }
    };
}
