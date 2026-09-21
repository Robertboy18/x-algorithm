// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 X.AI Corp.
use futures::future::join_all;
use std::{cmp, ffi::c_int, io, mem};
use tokio::task;

#[cfg(target_os = "linux")]
use nix::libc::{
    MADV_HUGEPAGE, MADV_NORMAL, MADV_SEQUENTIAL, MPOL_BIND, MPOL_INTERLEAVE, SYS_get_mempolicy,
    SYS_set_mempolicy, madvise, syscall,
};

pub const NODE_MASK_SIZE: usize = 16;
const MPOL_F_MEMS_ALLOWED: u64 = 4;

pub fn get_numa_policy(flags: u64) -> Result<(c_int, [u64; NODE_MASK_SIZE]), io::Error> {
    let _used = (flags, MPOL_F_MEMS_ALLOWED);

    #[cfg(target_os = "linux")]
    {
        let mut mode: c_int = 0;
        let mut node_mask = [0u64; NODE_MASK_SIZE];
        let ret = unsafe {
            syscall(
                SYS_get_mempolicy,
                &mut mode,
                node_mask.as_mut_ptr(),
                NODE_MASK_SIZE as u64 * 64,
                0usize,
                flags,
            )
        };
        if ret != 0 {
            return Err(io::Error::last_os_error());
        }
        Ok((mode, node_mask))
    }
    #[cfg(not(target_os = "linux"))]
    Ok((0, [0; NODE_MASK_SIZE]))
}

pub fn get_numa_node_count() -> Result<u32, io::Error> {
    #[cfg(target_os = "linux")]
    {
        let mut count = 0;
        for mask in get_numa_policy(MPOL_F_MEMS_ALLOWED)?.1 {
            if mask != u64::MAX {
                count += mask.trailing_ones();
                break;
            }
            count += 64;
        }
        Ok(count)
    }
    #[cfg(not(target_os = "linux"))]
    Ok(1)
}

pub fn set_numa_policy(mode: c_int, node_mask: [u64; NODE_MASK_SIZE]) -> Result<(), io::Error> {
    let _used = (mode, &node_mask);

    #[cfg(target_os = "linux")]
    {
        let ret = unsafe {
            syscall(
                SYS_set_mempolicy,
                mode,
                node_mask.as_ptr(),
                NODE_MASK_SIZE * 64,
            )
        };
        if ret != 0 {
            return Err(io::Error::last_os_error());
        }
    }
    Ok(())
}

#[cfg(target_arch = "aarch64")]
const HUGE_PAGE_SIZE: usize = 2usize.pow(29);
#[cfg(not(target_arch = "aarch64"))]
const HUGE_PAGE_SIZE: usize = 2usize.pow(21);

pub fn madvise_hugepage_internal(slice: &mut [u8]) -> Result<(), io::Error> {
    let _used = (&slice, HUGE_PAGE_SIZE);

    #[cfg(target_os = "linux")]
    unsafe {
        use std::ffi::c_void;

        let offset = (!(slice.as_ptr() as usize) + 1) & (HUGE_PAGE_SIZE - 1);
        let len = slice.len().saturating_sub(offset) & !(HUGE_PAGE_SIZE - 1);
        if len == 0 {
            return Ok(());
        }
        let ret = madvise(
            slice.as_mut_ptr().add(offset) as *mut c_void,
            len,
            MADV_HUGEPAGE,
        );
        if ret != 0 {
            return Err(io::Error::last_os_error());
        }
    }
    Ok(())
}

pub fn madvise_sequential_internal(slice: &mut [u8]) -> Result<(), io::Error> {
    #[cfg(target_os = "linux")]
    unsafe {
        use std::ffi::c_void;
        let ret = madvise(
            slice.as_mut_ptr() as *mut c_void,
            slice.len(),
            MADV_SEQUENTIAL,
        );
        if ret != 0 {
            return Err(io::Error::last_os_error());
        }
    }
    let _ = slice;
    Ok(())
}

pub fn madvise_normal_internal(slice: &mut [u8]) -> Result<(), io::Error> {
    #[cfg(target_os = "linux")]
    unsafe {
        use std::ffi::c_void;
        let ret = madvise(slice.as_mut_ptr() as *mut c_void, slice.len(), MADV_NORMAL);
        if ret != 0 {
            return Err(io::Error::last_os_error());
        }
    }
    let _ = slice;
    Ok(())
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub unsafe fn copy_nontemporal(dst: &mut [u8], src: &[u8]) {
    use std::arch::x86_64::{__m256i, _mm256_loadu_si256, _mm256_stream_si256};

    const NT_THRESHOLD: usize = 4096;
    const CHUNK: usize = 32;

    assert_eq!(dst.len(), src.len());
    let len = dst.len();

    if len < NT_THRESHOLD {
        dst.copy_from_slice(src);
        return;
    }

    let dst_ptr = dst.as_mut_ptr();
    let align_offset = dst_ptr.align_offset(CHUNK).min(len);

    dst[..align_offset].copy_from_slice(&src[..align_offset]);

    let middle_end = align_offset + ((len - align_offset) / CHUNK) * CHUNK;
    let mut i = align_offset;
    while i < middle_end {
        unsafe {
            let s = _mm256_loadu_si256(src.as_ptr().add(i) as *const __m256i);
            _mm256_stream_si256(dst_ptr.add(i) as *mut __m256i, s);
        }
        i += CHUNK;
    }

    dst[middle_end..].copy_from_slice(&src[middle_end..]);
}

#[cfg(not(target_arch = "x86_64"))]
pub fn copy_nontemporal(dst: &mut [u8], src: &[u8]) {
    dst.copy_from_slice(src);
}

#[derive(Copy, Clone, Debug)]
pub enum InitKind {
    Striped(usize),
    Interleaved,
    Default,
}

pub async fn init_mem(buf: &mut [u8], kind: InitKind) -> Result<(), io::Error> {
    madvise_hugepage_internal(buf)?;
    let mut buf: &'static mut [u8] = unsafe { mem::transmute(buf) };
    let (n, stripe_size) = if let InitKind::Striped(size) = kind {
        (get_numa_node_count()? as usize, size)
    } else {
        (1, usize::MAX)
    };
    let step = buf.len() / (n << 10);
    let mut bufz = Vec::with_capacity(n);
    for _ in 0..n {
        bufz.push(Vec::new());
    }
    let mut pos = 0usize;
    let mut i = n - 1;
    while !buf.is_empty() {
        let mut delta = pos.next_multiple_of(stripe_size) - pos;
        if delta == 0 {
            delta = stripe_size;
            i = (i + 1) % n;
        }
        let (buf0, buf1) = buf.split_at_mut(cmp::min(step, cmp::min(delta, buf.len())));
        pos += buf0.len();
        bufz[i].push(buf0);
        buf = buf1;
    }
    for (i, bufs) in bufz.into_iter().enumerate() {
        let _used = i;
        join_all(bufs.into_iter().map(|buf| {
            task::spawn(async move {
                let (mode, node_mask) = get_numa_policy(0)?;
                #[cfg(target_os = "linux")]
                match kind {
                    InitKind::Striped(_) => {
                        let mut node_mask = [0; NODE_MASK_SIZE];
                        node_mask[i / 64] = 1u64 << (i % 64);
                        set_numa_policy(MPOL_BIND, node_mask)?;
                    }
                    InitKind::Interleaved => {
                        set_numa_policy(MPOL_INTERLEAVE, get_numa_policy(MPOL_F_MEMS_ALLOWED)?.1)?;
                    }
                    InitKind::Default => {}
                }
                for i in (0..buf.len()).step_by(1 << 11) {
                    buf[i] = 0;
                }
                set_numa_policy(mode, node_mask)
            })
        }))
        .await
        .into_iter()
        .collect::<Result<Vec<_>, _>>()?
        .into_iter()
        .collect::<Result<Vec<()>, _>>()?;
    }
    Ok(())
}
