#![cfg_attr(not(test), expect(dead_code, reason = "TODO: no consumer wired yet"))]

use std::hash::{DefaultHasher, Hash, Hasher};
use std::mem::size_of;
use std::sync::{
    atomic::{AtomicU64, Ordering},
    Mutex,
};

const WAYS: usize = 32;
const SHARDS: usize = 64;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Lookup<V> {
    Miss,
    NotFound,
    Found(V),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct CapacityError {
    pub(crate) capacity: usize,
}

impl std::fmt::Display for CapacityError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "clock cache layout for {} entries exceeds the addressable range",
            self.capacity
        )
    }
}

impl std::error::Error for CapacityError {}

#[derive(Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
enum Mark {
    Empty,
    Cold,
    Referenced,
}

#[derive(Clone, Copy)]
struct Slot<V> {
    key: u64,
    generation: u64,
    value: Option<V>,
}

#[derive(Clone)]
struct Set<V> {
    slots: [Slot<V>; WAYS],
    marks: [Mark; WAYS],
    hand: u8,
}

struct Shard<V> {
    sets: Vec<Set<V>>,
    entries: usize,
}

pub(crate) struct ClockCache<V> {
    shards: Vec<Mutex<Shard<V>>>,
    set_count: usize,
    next_generation: AtomicU64,
}

impl<V: Copy> ClockCache<V> {
    pub(crate) fn new(capacity: usize) -> Result<Self, CapacityError> {
        if Self::inline_bytes(capacity).is_none() {
            return Err(CapacityError { capacity });
        }
        let set_count = capacity.div_ceil(WAYS);
        let shard_count = SHARDS.min(set_count);
        let shards = (0..shard_count)
            .map(|shard| {
                let sets = set_count / shard_count + usize::from(shard < set_count % shard_count);
                Mutex::new(Shard {
                    sets: vec![
                        Set {
                            slots: [Slot {
                                key: 0,
                                generation: 0,
                                value: None
                            }; WAYS],
                            marks: [Mark::Empty; WAYS],
                            hand: 0,
                        };
                        sets
                    ],
                    entries: 0,
                })
            })
            .collect();
        Ok(Self {
            shards,
            set_count,
            next_generation: AtomicU64::new(0),
        })
    }

    pub(crate) fn inline_bytes(capacity: usize) -> Option<usize> {
        let set_count = capacity.div_ceil(WAYS);
        let sets = set_count.checked_mul(size_of::<Set<V>>())?;
        let shards = SHARDS
            .min(set_count)
            .checked_mul(size_of::<Mutex<Shard<V>>>())?;
        sets.checked_add(shards)
            .filter(|total| isize::try_from(*total).is_ok())
    }

    pub(crate) fn begin_request(&self) -> u64 {
        self.next_generation.fetch_add(1, Ordering::Relaxed)
    }

    #[expect(
        clippy::indexing_slicing,
        reason = "global set modulo shard count selects an allocated shard"
    )]
    fn shard(&self, key: u64) -> Option<(&Mutex<Shard<V>>, usize)> {
        if self.set_count == 0 {
            return None;
        }
        let mut hasher = DefaultHasher::new();
        key.hash(&mut hasher);
        let set = hasher.finish() as usize % self.set_count;
        Some((
            &self.shards[set % self.shards.len()],
            set / self.shards.len(),
        ))
    }

    #[expect(
        clippy::indexing_slicing,
        reason = "local set is allocated; the CLOCK hand stays below the 32-slot array length"
    )]
    pub(crate) fn insert(&self, generation: u64, key: u64, value: Option<V>) {
        let Some((shard, index)) = self.shard(key) else {
            return;
        };
        let mut shard = shard
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let set = &mut shard.sets[index];
        if let Some((slot, mark)) = set
            .slots
            .iter_mut()
            .zip(&mut set.marks)
            .find(|(slot, mark)| **mark != Mark::Empty && slot.key == key)
        {
            if slot.generation > generation {
                return;
            }
            *slot = Slot {
                key,
                generation,
                value,
            };
            *mark = Mark::Referenced;
            return;
        }
        for _ in 0..WAYS {
            let hand = usize::from(set.hand);
            if set.marks[hand] != Mark::Referenced {
                break;
            }
            set.marks[hand] = Mark::Cold;
            set.hand = (set.hand + 1) % WAYS as u8;
        }
        let hand = usize::from(set.hand);
        let inserted = set.marks[hand] == Mark::Empty;
        set.slots[hand] = Slot {
            key,
            generation,
            value,
        };
        set.marks[hand] = Mark::Referenced;
        set.hand = (set.hand + 1) % WAYS as u8;
        shard.entries += usize::from(inserted);
    }

    #[expect(
        clippy::indexing_slicing,
        reason = "shard maps each global set to an allocated local set"
    )]
    pub(crate) fn get(&self, key: u64) -> Lookup<V> {
        let Some((shard, index)) = self.shard(key) else {
            return Lookup::Miss;
        };
        let mut shard = shard
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let set = &mut shard.sets[index];
        let Some((slot, mark)) = set
            .slots
            .iter()
            .zip(&mut set.marks)
            .find(|(slot, mark)| **mark != Mark::Empty && slot.key == key)
        else {
            return Lookup::Miss;
        };
        *mark = Mark::Referenced;
        match slot.value {
            Some(value) => Lookup::Found(value),
            None => Lookup::NotFound,
        }
    }

    pub(crate) fn entry_count(&self) -> usize {
        self.shards
            .iter()
            .map(|shard| {
                shard
                    .lock()
                    .unwrap_or_else(|poisoned| poisoned.into_inner())
                    .entries
            })
            .sum()
    }

    pub(crate) fn capacity(&self) -> usize {
        self.set_count * WAYS
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::num::NonZeroU64;

    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    #[repr(align(64))]
    struct Aligned(u8);

    fn cached<V>(value: Option<V>) -> Lookup<V> {
        value.map_or(Lookup::NotFound, Lookup::Found)
    }

    #[test]
    fn distinguishes_full_keys_not_found_and_empty() {
        let cache = ClockCache::new(32).unwrap();
        let generation = cache.begin_request();
        let author = NonZeroU64::new(u64::MAX);
        cache.insert(generation, 0, author);
        cache.insert(generation, u64::MAX, None);
        assert_eq!(cache.get(0), cached(author));
        assert_eq!(cache.get(u64::MAX), Lookup::NotFound);
        assert_eq!(cache.get(1), Lookup::Miss);
        assert_eq!(cache.entry_count(), 2);
    }

    #[test]
    fn plain_and_aligned_values_evict_and_tombstone_like_authors() {
        let cache = ClockCache::<u64>::new(32).unwrap();
        for id in 0..33 {
            cache.insert(cache.begin_request(), id, Some(0));
        }
        assert_eq!(cache.get(0), Lookup::Miss);
        assert_eq!(cache.get(32), Lookup::Found(0));

        let cache = ClockCache::<Aligned>::new(32).unwrap();
        for id in 0..33 {
            let value = if id % 2 == 0 {
                Some(Aligned(id as u8))
            } else {
                None
            };
            cache.insert(cache.begin_request(), id, value);
        }
        assert_eq!(cache.get(0), Lookup::Miss);
        assert_eq!(cache.get(31), Lookup::NotFound);
        assert_eq!(cache.get(32), Lookup::Found(Aligned(32)));
        assert_eq!(cache.entry_count(), 32);
    }

    #[test]
    fn resident_generation_order_preserves_found_and_not_found() {
        for newest in [None, NonZeroU64::new(9)] {
            let cache = ClockCache::new(32).unwrap();
            let older = cache.begin_request();
            let newer = cache.begin_request();
            let oldest = if newest.is_some() {
                None
            } else {
                NonZeroU64::new(7)
            };
            cache.insert(newer, 1, newest);
            cache.insert(older, 1, oldest);
            assert_eq!(cache.get(1), cached(newest));
            assert_eq!(cache.entry_count(), 1);
        }
    }

    #[test]
    fn full_set_churn_reuses_slots_without_aliasing_keys_or_not_found() {
        let cache = ClockCache::new(32).unwrap();
        for id in 0..160 {
            let key = u64::MAX - id;
            let author = if id % 2 == 0 {
                NonZeroU64::new(id + 1)
            } else {
                None
            };
            cache.insert(cache.begin_request(), key, author);
            assert_eq!(cache.get(key), cached(author));
            assert_eq!(cache.entry_count(), 32.min(id as usize + 1));
            for previous in 0..id {
                let found = cache.get(u64::MAX - previous);
                if found != Lookup::Miss {
                    let expected = if previous % 2 == 0 {
                        NonZeroU64::new(previous + 1)
                    } else {
                        None
                    };
                    assert_eq!(found, cached(expected));
                }
            }
        }
    }

    #[test]
    fn hits_and_refreshes_protect_old_keys_without_moving_the_hand() {
        for refresh in [false, true] {
            let cache = ClockCache::new(32).unwrap();
            for id in 0..33 {
                cache.insert(cache.begin_request(), id, NonZeroU64::new(id + 1));
            }
            assert_eq!(cache.get(0), Lookup::Miss);
            if refresh {
                cache.insert(cache.begin_request(), 2, None);
            } else {
                assert_eq!(cache.get(2), cached(NonZeroU64::new(3)));
            }
            cache.insert(cache.begin_request(), u64::MAX, None);
            assert_eq!(cache.get(1), Lookup::Miss);
            cache.insert(cache.begin_request(), u64::MAX - 1, None);
            assert_eq!(cache.get(3), Lookup::Miss);
            assert_eq!(
                cache.get(2),
                cached(if refresh { None } else { NonZeroU64::new(3) })
            );
        }
    }

    #[test]
    fn capacity_is_disabled_and_rounded() {
        for (requested, rounded) in [(0, 0), (1, 32), (31, 32), (32, 32), (33, 64), (2049, 2080)] {
            let cache = ClockCache::<NonZeroU64>::new(requested).unwrap();
            assert_eq!(cache.capacity(), rounded);
            cache.insert(cache.begin_request(), 7, None);
            assert_eq!(
                cache.get(7),
                if requested == 0 {
                    Lookup::Miss
                } else {
                    Lookup::NotFound
                }
            );
            if requested == 0 {
                assert_eq!(cache.entry_count(), 0);
            }
        }
    }

    #[test]
    fn inline_bytes_follow_the_value_layout_and_reject_unaddressable_sizes() {
        assert_eq!(size_of::<Slot<NonZeroU64>>(), 24);
        assert_eq!(ClockCache::<NonZeroU64>::inline_bytes(0), Some(0));
        assert_eq!(
            ClockCache::<NonZeroU64>::inline_bytes(8_000_000),
            Some(202_002_560)
        );
        assert!(
            ClockCache::<Aligned>::inline_bytes(8_000_000)
                > ClockCache::<u64>::inline_bytes(8_000_000)
        );

        assert_eq!(ClockCache::<NonZeroU64>::inline_bytes(usize::MAX), None);
        assert_eq!(ClockCache::<NonZeroU64>::inline_bytes(1 << 60), None);
        assert_eq!(ClockCache::<Aligned>::inline_bytes(1 << 56), None);
        assert_eq!(
            ClockCache::<NonZeroU64>::new(1 << 60).err(),
            Some(CapacityError { capacity: 1 << 60 })
        );
        assert!(ClockCache::<Aligned>::new(1 << 56).is_err());
    }

    #[test]
    fn uneven_shards_fill_and_churn_with_exact_identity() {
        let cache = ClockCache::new(2049).unwrap();
        for id in 0..20_000 {
            cache.insert(cache.begin_request(), id, NonZeroU64::new(id + 1));
            assert_eq!(cache.get(id), cached(NonZeroU64::new(id + 1)));
        }
        assert_eq!(cache.entry_count(), 2080);
    }

    #[test]
    fn concurrent_eviction_and_reinsertion_order_only_resident_generations() {
        use std::sync::{mpsc, Arc};
        use std::time::Duration;

        for newest in [None, NonZeroU64::new(99)] {
            let cache = Arc::new(ClockCache::new(32).unwrap());
            let older = cache.begin_request();
            let newer = cache.begin_request();
            let oldest = if newest.is_some() {
                None
            } else {
                NonZeroU64::new(7)
            };
            cache.insert(newer, 0, newest);
            let (resume, waiting) = mpsc::channel();
            let (completed, done) = mpsc::channel();
            let worker_cache = Arc::clone(&cache);
            let worker = std::thread::spawn(move || {
                for _ in 0..2 {
                    waiting.recv_timeout(Duration::from_secs(5)).unwrap();
                    worker_cache.insert(older, 0, oldest);
                    completed.send(()).unwrap();
                }
            });
            for id in 1..33 {
                cache.insert(cache.begin_request(), id, NonZeroU64::new(id));
            }
            assert_eq!(cache.get(0), Lookup::Miss);
            resume.send(()).unwrap();
            done.recv_timeout(Duration::from_secs(5)).unwrap();
            assert_eq!(cache.get(0), cached(oldest));
            assert_eq!(cache.get(1), Lookup::Miss);

            for id in 33..97 {
                cache.insert(cache.begin_request(), id, None);
            }
            assert_eq!(cache.get(0), Lookup::Miss);
            cache.insert(newer, 0, newest);
            resume.send(()).unwrap();
            done.recv_timeout(Duration::from_secs(5)).unwrap();
            worker.join().unwrap();
            assert_eq!(cache.get(0), cached(newest));
            assert_eq!(cache.entry_count(), 32);
        }
    }
}
