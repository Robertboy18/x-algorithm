use crate::hydration::batch::{Hydrated, HydrationBatch};
use std::collections::hash_map::Entry;
use std::collections::HashMap;

pub(super) struct Fetcher<V> {
    states: HashMap<u64, State<V>>,
    has_incomplete: bool,
}

enum State<V> {
    Pending,
    Landed(Hydrated<V>),
}

impl<V> Default for Fetcher<V> {
    fn default() -> Self {
        Self {
            states: HashMap::new(),
            has_incomplete: false,
        }
    }
}

impl<V> Fetcher<V> {
    pub(super) fn land(&mut self, claimed: &[u64], batch: HydrationBatch<u64, V>) {
        let mut answers = batch.into_hydrated();
        for &key in claimed {
            let answer = answers.remove(&key).unwrap_or(Hydrated::NotFound);
            self.has_incomplete |= !answer.is_complete();
            self.states.insert(key, State::Landed(answer));
        }
    }

    pub(super) fn get(&self, key: u64) -> Option<&V> {
        match self.states.get(&key)? {
            State::Landed(answer) => answer.value(),
            State::Pending => None,
        }
    }

    pub(super) fn take(&mut self, key: u64) -> Option<V> {
        match self.states.remove(&key)? {
            State::Landed(answer) => answer.into_value(),
            State::Pending => None,
        }
    }
}

pub(super) trait AnyFetcher {
    fn claim(&mut self, keys: Vec<u64>) -> Vec<u64>;
    fn has_claimed(&self) -> bool;
    fn is_claimed(&self, key: u64) -> bool;
    fn has_incomplete(&self) -> bool;
    fn is_incomplete(&self, key: u64) -> bool;
}

impl<V> AnyFetcher for Fetcher<V> {
    fn claim(&mut self, mut keys: Vec<u64>) -> Vec<u64> {
        self.states.reserve(keys.len());
        keys.retain(|&key| match self.states.entry(key) {
            Entry::Occupied(_) => false,
            Entry::Vacant(entry) => {
                entry.insert(State::Pending);
                true
            }
        });
        keys
    }

    fn has_claimed(&self) -> bool {
        !self.states.is_empty()
    }

    fn is_claimed(&self, key: u64) -> bool {
        self.states.contains_key(&key)
    }

    fn has_incomplete(&self) -> bool {
        self.has_incomplete
    }

    fn is_incomplete(&self, key: u64) -> bool {
        matches!(self.states.get(&key), Some(State::Landed(answer)) if !answer.is_complete())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hydration::batch::HydrationError;

    #[test]
    fn only_found_and_not_found_keys_are_complete() {
        let mut fetcher = Fetcher::default();
        let keys = fetcher.claim(vec![1, 2, 3, 4]);
        fetcher.land(
            &keys,
            HydrationBatch::from_hydrated(HashMap::from([
                (1, Hydrated::Found(7)),
                (2, Hydrated::NotFound),
                (3, Hydrated::Partial(7)),
                (4, Hydrated::Failed(HydrationError::Timeout)),
            ])),
        );

        let incomplete: Vec<u64> = keys
            .into_iter()
            .filter(|&key| fetcher.is_incomplete(key))
            .collect();
        assert_eq!(incomplete, [3, 4]);
        assert_eq!(fetcher.get(3), Some(&7));
    }
}
