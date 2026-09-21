use std::num::NonZeroU64;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct AuthorFeatures {
    pub is_suspended: bool,
    pub is_deactivated: bool,
    pub is_protected: bool,
    pub is_nsfw_user: bool,
    pub is_nsfw_admin: bool,
    pub is_erased: bool,
    pub is_offboarded: bool,
    pub user_labels: AuthorLabelSet,
}

#[allow(dead_code)]
#[derive(Clone, Copy, Debug)]
pub(crate) struct CompactAuthorFeatures(NonZeroU64);

const COMPACT_PRESENT_BIT: NonZeroU64 = match NonZeroU64::new(1 << 63) {
    Some(bit) => bit,
    None => unreachable!(),
};
const COMPACT_LABEL_SHIFT: u32 = 8;
const COMPACT_LABEL_MASK: u64 = (1 << (63 - COMPACT_LABEL_SHIFT)) - 1;

#[allow(dead_code)]
impl AuthorFeatures {
    pub(crate) fn to_compact(self) -> CompactAuthorFeatures {
        let flags = self.is_suspended as u64
            | (self.is_deactivated as u64) << 1
            | (self.is_protected as u64) << 2
            | (self.is_nsfw_user as u64) << 3
            | (self.is_nsfw_admin as u64) << 4
            | (self.is_erased as u64) << 5
            | (self.is_offboarded as u64) << 6;
        CompactAuthorFeatures(
            COMPACT_PRESENT_BIT | flags | (self.user_labels.0 << COMPACT_LABEL_SHIFT),
        )
    }

    pub(crate) fn from_compact(compact: CompactAuthorFeatures) -> Self {
        let word = compact.0.get();
        Self {
            is_suspended: word & 1 != 0,
            is_deactivated: word & (1 << 1) != 0,
            is_protected: word & (1 << 2) != 0,
            is_nsfw_user: word & (1 << 3) != 0,
            is_nsfw_admin: word & (1 << 4) != 0,
            is_erased: word & (1 << 5) != 0,
            is_offboarded: word & (1 << 6) != 0,
            user_labels: AuthorLabelSet((word >> COMPACT_LABEL_SHIFT) & COMPACT_LABEL_MASK),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum AuthorLabel {
    NsfwHighRecall,
    NsfwHighPrecision,
    NsfwNearPerfect,
    NsfwAvatarImage,
    NsfwBannerImage,
    SpamHighRecall,
    AbusiveHighRecall,
    Compromised,
    ReadOnly,
    ImpersonationHighPrecision,
    DoNotAmplify,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct AuthorLabelSet(u64);

impl AuthorLabelSet {
    #[inline]
    pub fn insert(&mut self, label: AuthorLabel) {
        self.0 |= 1 << label as u8;
    }

    #[inline]
    pub fn has_label(self, label: AuthorLabel) -> bool {
        self.0 & (1 << label as u8) != 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_copy<T: Copy>() {}

    const ALL_LABELS: [AuthorLabel; 11] = [
        AuthorLabel::NsfwHighRecall,
        AuthorLabel::NsfwHighPrecision,
        AuthorLabel::NsfwNearPerfect,
        AuthorLabel::NsfwAvatarImage,
        AuthorLabel::NsfwBannerImage,
        AuthorLabel::SpamHighRecall,
        AuthorLabel::AbusiveHighRecall,
        AuthorLabel::Compromised,
        AuthorLabel::ReadOnly,
        AuthorLabel::ImpersonationHighPrecision,
        AuthorLabel::DoNotAmplify,
    ];

    #[test]
    fn compact_is_copy_and_option_is_8_bytes() {
        assert_copy::<CompactAuthorFeatures>();
        assert_eq!(std::mem::size_of::<Option<CompactAuthorFeatures>>(), 8);
    }

    #[test]
    fn compact_round_trips_every_flag_combination_and_label_subset() {
        for flag_bits in 0u32..1 << 7 {
            for label_bits in 0u32..1 << ALL_LABELS.len() {
                let mut user_labels = AuthorLabelSet::default();
                for (i, label) in ALL_LABELS.iter().enumerate() {
                    if label_bits & (1 << i) != 0 {
                        user_labels.insert(*label);
                    }
                }
                let features = AuthorFeatures {
                    is_suspended: flag_bits & 1 != 0,
                    is_deactivated: flag_bits & (1 << 1) != 0,
                    is_protected: flag_bits & (1 << 2) != 0,
                    is_nsfw_user: flag_bits & (1 << 3) != 0,
                    is_nsfw_admin: flag_bits & (1 << 4) != 0,
                    is_erased: flag_bits & (1 << 5) != 0,
                    is_offboarded: flag_bits & (1 << 6) != 0,
                    user_labels,
                };
                assert_eq!(
                    AuthorFeatures::from_compact(features.to_compact()),
                    features
                );
            }
        }
    }

    #[test]
    fn author_features_is_copy_and_16_bytes_with_option_niche() {
        assert_copy::<AuthorFeatures>();
        assert_eq!(std::mem::size_of::<AuthorFeatures>(), 16);
        assert_eq!(std::mem::size_of::<Option<AuthorFeatures>>(), 16);
    }

    #[test]
    fn label_set_membership() {
        let mut set = AuthorLabelSet::default();
        assert!(!set.has_label(AuthorLabel::NsfwHighRecall));
        set.insert(AuthorLabel::NsfwHighRecall);
        set.insert(AuthorLabel::DoNotAmplify);
        assert!(set.has_label(AuthorLabel::NsfwHighRecall));
        assert!(set.has_label(AuthorLabel::DoNotAmplify));
        assert!(!set.has_label(AuthorLabel::Compromised));
    }
}
