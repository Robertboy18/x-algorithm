use crate::hydration::fallback_cache::FallbackCache;
use crate::models::{AuthorId, PureCore, TweetId};
use xai_core_entities::entities::PureCoreData;

pub(crate) type PureCoreFallbackCache = FallbackCache<PureCore>;

pub(crate) fn pure_core_fallback_cache(capacity: usize) -> PureCoreFallbackCache {
    FallbackCache::new("author_id", capacity)
}

pub(crate) fn pure_core(core: &PureCoreData) -> PureCore {
    PureCore {
        author_id: AuthorId(core.author_id),
        source_tweet_id: core.source_tweet_id.map(TweetId),
        source_author_id: core.source_user_id.filter(|&id| id != 0).map(AuthorId),
        direct_reply_root_author_id: direct_reply_root_author(core),
    }
}

fn direct_reply_root_author(core: &PureCoreData) -> Option<AuthorId> {
    core.in_reply_to_tweet_id
        .filter(|&replied_to| core.conversation_id == Some(replied_to))
        .and(core.in_reply_to_user_id)
        .map(AuthorId)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_share_without_its_user_id_leaves_the_source_author_unknown() {
        for source_user_id in [Some(0), None] {
            let core = PureCoreData {
                author_id: 10,
                source_tweet_id: Some(5),
                source_user_id,
                ..Default::default()
            };
            assert_eq!(pure_core(&core).source_author_id, None);
        }
    }
}
