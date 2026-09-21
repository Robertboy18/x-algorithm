#[derive(Clone, Debug, PartialEq)]
pub struct ExclusiveContentFeatures {
    pub conversation_author_id: u64,
    pub viewer_super_follows_author: bool,
}
