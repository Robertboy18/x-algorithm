#[derive(Clone, Debug, Default, PartialEq)]
pub struct CoreFeature {
    pub text: String,
    pub source_tweet_id: Option<u64>,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct MediaFeature {
    pub has_media: bool,
    pub has_dmca_media: bool,
    pub geo_allow_list: Vec<String>,
    pub geo_deny_list: Vec<String>,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct NsfwFeature {
    pub user: bool,
    pub admin: bool,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct TweetFeatures {
    pub core: CoreFeature,
    pub media: MediaFeature,
    pub takedown_reasons: Vec<xai_core_entities::entities::TakedownReason>,
    pub nsfw: NsfwFeature,
    pub is_nullcast: bool,
    pub is_community_tweet: bool,
    pub edit_control: Option<xai_core_entities::entities::EditControl>,
}
