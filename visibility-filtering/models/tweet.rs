use xai_core_entities::entities::{EditControl, TakedownReason};

#[derive(Clone, Debug, Default, PartialEq)]
pub struct MediaFeature {
    pub has_media: bool,
    pub has_uploaded_media: bool,
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
    pub source_tweet_id: Option<u64>,
    pub media: MediaFeature,
    pub takedown_reasons: Vec<TakedownReason>,
    pub nsfw: NsfwFeature,
    pub is_nullcast: bool,
    pub is_community_tweet: bool,
    pub edit_control: Option<EditControl>,
    pub exclusive_conversation_author_id: Option<u64>,
}

impl TweetFeatures {
    pub fn is_retweet(&self) -> bool {
        self.source_tweet_id.is_some()
    }

    pub fn has_media(&self) -> bool {
        self.media.has_media
    }

    pub fn has_dmca_media(&self) -> bool {
        self.media.has_dmca_media
    }

    pub fn is_superseded_edit(&self, tweet_id: u64) -> bool {
        let edit_tweet_ids = match &self.edit_control {
            Some(EditControl::Initial(initial)) => &initial.edit_tweet_ids,
            Some(EditControl::Edit(edit)) => match &edit.edit_control_initial {
                Some(initial) => &initial.edit_tweet_ids,
                None => return false,
            },
            None => return false,
        };
        edit_tweet_ids.last().is_some_and(|&last| last != tweet_id)
    }

    pub fn legal_takedown_in(&self, request_country: Option<&str>) -> bool {
        self.takedown_in(request_country, legal_takedown_country)
    }

    pub fn local_laws_takedown_in(&self, request_country: Option<&str>) -> bool {
        self.takedown_in(request_country, local_laws_takedown_country)
    }

    fn takedown_in(
        &self,
        request_country: Option<&str>,
        extractor: fn(&TakedownReason) -> Option<&str>,
    ) -> bool {
        self.takedown_reasons.iter().filter_map(extractor).any(|c| {
            is_worldwide_code(c) || request_country.is_some_and(|v| c.eq_ignore_ascii_case(v))
        })
    }

    pub fn media_restricted_in(&self, request_country: Option<&str>) -> bool {
        let country = request_country.unwrap_or(WORLDWIDE_COUNTRY_CODE);
        let allow = &self.media.geo_allow_list;
        let deny = &self.media.geo_deny_list;
        (!allow.is_empty() && !allow.iter().any(|c| c.eq_ignore_ascii_case(country)))
            || deny.iter().any(|c| c.eq_ignore_ascii_case(country))
    }
}

const WORLDWIDE_COUNTRY_CODE: &str = "xx";
const WORLDWIDE_COPYRIGHT_COUNTRY_CODE: &str = "xy";

fn is_worldwide_code(country_code: &str) -> bool {
    country_code.eq_ignore_ascii_case(WORLDWIDE_COUNTRY_CODE)
        || country_code.eq_ignore_ascii_case(WORLDWIDE_COPYRIGHT_COUNTRY_CODE)
}

fn legal_takedown_country(reason: &TakedownReason) -> Option<&str> {
    match reason {
        TakedownReason::LegalRequest { country_code }
            if country_code.eq_ignore_ascii_case(WORLDWIDE_COPYRIGHT_COUNTRY_CODE) =>
        {
            None
        }
        TakedownReason::LegalRequest { country_code }
        | TakedownReason::UnspecifiedReason { country_code } => Some(country_code),
        TakedownReason::Dmca => Some(WORLDWIDE_COPYRIGHT_COUNTRY_CODE),
        _ => None,
    }
}

fn local_laws_takedown_country(reason: &TakedownReason) -> Option<&str> {
    match reason {
        TakedownReason::BystanderReport { country_code } if !is_worldwide_code(country_code) => {
            Some(country_code)
        }
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use xai_core_entities::entities::{EditControlEdit, EditControlInitial};

    #[test]
    fn is_superseded_edit_only_when_a_later_edit_exists() {
        for (edit_control, superseded) in [
            (
                EditControl::Initial(EditControlInitial {
                    edit_tweet_ids: vec![20],
                    ..Default::default()
                }),
                false,
            ),
            (
                EditControl::Edit(EditControlEdit {
                    initial_tweet_id: 10,
                    edit_control_initial: Some(EditControlInitial {
                        edit_tweet_ids: vec![10, 20, 30],
                        ..Default::default()
                    }),
                }),
                true,
            ),
            (
                EditControl::Edit(EditControlEdit {
                    initial_tweet_id: 10,
                    edit_control_initial: None,
                }),
                false,
            ),
        ] {
            let features = TweetFeatures {
                edit_control: Some(edit_control),
                ..Default::default()
            };
            assert_eq!(
                features.is_superseded_edit(20),
                superseded,
                "{:?}",
                features.edit_control
            );
        }
    }
}
