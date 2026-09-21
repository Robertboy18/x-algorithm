use crate::models::MediaFeature;
use anyhow::{anyhow, Result};
use std::collections::HashMap;
use std::sync::Arc;
use thrift::protocol::{TInputProtocol, TOutputProtocol, TSerializable, TType};
use tonic::async_trait;
use xai_core_entities::entities::{
    EditControl, ExclusiveTweetControl, MediaEntity, Share, TakedownReason,
};
use xai_strato::strato_thrift::{strato_decode, StratoResult};
use xai_strato::{encode, MValCodec, StratoGrpc};

const COLUMN: &str = "tweetypie/federated/tweetForVisibility.Tweet";
const OPERATION: &str = "fetch";

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct TweetForVisibility {
    pub(crate) author_id: u64,
    pub(crate) source_tweet_id: Option<u64>,
    pub(crate) is_nullcast: bool,
    pub(crate) nsfw_user: bool,
    pub(crate) nsfw_admin: bool,
    pub(crate) has_takedown: bool,
    pub(crate) takedown_reasons: Vec<TakedownReason>,
    pub(crate) media: MediaFeature,
    pub(crate) is_community_tweet: bool,
    pub(crate) edit_control: Option<EditControl>,
    pub(crate) exclusive_conversation_author_id: Option<u64>,
}

#[async_trait]
pub(crate) trait TweetForVisibilitySource: Send + Sync {
    async fn get_tweets_for_visibility(
        &self,
        tweet_ids: &[u64],
    ) -> HashMap<u64, Result<Option<TweetForVisibility>>>;
}

pub(crate) struct ProdTweetForVisibilitySource {
    pub(crate) grpc_client: Arc<StratoGrpc>,
}

#[async_trait]
impl TweetForVisibilitySource for ProdTweetForVisibilitySource {
    async fn get_tweets_for_visibility(
        &self,
        tweet_ids: &[u64],
    ) -> HashMap<u64, Result<Option<TweetForVisibility>>> {
        let calls = tweet_ids
            .iter()
            .map(|tweet_id| {
                (
                    COLUMN.to_string(),
                    OPERATION.to_string(),
                    vec![encode(&(*tweet_id, ()))],
                )
            })
            .collect();

        let result_batch = self.grpc_client.batch_call(calls, None).await;
        tweet_ids
            .iter()
            .zip(result_batch)
            .map(|(tweet_id, bytes_result)| {
                let item = bytes_result.and_then(|bytes| decode_tweet_for_visibility(&bytes));
                (*tweet_id, item)
            })
            .collect()
    }
}

#[cfg(test)]
#[derive(Default)]
pub(crate) struct MockTweetForVisibilitySource {
    pub(crate) tweets: HashMap<u64, Option<TweetForVisibility>>,
    pub(crate) errors: HashMap<u64, String>,
    pub(crate) requests: std::sync::Mutex<Vec<Vec<u64>>>,
}

#[cfg(test)]
#[async_trait]
impl TweetForVisibilitySource for MockTweetForVisibilitySource {
    async fn get_tweets_for_visibility(
        &self,
        tweet_ids: &[u64],
    ) -> HashMap<u64, Result<Option<TweetForVisibility>>> {
        self.requests.lock().unwrap().push(tweet_ids.to_vec());
        tweet_ids
            .iter()
            .map(|id| {
                let item = match self.errors.get(id) {
                    Some(message) => Err(anyhow!("{message}")),
                    None => Ok(self.tweets.get(id).cloned().unwrap_or(None)),
                };
                (*id, item)
            })
            .collect()
    }
}

pub(crate) fn decode_tweet_for_visibility(bytes: &[u8]) -> Result<Option<TweetForVisibility>> {
    match std::panic::catch_unwind(|| strato_decode::<Tweet>(bytes))
        .map_err(|_| anyhow!("MVal decoder panicked"))?
    {
        Ok(StratoResult::Ok { value, .. }) => Ok(value.and_then(Tweet::project)),
        Ok(StratoResult::Err { code, message }) => {
            Err(anyhow!("Strato error code {code}: {message}"))
        }
        Err(e) => Err(anyhow!("MVal decode error: {e}")),
    }
}

#[derive(Default)]
struct Tweet {
    core_data: Option<CoreData>,
    media: Vec<MediaEntity>,
    takedown_reasons: Vec<TakedownReason>,
    has_communities: bool,
    exclusive_tweet_control: Option<ExclusiveTweetControl>,
    edit_control: Option<EditControl>,
    has_media_refs: bool,
    has_card_reference: bool,
}

#[derive(Default)]
struct CoreData {
    user_id: u64,
    share: Option<Share>,
    has_takedown: bool,
    nsfw_user: bool,
    nsfw_admin: bool,
    nullcast: bool,
}

impl Tweet {
    fn project(self) -> Option<TweetForVisibility> {
        let core_data = self.core_data?;
        Some(TweetForVisibility {
            author_id: core_data.user_id,
            source_tweet_id: core_data.share.map(|share| share.source_tweet_id),
            is_nullcast: core_data.nullcast,
            nsfw_user: core_data.nsfw_user,
            nsfw_admin: core_data.nsfw_admin,
            has_takedown: core_data.has_takedown,
            takedown_reasons: self.takedown_reasons,
            media: MediaFeature {
                has_media: self.has_media_refs || self.has_card_reference,
                ..super::tes_hydrator::media_feature(self.media)
            },
            is_community_tweet: self.has_communities,
            edit_control: self.edit_control,
            exclusive_conversation_author_id: self
                .exclusive_tweet_control
                .map(|control| control.conversation_author_id),
        })
    }
}

impl TSerializable for Tweet {
    fn read_from_in_protocol(proto: &mut dyn TInputProtocol) -> thrift::Result<Self> {
        proto.read_struct_begin()?;
        let mut tweet = Tweet::default();
        loop {
            let field = proto.read_field_begin()?;
            if field.field_type == TType::Stop {
                break;
            }
            match field.id {
                Some(2) => tweet.core_data = Some(read_core_data(proto)?),
                Some(7) => {
                    let list = proto.read_list_begin()?;
                    let mut media = Vec::with_capacity(list.size as usize);
                    for _ in 0..list.size {
                        media.push(MediaEntity::read_from_in_protocol(proto)?);
                    }
                    proto.read_list_end()?;
                    tweet.media = media;
                }
                Some(30) => tweet.takedown_reasons = Vec::<TakedownReason>::from_thrift(proto),
                Some(118) => {
                    proto.skip(field.field_type)?;
                    tweet.has_card_reference = true;
                }
                Some(125) => tweet.has_communities = read_communities_non_empty(proto)?,
                Some(155) => {
                    tweet.exclusive_tweet_control = Some(ExclusiveTweetControl::from_thrift(proto));
                }
                Some(157) => tweet.edit_control = Some(EditControl::from_thrift(proto)),
                Some(162) => tweet.has_media_refs = skip_list_non_empty(proto)?,
                _ => proto.skip(field.field_type)?,
            }
            proto.read_field_end()?;
        }
        proto.read_struct_end()?;
        Ok(tweet)
    }

    fn write_to_out_protocol(&self, _proto: &mut dyn TOutputProtocol) -> thrift::Result<()> {
        Err(thrift::new_protocol_error(
            thrift::ProtocolErrorKind::NotImplemented,
            "Tweet is decode-only",
        ))
    }
}

fn read_core_data(proto: &mut dyn TInputProtocol) -> thrift::Result<CoreData> {
    proto.read_struct_begin()?;
    let mut core_data = CoreData::default();
    loop {
        let field = proto.read_field_begin()?;
        if field.field_type == TType::Stop {
            break;
        }
        match field.id {
            Some(1) => core_data.user_id = proto.read_i64()? as u64,
            Some(7) => core_data.share = Some(Share::from_thrift(proto)),
            Some(8) => core_data.has_takedown = proto.read_bool()?,
            Some(9) => core_data.nsfw_user = proto.read_bool()?,
            Some(10) => core_data.nsfw_admin = proto.read_bool()?,
            Some(11) => core_data.nullcast = proto.read_bool()?,
            _ => proto.skip(field.field_type)?,
        }
        proto.read_field_end()?;
    }
    proto.read_struct_end()?;
    Ok(core_data)
}

fn skip_list_non_empty(proto: &mut dyn TInputProtocol) -> thrift::Result<bool> {
    let list = proto.read_list_begin()?;
    for _ in 0..list.size {
        proto.skip(list.element_type)?;
    }
    proto.read_list_end()?;
    Ok(list.size > 0)
}

fn read_communities_non_empty(proto: &mut dyn TInputProtocol) -> thrift::Result<bool> {
    proto.read_struct_begin()?;
    let mut non_empty = false;
    loop {
        let field = proto.read_field_begin()?;
        if field.field_type == TType::Stop {
            break;
        }
        match field.id {
            Some(1) => non_empty = skip_list_non_empty(proto)?,
            _ => proto.skip(field.field_type)?,
        }
        proto.read_field_end()?;
    }
    proto.read_struct_end()?;
    Ok(non_empty)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hydration::batch::TweetHydrationBatch;
    use crate::hydration::tes_hydrator::TesHydrator;
    use crate::models::{
        resolve_candidate, CoreFeature, NsfwFeature, RawCandidate, TweetFeatures, TweetId,
    };
    use thrift::protocol::{
        TBinaryOutputProtocol, TFieldIdentifier, TListIdentifier, TStructIdentifier,
    };
    use xai_core_entities::entities::{EditControlInitial, PureCoreData};
    use xai_core_entities::tweet_entity_service_client::MockTESClient;
    use xai_x_thrift::media_common::MediaKey;
    use xai_x_thrift::media_information::{AdditionalMetadata, GeoRestrictions, Restrictions};

    type Proto<'a> = TBinaryOutputProtocol<&'a mut Vec<u8>>;
    type Fields<'a> = &'a mut dyn FnMut(&mut Proto<'_>);

    const TWEET_ID: i64 = 10;
    const AUTHOR_ID: i64 = 7001;
    const SOURCE_TWEET_ID: i64 = 9;
    const PURE_CORE_SOURCE_TWEET_ID: u64 = 8;
    const CONVERSATION_AUTHOR_ID: i64 = 7003;
    const PURE_CORE_TEXT: &str = "pure core text";

    fn field(proto: &mut Proto<'_>, id: i16, ty: TType, value: impl FnOnce(&mut Proto<'_>)) {
        proto
            .write_field_begin(&TFieldIdentifier::new("", ty, id))
            .unwrap();
        value(proto);
        proto.write_field_end().unwrap();
    }

    fn bare_struct(proto: &mut Proto<'_>, fields: impl FnOnce(&mut Proto<'_>)) {
        proto
            .write_struct_begin(&TStructIdentifier::new(""))
            .unwrap();
        fields(proto);
        proto.write_field_stop().unwrap();
        proto.write_struct_end().unwrap();
    }

    fn structure(proto: &mut Proto<'_>, id: i16, fields: impl FnOnce(&mut Proto<'_>)) {
        field(proto, id, TType::Struct, |p| bare_struct(p, fields));
    }

    fn list(
        proto: &mut Proto<'_>,
        id: i16,
        ty: TType,
        len: usize,
        elements: impl FnOnce(&mut Proto<'_>),
    ) {
        field(proto, id, TType::List, |p| {
            p.write_list_begin(&TListIdentifier::new(ty, len as i32))
                .unwrap();
            elements(p);
            p.write_list_end().unwrap();
        });
    }

    fn i64_field(proto: &mut Proto<'_>, id: i16, value: i64) {
        field(proto, id, TType::I64, |p| p.write_i64(value).unwrap());
    }

    fn bool_field(proto: &mut Proto<'_>, id: i16, value: bool) {
        field(proto, id, TType::Bool, |p| p.write_bool(value).unwrap());
    }

    fn encode_option(option: Fields<'_>) -> Vec<u8> {
        let mut bytes = Vec::new();
        let mut proto = TBinaryOutputProtocol::new(&mut bytes, false);
        bare_struct(&mut proto, |p| {
            structure(p, 4, |p| {
                structure(p, 2556, |p| structure(p, 118, |p| option(p)));
            });
        });
        bytes
    }

    fn encode_tweet(core_data: Fields<'_>, rest: Fields<'_>) -> Vec<u8> {
        encode_option(&mut |p| {
            structure(p, 26900, |p| {
                i64_field(p, 1, TWEET_ID);
                structure(p, 2, |p| {
                    i64_field(p, 1, AUTHOR_ID);
                    field(p, 2, TType::String, |p| {
                        p.write_string("composite text").unwrap()
                    });
                    core_data(p);
                });
                rest(p);
            });
        })
    }

    fn media_fixture(has_refs: bool, has_card: bool, entities: &[MediaEntity]) -> Vec<u8> {
        encode_tweet(&mut |_| {}, &mut |p| {
            list(p, 7, TType::Struct, entities.len(), |p| {
                for entity in entities {
                    entity.write_to_out_protocol(p).unwrap();
                }
            });
            if has_card {
                structure(p, 118, |_| {});
            }
            list(p, 162, TType::Struct, usize::from(has_refs), |p| {
                if has_refs {
                    bare_struct(p, |_| {});
                }
            });
        })
    }

    fn restricted_media(has_key: bool, dmca: bool, allow: &[&str], deny: &[&str]) -> MediaEntity {
        MediaEntity {
            media_key: has_key.then(MediaKey::default),
            additional_metadata: Some(AdditionalMetadata {
                restrictions: Some(Restrictions {
                    is_dmca: Some(dmca),
                    geo_restrictions: Some(GeoRestrictions {
                        whitelisted_country_codes: Some(
                            allow.iter().map(|s| s.to_string()).collect(),
                        ),
                        blacklisted_country_codes: Some(
                            deny.iter().map(|s| s.to_string()).collect(),
                        ),
                    }),
                    ..Default::default()
                }),
                ..Default::default()
            }),
            ..Default::default()
        }
    }

    fn assemble_fixture(bytes: &[u8]) -> TweetFeatures {
        let core = HashMap::from([(
            TweetId(TWEET_ID as u64),
            PureCoreData {
                author_id: 100,
                text: PURE_CORE_TEXT.to_string(),
                source_tweet_id: Some(PURE_CORE_SOURCE_TWEET_ID),
                ..Default::default()
            },
        )]);
        let candidate = resolve_candidate(
            &RawCandidate {
                tweet_id: TweetId(TWEET_ID as u64),
                request_author_id: None,
            },
            &core,
            &HashMap::new(),
        )
        .unwrap();
        let composite = TweetHydrationBatch::from_results(
            [TweetId(TWEET_ID as u64)],
            HashMap::from([(
                TweetId(TWEET_ID as u64),
                Ok::<_, anyhow::Error>(decode_tweet_for_visibility(bytes).unwrap()),
            )]),
        );
        TesHydrator::new(
            Arc::new(MockTESClient::default()),
            Arc::new(MockTweetForVisibilitySource::default()),
            None,
        )
        .assemble_tweet_features(&[candidate], &core, &composite)
        .remove(&TweetId(TWEET_ID as u64))
        .unwrap()
    }

    fn pure_core_features() -> TweetFeatures {
        TweetFeatures {
            core: CoreFeature {
                text: PURE_CORE_TEXT.to_string(),
                source_tweet_id: None,
            },
            ..Default::default()
        }
    }

    #[test]
    fn projects_every_read_field_of_a_found_tweet() {
        let bytes = encode_tweet(
            &mut |p| {
                structure(p, 7, |p| i64_field(p, 1, SOURCE_TWEET_ID));
                bool_field(p, 8, true);
                bool_field(p, 9, true);
                bool_field(p, 10, false);
                bool_field(p, 11, true);
            },
            &mut |p| {
                list(p, 7, TType::Struct, 1, |p| {
                    restricted_media(true, true, &[], &[])
                        .write_to_out_protocol(p)
                        .unwrap()
                });
                list(p, 30, TType::Struct, 1, |p| {
                    bare_struct(p, |p| structure(p, 4, |_| {}))
                });
                structure(p, 118, |_| {});
                structure(p, 125, |p| {
                    list(p, 1, TType::I64, 1, |p| p.write_i64(500).unwrap())
                });
                structure(p, 155, |p| i64_field(p, 1, CONVERSATION_AUTHOR_ID));
                structure(p, 157, |p| {
                    structure(p, 1, |p| {
                        list(p, 1, TType::I64, 1, |p| p.write_i64(TWEET_ID).unwrap());
                        i64_field(p, 3, 4);
                    })
                });
                list(p, 162, TType::Struct, 1, |p| bare_struct(p, |_| {}));
            },
        );
        let media = MediaFeature {
            has_media: true,
            has_dmca_media: true,
            ..Default::default()
        };
        let edit_control = Some(EditControl::Initial(EditControlInitial {
            edit_tweet_ids: vec![TWEET_ID as u64],
            edits_remaining: Some(4),
            ..Default::default()
        }));

        assert_eq!(
            decode_tweet_for_visibility(&bytes).unwrap().unwrap(),
            TweetForVisibility {
                author_id: AUTHOR_ID as u64,
                source_tweet_id: Some(SOURCE_TWEET_ID as u64),
                is_nullcast: true,
                nsfw_user: true,
                nsfw_admin: false,
                has_takedown: true,
                takedown_reasons: vec![TakedownReason::Dmca],
                media: media.clone(),
                is_community_tweet: true,
                edit_control: edit_control.clone(),
                exclusive_conversation_author_id: Some(CONVERSATION_AUTHOR_ID as u64),
            }
        );
        assert_eq!(
            assemble_fixture(&bytes),
            TweetFeatures {
                core: CoreFeature {
                    text: PURE_CORE_TEXT.to_string(),
                    source_tweet_id: Some(SOURCE_TWEET_ID as u64),
                },
                media,
                takedown_reasons: vec![TakedownReason::Dmca],
                nsfw: NsfwFeature {
                    user: true,
                    admin: false
                },
                is_nullcast: true,
                is_community_tweet: true,
                edit_control,
            }
        );
    }

    #[test]
    fn plain_tweet_takes_text_from_pure_core_but_not_its_retweet_source() {
        let bytes = encode_tweet(&mut |_| {}, &mut |_| {});

        assert_eq!(assemble_fixture(&bytes), pure_core_features());
    }

    #[test]
    fn media_restrictions_fold_only_across_keyed_entities() {
        let bytes = media_fixture(
            true,
            false,
            &[
                restricted_media(true, true, &["us"], &["de"]),
                restricted_media(true, false, &["gb"], &["fr"]),
                restricted_media(false, true, &["ignored"], &["ignored"]),
                MediaEntity::default(),
            ],
        );

        assert_eq!(
            assemble_fixture(&bytes).media,
            MediaFeature {
                has_media: true,
                has_dmca_media: true,
                geo_allow_list: vec!["us".to_string(), "gb".to_string()],
                geo_deny_list: vec!["de".to_string(), "fr".to_string()],
            }
        );
    }

    #[test]
    fn media_presence_comes_from_refs_or_card_reference_never_the_media_list() {
        for (name, bytes, expected) in [
            ("refs", media_fixture(true, false, &[]), true),
            ("card", media_fixture(false, true, &[]), true),
            (
                "entities only",
                media_fixture(false, false, &[MediaEntity::default()]),
                false,
            ),
        ] {
            assert_eq!(assemble_fixture(&bytes).media.has_media, expected, "{name}");
        }
    }

    #[test]
    fn community_tweet_requires_a_nonempty_community_id_list() {
        for (community_ids, expected) in [(vec![500], true), (vec![], false)] {
            let bytes = encode_tweet(&mut |_| {}, &mut |p| {
                structure(p, 125, |p| {
                    list(p, 1, TType::I64, community_ids.len(), |p| {
                        for id in &community_ids {
                            p.write_i64(*id).unwrap();
                        }
                    });
                });
            });

            assert_eq!(assemble_fixture(&bytes).is_community_tweet, expected);
        }
    }

    #[test]
    fn truncated_share_is_an_error_not_a_panic() {
        let bytes = encode_tweet(
            &mut |p| structure(p, 7, |p| i64_field(p, 1, SOURCE_TWEET_ID)),
            &mut |_| {},
        );
        let end = bytes
            .windows(8)
            .position(|window| window == SOURCE_TWEET_ID.to_be_bytes())
            .unwrap()
            + 4;

        assert!(decode_tweet_for_visibility(&bytes[..end]).is_err());
    }

    #[test]
    fn missing_tweet_keeps_pure_core_text_and_defaults_the_rest() {
        for bytes in [
            encode_option(&mut |p| field(p, 9048, TType::Void, |_| {})),
            encode_option(&mut |p| structure(p, 26900, |p| i64_field(p, 1, TWEET_ID))),
        ] {
            assert!(decode_tweet_for_visibility(&bytes).unwrap().is_none());
            assert_eq!(assemble_fixture(&bytes), pure_core_features());
        }
    }
}
