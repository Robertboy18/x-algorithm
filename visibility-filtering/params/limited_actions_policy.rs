use crate::limited_actions_copy::{Prompt, PromptCopy};
use crate::models::LimitedEngagementReason;
use xai_feature_switches::FeatureSwitchResults;
use xai_stats_receiver::StatsReceiverExt;

pub(super) const REASON_FIELD: &str = "LimitedEngagementReason";
const LIMITED_ACTIONS_KEY: &str = "limited_actions_policy_limited_actions";
const COPY_NAMESPACE_KEY: &str = "limited_actions_policy_copy_namespace";
const PROMPT_TYPE_KEY: &str = "limited_actions_policy_prompt_type";
const LEARN_MORE_URL_KEY: &str = "limited_actions_policy_prompt_learn_more_url";
const MISSING_POLICY_COUNTER: &str = "limited_actions_policy_missing";

#[derive(Clone, Copy, Debug, PartialEq, Eq, strum::EnumString)]
#[strum(serialize_all = "snake_case")]
pub(crate) enum LimitedActionType {
    Reply,
    Retweet,
    QuoteTweet,
    Like,
    React,
    SendViaDm,
    AddToBookmarks,
    AddToMoment,
    PinToProfile,
    ViewTweetActivity,
    ShareTweetVia,
    Follow,
    ListsAddRemove,
    MuteConversation,
    Embed,
    ViewHiddenReplies,
    HideCommunityTweet,
    CopyLink,
    VoteOnPoll,
    RemoveFromCommunity,
    ShowRetweetActionMenu,
    ReplyDownVote,
    Autoplay,
    EditTweet,
    Highlight,
    ViewPostEngagements,
}

#[derive(Clone, Copy, strum::IntoStaticStr)]
#[strum(serialize_all = "snake_case")]
enum Missing {
    Actions,
    Copy,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct LimitedAction {
    pub(crate) action_type: LimitedActionType,
    pub(crate) prompt: Option<Prompt>,
}

#[derive(Default)]
pub(crate) struct LimitedActionsPolicies {
    policies: Vec<(LimitedEngagementReason, Vec<LimitedAction>)>,
}

impl LimitedActionsPolicies {
    pub(super) fn resolve(
        reasons: impl IntoIterator<Item = LimitedEngagementReason>,
        match_reason: impl Fn(LimitedEngagementReason) -> FeatureSwitchResults,
        prompt: impl Fn(LimitedActionType, &PromptCopy) -> Option<Prompt>,
        stats: Option<&dyn StatsReceiverExt>,
    ) -> Self {
        let mut matched: Vec<LimitedEngagementReason> = Vec::new();
        let mut policies = Vec::new();
        let count = |reason: LimitedEngagementReason, missing: Missing| {
            if let Some(stats) = stats {
                let labels = [("reason", reason.into()), ("missing", missing.into())];
                stats.incr(MISSING_POLICY_COUNTER, &labels, 1);
            }
        };
        for reason in reasons {
            if matched.contains(&reason) {
                continue;
            }
            matched.push(reason);
            let results = match_reason(reason);
            let action_types: Vec<LimitedActionType> = results
                .get_array_no_impression(LIMITED_ACTIONS_KEY)
                .into_iter()
                .flatten()
                .filter_map(|name| name.as_str()?.parse().ok())
                .collect();
            if action_types.is_empty() {
                count(reason, Missing::Actions);
                continue;
            }
            let copy = PromptCopy::from_switches(
                results.get_string_no_impression(COPY_NAMESPACE_KEY),
                results.get_string_no_impression(PROMPT_TYPE_KEY),
                results.get_string_no_impression(LEARN_MORE_URL_KEY),
            );
            let actions: Vec<LimitedAction> = action_types
                .into_iter()
                .map(|action_type| LimitedAction {
                    action_type,
                    prompt: copy.as_ref().and_then(|copy| prompt(action_type, copy)),
                })
                .collect();
            if copy.is_some() && actions.iter().any(|action| action.prompt.is_none()) {
                count(reason, Missing::Copy);
            }
            policies.push((reason, actions));
        }
        Self { policies }
    }

    pub(crate) fn actions(&self, reason: LimitedEngagementReason) -> &[LimitedAction] {
        self.policies
            .iter()
            .find(|(held, _)| *held == reason)
            .map_or(&[], |(_, actions)| actions)
    }

    #[cfg(test)]
    pub(crate) fn for_tests(
        policies: Vec<(LimitedEngagementReason, Vec<LimitedActionType>)>,
    ) -> Self {
        let unprompted = |action_type| LimitedAction {
            action_type,
            prompt: None,
        };
        Self {
            policies: policies
                .into_iter()
                .map(|(reason, types)| (reason, types.into_iter().map(unprompted).collect()))
                .collect(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::tests::Counted;
    use super::*;
    use crate::limited_actions_copy::LimitedActionsCopy;
    use crate::params::ClientSwitches;
    use arc_swap::ArcSwap;
    use std::sync::Arc;
    use xai_feature_switches::FeatureSwitches;
    use LimitedEngagementReason::{
        BlockedViewer, ConversationControl, ReadonlyViewer, RootAuthorBlockedViewer, StaleTweet,
    };

    #[test]
    fn each_reason_resolves_its_policy_from_the_scala_file() {
        use LimitedActionType as T;
        let expected = [
            (ConversationControl, vec![T::Reply]),
            (
                BlockedViewer,
                vec![
                    T::Reply,
                    T::Retweet,
                    T::QuoteTweet,
                    T::Like,
                    T::React,
                    T::SendViaDm,
                    T::AddToBookmarks,
                    T::AddToMoment,
                    T::PinToProfile,
                    T::ShareTweetVia,
                    T::Follow,
                    T::ListsAddRemove,
                    T::Embed,
                    T::CopyLink,
                    T::VoteOnPoll,
                    T::ShowRetweetActionMenu,
                    T::ReplyDownVote,
                    T::EditTweet,
                    T::Highlight,
                ],
            ),
            (RootAuthorBlockedViewer, vec![T::Reply]),
            (
                ReadonlyViewer,
                vec![
                    T::AddToBookmarks,
                    T::AddToMoment,
                    T::Embed,
                    T::HideCommunityTweet,
                    T::Like,
                    T::ListsAddRemove,
                    T::PinToProfile,
                    T::QuoteTweet,
                    T::React,
                    T::Reply,
                    T::Retweet,
                    T::VoteOnPoll,
                    T::EditTweet,
                    T::Highlight,
                ],
            ),
            (
                StaleTweet,
                vec![
                    T::Reply,
                    T::Retweet,
                    T::QuoteTweet,
                    T::Like,
                    T::React,
                    T::SendViaDm,
                    T::AddToBookmarks,
                    T::ShareTweetVia,
                    T::AddToMoment,
                    T::Embed,
                    T::HideCommunityTweet,
                    T::PinToProfile,
                    T::ViewTweetActivity,
                    T::VoteOnPoll,
                    T::CopyLink,
                ],
            ),
        ];
        let stats = Counted::default();
        let policies = ClientSwitches::for_tests().limited_actions_policies(
            None,
            None,
            None,
            expected.iter().map(|(reason, _)| *reason),
            &LimitedActionsCopy::from_json("[]"),
            Some(&stats),
        );
        for (reason, actions) in expected {
            let action_types: Vec<LimitedActionType> = policies
                .actions(reason)
                .iter()
                .map(|action| action.action_type)
                .collect();
            assert_eq!(action_types, actions, "{reason:?}");
        }
        let missing_copy = |reason: &str| {
            (
                MISSING_POLICY_COUNTER.to_string(),
                format!("reason={reason},missing=copy"),
                1,
            )
        };
        assert_eq!(
            stats.take(),
            [
                missing_copy("conversation_control"),
                missing_copy("blocked_viewer"),
                missing_copy("root_author_blocked_viewer"),
                missing_copy("readonly_viewer"),
            ]
        );
    }

    #[test]
    fn a_reason_without_a_known_action_has_no_policy_and_counts_once() {
        const POLICY: &str = r#"
limited_actions_policy:
  parameters:
    limited_actions_policy_limited_actions: {type: array, default: []}
  rules:
  - query: "[$LimitedEngagementReason eq limited_replies]"
    values:
      limited_actions_policy_limited_actions: ["reply"]
  - query: "[$LimitedEngagementReason eq blocked_viewer]"
    values:
      limited_actions_policy_limited_actions: ["no_such_action"]
"#;
        let switches = ClientSwitches::new(Arc::new(ArcSwap::from_pointee(
            FeatureSwitches::load_string(POLICY).unwrap(),
        )));
        let stats = Counted::default();
        let policies = switches.limited_actions_policies(
            None,
            None,
            None,
            [StaleTweet, ConversationControl, BlockedViewer, StaleTweet],
            &LimitedActionsCopy::from_json("[]"),
            Some(&stats),
        );
        assert_eq!(
            policies.actions(ConversationControl),
            [LimitedAction {
                action_type: LimitedActionType::Reply,
                prompt: None,
            }]
        );
        assert!(policies.actions(StaleTweet).is_empty());
        assert!(policies.actions(BlockedViewer).is_empty());
        let missing = |reason: &str, missing: &str| {
            (
                MISSING_POLICY_COUNTER.to_string(),
                format!("reason={reason},missing={missing}"),
                1,
            )
        };
        assert_eq!(
            stats.take(),
            [
                missing("stale_tweet", "actions"),
                missing("conversation_control", "copy"),
                missing("blocked_viewer", "actions"),
            ]
        );
    }
}
