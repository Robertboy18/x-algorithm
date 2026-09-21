use crate::models::VfAction;
use crate::rules::context::{AuthorPredicates, TweetPredicates};
use crate::rules::RuleContext;
use xai_visibility_filtering::models::FilteredReason;

pub(super) enum RuleSpec {
    Tweet {
        name: &'static str,
        when: fn(TweetPredicates<'_>) -> bool,
        action: RuleAction,
        exempt_author: bool,
    },
    Author {
        name: &'static str,
        when: fn(AuthorPredicates<'_>) -> bool,
        reason: FilteredReason,
        exempt_follower: bool,
    },
    Custom {
        name: &'static str,
        evaluate: fn(&RuleContext<'_>) -> VfAction,
    },
}

pub(super) enum RuleAction {
    Drop(FilteredReason),
    SensitiveMediaInterstitial(FilteredReason),
}

impl RuleSpec {
    pub(super) fn name(&self) -> &'static str {
        match self {
            RuleSpec::Tweet { name, .. }
            | RuleSpec::Author { name, .. }
            | RuleSpec::Custom { name, .. } => name,
        }
    }

    pub(super) fn evaluate(&self, context: &RuleContext<'_>) -> VfAction {
        match self {
            RuleSpec::Tweet {
                when,
                action,
                exempt_author,
                ..
            } => {
                if !when(context.tweet()) {
                    return VfAction::Allow;
                }
                if *exempt_author && context.viewer().is_author() {
                    return VfAction::Allow;
                }
                match action {
                    RuleAction::Drop(reason) => VfAction::Drop(reason.clone()),
                    RuleAction::SensitiveMediaInterstitial(reason) => {
                        if context.viewer().allows_sensitive_media() {
                            VfAction::Allow
                        } else {
                            VfAction::Interstitial(reason.clone())
                        }
                    }
                }
            }
            RuleSpec::Author {
                when,
                reason,
                exempt_follower,
                ..
            } => {
                if !when(context.author()) {
                    return VfAction::Allow;
                }
                if context.viewer().is_author() {
                    return VfAction::Allow;
                }
                if *exempt_follower
                    && !context.viewer().is_logged_out()
                    && context.viewer().follows_author()
                {
                    return VfAction::Allow;
                }
                VfAction::Drop(reason.clone())
            }
            RuleSpec::Custom { evaluate, .. } => evaluate(context),
        }
    }
}
