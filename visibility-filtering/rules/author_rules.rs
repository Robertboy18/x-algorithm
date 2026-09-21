use crate::models::AuthorLabel;
use crate::models::VfAction;
use crate::rules::rule_spec::RuleSpec;
use crate::rules::RuleContext;
use xai_visibility_filtering::models::FilteredReason;

pub(super) const AUTHOR_STATE_DROPS: &[RuleSpec] = &[
    RuleSpec::Author {
        name: "SuspendedAuthorRule",
        when: |author| author.is_suspended(),
        reason: FilteredReason::AuthorIsSuspended,
        exempt_follower: false,
    },
    RuleSpec::Author {
        name: "DeactivatedAuthorRule",
        when: |author| author.is_deactivated(),
        reason: FilteredReason::AuthorIsDeactivated,
        exempt_follower: false,
    },
    RuleSpec::Author {
        name: "ErasedAuthorRule",
        when: |author| author.is_erased(),
        reason: FilteredReason::AuthorAccountIsInactive,
        exempt_follower: false,
    },
    RuleSpec::Author {
        name: "OffboardedAuthorRule",
        when: |author| author.is_offboarded(),
        reason: FilteredReason::AuthorAccountIsInactive,
        exempt_follower: false,
    },
    RuleSpec::Author {
        name: "ProtectedAuthorDropRule",
        when: |author| author.is_protected(),
        reason: FilteredReason::AuthorIsProtected,
        exempt_follower: true,
    },
];

pub(super) const OON_NSFW_AUTHOR_DROPS: &[RuleSpec] = &[
    RuleSpec::Author {
        name: "DropNsfwUserAuthorRule",
        when: |author| author.is_nsfw_user(),
        reason: FilteredReason::ContainNsfwMedia,
        exempt_follower: false,
    },
    RuleSpec::Author {
        name: "DropNsfwAdminAuthorRule",
        when: |author| author.is_nsfw_admin(),
        reason: FilteredReason::ContainNsfwMedia,
        exempt_follower: false,
    },
];

pub(super) const OON_USER_LABEL_DROPS: &[RuleSpec] = &[
    RuleSpec::Author {
        name: "NsfwHighRecallUserLabelRule",
        when: |author| author.has_user_label(AuthorLabel::NsfwHighRecall),
        reason: FilteredReason::UnspecifiedReason,
        exempt_follower: false,
    },
    RuleSpec::Author {
        name: "NsfwHighPrecisionUserLabelRule",
        when: |author| author.has_user_label(AuthorLabel::NsfwHighPrecision),
        reason: FilteredReason::UnspecifiedReason,
        exempt_follower: false,
    },
    RuleSpec::Author {
        name: "SpamHighRecallUserLabelRule",
        when: |author| author.has_user_label(AuthorLabel::SpamHighRecall),
        reason: FilteredReason::UnspecifiedReason,
        exempt_follower: false,
    },
    RuleSpec::Author {
        name: "CompromisedUserLabelRule",
        when: |author| author.has_user_label(AuthorLabel::Compromised),
        reason: FilteredReason::UnspecifiedReason,
        exempt_follower: false,
    },
    RuleSpec::Author {
        name: "ReadOnlyUserLabelRule",
        when: |author| author.has_user_label(AuthorLabel::ReadOnly),
        reason: FilteredReason::UnspecifiedReason,
        exempt_follower: false,
    },
    RuleSpec::Author {
        name: "ImpersonationHighPrecisionUserLabelRule",
        when: |author| author.has_user_label(AuthorLabel::ImpersonationHighPrecision),
        reason: FilteredReason::UnspecifiedReason,
        exempt_follower: false,
    },
    RuleSpec::Author {
        name: "NsfwAvatarImageRule",
        when: |author| author.has_user_label(AuthorLabel::NsfwAvatarImage),
        reason: FilteredReason::UnspecifiedReason,
        exempt_follower: false,
    },
    RuleSpec::Author {
        name: "NsfwBannerImageRule",
        when: |author| author.has_user_label(AuthorLabel::NsfwBannerImage),
        reason: FilteredReason::UnspecifiedReason,
        exempt_follower: false,
    },
    RuleSpec::Author {
        name: "AbusiveHighRecallRule",
        when: |author| author.has_user_label(AuthorLabel::AbusiveHighRecall),
        reason: FilteredReason::UnspecifiedReason,
        exempt_follower: true,
    },
    RuleSpec::Author {
        name: "NsfwNearPerfectAuthorRule",
        when: |author| author.has_user_label(AuthorLabel::NsfwNearPerfect),
        reason: FilteredReason::UnspecifiedReason,
        exempt_follower: false,
    },
    RuleSpec::Author {
        name: "DoNotAmplifyNonFollowerRule",
        when: |author| author.has_user_label(AuthorLabel::DoNotAmplify),
        reason: FilteredReason::UnspecifiedReason,
        exempt_follower: true,
    },
];

fn viewer_blocks_author(context: &RuleContext<'_>) -> VfAction {
    if context.viewer().is_logged_out() {
        return VfAction::Allow;
    }
    if context.viewer().blocks_author() {
        return VfAction::Drop(FilteredReason::ViewerBlocksAuthor);
    }
    VfAction::Allow
}

fn viewer_mutes_author(context: &RuleContext<'_>) -> VfAction {
    if context.viewer().is_logged_out() {
        return VfAction::Allow;
    }
    if context.viewer().mutes_author() {
        return VfAction::Drop(FilteredReason::ViewerMutesAuthor);
    }
    VfAction::Allow
}

fn muted_retweets(context: &RuleContext<'_>) -> VfAction {
    if context.viewer().is_logged_out() {
        return VfAction::Allow;
    }
    if context.tweet().is_retweet() && context.viewer().mutes_retweets_from_author() {
        return VfAction::Drop(FilteredReason::UnspecifiedReason);
    }
    VfAction::Allow
}

pub(super) const SOCIALGRAPH_DROPS: &[RuleSpec] = &[
    RuleSpec::Custom {
        name: "ViewerBlocksAuthorRule",
        evaluate: viewer_blocks_author,
    },
    RuleSpec::Custom {
        name: "ViewerMutesAuthorRule",
        evaluate: viewer_mutes_author,
    },
    RuleSpec::Custom {
        name: "MutedRetweetsRule",
        evaluate: muted_retweets,
    },
];

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::{AuthorFeatures, ViewerAuthorRelationship};
    use crate::rules::fixtures::{
        assert_allows, assert_drops, author_viewer, candidate, logged_out_viewer, viewer, VIEWER_ID,
    };

    fn author_flag_features(name: &str) -> AuthorFeatures {
        let mut features = AuthorFeatures::default();
        match name {
            "SuspendedAuthorRule" => features.is_suspended = true,
            "DeactivatedAuthorRule" => features.is_deactivated = true,
            "ErasedAuthorRule" => features.is_erased = true,
            "OffboardedAuthorRule" => features.is_offboarded = true,
            "DropNsfwUserAuthorRule" => features.is_nsfw_user = true,
            "DropNsfwAdminAuthorRule" => features.is_nsfw_admin = true,
            "ProtectedAuthorDropRule" => features.is_protected = true,
            _ => panic!("no trigger flags for rule {name}"),
        }
        features
    }

    #[test]
    fn author_flag_drop_axis() {
        for spec in AUTHOR_STATE_DROPS.iter().chain(OON_NSFW_AUTHOR_DROPS) {
            let RuleSpec::Author {
                name,
                reason,
                exempt_follower,
                ..
            } = spec
            else {
                panic!("{} is not an author drop row", spec.name());
            };
            let firing = candidate()
                .with_author_features(author_flag_features(name))
                .build();
            for v in [viewer(VIEWER_ID), logged_out_viewer()] {
                assert_drops(spec, &v, &firing, reason);
            }
            let followed = candidate()
                .with_author_features(author_flag_features(name))
                .followed()
                .build();
            if *exempt_follower {
                assert_allows(spec, &viewer(VIEWER_ID), &followed);
                assert_drops(spec, &logged_out_viewer(), &followed, reason);
            } else {
                assert_drops(spec, &viewer(VIEWER_ID), &followed, reason);
            }
            let unflagged = candidate().build();
            assert_allows(spec, &viewer(VIEWER_ID), &unflagged);
            assert_allows(spec, &author_viewer(), &firing);
        }
    }

    fn trigger_user_label(name: &str) -> AuthorLabel {
        match name {
            "NsfwHighRecallUserLabelRule" => AuthorLabel::NsfwHighRecall,
            "NsfwHighPrecisionUserLabelRule" => AuthorLabel::NsfwHighPrecision,
            "SpamHighRecallUserLabelRule" => AuthorLabel::SpamHighRecall,
            "CompromisedUserLabelRule" => AuthorLabel::Compromised,
            "ReadOnlyUserLabelRule" => AuthorLabel::ReadOnly,
            "ImpersonationHighPrecisionUserLabelRule" => AuthorLabel::ImpersonationHighPrecision,
            "NsfwAvatarImageRule" => AuthorLabel::NsfwAvatarImage,
            "NsfwBannerImageRule" => AuthorLabel::NsfwBannerImage,
            "AbusiveHighRecallRule" => AuthorLabel::AbusiveHighRecall,
            "NsfwNearPerfectAuthorRule" => AuthorLabel::NsfwNearPerfect,
            "DoNotAmplifyNonFollowerRule" => AuthorLabel::DoNotAmplify,
            _ => panic!("no trigger user label for rule {name}"),
        }
    }

    #[test]
    fn user_label_drop_axis() {
        for spec in OON_USER_LABEL_DROPS {
            let RuleSpec::Author {
                name,
                reason,
                exempt_follower,
                ..
            } = spec
            else {
                panic!("{} is not a user-label drop row", spec.name());
            };
            let firing = candidate()
                .with_author_user_label(trigger_user_label(name))
                .build();
            for v in [viewer(VIEWER_ID), logged_out_viewer()] {
                assert_drops(spec, &v, &firing, reason);
            }
            let followed = candidate()
                .with_author_user_label(trigger_user_label(name))
                .followed()
                .build();
            if *exempt_follower {
                assert_allows(spec, &viewer(VIEWER_ID), &followed);
                assert_drops(spec, &logged_out_viewer(), &followed, reason);
            } else {
                assert_drops(spec, &viewer(VIEWER_ID), &followed, reason);
            }
            let nonmatching = if trigger_user_label(name) == AuthorLabel::Compromised {
                AuthorLabel::ReadOnly
            } else {
                AuthorLabel::Compromised
            };
            let unrelated = candidate().with_author_user_label(nonmatching).build();
            assert_allows(spec, &viewer(VIEWER_ID), &unrelated);
            assert_allows(spec, &author_viewer(), &firing);
        }
    }

    fn relationship_trigger(name: &str) -> (ViewerAuthorRelationship, bool, FilteredReason) {
        match name {
            "ViewerBlocksAuthorRule" => (
                ViewerAuthorRelationship {
                    viewer_blocks_author: true,
                    ..Default::default()
                },
                false,
                FilteredReason::ViewerBlocksAuthor,
            ),
            "ViewerMutesAuthorRule" => (
                ViewerAuthorRelationship {
                    viewer_mutes_author: true,
                    ..Default::default()
                },
                false,
                FilteredReason::ViewerMutesAuthor,
            ),
            "MutedRetweetsRule" => (
                ViewerAuthorRelationship {
                    viewer_mutes_retweets_from_author: true,
                    ..Default::default()
                },
                true,
                FilteredReason::UnspecifiedReason,
            ),
            _ => panic!("no relationship trigger for rule {name}"),
        }
    }

    #[test]
    fn socialgraph_relationship_axis() {
        for spec in SOCIALGRAPH_DROPS {
            let RuleSpec::Custom { name, .. } = spec else {
                panic!("{} is not a custom socialgraph row", spec.name());
            };
            let (rel, retweet, reason) = relationship_trigger(name);
            let mut firing = candidate().with_relationship(rel.clone());
            if retweet {
                firing = firing.retweet_of(99);
            }
            let firing = firing.build();
            assert_drops(spec, &viewer(VIEWER_ID), &firing, &reason);
            assert_allows(spec, &logged_out_viewer(), &firing);
            assert_allows(spec, &viewer(VIEWER_ID), &candidate().build());
            if *name == "MutedRetweetsRule" {
                let non_retweet = candidate().with_relationship(rel).build();
                assert_allows(spec, &viewer(VIEWER_ID), &non_retweet);
            }
        }
    }
}
