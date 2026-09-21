use std::collections::HashMap;

use xai_visibility_filtering_proto as vf_pb;
use xai_x_thrift::tweet_safety_label::{SafetyLabel, SafetyLabelSource};

pub(crate) fn label_to_proto(label: SafetyLabel) -> vf_pb::SafetyLabel {
    let safety_label_source = label.safety_label_source.and_then(|src| match src {
        SafetyLabelSource::BotMakerAction(a) => {
            Some(vf_pb::safety_label::SafetyLabelSource::BotmakerAction(
                vf_pb::BotmakerAction { rule_id: a.rule_id },
            ))
        }
        SafetyLabelSource::ToolAction(a) => Some(
            vf_pb::safety_label::SafetyLabelSource::ToolAction(vf_pb::ToolAction {
                agent_tool: i32::from(a.agent_tool),
                actor_ldap: a.actor_ldap,
            }),
        ),
        SafetyLabelSource::GrokAnnotationAction(a) => Some(
            vf_pb::safety_label::SafetyLabelSource::GrokAnnotationAction(
                vf_pb::GrokAnnotationAction {
                    source: i32::from(a.grok_annotation_source),
                },
            ),
        ),
        _ => None,
    });

    vf_pb::SafetyLabel {
        score: label.score.map(|f| f.into_inner()),
        applicable_users: label.applicable_users.map_or_else(Vec::new, |users| {
            users.into_iter().map(|u| u.user_id).collect()
        }),
        holdback_experiment: label.holdback_experiment,
        source: label.source,
        created_at_msec: label.created_at_msec,
        expires_at_msec: label.expires_at_msec,
        applicable_countries: label.applicable_countries.unwrap_or_default(),
        safety_label_source,
    }
}

pub(crate) fn label_map_to_proto(
    labels: xai_safety_label_store::types::SafetyLabelMap,
) -> vf_pb::SafetyLabelMap {
    let map: HashMap<i32, vf_pb::SafetyLabel> = labels
        .into_iter()
        .map(|(lt, label)| (i32::from(lt), label_to_proto(label)))
        .collect();
    vf_pb::SafetyLabelMap { labels: map }
}

#[cfg(test)]
mod tests {
    use super::*;
    use xai_x_thrift::tweet_safety_label::{
        AgentTool, BotMakerAction, GraphId, GrokAnnotationAction, GrokAnnotationSource,
        StreamertailAction, ToolAction,
    };

    fn empty_label(src: SafetyLabelSource) -> SafetyLabel {
        SafetyLabel::new(None, None, None, None, Some(src), None, None, None)
    }

    #[test]
    fn botmaker_action_to_proto() {
        let label = empty_label(SafetyLabelSource::BotMakerAction(BotMakerAction::new(42)));
        let proto = label_to_proto(label);
        assert_eq!(
            proto.safety_label_source,
            Some(vf_pb::safety_label::SafetyLabelSource::BotmakerAction(
                vf_pb::BotmakerAction { rule_id: 42 }
            ))
        );
    }

    #[test]
    fn tool_action_to_proto() {
        let label = empty_label(SafetyLabelSource::ToolAction(ToolAction::new(
            AgentTool::BME,
            "someone".to_string(),
        )));
        let proto = label_to_proto(label);
        assert_eq!(
            proto.safety_label_source,
            Some(vf_pb::safety_label::SafetyLabelSource::ToolAction(
                vf_pb::ToolAction {
                    agent_tool: 15,
                    actor_ldap: "someone".to_string(),
                }
            ))
        );
    }

    #[test]
    fn grok_annotation_action_to_proto() {
        let label = empty_label(SafetyLabelSource::GrokAnnotationAction(
            GrokAnnotationAction::new(GrokAnnotationSource::GROX_PTOS),
        ));
        let proto = label_to_proto(label);
        assert_eq!(
            proto.safety_label_source,
            Some(
                vf_pb::safety_label::SafetyLabelSource::GrokAnnotationAction(
                    vf_pb::GrokAnnotationAction { source: 1 }
                )
            )
        );
    }

    #[test]
    fn unmapped_variant_produces_none() {
        let label = empty_label(SafetyLabelSource::StreamertailAction(
            StreamertailAction::new(GraphId::TEST),
        ));
        let proto = label_to_proto(label);
        assert_eq!(proto.safety_label_source, None);
    }

    #[test]
    fn no_source_produces_none() {
        let label = SafetyLabel::new(None, None, None, None, None, None, None, None);
        let proto = label_to_proto(label);
        assert_eq!(proto.safety_label_source, None);
    }
}
