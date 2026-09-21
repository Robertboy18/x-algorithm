pub use xai_x_thrift::tweet_safety_label::SafetyLabelType;

use std::collections::HashSet;
use xai_visibility_filtering_proto as vf_pb;

#[derive(Clone, Debug, Default)]
pub struct SafetyLabelMap(HashSet<SafetyLabelType>);

impl SafetyLabelMap {
    #[cfg(test)]
    pub fn new(label_types: HashSet<SafetyLabelType>) -> Self {
        Self(label_types)
    }

    pub fn from_proto_label_types(proto: &vf_pb::SafetyLabelMap) -> Self {
        Self(
            proto
                .labels
                .keys()
                .map(|label_type| SafetyLabelType(*label_type))
                .collect(),
        )
    }

    #[inline]
    pub fn has_label(&self, label_type: SafetyLabelType) -> bool {
        self.0.contains(&label_type)
    }
}
