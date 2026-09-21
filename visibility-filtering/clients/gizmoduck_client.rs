use std::collections::HashMap;
use std::sync::Arc;
use xai_core_entities::entities::GizmoduckUserResult;
use xai_core_entities::gizmoduck_client::{GizmoduckClient, LookupContext, QueryFields};

pub struct GizmoduckLookup {
    inner: Arc<dyn GizmoduckClient + Send + Sync>,
}

fn author_hydration_lookup_context() -> LookupContext {
    LookupContext {
        for_user_id: None,
        include_deactivated: true,
        include_failed: true,
        include_erased: true,
        include_no_screen_name_users: true,
        include_offboarded: true,
        ..Default::default()
    }
}

impl GizmoduckLookup {
    pub fn new(inner: Arc<dyn GizmoduckClient + Send + Sync>) -> Self {
        Self { inner }
    }

    pub async fn get_users(
        &self,
        user_ids: Vec<u64>,
        fields: &[QueryFields],
    ) -> HashMap<u64, anyhow::Result<Option<GizmoduckUserResult>>> {
        let ids: Vec<i64> = user_ids.into_iter().map(|id| id as i64).collect();
        self.inner
            .get_users_with_context(ids, Some(author_hydration_lookup_context()), fields)
            .await
            .into_iter()
            .map(|(id, result)| (id as u64, result))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn author_lookup_context_sets_include_flags() {
        let ctx = author_hydration_lookup_context();
        assert_eq!(ctx.for_user_id, None);
        assert!(ctx.include_deactivated);
        assert!(ctx.include_failed);
        assert!(ctx.include_erased);
        assert!(ctx.include_no_screen_name_users);
        assert!(ctx.include_offboarded);
        assert!(!ctx.include_soft_users);
        assert!(!ctx.include_login_email);
        assert!(ctx.filter_perspectival.is_empty());
    }
}
