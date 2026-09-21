use crate::models::ViewerAuthorRelationship;
use std::collections::{HashMap, HashSet};
use tonic::async_trait;
use tracing::warn;
use xai_flock_client::FlockClient;
use xai_flock_proto::{
    EdgeState, LongList, Page, QueryTerm, Results, SelectOperation, SelectOperationType,
    SelectQuery, SelectRequest,
};

const FOLLOWS_GRAPH_ID: i32 = 1;
const BLOCKS_GRAPH_ID: i32 = 3;
const MUTE_GRAPH_ID: i32 = 23;
const MUTE_RETWEETS_GRAPH_ID: i32 = 10;
const SUPER_FOLLOWS_GRAPH_ID: i32 = 55;

#[async_trait]
pub trait SocialgraphClient: Send + Sync {
    async fn batch_check_relationships(
        &self,
        viewer_id: u64,
        author_ids: &[u64],
    ) -> HashMap<u64, ViewerAuthorRelationship>;

    async fn batch_check_super_follows(
        &self,
        viewer_id: u64,
        author_ids: &[u64],
    ) -> Option<HashMap<u64, bool>>;
}

#[cfg(test)]
pub struct FakeSocialgraphClient;

#[cfg(test)]
#[async_trait]
impl SocialgraphClient for FakeSocialgraphClient {
    async fn batch_check_relationships(
        &self,
        _viewer_id: u64,
        author_ids: &[u64],
    ) -> HashMap<u64, ViewerAuthorRelationship> {
        author_ids
            .iter()
            .map(|&author_id| (author_id, ViewerAuthorRelationship::default()))
            .collect()
    }

    async fn batch_check_super_follows(
        &self,
        _viewer_id: u64,
        author_ids: &[u64],
    ) -> Option<HashMap<u64, bool>> {
        Some(
            author_ids
                .iter()
                .map(|&author_id| (author_id, false))
                .collect(),
        )
    }
}

fn decode_packed_ids(packed: &[u8]) -> HashSet<u64> {
    let (chunks, _remainder) = packed.as_chunks::<8>();
    chunks
        .iter()
        .map(|&arr| i64::from_le_bytes(arr) as u64)
        .collect()
}

fn edge_membership_query(source_id: u64, graph_id: i32, destination_ids: &[i64]) -> SelectQuery {
    SelectQuery {
        operations: vec![SelectOperation {
            operation_type: SelectOperationType::SimpleQuery as i32,
            term: Some(QueryTerm {
                source_id: source_id as i64,
                graph_id,
                is_forward: true,
                destination_ids: Some(LongList {
                    ids: destination_ids.to_vec(),
                }),
                state_ids: vec![EdgeState::Positive as i32],
                size_hint: None,
                cursor_hint: None,
            }),
        }],
        page: Some(Page {
            count: destination_ids.len() as i32,
            cursor: -1,
        }),
    }
}

#[derive(Debug, Default, PartialEq, Eq)]
struct RelationshipEdges {
    follows: HashSet<u64>,
    blocks: HashSet<u64>,
    mutes: HashSet<u64>,
    mute_retweets: HashSet<u64>,
}

fn relationship_select_request(viewer_id: u64, destination_ids: &[i64]) -> SelectRequest {
    SelectRequest {
        queries: vec![
            edge_membership_query(viewer_id, FOLLOWS_GRAPH_ID, destination_ids),
            edge_membership_query(viewer_id, BLOCKS_GRAPH_ID, destination_ids),
            edge_membership_query(viewer_id, MUTE_GRAPH_ID, destination_ids),
            edge_membership_query(viewer_id, MUTE_RETWEETS_GRAPH_ID, destination_ids),
        ],
        ancestor_client_id: None,
        service_account: None,
        quota_name: None,
    }
}

fn next_edge_set(results: &mut impl Iterator<Item = Results>) -> HashSet<u64> {
    results
        .next()
        .map(|r| decode_packed_ids(&r.ids))
        .unwrap_or_default()
}

fn decode_relationship_edges(results: impl IntoIterator<Item = Results>) -> RelationshipEdges {
    let mut results = results.into_iter();
    RelationshipEdges {
        follows: next_edge_set(&mut results),
        blocks: next_edge_set(&mut results),
        mutes: next_edge_set(&mut results),
        mute_retweets: next_edge_set(&mut results),
    }
}

async fn select_edge_set(
    client: &FlockClient,
    query: SelectQuery,
    label: &'static str,
) -> Option<HashSet<u64>> {
    let request = SelectRequest {
        queries: vec![query],
        ancestor_client_id: None,
        service_account: None,
        quota_name: None,
    };
    match client.inner().clone().select(request).await {
        Ok(resp) => Some(
            resp.into_inner()
                .results
                .first()
                .map(|r| decode_packed_ids(&r.ids))
                .unwrap_or_default(),
        ),
        Err(e) => {
            warn!(error = %e, label, "FlockDB select failed, defaulting to empty set");
            None
        }
    }
}

pub struct ProdSocialgraphClient {
    flock_client: FlockClient,
}

impl ProdSocialgraphClient {
    pub async fn new(
        datacenter: &str,
        ca_cert_path: &str,
        client_cert_path: &str,
        client_key_path: &str,
    ) -> anyhow::Result<Self> {
        let flock_client =
            FlockClient::from_s2s(datacenter, ca_cert_path, client_cert_path, client_key_path)
                .await?;
        Ok(Self { flock_client })
    }
}

#[async_trait]
impl SocialgraphClient for ProdSocialgraphClient {
    async fn batch_check_relationships(
        &self,
        viewer_id: u64,
        author_ids: &[u64],
    ) -> HashMap<u64, ViewerAuthorRelationship> {
        if author_ids.is_empty() {
            return HashMap::new();
        }

        let dest_ids: Vec<i64> = author_ids.iter().map(|&id| id as i64).collect();
        let request = relationship_select_request(viewer_id, &dest_ids);

        let edges = match self.flock_client.inner().clone().select(request).await {
            Ok(resp) => decode_relationship_edges(resp.into_inner().results),
            Err(e) => {
                warn!(
                    error = %e,
                    "FlockDB multi-query select failed, returning no relationships"
                );
                return HashMap::new();
            }
        };

        author_ids
            .iter()
            .map(|&author_id| {
                (
                    author_id,
                    ViewerAuthorRelationship {
                        viewer_follows_author: edges.follows.contains(&author_id),
                        viewer_blocks_author: edges.blocks.contains(&author_id),
                        viewer_mutes_author: edges.mutes.contains(&author_id),
                        viewer_mutes_retweets_from_author: edges.mute_retweets.contains(&author_id),
                    },
                )
            })
            .collect()
    }

    async fn batch_check_super_follows(
        &self,
        viewer_id: u64,
        author_ids: &[u64],
    ) -> Option<HashMap<u64, bool>> {
        let dest_ids: Vec<i64> = author_ids.iter().map(|&id| id as i64).collect();

        let super_set = select_edge_set(
            &self.flock_client,
            edge_membership_query(viewer_id, SUPER_FOLLOWS_GRAPH_ID, &dest_ids),
            "super_follows",
        )
        .await?;

        Some(
            author_ids
                .iter()
                .map(|&author_id| (author_id, super_set.contains(&author_id)))
                .collect(),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decode_packed_ids_empty() {
        assert!(decode_packed_ids(&[]).is_empty());
    }

    #[test]
    fn decode_packed_ids_single() {
        let id: i64 = 42;
        let packed = id.to_le_bytes();
        let result = decode_packed_ids(&packed);
        assert_eq!(result.len(), 1);
        assert!(result.contains(&42u64));
    }

    #[test]
    fn decode_packed_ids_multiple() {
        let ids: Vec<i64> = vec![100, 200, 300];
        let packed: Vec<u8> = ids.iter().flat_map(|id| id.to_le_bytes()).collect();
        let result = decode_packed_ids(&packed);
        assert_eq!(result.len(), 3);
        assert!(result.contains(&100u64));
        assert!(result.contains(&200u64));
        assert!(result.contains(&300u64));
    }

    #[test]
    fn decode_packed_ids_ignores_trailing_bytes() {
        let mut packed = 42i64.to_le_bytes().to_vec();
        packed.extend_from_slice(&[0xff, 0xff]);
        let result = decode_packed_ids(&packed);
        assert_eq!(result.len(), 1);
        assert!(result.contains(&42u64));
    }

    #[test]
    fn relationship_select_request_packs_four_graphs_in_order() {
        let dest = vec![10i64, 20];
        let request = relationship_select_request(999, &dest);
        assert_eq!(request.queries.len(), 4);
        let expected_graphs = [
            FOLLOWS_GRAPH_ID,
            BLOCKS_GRAPH_ID,
            MUTE_GRAPH_ID,
            MUTE_RETWEETS_GRAPH_ID,
        ];
        for (query, &graph_id) in request.queries.iter().zip(expected_graphs.iter()) {
            let term = query.operations[0].term.as_ref().unwrap();
            assert_eq!(term.source_id, 999);
            assert_eq!(term.graph_id, graph_id);
            assert_eq!(term.destination_ids.as_ref().unwrap().ids, dest);
        }
    }

    #[test]
    fn decode_relationship_edges_by_named_fields() {
        let pack = |ids: &[i64]| -> Results {
            Results {
                ids: ids.iter().flat_map(|id| id.to_le_bytes()).collect(),
                next_cursor: 0,
                prev_cursor: 0,
            }
        };
        let edges =
            decode_relationship_edges([pack(&[1, 2]), pack(&[3]), pack(&[]), pack(&[4, 5, 6])]);
        assert!(edges.follows.contains(&1) && edges.follows.contains(&2));
        assert!(edges.blocks.contains(&3));
        assert!(edges.mutes.is_empty());
        assert_eq!(edges.mute_retweets.len(), 3);
        assert!(edges.mute_retweets.contains(&4) && edges.mute_retweets.contains(&6));
    }

    #[test]
    fn decode_relationship_edges_missing_slots_fail_open() {
        let pack = |ids: &[i64]| -> Results {
            Results {
                ids: ids.iter().flat_map(|id| id.to_le_bytes()).collect(),
                next_cursor: 0,
                prev_cursor: 0,
            }
        };
        let edges = decode_relationship_edges([pack(&[1])]);
        assert_eq!(
            edges,
            RelationshipEdges {
                follows: HashSet::from([1]),
                ..RelationshipEdges::default()
            }
        );
    }
}
