// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 X.AI Corp.
use crate::{ContinuousActionName, PredictNextActionsRequest, PurchaseValueBaseline};

pub const ACTION_INDEX: usize = ContinuousActionName::AdsWebCtPurchaseValue as usize;

pub fn is_valid_baseline(baseline: &PurchaseValueBaseline) -> bool {
    baseline.impression_id > 0
        && baseline.advertiser_account_id > 0
        && baseline.mean_value_usd_28d.is_finite()
        && baseline.mean_value_usd_28d > 0.0
}

pub fn find_baseline(
    request: &PredictNextActionsRequest,
    impression_id: i64,
    advertiser_account_id: i64,
) -> Option<&PurchaseValueBaseline> {
    request.purchase_value_baselines.iter().find(|b| {
        b.impression_id == impression_id
            && b.advertiser_account_id == advertiser_account_id
            && is_valid_baseline(b)
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use prost::Message;

    fn baseline(impression_id: i64, advertiser_account_id: i64) -> PurchaseValueBaseline {
        PurchaseValueBaseline {
            impression_id,
            advertiser_account_id,
            mean_value_usd_28d: 10.0,
        }
    }

    #[test]
    fn slot_five_and_existing_slots_are_stable() {
        assert_eq!(ACTION_INDEX, 5);
        assert_eq!(ContinuousActionName::DwellTime as u32, 1);
        assert_eq!(ContinuousActionName::ClickDwellTime as u32, 2);
        assert_eq!(ContinuousActionName::ActiveSecs5mResidualNorm as u32, 3);
        assert_eq!(ContinuousActionName::BridgeProbability as u32, 4);
    }

    #[test]
    fn old_requests_decode_with_no_baselines() {
        let old = PredictNextActionsRequest {
            return_logits_list: true,
            ..Default::default()
        };
        let decoded = PredictNextActionsRequest::decode(old.encode_to_vec().as_slice()).unwrap();
        assert!(decoded.return_logits_list);
        assert!(decoded.purchase_value_baselines.is_empty());
    }

    #[test]
    fn baselines_and_return_logits_list_use_independent_wire_tags() {
        let request = PredictNextActionsRequest {
            return_logits_list: true,
            purchase_value_baselines: vec![baseline(7, 9)],
            ..Default::default()
        };
        let bytes = request.encode_to_vec();
        assert!(bytes.windows(3).any(|w| w == [0xB0, 0x01, 0x01]));
        assert!(bytes.windows(2).any(|w| w == [0xBA, 0x01]));

        let decoded = PredictNextActionsRequest::decode(bytes.as_slice()).unwrap();
        assert!(decoded.return_logits_list);
        assert_eq!(decoded.purchase_value_baselines, vec![baseline(7, 9)]);
    }

    #[test]
    fn baseline_wire_tags_are_stable() {
        let bytes = baseline(1, 2).encode_to_vec();
        assert_eq!(bytes[0], 0x08);
        assert_eq!(bytes[2], 0x10);
        assert_eq!(bytes[4], 0x19);
        assert_eq!(bytes.len(), 5 + 8);
    }

    #[test]
    fn valid_baseline_requires_positive_ids_and_positive_finite_mean() {
        assert!(is_valid_baseline(&baseline(1, 2)));
        assert!(!is_valid_baseline(&baseline(0, 2)));
        assert!(!is_valid_baseline(&baseline(1, 0)));
        assert!(!is_valid_baseline(&baseline(-1, 2)));
        for mean in [0.0, -1.0, f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            let mut b = baseline(1, 2);
            b.mean_value_usd_28d = mean;
            assert!(!is_valid_baseline(&b), "mean {mean} must be invalid");
        }
    }

    #[test]
    fn find_baseline_matches_identity_not_order_and_skips_invalid() {
        let mut stale = baseline(7, 9);
        stale.mean_value_usd_28d = 0.0;
        let request = PredictNextActionsRequest {
            purchase_value_baselines: vec![baseline(3, 4), stale, baseline(7, 8)],
            ..Default::default()
        };
        assert_eq!(find_baseline(&request, 7, 8), Some(&baseline(7, 8)));
        assert_eq!(find_baseline(&request, 3, 4), Some(&baseline(3, 4)));
        assert_eq!(find_baseline(&request, 7, 9), None);
        assert_eq!(find_baseline(&request, 3, 8), None);
        assert_eq!(
            find_baseline(&PredictNextActionsRequest::default(), 3, 4),
            None
        );
    }
}
