use anyhow::{ensure, Result};
use thrift::protocol::{TCompactInputProtocol, TSerializable};
use xai_x_thrift::action::Action;

pub const MAX_ACTION_BYTES: usize = 64 * 1024;

#[derive(Debug, PartialEq)]
pub enum EvaluationResult {
    Evaluated(Box<Action>),
    NotEvaluated,
    Failed,
}

pub fn decode_action(bytes: &[u8]) -> Result<Action> {
    ensure!(
        bytes.len() <= MAX_ACTION_BYTES,
        "Action byte limit exceeded"
    );
    let element_size = size_of::<xai_x_thrift::action::MessageLink>()
        .max(size_of::<xai_x_thrift::action::LimitedAction>())
        .max(size_of::<xai_x_thrift::action::TweetVisibilityNudgeAction>());
    let config = thrift::TConfiguration::builder()
        .max_string_size(Some(MAX_ACTION_BYTES))
        .max_message_size(Some(MAX_ACTION_BYTES))
        .max_frame_size(Some(MAX_ACTION_BYTES))
        .max_container_size(Some(MAX_ACTION_BYTES / element_size))
        .build()?;
    let mut cursor = std::io::Cursor::new(bytes);
    let action = Action::read_from_in_protocol(&mut TCompactInputProtocol::with_config(
        &mut cursor,
        config,
    ))?;
    ensure!(
        cursor.position() == bytes.len() as u64,
        "trailing Action bytes"
    );
    ensure!(
        !matches!(action, Action::NotEvaluated(_)),
        "not an evaluated action"
    );
    Ok(action)
}

#[cfg(test)]
mod tests {
    use super::*;
    use xai_x_thrift::action;

    #[test]
    fn declared_lengths_are_rejected_before_reading_payloads() {
        for bytes in [
            &[0x3c, 0x29, 0xf8, 0xa0, 0x8d, 0x06][..],
            &[0x2c, 0x18, 0xc0, 0x84, 0x3d][..],
            &[0x2c, 0x19, 0xf8, 0xa0, 0x8d, 0x06][..],
        ] {
            let error = decode_action(bytes).unwrap_err();
            assert!(
                matches!(
                    error.downcast_ref::<thrift::Error>(),
                    Some(thrift::Error::Protocol(thrift::ProtocolError {
                        kind: thrift::ProtocolErrorKind::SizeLimit,
                        ..
                    }))
                ),
                "expected a size limit before payload read: {error:?}"
            );
        }
    }

    #[test]
    fn decode_action_roundtrips_and_rejects_decode_errors() {
        let drop = Action::Drop(action::Drop::new(
            Some(action::DropReason::LegalDemandsWithheld(true)),
            Some(vec!["US".into(), "DE".into()]),
        ));
        let bytes = xai_x_thrift::serialize_compact(&drop).unwrap();
        assert_eq!(decode_action(&bytes).unwrap(), drop);
        assert_eq!(
            xai_x_thrift::deserialize_binary::<Action>(
                &xai_x_thrift::serialize_binary(&drop).unwrap()
            )
            .unwrap(),
            drop
        );
        assert_eq!(
            decode_action(&[0x2c, 0, 0]).unwrap(),
            Action::Allow(action::Allow::new())
        );
        assert_eq!(
            decode_action(&[0x2c, 0x15, 0x02, 0, 0]).unwrap(),
            Action::Allow(action::Allow::new())
        );
        let avoid = Action::Avoid(action::Avoid::new(
            None,
            None,
            Some(std::collections::BTreeSet::from([Box::new(
                action::BrandSafetyAttributes::IsIASSuitable(true),
            )])),
        ));
        assert_eq!(
            decode_action(&xai_x_thrift::serialize_compact(&avoid).unwrap()).unwrap(),
            avoid
        );
        assert_eq!(
            xai_x_thrift::deserialize_binary::<Action>(
                &xai_x_thrift::serialize_binary(&avoid).unwrap()
            )
            .unwrap(),
            avoid
        );
        for bytes in [
            vec![],
            vec![0x2c],
            vec![0x2c, 0, 0, 0xff],
            vec![0x1c, 0, 0],
            vec![0x0c, 0x90, 0x03, 0, 0],
            vec![0; MAX_ACTION_BYTES + 1],
        ] {
            assert!(decode_action(&bytes).is_err(), "accepted {bytes:?}");
        }
    }
}
