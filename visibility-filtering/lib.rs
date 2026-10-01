#![cfg_attr(not(test), deny(clippy::allow_attributes))]
#![deny(
    clippy::allow_attributes_without_reason,
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss,
    clippy::dbg_macro,
    clippy::expect_used,
    clippy::indexing_slicing,
    clippy::panic,
    clippy::print_stderr,
    clippy::print_stdout,
    clippy::todo,
    clippy::unimplemented,
    clippy::unwrap_used
)]
#![cfg_attr(
    test,
    allow(
        clippy::cast_possible_truncation,
        clippy::cast_possible_wrap,
        clippy::cast_sign_loss,
        clippy::dbg_macro,
        clippy::expect_used,
        clippy::indexing_slicing,
        clippy::panic,
        clippy::print_stderr,
        clippy::print_stdout,
        clippy::unwrap_used,
        reason = "fixtures may panic and cast freely"
    )
)]

pub(crate) mod caller_identity;
pub(crate) mod clients;
pub mod config;
pub mod dark_traffic_setup;
pub(crate) mod evaluate_tweets;
pub(crate) mod filter;
pub(crate) mod filter_tweets;
pub(crate) mod get_safety_labels;
pub(crate) mod hydration;
pub(crate) mod limited_actions_copy;
pub(crate) mod models;
pub mod params;
pub mod reference;
pub(crate) mod reference_compare;
pub(crate) mod retweet;
pub(crate) mod rules;
pub(crate) mod safety_label_source;
pub mod server;
pub(crate) mod server_deps;
pub(crate) mod treatment;
