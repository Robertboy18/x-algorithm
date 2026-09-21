#![deny(
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
        clippy::dbg_macro,
        clippy::expect_used,
        clippy::indexing_slicing,
        clippy::panic,
        clippy::print_stderr,
        clippy::print_stdout,
        clippy::unwrap_used
    )
)]

pub(crate) mod clients;
pub(crate) mod clock_cache;
pub mod config;
pub mod dark_traffic_setup;
pub(crate) mod evaluate_tweets;
pub(crate) mod filter;
pub(crate) mod filter_tweets;
pub(crate) mod get_safety_labels;
pub(crate) mod hydration;
pub(crate) mod models;
pub mod params;
pub(crate) mod reference_compare;
pub(crate) mod rules;
pub(crate) mod safety_label_source;
pub mod server;
pub(crate) mod server_deps;
