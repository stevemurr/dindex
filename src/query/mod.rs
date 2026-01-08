//! Query coordination and result aggregation
//!
//! Handles distributed queries across P2P network

mod coordinator;
mod executor;

pub use coordinator::*;
pub use executor::*;
