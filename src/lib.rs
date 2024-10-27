use clap::Command;
use std::env;

pub fn run() -> Result<(), Box<dyn std::error::Error>> {
    let _matches = Command::new("panlabel")
        .version(env!("CARGO_PKG_VERSION"))
        .author(env!("CARGO_PKG_AUTHORS"))
        .about(env!("CARGO_PKG_DESCRIPTION"))
        .get_matches();

    Ok(())
}
