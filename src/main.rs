fn main() {
    if let Err(e) = panlabel::run() {
        eprintln!("Error: {}", e);
        std::process::exit(1);
    }
}
